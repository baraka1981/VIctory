use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use objc2::rc::autoreleasepool;
use objc2_metal::MTLBuffer;
use victory_host::{Barrier, ComputeDesc, Device, Point, Recorder, ShaderCode};
use victory_metal::{MetalDevice, MetalPipeline, MetalRecorder, MetalResource};
use victory_wine::{
    Command, Committed, ComputePipeline, Contents, GpuAddress, Message, Open, Reader, Release,
    Submit, code, kind, memory_of, op, status,
};

type Entry = extern "C" fn(*mut c_void) -> u32;

/// Wine's unixlib call table: indexed by `victory_wine::op`.
#[unsafe(no_mangle)]
#[used]
pub static __wine_unix_call_funcs: [Entry; op::COUNT] = [
    open,
    release,
    committed,
    contents,
    gpu_address,
    compute_pipeline,
    submit,
];

/// Runs `f` on the block at `args` inside an autorelease pool; a panic
/// becomes `STATUS_UNSUCCESSFUL` rather than unwinding into Wine.
fn call<T>(args: *mut c_void, f: impl FnOnce(&mut T) -> u32) -> u32 {
    if args.is_null() {
        return status::INVALID;
    }
    // SAFETY: the Windows side passes the block this entry reads (same
    // process, `repr(C)` on both sides).
    let a = unsafe { &mut *args.cast::<T>() };
    catch_unwind(AssertUnwindSafe(|| autoreleasepool(|_| f(a)))).unwrap_or(status::FAILED)
}

fn fail(m: &mut Message, e: impl std::fmt::Display) -> u32 {
    m.set(&e.to_string());
    status::FAILED
}

// Handles are boxed objects' addresses, owned by the Windows side until
// released. SAFETY (for the three below): a handle the Windows side got
// from this library and has not released.
fn device<'a>(h: u64) -> Option<&'a MetalDevice> {
    unsafe { (h as *const MetalDevice).as_ref() }
}

fn resource<'a>(h: u64) -> Option<&'a MetalResource> {
    unsafe { (h as *const MetalResource).as_ref() }
}

fn pipeline<'a>(h: u64) -> Option<&'a MetalPipeline> {
    unsafe { (h as *const MetalPipeline).as_ref() }
}

fn handle<T>(v: T) -> u64 {
    Box::into_raw(Box::new(v)) as u64
}

/// Bytes the Windows side points at (empty for a null pointer).
///
/// # Safety
/// `ptr` is valid for `len` bytes for the call's length.
unsafe fn bytes<'a>(ptr: u64, len: u64) -> &'a [u8] {
    if ptr == 0 || len == 0 {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) }
}

extern "C" fn open(args: *mut c_void) -> u32 {
    call(args, |a: &mut Open| match MetalDevice::system_default() {
        Ok(d) => {
            a.device = handle(d);
            status::SUCCESS
        }
        Err(e) => fail(&mut a.error, e),
    })
}

extern "C" fn release(args: *mut c_void) -> u32 {
    call(args, |a: &mut Release| {
        if a.handle == 0 {
            return status::SUCCESS;
        }
        // SAFETY: a live handle of this kind, released once.
        unsafe {
            match a.kind {
                kind::DEVICE => drop(Box::from_raw(a.handle as *mut MetalDevice)),
                kind::RESOURCE => drop(Box::from_raw(a.handle as *mut MetalResource)),
                kind::PIPELINE => drop(Box::from_raw(a.handle as *mut MetalPipeline)),
                _ => return status::INVALID,
            }
        }
        status::SUCCESS
    })
}

extern "C" fn committed(args: *mut c_void) -> u32 {
    call(args, |a: &mut Committed| {
        let (Some(d), Some(memory)) = (device(a.device), memory_of(a.memory)) else {
            return status::INVALID;
        };
        match d.committed(memory, &a.desc, a.state, None) {
            Ok(r) => {
                a.resource = handle(r);
                status::SUCCESS
            }
            Err(e) => fail(&mut a.error, e),
        }
    })
}

extern "C" fn contents(args: *mut c_void) -> u32 {
    call(args, |a: &mut Contents| {
        let Some(MetalResource::Buffer(b)) = resource(a.resource) else {
            return status::INVALID;
        };
        a.ptr = b.contents().as_ptr() as u64;
        a.len = b.length() as u64;
        status::SUCCESS
    })
}

extern "C" fn gpu_address(args: *mut c_void) -> u32 {
    call(args, |a: &mut GpuAddress| {
        // Buffers only, as on D3D12.
        let Some(MetalResource::Buffer(b)) = resource(a.resource) else {
            return status::INVALID;
        };
        a.address = b.gpuAddress();
        status::SUCCESS
    })
}

extern "C" fn compute_pipeline(args: *mut c_void) -> u32 {
    call(args, |a: &mut ComputePipeline| {
        let Some(d) = device(a.device) else {
            return status::INVALID;
        };
        // SAFETY: the Windows side's buffers, alive for the call.
        let (code, entry, root) = unsafe {
            (
                bytes(a.code, a.code_len),
                bytes(a.entry, a.entry_len),
                bytes(a.root, a.root_len),
            )
        };
        let Ok(entry) = std::str::from_utf8(entry) else {
            return status::INVALID;
        };
        let code = match a.code_kind {
            code::DXIL => ShaderCode::Dxil(code),
            code::MSL => match std::str::from_utf8(code) {
                Ok(s) => ShaderCode::Msl(s),
                Err(_) => return status::INVALID,
            },
            code::METALLIB => ShaderCode::MetalLib(code),
            _ => return status::INVALID,
        };
        let desc = ComputeDesc {
            code,
            entry,
            root_signature: root,
            threads: a.threads,
            cache: None,
        };
        match d.compute_pipeline(&desc) {
            Ok(p) => {
                a.pipeline = handle(p);
                status::SUCCESS
            }
            Err(e) => fail(&mut a.error, e),
        }
    })
}

fn point(p: u32) -> Option<Point> {
    match p {
        0 => Some(Point::Compute),
        1 => Some(Point::Graphics),
        _ => None,
    }
}

/// One command onto `rec`; `Err` names what was wrong with it.
fn replay(rec: &MetalRecorder, c: &Command) -> Result<(), &'static str> {
    // SAFETY: the Windows side's live handles and addresses; recording on
    // one thread, the recorder's own.
    unsafe {
        match c {
            Command::SetComputeRoot(p) => rec.set_compute_root(pipeline(*p).ok_or("no pipeline")?),
            Command::SetComputePipeline(p) => {
                rec.set_compute_pipeline(pipeline(*p).ok_or("no pipeline")?)
            }
            Command::SetRootConstants {
                point: p,
                param,
                offset,
                values,
            } => rec.set_root_constants(point(*p).ok_or("bad point")?, *param, *offset, values),
            Command::SetRootView {
                point: p,
                param,
                kind,
                address,
            } => rec.set_root_view(point(*p).ok_or("bad point")?, *param, *kind, *address),
            Command::SetRootTable {
                point: p,
                param,
                handle,
            } => rec.set_root_table(point(*p).ok_or("bad point")?, *param, *handle),
            Command::Dispatch(groups) => rec.dispatch_groups(*groups),
            Command::BarrierUnordered => rec.barriers(&[Barrier::Unordered(None)]),
            Command::CopyBuffer {
                to,
                to_offset,
                from,
                from_offset,
                bytes,
            } => rec.copy_buffer(
                resource(*to).ok_or("no resource")?,
                *to_offset,
                resource(*from).ok_or("no resource")?,
                *from_offset,
                *bytes,
            ),
        }
    }
    Ok(())
}

extern "C" fn submit(args: *mut c_void) -> u32 {
    call(args, |a: &mut Submit| {
        let Some(d) = device(a.device) else {
            return status::INVALID;
        };
        let rec = match d.recorder() {
            Ok(r) => r,
            Err(e) => return fail(&mut a.error, e),
        };
        // SAFETY: the Windows side's stream, alive for the call.
        let stream = unsafe { bytes(a.stream, a.len) };
        for (n, c) in Reader::new(stream).enumerate() {
            let result = match c {
                Ok(c) => replay(&rec, &c),
                Err(at) => {
                    return fail(&mut a.error, format!("unreadable command {n} at byte {at}"));
                }
            };
            if let Err(why) = result {
                return fail(&mut a.error, format!("command {n}: {why}"));
            }
        }
        if a.wait != 0 {
            if let Err(e) = rec.commit_and_wait() {
                return fail(&mut a.error, e);
            }
        } else {
            rec.commit();
        }
        status::SUCCESS
    })
}
