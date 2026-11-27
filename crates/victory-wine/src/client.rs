//! The Windows side: finds `victory_metal.dll`'s Unix library and calls
//! into it.

use std::ffi::c_void;

use victory_host::{Memory, ResourceDesc};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};

use crate::{
    Committed, ComputePipeline, Contents, GpuAddress, Message, Open, Release, Stream, Submit, code,
    memory_code, op, status,
};

type UnixCall = unsafe extern "system" fn(u64, u32, *mut c_void) -> u32;
type QueryMemory =
    unsafe extern "system" fn(isize, *const c_void, u32, *mut c_void, usize, *mut usize) -> i32;

/// `MemoryWineUnixFuncs`: the information class that hands out a builtin
/// module's Unix call table (Wine 7+).
const MEMORY_WINE_UNIX_FUNCS: u32 = 1000;

/// How ntdll takes Unix calls.
#[derive(Clone, Copy)]
enum Call {
    /// Wine 7.x: `__wine_unix_call`, exported.
    Direct(UnixCall),
    /// Wine 8+: `__wine_unix_call_dispatcher`, a pointer to the function.
    Dispatcher(*const UnixCall),
}

/// The bridge, loaded.
pub struct Wine {
    table: u64,
    call: Call,
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

/// An error from a block's message (or a status without one).
fn failed(status: u32, message: &Message, what: &str) -> String {
    let text = message.get();
    if text.is_empty() {
        format!("{what}: status {status:#x}")
    } else {
        format!("{what}: {text}")
    }
}

impl Wine {
    /// Loads `victory_metal.dll` (found as a builtin through
    /// `WINEDLLPATH`) and its Unix library.
    pub fn load() -> Result<Wine, String> {
        // SAFETY: plain Win32 calls with NUL-ended names; the functions
        // got are called with their documented signatures.
        unsafe {
            let dll = LoadLibraryW(wide("victory_metal.dll").as_ptr());
            if dll.is_null() {
                return Err(
                    "victory_metal.dll not found (is WINEDLLPATH set to the bridge's folder?)"
                        .into(),
                );
            }
            let ntdll = GetModuleHandleW(wide("ntdll.dll").as_ptr());
            let query = GetProcAddress(ntdll, c"NtQueryVirtualMemory".as_ptr().cast())
                .ok_or("ntdll has no NtQueryVirtualMemory")?;
            let query: QueryMemory = std::mem::transmute(query);
            let mut table = 0u64;
            let got = query(
                -1,
                dll,
                MEMORY_WINE_UNIX_FUNCS,
                std::ptr::from_mut(&mut table).cast(),
                8,
                std::ptr::null_mut(),
            );
            if got != 0 || table == 0 {
                return Err(format!(
                    "no Unix library for victory_metal.dll (status {got:#x}): not under Wine, the DLL not marked builtin, or victory_metal.so missing"
                ));
            }
            let call = if let Some(f) = GetProcAddress(ntdll, c"__wine_unix_call".as_ptr().cast()) {
                Call::Direct(std::mem::transmute::<
                    unsafe extern "system" fn() -> isize,
                    UnixCall,
                >(f))
            } else if let Some(p) =
                GetProcAddress(ntdll, c"__wine_unix_call_dispatcher".as_ptr().cast())
            {
                Call::Dispatcher(p as *const UnixCall)
            } else {
                return Err(
                    "ntdll exports neither __wine_unix_call nor __wine_unix_call_dispatcher".into(),
                );
            };
            Ok(Wine { table, call })
        }
    }

    /// One Unix call with `args` as its block.
    fn call<T>(&self, code: u32, args: &mut T) -> u32 {
        let args = std::ptr::from_mut(args).cast();
        // SAFETY: the table is victory_metal.so's, `code` one of its
        // entries, and `args` the block that entry reads.
        unsafe {
            match self.call {
                Call::Direct(f) => f(self.table, code, args),
                Call::Dispatcher(p) => (*p)(self.table, code, args),
            }
        }
    }

    /// The system's Metal device.
    pub fn open(&self) -> Result<u64, String> {
        let mut a = Open::default();
        match self.call(op::OPEN, &mut a) {
            status::SUCCESS => Ok(a.device),
            s => Err(failed(s, &a.error, "opening the Metal device")),
        }
    }

    /// Frees an object of [`crate::kind`] `kind`.
    pub fn release(&self, kind: u32, handle: u64) {
        let mut a = Release { kind, handle };
        self.call(op::RELEASE, &mut a);
    }

    pub fn committed(
        &self,
        device: u64,
        memory: Memory,
        desc: &ResourceDesc,
        state: u32,
    ) -> Result<u64, String> {
        let mut a = Committed {
            device,
            memory: memory_code(memory),
            state,
            desc: *desc,
            resource: 0,
            error: Message::default(),
        };
        match self.call(op::COMMITTED, &mut a) {
            status::SUCCESS => Ok(a.resource),
            s => Err(failed(s, &a.error, "a committed resource")),
        }
    }

    /// A CPU-visible buffer's contents, in this process.
    ///
    /// # Safety
    /// The slice aliases the buffer: no other slice of it may be alive,
    /// and the GPU must not be writing it meanwhile; the resource outlives
    /// the slice.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn contents(&self, resource: u64) -> Option<&mut [u8]> {
        let mut a = Contents {
            resource,
            ..Default::default()
        };
        self.call(op::CONTENTS, &mut a);
        // SAFETY: the Unix side's buffer contents, same address space; the
        // caller's aliasing and lifetime guarantees.
        (a.ptr != 0)
            .then(|| unsafe { std::slice::from_raw_parts_mut(a.ptr as *mut u8, a.len as usize) })
    }

    pub fn gpu_address(&self, resource: u64) -> u64 {
        let mut a = GpuAddress {
            resource,
            address: 0,
        };
        self.call(op::GPU_ADDRESS, &mut a);
        a.address
    }

    /// A compute pipeline from DXIL (`root`: a serialized root signature,
    /// empty for the embedded one).
    pub fn compute_pipeline_dxil(
        &self,
        device: u64,
        dxil: &[u8],
        entry: &str,
        root: &[u8],
    ) -> Result<u64, String> {
        let mut a = ComputePipeline {
            device,
            code_kind: code::DXIL,
            code: dxil.as_ptr() as u64,
            code_len: dxil.len() as u64,
            entry: entry.as_ptr() as u64,
            entry_len: entry.len() as u64,
            root: root.as_ptr() as u64,
            root_len: root.len() as u64,
            ..Default::default()
        };
        match self.call(op::COMPUTE_PIPELINE, &mut a) {
            status::SUCCESS => Ok(a.pipeline),
            s => Err(failed(s, &a.error, "a compute pipeline")),
        }
    }

    /// Replays `stream` into a command buffer and commits it.
    pub fn submit(&self, device: u64, stream: &Stream, wait: bool) -> Result<(), String> {
        let mut a = Submit {
            device,
            stream: stream.0.as_ptr() as u64,
            len: stream.0.len() as u64,
            wait: u32::from(wait),
            error: Message::default(),
        };
        match self.call(op::SUBMIT, &mut a) {
            status::SUCCESS => Ok(()),
            s => Err(failed(s, &a.error, "a submission")),
        }
    }
}
