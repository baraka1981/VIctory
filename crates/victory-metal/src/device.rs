//! [`victory_host::Device`] on Metal.
//!
//! Console states have no Metal counterpart (Metal tracks hazards itself)
//! and are ignored; clear values belong to Metal render passes and are
//! ignored here too.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use objc2::rc::{Retained, Weak};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLFunction, MTLHeap, MTLHeapDescriptor, MTLHeapType, MTLLibrary, MTLPixelFormat, MTLResourceOptions,
    MTLSamplerState, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType,
    MTLTextureUsage,
};
use victory_host::{
    Allocation, ClearValue, ComputeDesc, DepthViewDesc, Descriptors, Device, HeapUse, Memory,
    ResourceDesc, SamplerDesc, ShaderCode, ShaderViewDesc, TargetViewDesc, UnorderedViewDesc,
    dimension, flags, view,
};

use crate::record::MetalRecorder;
use crate::render::MetalGraphicsPipeline;
use crate::toplevel::TopLevel;
use crate::views::{self, Kept, Slots, Target, resource_id};
use crate::{DESCRIPTOR_BYTES, MetalError, format, layout};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Heap = Retained<ProtocolObject<dyn MTLHeap>>;

/// A buffer or a texture.
pub enum MetalResource {
    Buffer(Buffer),
    Texture(Texture),
}

/// A descriptor heap: a table of [`DESCRIPTOR_BYTES`] entries in shared
/// memory, which translated shaders index (shader-visible heaps) or the
/// backend reads when recording (render target and depth views).
pub struct MetalDescriptorHeap {
    pub kind: Descriptors,
    pub count: u32,
    pub table: Buffer,
}

/// How a pipeline's shader finds its resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Abi {
    /// Our own shaders: constants at buffer 0, bindings from 1.
    Direct,
    /// Converted DXIL: a top-level argument buffer at
    /// `irconverter::ARGUMENT_BUFFER_INDEX`.
    Converted,
}

/// A compute pipeline and the group size its shader declares.
pub struct MetalPipeline {
    pub state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub threads: [u32; 3],
    pub abi: Abi,
    /// The top-level argument buffer's layout ([`Abi::Converted`]; empty
    /// for [`Abi::Direct`]).
    pub top: TopLevel,
}

pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Metal objects descriptor slots refer to, by slot address.
    slots: Arc<Slots>,
    /// Buffers made here, by GPU address, for constant views (which name
    /// an address, not a buffer).
    buffers: Mutex<BTreeMap<u64, (u64, Weak<ProtocolObject<dyn MTLBuffer>>)>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalDevice {
    /// The system's default GPU (the only one on Apple silicon).
    pub fn system_default() -> Result<MetalDevice, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::Failed("a command queue"))?;
        Ok(MetalDevice {
            device,
            slots: Arc::default(),
            buffers: Mutex::default(),
            queue,
        })
    }

    /// A new command buffer to record into.
    pub fn recorder(&self) -> Result<MetalRecorder, MetalError> {
        let buffer = self
            .queue
            .commandBuffer()
            .ok_or(MetalError::Failed("a command buffer"))?;
        Ok(MetalRecorder::new(buffer, self.slots.clone()))
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// The render target or depth view written at slot `at`, for render
    /// pass attachments.
    pub fn target(&self, at: usize) -> Option<Target> {
        self.slots.target(at)
    }

    /// Writes an entry and what it names (`write`: an unordered access
    /// view).
    ///
    /// # Safety
    /// `at` is an entry of a live descriptor table.
    unsafe fn put(&self, at: usize, words: [u64; 3], kept: Option<Kept>, write: bool) {
        self.slots.set(at, kept, write);
        // SAFETY: the caller's entry.
        unsafe { views::write(at, words) }
    }

    /// Notes a new buffer's GPU range (dropping entries whose buffers are
    /// gone now and then).
    fn remember(&self, buffer: &Buffer) {
        let mut all = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        all.insert(
            buffer.gpuAddress(),
            (buffer.length() as u64, Weak::from_retained(buffer)),
        );
        if all.len().is_multiple_of(1024) {
            all.retain(|_, (_, b)| b.load().is_some());
        }
    }

    /// The live buffer made here that holds GPU `address`.
    fn buffer_at(&self, address: u64) -> Option<Buffer> {
        let all = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let (start, (len, b)) = all.range(..=address).next_back()?;
        if address >= start + len {
            return None;
        }
        b.load()
    }

    /// Function `entry` of a library made from MSL source or a metallib.
    pub(crate) fn function(
        &self,
        code: ShaderCode<'_>,
        entry: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, MetalError> {
        let shader = |e: Retained<objc2_foundation::NSError>| {
            MetalError::Shader(e.localizedDescription().to_string())
        };
        let library: Retained<ProtocolObject<dyn MTLLibrary>> = match code {
            ShaderCode::Msl(source) => self
                .device
                .newLibraryWithSource_options_error(&NSString::from_str(source), None)
                .map_err(shader)?,
            ShaderCode::MetalLib(bytes) => {
                // Metal reads libraries from files or dispatch data; a file
                // keeps this free of libdispatch bindings.
                let path =
                    std::env::temp_dir().join(format!("victory-{:x}.metallib", fxhash(bytes)));
                std::fs::write(&path, bytes).map_err(|e| MetalError::Shader(e.to_string()))?;
                let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
                let made = self.device.newLibraryWithURL_error(&url).map_err(shader);
                let _ = std::fs::remove_file(&path);
                made?
            }
            ShaderCode::Dxil(_) => {
                return Err(MetalError::Shader("DXIL needs converting first".into()));
            }
        };
        library
            .newFunctionWithName(&NSString::from_str(entry))
            .ok_or_else(|| MetalError::Shader(format!("no function {entry}")))
    }

    fn texture_entry(
        texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    ) -> ([u64; 3], Option<Kept>) {
        match texture {
            Some(t) => (
                [0, resource_id(t.gpuResourceID()), 0],
                Some(Kept::Texture(t)),
            ),
            // A view Metal refused reads as a null one.
            None => ([0; 3], None),
        }
    }
}

/// The root signature a converted pipeline uses: the one handed in, else
/// the one embedded in `dxil`; and whether it was handed in.
pub(crate) fn root_signature_for(
    given: &[u8],
    dxil: &[u8],
) -> Result<(victory_host::root::RootSignature, bool), MetalError> {
    let blob = if given.is_empty() {
        victory_host::root::embedded(dxil).ok_or_else(|| {
            MetalError::Shader("no root signature: none given, none embedded".into())
        })?
    } else {
        given
    };
    let rs = victory_host::root::parse(blob).map_err(|e| MetalError::Shader(e.to_string()))?;
    Ok((rs, !given.is_empty()))
}

pub(crate) fn c_entry(entry: &str) -> Result<std::ffi::CString, MetalError> {
    std::ffi::CString::new(entry).map_err(|e| MetalError::Shader(e.to_string()))
}

/// The Metal function name converted shaders have.
pub(crate) fn converted_entry() -> &'static str {
    crate::irconverter::ENTRY.to_str().unwrap_or("victory_main")
}

/// A short hash for temporary file names.
fn fxhash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

fn storage(memory: Memory) -> MTLStorageMode {
    match memory {
        Memory::Device => MTLStorageMode::Private,
        // Unified memory: everything the CPU touches is shared.
        Memory::Upload | Memory::Readback | Memory::DeviceUpload | Memory::Shared => {
            MTLStorageMode::Shared
        }
    }
}

fn buffer_options(memory: Memory) -> MTLResourceOptions {
    match memory {
        Memory::Device => MTLResourceOptions::StorageModePrivate,
        // Written by the CPU, never read back: write-combined.
        Memory::Upload | Memory::DeviceUpload => {
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined
        }
        Memory::Readback | Memory::Shared => MTLResourceOptions::StorageModeShared,
    }
}

fn texture_type(desc: &ResourceDesc) -> Result<MTLTextureType, MetalError> {
    let array = desc.depth_or_array_size > 1;
    let multisample = desc.sample_count > 1;
    Ok(match desc.dimension {
        dimension::TEXTURE_1D if array => MTLTextureType::Type1DArray,
        dimension::TEXTURE_1D => MTLTextureType::Type1D,
        dimension::TEXTURE_2D if multisample && array => MTLTextureType::Type2DMultisampleArray,
        dimension::TEXTURE_2D if multisample => MTLTextureType::Type2DMultisample,
        dimension::TEXTURE_2D if array => MTLTextureType::Type2DArray,
        dimension::TEXTURE_2D => MTLTextureType::Type2D,
        dimension::TEXTURE_3D => MTLTextureType::Type3D,
        other => return Err(MetalError::Dimension(other)),
    })
}

/// A Metal texture description for `desc` in `storage`.
fn texture_descriptor(
    desc: &ResourceDesc,
    storage: MTLStorageMode,
) -> Result<Retained<MTLTextureDescriptor>, MetalError> {
    let depth = desc.flags & flags::DEPTH_STENCIL != 0;
    let mapped = format::map(desc.format, depth).ok_or(MetalError::Format(desc.format))?;
    let d = MTLTextureDescriptor::new();
    d.setTextureType(texture_type(desc)?);
    d.setPixelFormat(MTLPixelFormat(mapped.pixel_format));
    let layers = usize::from(desc.depth_or_array_size.max(1));
    // SAFETY: sizes are the title's own (validated by the console side);
    // Metal rejects out-of-range ones by returning nil at creation.
    unsafe {
        d.setWidth(desc.width.max(1) as usize);
        d.setHeight(desc.height.max(1) as usize);
        if desc.dimension == dimension::TEXTURE_3D {
            d.setDepth(layers);
        } else {
            d.setArrayLength(layers);
        }
        d.setMipmapLevelCount(layout::mips(desc) as usize);
        d.setSampleCount(desc.sample_count.max(1) as usize);
    }
    let mut usage = MTLTextureUsage::empty();
    if desc.flags & flags::DENY_SHADER_RESOURCE == 0 {
        usage |= MTLTextureUsage::ShaderRead;
    }
    if desc.flags & flags::UNORDERED_ACCESS != 0 {
        usage |= MTLTextureUsage::ShaderWrite;
    }
    if desc.flags & (flags::RENDER_TARGET | flags::DEPTH_STENCIL) != 0 {
        usage |= MTLTextureUsage::RenderTarget;
    }
    if mapped.typeless {
        usage |= MTLTextureUsage::PixelFormatView;
    }
    d.setUsage(usage);
    d.setStorageMode(storage);
    Ok(d)
}

impl Device for MetalDevice {
    type Resource = MetalResource;
    type Heap = Heap;
    type DescriptorHeap = MetalDescriptorHeap;
    type Pipeline = MetalPipeline;
    type GraphicsPipeline = MetalGraphicsPipeline;
    type Error = MetalError;

    fn committed(
        &self,
        memory: Memory,
        desc: &ResourceDesc,
        _state: u32,
        _clear: Option<&ClearValue>,
    ) -> Result<MetalResource, MetalError> {
        if desc.dimension == dimension::BUFFER {
            let buffer = self
                .device
                .newBufferWithLength_options(desc.width.max(1) as usize, buffer_options(memory))
                .ok_or(MetalError::Failed("a buffer"))?;
            self.remember(&buffer);
            return Ok(MetalResource::Buffer(buffer));
        }
        let d = texture_descriptor(desc, storage(memory))?;
        let texture = self
            .device
            .newTextureWithDescriptor(&d)
            .ok_or(MetalError::Failed("a texture"))?;
        Ok(MetalResource::Texture(texture))
    }

    /// For now each reserved texture gets a sparse heap big enough to back
    /// all of it; tiles are mapped in later (with command recording).
    fn reserved(&self, desc: &ResourceDesc, _state: u32) -> Result<MetalResource, MetalError> {
        let d = texture_descriptor(desc, MTLStorageMode::Private)?;
        let size = self.device.heapTextureSizeAndAlignWithDescriptor(&d).size;
        let tile = self.device.sparseTileSizeInBytes().max(1);
        let h = MTLHeapDescriptor::new();
        h.setType(MTLHeapType::Sparse);
        h.setStorageMode(MTLStorageMode::Private);
        h.setSize(size.next_multiple_of(tile));
        let heap = self
            .device
            .newHeapWithDescriptor(&h)
            .ok_or(MetalError::Failed("a sparse heap"))?;
        let texture = heap
            .newTextureWithDescriptor(&d)
            .ok_or(MetalError::Failed("a sparse texture"))?;
        Ok(MetalResource::Texture(texture))
    }

    /// Size and alignment in a private placement heap; zero for formats
    /// Metal cannot make.
    fn allocation(&self, desc: &ResourceDesc) -> Allocation {
        let got = if desc.dimension == dimension::BUFFER {
            self.device.heapBufferSizeAndAlignWithLength_options(
                desc.width.max(1) as usize,
                MTLResourceOptions::StorageModePrivate,
            )
        } else {
            match texture_descriptor(desc, MTLStorageMode::Private) {
                Ok(d) => self.device.heapTextureSizeAndAlignWithDescriptor(&d),
                Err(_) => {
                    return Allocation {
                        size: 0,
                        alignment: 0,
                    };
                }
            }
        };
        Allocation {
            size: got.size as u64,
            alignment: got.align as u64,
        }
    }

    /// Metal placement heaps hold any kind of resource, so `use_` only
    /// matters to D3D12; their alignment is the device's, at least a page.
    fn heap(
        &self,
        memory: Memory,
        bytes: u64,
        _alignment: u64,
        _use: HeapUse,
    ) -> Result<Heap, MetalError> {
        let h = MTLHeapDescriptor::new();
        h.setType(MTLHeapType::Placement);
        h.setStorageMode(storage(memory));
        h.setSize(bytes as usize);
        self.device
            .newHeapWithDescriptor(&h)
            .ok_or(MetalError::Failed("a heap"))
    }

    fn placed(
        &self,
        heap: &Heap,
        offset: u64,
        desc: &ResourceDesc,
        _state: u32,
    ) -> Result<MetalResource, MetalError> {
        if desc.dimension == dimension::BUFFER {
            // SAFETY: the caller keeps `offset` aligned (as `allocation`
            // says) and inside the heap; options are the heap's own.
            let buffer = unsafe {
                heap.newBufferWithLength_options_offset(
                    desc.width.max(1) as usize,
                    heap.resourceOptions(),
                    offset as usize,
                )
            };
            let buffer = buffer.ok_or(MetalError::Failed("a placed buffer"))?;
            self.remember(&buffer);
            return Ok(MetalResource::Buffer(buffer));
        }
        let d = texture_descriptor(desc, heap.storageMode())?;
        // SAFETY: as above.
        let texture = unsafe { heap.newTextureWithDescriptor_offset(&d, offset as usize) };
        texture
            .map(MetalResource::Texture)
            .ok_or(MetalError::Failed("a placed texture"))
    }

    fn copyable_bytes(&self, desc: &ResourceDesc, first: u32, count: u32) -> u64 {
        layout::copyable_bytes(desc, first, count).unwrap_or(0)
    }

    fn descriptor_heap(
        &self,
        kind: Descriptors,
        count: u32,
        _shader_visible: bool,
    ) -> Result<MetalDescriptorHeap, MetalError> {
        let bytes = count.max(1) as usize * DESCRIPTOR_BYTES as usize;
        let table = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::Failed("a descriptor table"))?;
        Ok(MetalDescriptorHeap { kind, count, table })
    }

    fn descriptor_step(&self, _kind: Descriptors) -> u32 {
        DESCRIPTOR_BYTES
    }

    fn compute_pipeline(&self, desc: &ComputeDesc<'_>) -> Result<MetalPipeline, MetalError> {
        let shader = |e: Retained<objc2_foundation::NSError>| {
            MetalError::Shader(e.localizedDescription().to_string())
        };
        let (function, threads, abi, top) = match desc.code {
            ShaderCode::Dxil(dxil) => {
                let (rs, given) = root_signature_for(desc.root_signature, dxil)?;
                let c = crate::irconverter::convert(dxil, &c_entry(desc.entry)?, given.then_some(&rs))?;
                let top = TopLevel::new(&rs).map_err(|e| MetalError::Shader(e.to_string()))?;
                let function = self.function(ShaderCode::MetalLib(&c.metallib), converted_entry())?;
                (function, c.threads, Abi::Converted, top)
            }
            code => (
                self.function(code, desc.entry)?,
                desc.threads,
                Abi::Direct,
                TopLevel::default(),
            ),
        };
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(shader)?;
        Ok(MetalPipeline {
            state,
            threads,
            abi,
            top,
        })
    }

    fn graphics_pipeline(
        &self,
        desc: &victory_host::GraphicsDesc<'_>,
    ) -> Result<MetalGraphicsPipeline, MetalError> {
        crate::render::pipeline(self, desc)
    }

    fn copy_layout(
        &self,
        desc: &ResourceDesc,
        first: u32,
        out: &mut [victory_host::Footprint],
    ) -> u64 {
        layout::layout(desc, first, out).unwrap_or(0)
    }

    fn descriptor_start(&self, heap: &MetalDescriptorHeap) -> usize {
        heap.table.contents().as_ptr() as usize
    }

    unsafe fn shader_view(
        &self,
        resource: Option<&MetalResource>,
        desc: Option<&ShaderViewDesc>,
        at: usize,
    ) {
        let (words, kept) = match resource {
            None => ([0; 3], None),
            Some(MetalResource::Buffer(b)) => match desc {
                Some(d) if d.dimension == view::BUFFER => (
                    views::buffer_entry(b, d.buffer(), d.format),
                    Some(Kept::Buffer(b.clone())),
                ),
                _ => (
                    [b.gpuAddress(), 0, b.length() as u64],
                    Some(Kept::Buffer(b.clone())),
                ),
            },
            Some(MetalResource::Texture(t)) => Self::texture_entry(views::shader_texture(t, desc)),
        };
        // SAFETY: the caller's slot.
        unsafe { self.put(at, words, kept, false) }
    }

    unsafe fn unordered_view(
        &self,
        resource: Option<&MetalResource>,
        desc: Option<&UnorderedViewDesc>,
        at: usize,
    ) {
        let (words, kept) = match resource {
            None => ([0; 3], None),
            Some(MetalResource::Buffer(b)) => match desc {
                Some(d) if d.dimension == view::BUFFER => (
                    views::buffer_entry(b, d.buffer(), d.format),
                    Some(Kept::Buffer(b.clone())),
                ),
                _ => (
                    [b.gpuAddress(), 0, b.length() as u64],
                    Some(Kept::Buffer(b.clone())),
                ),
            },
            Some(MetalResource::Texture(t)) => {
                Self::texture_entry(views::unordered_texture(t, desc))
            }
        };
        // SAFETY: the caller's slot.
        unsafe { self.put(at, words, kept, true) }
    }

    /// The buffer holding `address` is looked up among those made here,
    /// so dispatches can make it resident.
    unsafe fn constant_view(&self, address: u64, bytes: u32, at: usize) {
        let kept = self.buffer_at(address).map(Kept::Buffer);
        // SAFETY: the caller's slot.
        unsafe { self.put(at, [address, 0, u64::from(bytes)], kept, false) }
    }

    unsafe fn sampler(&self, desc: &SamplerDesc, at: usize) {
        let (words, kept) = match views::sampler(&self.device, desc) {
            Some(s) => (
                [
                    resource_id(s.gpuResourceID()),
                    0,
                    u64::from(desc.mip_lod_bias.to_bits()),
                ],
                Some(Kept::Sampler(s)),
            ),
            None => ([0; 3], None),
        };
        // SAFETY: the caller's slot.
        unsafe { self.put(at, words, kept, false) }
    }

    unsafe fn target_view(
        &self,
        resource: Option<&MetalResource>,
        desc: Option<&TargetViewDesc>,
        at: usize,
    ) {
        let target = match resource {
            Some(MetalResource::Texture(t)) => views::target(t, desc),
            _ => None,
        };
        let (words, kept) = match target {
            Some(t) => (
                [0, resource_id(t.texture.gpuResourceID()), 0],
                Some(Kept::Target(t)),
            ),
            None => ([0; 3], None),
        };
        // SAFETY: the caller's slot.
        unsafe { self.put(at, words, kept, false) }
    }

    unsafe fn depth_view(
        &self,
        resource: Option<&MetalResource>,
        desc: Option<&DepthViewDesc>,
        at: usize,
    ) {
        let (words, kept) = match resource {
            Some(MetalResource::Texture(t)) => {
                let target = views::depth(t, desc);
                (
                    [0, resource_id(target.texture.gpuResourceID()), 0],
                    Some(Kept::Target(target)),
                )
            }
            _ => ([0; 3], None),
        };
        // SAFETY: the caller's slot.
        unsafe { self.put(at, words, kept, false) }
    }

    unsafe fn copy_descriptors(&self, count: u32, to: usize, from: usize, _kind: Descriptors) {
        let step = DESCRIPTOR_BYTES as usize;
        // SAFETY: the caller's ranges, in live tables; they may overlap.
        unsafe { core::ptr::copy(from as *const u8, to as *mut u8, count as usize * step) };
        self.slots.copy(count, to, from, step);
    }
}

/// Run on a Mac: `cargo test -p victory-metal`. Needs a GPU, so not in CI.
#[cfg(test)]
mod tests {
    use super::*;
    use victory_host::{state, view};

    fn texture(width: u64, height: u32, format: u32, flags: u32) -> ResourceDesc {
        ResourceDesc {
            dimension: dimension::TEXTURE_2D,
            alignment: 0,
            width,
            height,
            depth_or_array_size: 1,
            mip_levels: 0,
            format,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags,
        }
    }

    #[test]
    fn makes_buffers_textures_heaps_and_tables() {
        let gpu = MetalDevice::system_default().expect("a Metal device");

        let Ok(MetalResource::Buffer(b)) = gpu.committed(
            Memory::Upload,
            &ResourceDesc::buffer(4096, 0),
            state::GENERIC_READ,
            None,
        ) else {
            panic!("upload buffer")
        };
        assert_eq!(b.length(), 4096);
        // Upload buffers are CPU-visible: write a byte and read it back.
        let at = b.contents().as_ptr().cast::<u8>();
        // SAFETY: a live shared buffer of 4096 bytes.
        unsafe {
            at.write(7);
            assert_eq!(at.read(), 7);
        }

        let desc = texture(256, 128, 28, flags::RENDER_TARGET);
        let Ok(MetalResource::Texture(t)) =
            gpu.committed(Memory::Device, &desc, state::COMMON, None)
        else {
            panic!("render target")
        };
        assert_eq!((t.width(), t.height(), t.mipmapLevelCount()), (256, 128, 9));
        assert_eq!(t.pixelFormat(), MTLPixelFormat::RGBA8Unorm);

        // Depth: typeless R32 family as a depth target.
        let depth = texture(64, 64, 39, flags::DEPTH_STENCIL);
        let Ok(MetalResource::Texture(d)) =
            gpu.committed(Memory::Device, &depth, state::COMMON, None)
        else {
            panic!("depth target")
        };
        assert_eq!(d.pixelFormat(), MTLPixelFormat::Depth32Float);

        // Placement: two small BC7 textures back to back in one heap.
        let small = texture(64, 64, 98, 0);
        let a = gpu.allocation(&small);
        assert!(a.size > 0 && a.alignment > 0);
        let heap = gpu
            .heap(
                Memory::Device,
                a.size.next_multiple_of(a.alignment) * 2,
                0,
                HeapUse::Textures,
            )
            .expect("heap");
        for i in 0..2 {
            let offset = a.size.next_multiple_of(a.alignment) * i;
            assert!(matches!(
                gpu.placed(&heap, offset, &small, state::COMMON),
                Ok(MetalResource::Texture(_))
            ));
        }

        // Sparse (reserved) texture, where the GPU has them.
        let reserved = texture(1024, 1024, 28, 0);
        match gpu.reserved(&reserved, state::COMMON) {
            Ok(MetalResource::Texture(t)) => assert_eq!(t.width(), 1024),
            Ok(MetalResource::Buffer(_)) => panic!("a buffer for a texture"),
            Err(e) => eprintln!("no sparse texture here: {e}"),
        }

        let table = gpu
            .descriptor_heap(Descriptors::Views, 1000, true)
            .expect("table");
        assert_eq!(table.table.length(), 1000 * DESCRIPTOR_BYTES as usize);
        assert_eq!(gpu.descriptor_step(Descriptors::Views), DESCRIPTOR_BYTES);
    }

    #[test]
    fn views_and_samplers_fill_table_entries() {
        let gpu = MetalDevice::system_default().expect("a Metal device");
        let table = gpu.descriptor_heap(Descriptors::Views, 8, true).unwrap();
        let start = gpu.descriptor_start(&table);
        let step = gpu.descriptor_step(Descriptors::Views) as usize;
        let entry = |i: usize| unsafe { ((start + i * step) as *const [u64; 3]).read() };

        let Ok(buffer) = gpu.committed(
            Memory::Device,
            &ResourceDesc::buffer(4096, 0),
            state::COMMON,
            None,
        ) else {
            panic!("buffer")
        };
        let MetalResource::Buffer(b) = &buffer else {
            unreachable!()
        };
        // Raw view: elements 16.. (64 bytes in), 32 elements of 4 bytes.
        let raw = ShaderViewDesc {
            format: 39,
            dimension: view::BUFFER,
            mapping: 0x1688,
            _pad: 0,
            fields: [16, 32, u64::from(view::BUFFER_RAW)],
        };
        unsafe { gpu.shader_view(Some(&buffer), Some(&raw), start) };
        assert_eq!(entry(0), [b.gpuAddress() + 64, 0, 128]);

        // A texture, whole, then as a one-mip view: both name a resource id.
        let desc = texture(64, 64, 28, 0);
        let tex = gpu
            .committed(Memory::Device, &desc, state::COMMON, None)
            .unwrap();
        unsafe { gpu.shader_view(Some(&tex), None, start + step) };
        assert_ne!(entry(1)[1], 0);
        let one_mip = ShaderViewDesc {
            format: 29, // sRGB view of an RGBA8 texture
            dimension: view::TEXTURE_2D,
            mapping: 0x1688,
            _pad: 0,
            fields: [2 | (1 << 32), 0, 0],
        };
        unsafe { gpu.shader_view(Some(&tex), Some(&one_mip), start + 2 * step) };
        assert_ne!(entry(2)[1], 0);
        assert_ne!(entry(2)[1], entry(1)[1], "a view is its own resource");

        // Null view, constant view, copy.
        unsafe { gpu.shader_view(None, None, start + 3 * step) };
        assert_eq!(entry(3), [0; 3]);
        unsafe { gpu.constant_view(b.gpuAddress(), 256, start + 4 * step) };
        assert_eq!(entry(4), [b.gpuAddress(), 0, 256]);
        unsafe { gpu.copy_descriptors(2, start + 5 * step, start + step, Descriptors::Views) };
        assert_eq!(entry(5), entry(1));
        assert_eq!(entry(6), entry(2));

        // A linear wrap sampler.
        let samplers = gpu.descriptor_heap(Descriptors::Samplers, 1, true).unwrap();
        let at = gpu.descriptor_start(&samplers);
        let linear = SamplerDesc {
            filter: 0x15,
            address: [1, 1, 1],
            mip_lod_bias: 0.5,
            max_anisotropy: 1,
            comparison: 0,
            border: [0.0; 4],
            min_lod: 0.0,
            max_lod: 1000.0,
        };
        unsafe { gpu.sampler(&linear, at) };
        let s = unsafe { (at as *const [u64; 3]).read() };
        assert_ne!(s[0], 0);
        assert_eq!(s[2], u64::from(0.5f32.to_bits()));

        // A render target view is remembered for render passes.
        let targets = gpu.descriptor_heap(Descriptors::Targets, 1, false).unwrap();
        let rt = gpu
            .committed(
                Memory::Device,
                &texture(32, 32, 28, flags::RENDER_TARGET),
                state::COMMON,
                None,
            )
            .unwrap();
        let at = gpu.descriptor_start(&targets);
        unsafe { gpu.target_view(Some(&rt), None, at) };
        assert!(gpu.target(at).is_some());
    }

    #[test]
    fn copies_round_trip_through_a_texture() {
        use victory_host::{Footprint, Location, Recorder as _};
        let gpu = MetalDevice::system_default().expect("a Metal device");
        // 16x16 RGBA8: 64 bytes a row, laid out with 256-byte rows.
        let desc = texture(16, 16, 28, 0);
        let desc = ResourceDesc {
            mip_levels: 1,
            ..desc
        };
        let mut rows = [Footprint::default(); 1];
        let total = gpu.copy_layout(&desc, 0, &mut rows);
        assert_eq!(
            (rows[0].row_pitch, rows[0].height, total),
            (256, 16, 256 * 15 + 64)
        );

        let upload = gpu
            .committed(
                Memory::Upload,
                &ResourceDesc::buffer(total, 0),
                state::GENERIC_READ,
                None,
            )
            .unwrap();
        let staging = gpu
            .committed(
                Memory::Device,
                &ResourceDesc::buffer(total, 0),
                state::COMMON,
                None,
            )
            .unwrap();
        let back = gpu
            .committed(
                Memory::Readback,
                &ResourceDesc::buffer(total, 0),
                state::COPY_DEST,
                None,
            )
            .unwrap();
        let tex = gpu
            .committed(Memory::Device, &desc, state::COMMON, None)
            .unwrap();
        let (MetalResource::Buffer(up), MetalResource::Buffer(rb)) = (&upload, &back) else {
            unreachable!()
        };
        // Each texel's bytes: its x, its y, 7, 255.
        let at = up.contents().as_ptr().cast::<u8>();
        for y in 0..16usize {
            for x in 0..16usize {
                let p = y * 256 + x * 4;
                unsafe { at.add(p).copy_from([x as u8, y as u8, 7, 255].as_ptr(), 4) };
            }
        }

        let rec = gpu.recorder().unwrap();
        unsafe {
            rec.copy_buffer(&staging, 0, &upload, 0, total);
            rec.copy_texture(
                Location::Subresource(&tex, 0),
                [0; 3],
                Location::Footprint(&staging, rows[0]),
                None,
            );
            rec.copy_texture(
                Location::Footprint(&back, rows[0]),
                [0; 3],
                Location::Subresource(&tex, 0),
                None,
            );
        }
        rec.commit_and_wait().expect("the copies run");

        let got = rb.contents().as_ptr().cast::<u8>();
        for y in 0..16usize {
            for x in 0..16usize {
                let mut texel = [0u8; 4];
                unsafe { got.add(y * 256 + x * 4).copy_to(texel.as_mut_ptr(), 4) };
                assert_eq!(texel, [x as u8, y as u8, 7, 255], "texel {x},{y}");
            }
        }
    }

    #[test]
    fn compute_dispatch_runs_on_the_gpu() {
        use victory_host::{Binding, ComputeDesc, Recorder as _, ShaderCode};
        let gpu = MetalDevice::system_default().expect("a Metal device");
        let source = "#include <metal_stdlib>\nusing namespace metal;\n\
            kernel void add(constant uint& base [[buffer(0)]], device const uint* input [[buffer(1)]],\n\
                            device uint* out [[buffer(2)]], uint id [[thread_position_in_grid]]) {\n\
                out[id] = input[id] * 2 + base;\n\
            }\n";
        let pipeline = gpu
            .compute_pipeline(&ComputeDesc {
                code: ShaderCode::Msl(source),
                entry: "add",
                root_signature: &[],
                threads: [64, 1, 1],
            })
            .expect("the shader compiles");
        let bytes = 256 * 4;
        let input = gpu
            .committed(
                Memory::Upload,
                &ResourceDesc::buffer(bytes, 0),
                state::GENERIC_READ,
                None,
            )
            .unwrap();
        let out = gpu
            .committed(
                Memory::Shared,
                &ResourceDesc::buffer(bytes, flags::UNORDERED_ACCESS),
                state::COMMON,
                None,
            )
            .unwrap();
        let (MetalResource::Buffer(i), MetalResource::Buffer(o)) = (&input, &out) else {
            unreachable!()
        };
        let values = i.contents().as_ptr().cast::<u32>();
        for n in 0..256 {
            unsafe { values.add(n).write(n as u32) };
        }
        let rec = gpu.recorder().unwrap();
        unsafe {
            rec.dispatch(
                &pipeline,
                &[1000],
                &[Binding::Read(&input, 0), Binding::Write(&out, 0)],
                &[],
                [4, 1, 1],
            )
        };
        rec.commit_and_wait().expect("the dispatch runs");
        let got = o.contents().as_ptr().cast::<u32>();
        for n in 0..256 {
            assert_eq!(
                unsafe { got.add(n).read() },
                n as u32 * 2 + 1000,
                "element {n}"
            );
        }
        // Bytes that are not DXIL fail as a shader error, not a crash.
        // (saxaboom panics on a null IRObject; keep this out until guarded.)
    }

    /// Converted DXIL end to end. Needs Metal Shader Converter installed
    /// and the fixture `tests/fixtures/add.dxil` (tests/fixtures/add.hlsl,
    /// compiled with `dxc -T cs_6_0 -E main add.hlsl -Fo add.dxil`); skips
    /// without the fixture.
    #[test]
    fn converted_dxil_runs_on_the_gpu() {
        use victory_host::{Binding, ComputeDesc, Recorder as _, ShaderCode};
        let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/add.dxil");
        let Ok(dxil) = std::fs::read(fixture) else {
            eprintln!("skipped: no {fixture}");
            return;
        };
        let gpu = MetalDevice::system_default().expect("a Metal device");
        let pipeline = gpu
            .compute_pipeline(&ComputeDesc {
                code: ShaderCode::Dxil(&dxil),
                entry: "main",
                root_signature: &[],
                threads: [0; 3],
            })
            .expect("Metal Shader Converter converts the shader");
        assert_eq!(pipeline.threads, [64, 1, 1], "group size from reflection");
        let bytes = 256 * 4;
        let input = gpu
            .committed(
                Memory::Upload,
                &ResourceDesc::buffer(bytes, 0),
                state::GENERIC_READ,
                None,
            )
            .unwrap();
        let out = gpu
            .committed(
                Memory::Shared,
                &ResourceDesc::buffer(bytes, flags::UNORDERED_ACCESS),
                state::COMMON,
                None,
            )
            .unwrap();
        let (MetalResource::Buffer(i), MetalResource::Buffer(o)) = (&input, &out) else {
            unreachable!()
        };
        let values = i.contents().as_ptr().cast::<u32>();
        for n in 0..256 {
            unsafe { values.add(n).write(n as u32) };
        }
        let rec = gpu.recorder().unwrap();
        unsafe {
            rec.dispatch(
                &pipeline,
                &[1000],
                &[Binding::Read(&input, 0), Binding::Write(&out, 0)],
                &[],
                [4, 1, 1],
            )
        };
        rec.commit_and_wait().expect("the dispatch runs");
        let got = o.contents().as_ptr().cast::<u32>();
        for n in 0..256 {
            assert_eq!(
                unsafe { got.add(n).read() },
                n as u32 * 2 + 1000,
                "element {n}"
            );
        }
    }

    /// A root signature handed in separately (the DXIL has none), in an
    /// order unlike add.hlsl's: UAV, constants(2), SRV. Needs Metal Shader
    /// Converter and `tests/fixtures/swap.dxil` (see swap.hlsl); skips
    /// without the fixture.
    #[test]
    fn separate_root_signature_runs_on_the_gpu() {
        use victory_host::{Binding, ComputeDesc, Recorder as _, ShaderCode};
        let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/swap.dxil");
        let Ok(dxil) = std::fs::read(fixture) else {
            eprintln!("skipped: no {fixture}");
            return;
        };
        assert!(victory_host::root::embedded(&dxil).is_none(), "stripped");
        let rs = crate::toplevel::swap_root_signature().serialize();
        let gpu = MetalDevice::system_default().expect("a Metal device");
        let pipeline = gpu
            .compute_pipeline(&ComputeDesc {
                code: ShaderCode::Dxil(&dxil),
                entry: "main",
                root_signature: &rs,
                threads: [0; 3],
            })
            .expect("Metal Shader Converter converts the shader");
        assert_eq!(pipeline.threads, [32, 1, 1], "group size from reflection");
        let offsets: Vec<u32> = pipeline.top.entries.iter().map(|e| e.offset).collect();
        assert_eq!(offsets, [0, 8, 16]);
        let bytes = 256 * 4;
        let input = gpu
            .committed(
                Memory::Upload,
                &ResourceDesc::buffer(bytes, 0),
                state::GENERIC_READ,
                None,
            )
            .unwrap();
        let out = gpu
            .committed(
                Memory::Shared,
                &ResourceDesc::buffer(bytes, flags::UNORDERED_ACCESS),
                state::COMMON,
                None,
            )
            .unwrap();
        let (MetalResource::Buffer(i), MetalResource::Buffer(o)) = (&input, &out) else {
            unreachable!()
        };
        let values = i.contents().as_ptr().cast::<u32>();
        for n in 0..256 {
            unsafe { values.add(n).write(n as u32) };
        }
        let rec = gpu.recorder().unwrap();
        // Bindings in root-parameter order: the UAV, then the SRV.
        unsafe {
            rec.dispatch(
                &pipeline,
                &[3, 1000],
                &[Binding::Write(&out, 0), Binding::Read(&input, 0)],
                &[],
                [8, 1, 1],
            )
        };
        rec.commit_and_wait().expect("the dispatch runs");
        let got = o.contents().as_ptr().cast::<u32>();
        for n in 0..256 {
            assert_eq!(
                unsafe { got.add(n).read() },
                n as u32 * 3 + 1000,
                "element {n}"
            );
        }
    }

    /// Resources through a descriptor table: SRV, UAV and CBV at slots
    /// 5..8 of a view heap, the input buffer placed in a heap (`useHeap`),
    /// the others committed (`useResource`), the CBV found by address.
    /// Needs Metal Shader Converter and `tests/fixtures/table.dxil` (see
    /// table.hlsl); skips without the fixture.
    #[test]
    fn descriptor_table_runs_on_the_gpu() {
        use objc2_metal::MTLResource as _;
        use victory_host::{ComputeDesc, Recorder as _, ShaderCode, Table, UnorderedViewDesc};
        let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/table.dxil");
        let Ok(dxil) = std::fs::read(fixture) else {
            eprintln!("skipped: no {fixture}");
            return;
        };
        let gpu = MetalDevice::system_default().expect("a Metal device");
        let pipeline = gpu
            .compute_pipeline(&ComputeDesc {
                code: ShaderCode::Dxil(&dxil),
                entry: "main",
                root_signature: &[],
                threads: [0; 3],
            })
            .expect("Metal Shader Converter converts the shader");
        let offsets: Vec<u32> = pipeline.top.entries.iter().map(|e| e.offset).collect();
        assert_eq!(offsets, [0, 8], "constants, then the table's address");
        let count = 256u32;
        let bytes = u64::from(count) * 4;
        let buffer = ResourceDesc::buffer(bytes, 0);
        let alloc = gpu.allocation(&buffer);
        let heap = gpu
            .heap(Memory::Upload, alloc.size.max(65536), alloc.alignment, HeapUse::Buffers)
            .unwrap();
        let input = gpu.placed(&heap, 0, &buffer, state::GENERIC_READ).unwrap();
        let out = gpu
            .committed(
                Memory::Shared,
                &ResourceDesc::buffer(bytes, flags::UNORDERED_ACCESS),
                state::COMMON,
                None,
            )
            .unwrap();
        let cb = gpu
            .committed(
                Memory::Upload,
                &ResourceDesc::buffer(256, 0),
                state::GENERIC_READ,
                None,
            )
            .unwrap();
        let (MetalResource::Buffer(i), MetalResource::Buffer(o), MetalResource::Buffer(c)) =
            (&input, &out, &cb)
        else {
            unreachable!()
        };
        assert!(i.heap().is_some(), "the input is placed");
        let values = i.contents().as_ptr().cast::<u32>();
        for n in 0..count as usize {
            unsafe { values.add(n).write(n as u32) };
        }
        unsafe { c.contents().as_ptr().cast::<u32>().write(5) };
        let views = gpu.descriptor_heap(Descriptors::Views, 16, true).unwrap();
        let at = |n: usize| gpu.descriptor_start(&views) + n * DESCRIPTOR_BYTES as usize;
        let structured = u64::from(count) | (4 << 32);
        unsafe {
            gpu.shader_view(
                Some(&input),
                Some(&ShaderViewDesc {
                    format: 0,
                    dimension: view::BUFFER,
                    mapping: 0x1688,
                    _pad: 0,
                    fields: [0, structured, 0],
                }),
                at(5),
            );
            gpu.unordered_view(
                Some(&out),
                Some(&UnorderedViewDesc {
                    format: 0,
                    dimension: view::BUFFER,
                    fields: [0, structured, 0, 0],
                }),
                at(6),
            );
            gpu.constant_view(c.gpuAddress(), 256, at(7));
        }
        let rec = gpu.recorder().unwrap();
        unsafe {
            rec.dispatch(
                &pipeline,
                &[1000],
                &[],
                &[Table {
                    heap: &views,
                    first: 5,
                }],
                [count / 64, 1, 1],
            )
        };
        rec.commit_and_wait().expect("the dispatch runs");
        let got = o.contents().as_ptr().cast::<u32>();
        for n in 0..count as usize {
            assert_eq!(
                unsafe { got.add(n).read() },
                n as u32 * 5 + 1000,
                "element {n}"
            );
        }
    }

    fn fixture(name: &str) -> Option<Vec<u8>> {
        let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        let got = std::fs::read(&path).ok();
        if got.is_none() {
            eprintln!("skipped: no {path}");
        }
        got
    }

    /// An 8x8 RGBA8 render target: `record` draws into it (given the RTV
    /// slot), then it is read back as rows of pixels.
    fn render_8x8(
        gpu: &MetalDevice,
        record: impl FnOnce(&MetalRecorder, usize),
    ) -> Vec<[u8; 4]> {
        use victory_host::{Location, Recorder as _};
        let desc = texture(8, 8, 28, flags::RENDER_TARGET);
        let target = gpu
            .committed(Memory::Device, &desc, state::COMMON, None)
            .unwrap();
        let rtvs = gpu.descriptor_heap(Descriptors::Targets, 1, false).unwrap();
        let at = gpu.descriptor_start(&rtvs);
        unsafe { gpu.target_view(Some(&target), None, at) };
        let mut fp = [victory_host::Footprint::default()];
        let bytes = gpu.copy_layout(&desc, 0, &mut fp);
        let back = gpu
            .committed(
                Memory::Readback,
                &ResourceDesc::buffer(bytes, 0),
                state::COPY_DEST,
                None,
            )
            .unwrap();
        let rec = gpu.recorder().unwrap();
        record(&rec, at);
        unsafe {
            rec.copy_texture(
                Location::Footprint(&back, fp[0]),
                [0, 0, 0],
                Location::Subresource(&target, 0),
                None,
            )
        };
        rec.commit_and_wait().expect("the draws run");
        let MetalResource::Buffer(b) = &back else {
            unreachable!()
        };
        let base = b.contents().as_ptr().cast::<u8>();
        let mut out = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                let p = unsafe { base.add(fp[0].row_pitch as usize * y + x * 4) };
                out.push(unsafe { [p.read(), p.add(1).read(), p.add(2).read(), p.add(3).read()] });
            }
        }
        out
    }

    fn colour(rgba: [f32; 4]) -> [u32; 4] {
        rgba.map(f32::to_bits)
    }

    fn whole_8x8(at: usize, clear: Option<[f32; 4]>) -> victory_host::Pass<'static> {
        // Leaked: a test's few bytes, so the pass can be 'static.
        let targets = Box::leak(Box::new([victory_host::Attachment { at, clear }]));
        victory_host::Pass {
            targets,
            depth: None,
            viewport: victory_host::Viewport {
                x: 0.0,
                y: 0.0,
                width: 8.0,
                height: 8.0,
                min_depth: 0.0,
                max_depth: 1.0,
            },
            scissor: None,
        }
    }

    fn upload(gpu: &MetalDevice, words: &[u32]) -> MetalResource {
        let r = gpu
            .committed(
                Memory::Upload,
                &ResourceDesc::buffer(words.len() as u64 * 4, 0),
                state::GENERIC_READ,
                None,
            )
            .unwrap();
        let MetalResource::Buffer(b) = &r else {
            unreachable!()
        };
        let p = b.contents().as_ptr().cast::<u32>();
        for (i, w) in words.iter().enumerate() {
            unsafe { p.add(i).write(*w) };
        }
        r
    }

    /// Vertex + pixel through the seam: a cleared target, the left half
    /// drawn red from a vertex buffer, then the right half green with an
    /// index buffer (loading what is there). Needs Metal Shader Converter
    /// and quad_vs.dxil / quad_ps.dxil.
    #[test]
    fn vertex_pixel_pipeline_draws() {
        use victory_host::{
            APPEND_ALIGNED, Arguments, Blend, DepthTest, Draw, Geometry, GraphicsDesc, IndexBuffer,
            InputElement, Raster, Recorder as _, ShaderCode, Stage, VertexBuffer, Vertices,
        };
        let (Some(vs), Some(ps)) = (fixture("quad_vs.dxil"), fixture("quad_ps.dxil")) else {
            return;
        };
        let gpu = MetalDevice::system_default().expect("a Metal device");
        let inputs = [InputElement {
            semantic: "POSITION",
            index: 0,
            format: 16, // R32G32_FLOAT
            slot: 0,
            offset: APPEND_ALIGNED,
            per_instance: false,
            step_rate: 0,
        }];
        let pipeline = gpu
            .graphics_pipeline(&GraphicsDesc {
                geometry: Geometry::Vertex {
                    shader: Stage {
                        code: ShaderCode::Dxil(&vs),
                        entry: "main",
                    },
                    inputs: &inputs,
                },
                pixel: Some(Stage {
                    code: ShaderCode::Dxil(&ps),
                    entry: "main",
                }),
                root_signature: &[],
                targets: &[28],
                depth_format: 0,
                samples: 1,
                topology: 3,
                blend: [Blend::OPAQUE; 8],
                raster: Raster {
                    cull: 1,
                    ..Raster::DEFAULT
                },
                depth: DepthTest::OFF,
            })
            .expect("Metal Shader Converter converts both stages");
        let f = |v: f32| v.to_bits();
        // Left half: two triangles. Right half: four corners, indexed.
        let left = upload(
            &gpu,
            &[-1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0, -1.0, -1.0, 1.0, 0.0, 1.0].map(f),
        );
        let right = upload(&gpu, &[0.0, -1.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0].map(f));
        // u16 indices 0 1 2, 2 1 3, two to a word (low half first).
        let pair = |a: u32, b: u32| a | (b << 16);
        let indices = upload(&gpu, &[pair(0, 1), pair(2, 2), pair(1, 3)]);
        let red = colour([1.0, 0.0, 0.0, 1.0]);
        let green = colour([0.0, 1.0, 0.0, 1.0]);
        let pixels = render_8x8(&gpu, |rec, at| unsafe {
            rec.draw(
                &whole_8x8(at, Some([0.0, 0.0, 1.0, 1.0])),
                &pipeline,
                &Arguments {
                    constants: &red,
                    bindings: &[],
                    tables: &[],
                },
                &Vertices {
                    buffers: &[VertexBuffer {
                        resource: &left,
                        offset: 0,
                        stride: 8,
                    }],
                    index: None,
                    primitive: 4,
                },
                &Draw {
                    count: 6,
                    instances: 1,
                    first: 0,
                    base_vertex: 0,
                    first_instance: 0,
                },
            );
            rec.draw(
                &whole_8x8(at, None),
                &pipeline,
                &Arguments {
                    constants: &green,
                    bindings: &[],
                    tables: &[],
                },
                &Vertices {
                    buffers: &[VertexBuffer {
                        resource: &right,
                        offset: 0,
                        stride: 8,
                    }],
                    index: Some(IndexBuffer {
                        resource: &indices,
                        offset: 0,
                        format: 57, // R16_UINT
                    }),
                    primitive: 4,
                },
                &Draw {
                    count: 6,
                    instances: 1,
                    first: 0,
                    base_vertex: 0,
                    first_instance: 0,
                },
            );
        });
        assert_eq!(pixels[8 + 1], [255, 0, 0, 255], "left half red");
        assert_eq!(pixels[5 * 8 + 6], [0, 255, 0, 255], "right half green");
    }

    /// Mesh + pixel through the seam: one mesh group covers the left
    /// half red; the right half keeps the clear colour. Needs Metal Shader
    /// Converter and mesh.dxil / quad_ps.dxil.
    #[test]
    fn mesh_pipeline_draws() {
        use victory_host::{
            Arguments, Blend, DepthTest, Geometry, GraphicsDesc, Raster, Recorder as _, ShaderCode,
            Stage,
        };
        let (Some(ms), Some(ps)) = (fixture("mesh.dxil"), fixture("quad_ps.dxil")) else {
            return;
        };
        let gpu = MetalDevice::system_default().expect("a Metal device");
        let pipeline = gpu
            .graphics_pipeline(&GraphicsDesc {
                geometry: Geometry::Mesh {
                    amplification: None,
                    shader: Stage {
                        code: ShaderCode::Dxil(&ms),
                        entry: "main",
                    },
                },
                pixel: Some(Stage {
                    code: ShaderCode::Dxil(&ps),
                    entry: "main",
                }),
                root_signature: &[],
                targets: &[28],
                depth_format: 0,
                samples: 1,
                topology: 3,
                blend: [Blend::OPAQUE; 8],
                raster: Raster {
                    cull: 1,
                    ..Raster::DEFAULT
                },
                depth: DepthTest::OFF,
            })
            .expect("Metal Shader Converter converts both stages");
        assert!(matches!(
            pipeline.kind,
            crate::render::Kind::Mesh {
                mesh: [1, 1, 1],
                amplified: false,
                ..
            }
        ));
        let red = colour([1.0, 0.0, 0.0, 1.0]);
        let pixels = render_8x8(&gpu, |rec, at| unsafe {
            rec.draw_mesh(
                &whole_8x8(at, Some([0.0, 0.0, 1.0, 1.0])),
                &pipeline,
                &Arguments {
                    constants: &red,
                    bindings: &[],
                    tables: &[],
                },
                [1, 1, 1],
            )
        });
        assert_eq!(pixels[8 + 1], [255, 0, 0, 255], "left half red");
        assert_eq!(pixels[5 * 8 + 6], [0, 0, 255, 255], "right half cleared blue");
    }

    #[test]
    fn vertex_formats_match_metal() {
        use objc2_metal::MTLVertexFormat as F;
        for (dxgi, want) in [
            (2, F::Float4),
            (6, F::Float3),
            (16, F::Float2),
            (41, F::Float),
            (10, F::Half4),
            (34, F::Half2),
            (54, F::Half),
            (28, F::UChar4Normalized),
            (87, F::UChar4Normalized_BGRA),
            (31, F::Char4Normalized),
            (24, F::UInt1010102Normalized),
            (26, F::FloatRG11B10),
            (67, F::FloatRGB9E5),
            (42, F::UInt),
            (3, F::UInt4),
            (4, F::Int4),
            (12, F::UShort4),
            (11, F::UShort4Normalized),
            (13, F::Short4Normalized),
            (35, F::UShort2Normalized),
            (37, F::Short2Normalized),
            (49, F::UChar2Normalized),
            (61, F::UCharNormalized),
            (57, F::UShort),
        ] {
            assert_eq!(crate::vertex::format(dxgi).map(|f| f.0), Some(want.0), "DXGI {dxgi}");
        }
    }

    /// crate::toplevel's offsets against where MSC itself says each
    /// resource goes, for every parameter kind (RS10: constants, root
    /// CBV/SRV/UAV, tables incl. unbounded and samplers, a static sampler;
    /// RS11: version 1.1 flags; swap: addresses before constants). Needs
    /// Metal Shader Converter, no GPU work.
    #[test]
    fn irconverter_layout_matches() {
        use saxaboom::ffi::IRResourceType as T;
        use victory_host::root::{self, Kind, RootSignature, parameter, range, samples};
        let covers = |r: &root::Range, ty: u32, slot: u32, space: u32| {
            r.kind == ty
                && r.space == space
                && slot >= r.register
                && (r.count == root::UNBOUNDED || slot < r.register + r.count)
        };
        let check = |name: &str, rs: RootSignature| {
            let top = crate::toplevel::TopLevel::new(&rs).unwrap();
            let locations = crate::irconverter::locations(&rs).expect("MSC takes the root signature");
            eprintln!("{name}: ours {:?}", top.entries);
            let mut matched = 0;
            for l in &locations {
                eprintln!(
                    "{name}: MSC {:?} space {} slot {} at {} ({} bytes)",
                    l.resourceType, l.space, l.slot, l.topLevelOffset, l.sizeBytes
                );
                let (param_ty, range_ty) = match l.resourceType {
                    T::Constant => (Some(parameter::CONSTANTS), None),
                    T::CBV => (Some(parameter::CBV), Some(range::CBV)),
                    T::SRV => (Some(parameter::SRV), Some(range::SRV)),
                    T::UAV => (Some(parameter::UAV), Some(range::UAV)),
                    T::Sampler => (None, Some(range::SAMPLER)),
                    T::Table => (None, None),
                    _ => (None, None),
                };
                let found = rs.parameters.iter().position(|p| match &p.kind {
                    Kind::Constants { register, space, .. } => {
                        param_ty == Some(parameter::CONSTANTS)
                            && (*register, *space) == (l.slot, l.space)
                    }
                    Kind::Descriptor { kind, register, space, .. } => {
                        param_ty == Some(*kind) && (*register, *space) == (l.slot, l.space)
                    }
                    Kind::Table(ranges) => {
                        l.resourceType == T::Table
                            && ranges
                                .first()
                                .is_some_and(|r| (r.register, r.space) == (l.slot, l.space))
                            || range_ty.is_some_and(|t| {
                                ranges.iter().any(|r| covers(r, t, l.slot, l.space))
                            })
                    }
                });
                let static_sampler = l.resourceType == T::Sampler
                    && rs.samplers.iter().any(|s| (s.register, s.space) == (l.slot, l.space));
                match found {
                    Some(i) => {
                        assert_eq!(
                            l.topLevelOffset, top.entries[i].offset,
                            "{name}: parameter {i} ({:?})",
                            l.resourceType
                        );
                        matched += 1;
                    }
                    None if static_sampler => {}
                    None => panic!("{name}: MSC location {l:?} matches no parameter"),
                }
            }
            assert!(matched > 0, "{name}: MSC reported nothing to compare");
        };
        if let Err(MetalError::Shader(why)) =
            crate::irconverter::locations(&crate::toplevel::swap_root_signature())
            && why.contains("not loaded")
        {
            eprintln!("skipped: {why}");
            return;
        }
        check("RS10", root::parse(&samples::RS10).unwrap());
        check("RS11", root::parse(&samples::RS11).unwrap());
        check("swap", crate::toplevel::swap_root_signature());
    }

    #[test]
    fn unsupported_formats_are_errors_not_crashes() {
        let gpu = MetalDevice::system_default().expect("a Metal device");
        let rgb32 = texture(4, 4, 6, 0);
        assert!(matches!(
            gpu.committed(Memory::Device, &rgb32, state::COMMON, None),
            Err(MetalError::Format(6))
        ));
        assert_eq!(gpu.allocation(&rgb32).size, 0);
    }
}
