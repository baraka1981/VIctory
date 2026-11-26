//! [`victory_host::Device`] on Metal.
//!
//! Console states have no Metal counterpart (Metal tracks hazards itself)
//! and are ignored; clear values belong to Metal render passes and are
//! ignored here too.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLHeap, MTLHeapDescriptor, MTLHeapType,
    MTLPixelFormat, MTLResourceOptions, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};
use victory_host::{
    Allocation, ClearValue, Descriptors, Device, HeapUse, Memory, ResourceDesc, dimension, flags,
};

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

pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalDevice {
    /// The system's default GPU (the only one on Apple silicon).
    pub fn system_default() -> Result<MetalDevice, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        Ok(MetalDevice { device })
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }
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
                Err(_) => return Allocation { size: 0, alignment: 0 },
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
            return buffer
                .map(MetalResource::Buffer)
                .ok_or(MetalError::Failed("a placed buffer"));
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
}

/// Run on a Mac: `cargo test -p victory-metal`. Needs a GPU, so not in CI.
#[cfg(test)]
mod tests {
    use super::*;
    use victory_host::state;

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

        let Ok(MetalResource::Buffer(b)) =
            gpu.committed(Memory::Upload, &ResourceDesc::buffer(4096, 0), state::GENERIC_READ, None)
        else {
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
        let Ok(MetalResource::Texture(t)) = gpu.committed(Memory::Device, &desc, state::COMMON, None)
        else {
            panic!("render target")
        };
        assert_eq!((t.width(), t.height(), t.mipmapLevelCount()), (256, 128, 9));
        assert_eq!(t.pixelFormat(), MTLPixelFormat::RGBA8Unorm);

        // Depth: typeless R32 family as a depth target.
        let depth = texture(64, 64, 39, flags::DEPTH_STENCIL);
        let Ok(MetalResource::Texture(d)) = gpu.committed(Memory::Device, &depth, state::COMMON, None)
        else {
            panic!("depth target")
        };
        assert_eq!(d.pixelFormat(), MTLPixelFormat::Depth32Float);

        // Placement: two small BC7 textures back to back in one heap.
        let small = texture(64, 64, 98, 0);
        let a = gpu.allocation(&small);
        assert!(a.size > 0 && a.alignment > 0);
        let heap = gpu
            .heap(Memory::Device, a.size.next_multiple_of(a.alignment) * 2, 0, HeapUse::Textures)
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

        let table = gpu.descriptor_heap(Descriptors::Views, 1000, true).expect("table");
        assert_eq!(table.table.length(), 1000 * DESCRIPTOR_BYTES as usize);
        assert_eq!(gpu.descriptor_step(Descriptors::Views), DESCRIPTOR_BYTES);
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
