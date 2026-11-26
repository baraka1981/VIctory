//! The host graphics seam.
//!
//! victory-gpu turns the title's D3D12.X calls into host GPU work. Until
//! now that meant D3D12 calls made straight from the console-object code.
//! This crate names what the console side needs from *a* host API, so a
//! second backend (Metal, for Apple silicon) can sit beside D3D12.
//!
//! The vocabulary stays the console's: resource descriptions, states and
//! format numbers are D3D12.X's (D3D12's numbering, DXGI formats plus the
//! console's own). D3D12 passes them through; other backends translate.
//!
//! Slice 1 covers resource creation and descriptor heaps. Views, pipelines
//! and command recording follow (docs/metal-port.md).

use core::fmt;

/// A resource's shape and use, laid out as D3D12.X's (and D3D12's)
/// `D3D12_RESOURCE_DESC`, so the title's descriptions are read as is.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceDesc {
    pub dimension: u32,
    pub alignment: u64,
    pub width: u64,
    pub height: u32,
    pub depth_or_array_size: u16,
    pub mip_levels: u16,
    pub format: u32,
    pub sample_count: u32,
    pub sample_quality: u32,
    pub layout: u32,
    pub flags: u32,
}

const _: () = assert!(size_of::<ResourceDesc>() == 56);

/// `ResourceDesc::dimension` values.
pub mod dimension {
    pub const BUFFER: u32 = 1;
    pub const TEXTURE_1D: u32 = 2;
    pub const TEXTURE_2D: u32 = 3;
    pub const TEXTURE_3D: u32 = 4;
}

/// `ResourceDesc::flags` bits the backends act on.
pub mod flags {
    pub const RENDER_TARGET: u32 = 0x1;
    pub const DEPTH_STENCIL: u32 = 0x2;
    pub const UNORDERED_ACCESS: u32 = 0x4;
    pub const DENY_SHADER_RESOURCE: u32 = 0x8;
}

/// `ResourceDesc::layout`: row-major, the only layout a buffer has.
pub const ROW_MAJOR: u32 = 1;

/// Resource states (D3D12 numbering) the backends are handed.
pub mod state {
    pub const COMMON: u32 = 0;
    pub const UNORDERED_ACCESS: u32 = 0x8;
    pub const COPY_DEST: u32 = 0x400;
    pub const GENERIC_READ: u32 = 0xac3;
}

impl ResourceDesc {
    /// A buffer of `bytes` with resource `flags`.
    pub const fn buffer(bytes: u64, flags: u32) -> ResourceDesc {
        ResourceDesc {
            dimension: dimension::BUFFER,
            alignment: 0,
            width: bytes,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: ROW_MAJOR,
            flags,
        }
    }
}

/// A clear value as D3D12.X lays it out: the format, then four words of
/// colour (or depth as f32 and the stencil byte). Backends with no clear
/// value at creation (Metal: it is a render pass's) may ignore it.
pub type ClearValue = [u32; 5];

/// Where a resource's memory lives, from the CPU's side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Memory {
    /// The GPU's own (D3D12 default heap, Metal private storage).
    Device,
    /// Written by the CPU, read by the GPU (upload heap, shared storage).
    Upload,
    /// Written by the GPU, read by the CPU (readback heap, shared storage).
    Readback,
    /// Written by the CPU straight into the GPU's memory (D3D12 GPU upload
    /// heap, resizable BAR); shared storage on unified memory.
    DeviceUpload,
    /// System memory cached for the CPU that the GPU also reads and writes
    /// (D3D12 custom write-back heap in L0); shared storage on unified
    /// memory.
    Shared,
}

/// What a heap's placed resources may be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeapUse {
    Buffers,
    /// Textures that are neither render targets nor depth-stencil.
    Textures,
    /// Render target and depth-stencil textures.
    Targets,
}

/// Bytes and alignment a placed resource takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocation {
    pub size: u64,
    pub alignment: u64,
}

/// Kinds of descriptor heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Descriptors {
    /// Constant buffer, shader resource and unordered access views.
    Views,
    Samplers,
    /// Render target views.
    Targets,
    /// Depth-stencil views.
    Depth,
}

/// What a host graphics API provides the console side (slice 1).
///
/// Methods take `&self` and are called from the title's threads at once:
/// implementations are free-threaded, as D3D12 devices and Metal devices
/// are.
pub trait Device {
    /// A buffer or texture.
    type Resource;
    /// Memory resources are placed in (D3D12 heap, Metal placement heap).
    type Heap;
    /// A block of descriptors (D3D12 descriptor heap, Metal argument table).
    type DescriptorHeap;
    type Error: fmt::Display + fmt::Debug;

    /// A resource with memory of its own, in `state` (D3D12 numbering),
    /// optimised for `clear` when given.
    fn committed(
        &self,
        memory: Memory,
        desc: &ResourceDesc,
        state: u32,
        clear: Option<&ClearValue>,
    ) -> Result<Self::Resource, Self::Error>;

    /// A texture with no memory behind it yet (tiled / sparse); its tiles
    /// are mapped later.
    fn reserved(&self, desc: &ResourceDesc, state: u32) -> Result<Self::Resource, Self::Error>;

    /// Bytes and alignment a placed resource of `desc` takes.
    fn allocation(&self, desc: &ResourceDesc) -> Allocation;

    /// `bytes` of `memory`, aligned to `alignment`, to place resources of
    /// one `use_` in.
    fn heap(
        &self,
        memory: Memory,
        bytes: u64,
        alignment: u64,
        use_: HeapUse,
    ) -> Result<Self::Heap, Self::Error>;

    /// A resource at `offset` in `heap` (aligned as [`Device::allocation`]
    /// says), in `state`.
    fn placed(
        &self,
        heap: &Self::Heap,
        offset: u64,
        desc: &ResourceDesc,
        state: u32,
    ) -> Result<Self::Resource, Self::Error>;

    /// Bytes a buffer needs to hold subresources `first..first + count` of
    /// `desc` laid out for a copy (D3D12's copyable footprints).
    fn copyable_bytes(&self, desc: &ResourceDesc, first: u32, count: u32) -> u64;

    /// A heap of `count` descriptors of `kind`; `shader_visible` for one
    /// shaders index directly.
    fn descriptor_heap(
        &self,
        kind: Descriptors,
        count: u32,
        shader_visible: bool,
    ) -> Result<Self::DescriptorHeap, Self::Error>;

    /// Bytes between two descriptors of `kind` in a heap.
    fn descriptor_step(&self, kind: Descriptors) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_d3d12_resource_desc() {
        // Field offsets of D3D12_RESOURCE_DESC (x64).
        assert_eq!(core::mem::offset_of!(ResourceDesc, alignment), 8);
        assert_eq!(core::mem::offset_of!(ResourceDesc, width), 16);
        assert_eq!(core::mem::offset_of!(ResourceDesc, height), 24);
        assert_eq!(core::mem::offset_of!(ResourceDesc, depth_or_array_size), 28);
        assert_eq!(core::mem::offset_of!(ResourceDesc, mip_levels), 30);
        assert_eq!(core::mem::offset_of!(ResourceDesc, format), 32);
        assert_eq!(core::mem::offset_of!(ResourceDesc, sample_count), 36);
        assert_eq!(core::mem::offset_of!(ResourceDesc, layout), 44);
        assert_eq!(core::mem::offset_of!(ResourceDesc, flags), 48);
    }

    #[test]
    fn buffer_is_row_major_and_single() {
        let b = ResourceDesc::buffer(4096, flags::UNORDERED_ACCESS);
        assert_eq!((b.dimension, b.width, b.height, b.layout), (dimension::BUFFER, 4096, 1, ROW_MAJOR));
        assert_eq!((b.mip_levels, b.depth_or_array_size, b.sample_count), (1, 1, 1));
    }
}
