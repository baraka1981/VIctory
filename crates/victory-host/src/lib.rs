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

pub mod root;

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

/// A shader resource view description in D3D12.X's layout
/// (`D3D12_SHADER_RESOURCE_VIEW_DESC`): format, view dimension, component
/// mapping, then the dimension's own fields (first element / mip, counts,
/// flags) as D3D12 lays them out.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShaderViewDesc {
    pub format: u32,
    pub dimension: u32,
    pub mapping: u32,
    pub _pad: u32,
    pub fields: [u64; 3],
}

/// An unordered access view description (`D3D12_UNORDERED_ACCESS_VIEW_DESC`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnorderedViewDesc {
    pub format: u32,
    pub dimension: u32,
    pub fields: [u64; 4],
}

/// A render target view description (`D3D12_RENDER_TARGET_VIEW_DESC`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetViewDesc {
    pub format: u32,
    pub dimension: u32,
    pub fields: [u64; 2],
}

/// A depth-stencil view description (`D3D12_DEPTH_STENCIL_VIEW_DESC`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DepthViewDesc {
    pub format: u32,
    pub dimension: u32,
    pub flags: u32,
    pub fields: [u32; 3],
}

/// A sampler description (`D3D12_SAMPLER_DESC`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerDesc {
    pub filter: u32,
    pub address: [u32; 3],
    pub mip_lod_bias: f32,
    pub max_anisotropy: u32,
    pub comparison: u32,
    pub border: [f32; 4],
    pub min_lod: f32,
    pub max_lod: f32,
}

const _: () = assert!(size_of::<ShaderViewDesc>() == 40);
const _: () = assert!(size_of::<UnorderedViewDesc>() == 40);
const _: () = assert!(size_of::<TargetViewDesc>() == 24);
const _: () = assert!(size_of::<DepthViewDesc>() == 24);
const _: () = assert!(size_of::<SamplerDesc>() == 52);

/// View dimension values (D3D12 numbering) the backends read.
pub mod view {
    pub const BUFFER: u32 = 1;
    pub const TEXTURE_1D: u32 = 2;
    pub const TEXTURE_1D_ARRAY: u32 = 3;
    pub const TEXTURE_2D: u32 = 4;
    pub const TEXTURE_2D_ARRAY: u32 = 5;
    pub const TEXTURE_2D_MS: u32 = 6;
    pub const TEXTURE_2D_MS_ARRAY: u32 = 7;
    pub const TEXTURE_3D: u32 = 8;
    pub const TEXTURE_CUBE: u32 = 9;
    pub const TEXTURE_CUBE_ARRAY: u32 = 10;
    /// Buffer view flag: raw (byte address) access.
    pub const BUFFER_RAW: u32 = 1;
}

impl ShaderViewDesc {
    /// For a buffer view: (first element, element count, stride, flags).
    pub fn buffer(&self) -> (u64, u32, u32, u32) {
        let f = self.fields;
        (f[0], f[1] as u32, (f[1] >> 32) as u32, f[2] as u32)
    }

    /// For a texture view: (most detailed mip, mip count, first slice,
    /// slice count); cube views count faces in slices of six.
    pub fn texture(&self) -> (u32, u32, u32, u32) {
        let w = |i: usize| (self.fields[i / 2] >> (32 * (i % 2))) as u32;
        match self.dimension {
            view::TEXTURE_1D | view::TEXTURE_2D | view::TEXTURE_3D | view::TEXTURE_CUBE => {
                (w(0), w(1), 0, 1)
            }
            view::TEXTURE_1D_ARRAY | view::TEXTURE_CUBE_ARRAY => (w(0), w(1), w(2), w(3)),
            view::TEXTURE_2D_ARRAY => (w(0), w(1), w(2), w(3)),
            view::TEXTURE_2D_MS_ARRAY => (0, 1, w(0), w(1)),
            _ => (0, 1, 0, 1),
        }
    }
}

impl UnorderedViewDesc {
    /// For a buffer view: (first element, element count, stride, flags).
    pub fn buffer(&self) -> (u64, u32, u32, u32) {
        let f = self.fields;
        (f[0], f[1] as u32, (f[1] >> 32) as u32, f[3] as u32)
    }
}

/// Where one subresource lies in a buffer laid out for a copy
/// (`D3D12_PLACED_SUBRESOURCE_FOOTPRINT`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Footprint {
    pub offset: u64,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub row_pitch: u32,
}

const _: () = assert!(size_of::<Footprint>() == 32);

/// A box of texels (`D3D12_BOX`): left, top, front inclusive; right,
/// bottom, back exclusive.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub left: u32,
    pub top: u32,
    pub front: u32,
    pub right: u32,
    pub bottom: u32,
    pub back: u32,
}

/// One end of a texture copy.
#[derive(Debug)]
pub enum Location<'a, R> {
    /// Subresource `index` of a texture (mip + slice * mips + plane * ...).
    Subresource(&'a R, u32),
    /// A buffer region laid out as `Footprint` says.
    Footprint(&'a R, Footprint),
}

/// Every subresource, in a transition barrier.
pub const ALL_SUBRESOURCES: u32 = u32::MAX;

/// An ordering point between commands of one list.
#[derive(Debug)]
pub enum Barrier<'a, R> {
    /// `resource` (or one subresource) moves from state `before` to `after`
    /// (D3D12 numbering). Backends that track hazards themselves (Metal)
    /// may record nothing.
    Transition {
        resource: &'a R,
        subresource: u32,
        before: u32,
        after: u32,
    },
    /// Unordered-access writes to `resource` (any, when `None`) finish
    /// before later commands read or write it.
    Unordered(Option<&'a R>),
    /// `after` starts using memory `before` used (placed resources).
    Aliasing(Option<&'a R>, Option<&'a R>),
}

/// Shader code in a form one backend takes.
#[derive(Debug, Clone, Copy)]
pub enum ShaderCode<'a> {
    /// A DXIL container (what victory-shader emits). D3D12 takes it as is;
    /// Metal needs it converted first (Metal Shader Converter, slice 3b).
    Dxil(&'a [u8]),
    /// Metal Shading Language source (Metal only: the backend's own
    /// helper shaders).
    Msl(&'a str),
    /// A compiled Metal library (Metal only).
    MetalLib(&'a [u8]),
}

/// A compute pipeline.
///
/// Its root signature says where a dispatch's arguments go: walking its
/// parameters in order, each root constants parameter takes the next
/// `count` of the dispatch's constants, and each root CBV / SRV / UAV the
/// next binding, and each descriptor table the next [`Table`].
///
/// D3D12 and converted DXIL on Metal: `root_signature` is serialized
/// ([`root`]: a container or a bare `RTS0` part); empty means the one
/// embedded in the DXIL. Metal's own shaders (MSL, metallib) have none:
/// constants at buffer index 0, bindings from 1.
#[derive(Debug, Clone, Copy)]
pub struct ComputeDesc<'a> {
    pub code: ShaderCode<'a>,
    /// Entry point name (Metal; D3D12 takes the container's).
    pub entry: &'a str,
    pub root_signature: &'a [u8],
    /// Threads per group, as the shader declares them.
    pub threads: [u32; 3],
}

/// A buffer bound for a dispatch.
#[derive(Debug)]
pub enum Binding<'a, R> {
    /// Read-only (root SRV or CBV) from `offset`.
    Read(&'a R, u64),
    /// Read-write (root UAV) from `offset`.
    Write(&'a R, u64),
}

/// A descriptor table for a dispatch: descriptors of `heap` (a
/// shader-visible view or sampler heap) from index `first`. The root
/// signature's ranges say which of them the shader reads.
#[derive(Debug)]
pub struct Table<'a, H> {
    pub heap: &'a H,
    pub first: u32,
}

impl<H> Clone for Table<'_, H> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<H> Copy for Table<'_, H> {}

impl<R> Clone for Binding<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R> Copy for Binding<'_, R> {}

// Both hold only references: copyable whatever the resource type is.
impl<R> Clone for Location<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R> Copy for Location<'_, R> {}
impl<R> Clone for Barrier<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R> Copy for Barrier<'_, R> {}

/// Recording into one open command list (slice 2a: barriers and copies).
/// Methods are unsafe: the caller keeps the list open and every resource
/// alive until the list has run, and ranges inside their resources.
pub trait Recorder {
    type Resource;
    type Pipeline;
    type DescriptorHeap;

    /// Runs `pipeline` over `groups` thread groups with `constants`,
    /// `bindings` and `tables` (see [`ComputeDesc`]).
    ///
    /// # Safety
    /// See the trait; the bindings stay inside their buffers, the tables
    /// inside their heaps, and every descriptor the shader reads through a
    /// table names a live resource (or is null).
    unsafe fn dispatch(
        &self,
        pipeline: &Self::Pipeline,
        constants: &[u32],
        bindings: &[Binding<'_, Self::Resource>],
        tables: &[Table<'_, Self::DescriptorHeap>],
        groups: [u32; 3],
    );

    /// # Safety
    /// See the trait.
    unsafe fn barriers(&self, barriers: &[Barrier<'_, Self::Resource>]);

    /// `bytes` from `from` at `from_offset` to `to` at `to_offset`.
    ///
    /// # Safety
    /// See the trait.
    unsafe fn copy_buffer(
        &self,
        to: &Self::Resource,
        to_offset: u64,
        from: &Self::Resource,
        from_offset: u64,
        bytes: u64,
    );

    /// The whole of `from` to `to` (same shape).
    ///
    /// # Safety
    /// See the trait.
    unsafe fn copy_resource(&self, to: &Self::Resource, from: &Self::Resource);

    /// `region` of `from` (all of it when `None`) to `to` at texel `at`.
    /// At least one end is a texture subresource.
    ///
    /// # Safety
    /// See the trait.
    unsafe fn copy_texture(
        &self,
        to: Location<'_, Self::Resource>,
        at: [u32; 3],
        from: Location<'_, Self::Resource>,
        region: Option<&Region>,
    );
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
    /// A compiled compute pipeline.
    type Pipeline;
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

    /// The CPU address of `heap`'s first descriptor; slot `i` is at
    /// `start + i * step`. The view methods below write at such addresses.
    fn descriptor_start(&self, heap: &Self::DescriptorHeap) -> usize;

    /// A shader resource view of `resource` at `at`; a null view (reads
    /// zeros) without one. With no `desc`, the whole resource as it is.
    ///
    /// # Safety
    /// `at` is a slot of a live view heap of this device.
    unsafe fn shader_view(
        &self,
        resource: Option<&Self::Resource>,
        desc: Option<&ShaderViewDesc>,
        at: usize,
    );

    /// An unordered access view of `resource` at `at` (null: reads zeros,
    /// writes dropped).
    ///
    /// # Safety
    /// As for [`Device::shader_view`].
    unsafe fn unordered_view(
        &self,
        resource: Option<&Self::Resource>,
        desc: Option<&UnorderedViewDesc>,
        at: usize,
    );

    /// A constant buffer view of `bytes` at GPU `address`.
    ///
    /// # Safety
    /// As for [`Device::shader_view`]; the memory stays live while the
    /// view is used.
    unsafe fn constant_view(&self, address: u64, bytes: u32, at: usize);

    /// A sampler at `at`.
    ///
    /// # Safety
    /// `at` is a slot of a live sampler heap of this device.
    unsafe fn sampler(&self, desc: &SamplerDesc, at: usize);

    /// A render target view at `at`.
    ///
    /// # Safety
    /// `at` is a slot of a live target heap of this device.
    unsafe fn target_view(
        &self,
        resource: Option<&Self::Resource>,
        desc: Option<&TargetViewDesc>,
        at: usize,
    );

    /// A depth-stencil view at `at`.
    ///
    /// # Safety
    /// `at` is a slot of a live depth heap of this device.
    unsafe fn depth_view(
        &self,
        resource: Option<&Self::Resource>,
        desc: Option<&DepthViewDesc>,
        at: usize,
    );

    /// A compute pipeline (slice 3a's binding model, [`ComputeDesc`]).
    fn compute_pipeline(&self, desc: &ComputeDesc<'_>) -> Result<Self::Pipeline, Self::Error>;

    /// Copy footprints of subresources `first..first + count` of `desc`,
    /// one per entry of `out` (its length is `count`), placed from offset
    /// 0; returns the total bytes (as [`Device::copyable_bytes`]).
    fn copy_layout(&self, desc: &ResourceDesc, first: u32, out: &mut [Footprint]) -> u64;

    /// Copies `count` descriptors of `kind` from `from` to `to`.
    ///
    /// # Safety
    /// Both ranges are slots of live heaps of `kind` of this device.
    unsafe fn copy_descriptors(&self, count: u32, to: usize, from: usize, kind: Descriptors);
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
    fn view_descriptions_read_their_fields() {
        // A raw buffer SRV: first element 16, 64 elements, stride 0, raw.
        let v = ShaderViewDesc {
            format: 39,
            dimension: view::BUFFER,
            mapping: 0x1688,
            _pad: 0,
            fields: [16, 64, u64::from(view::BUFFER_RAW)],
        };
        assert_eq!(v.buffer(), (16, 64, 0, view::BUFFER_RAW));
        // A 2D array view: mip 1, 3 mips, slices 2..6.
        let a = ShaderViewDesc {
            format: 28,
            dimension: view::TEXTURE_2D_ARRAY,
            mapping: 0x1688,
            _pad: 0,
            fields: [1 | (3 << 32), 2 | (4 << 32), 0],
        };
        assert_eq!(a.texture(), (1, 3, 2, 4));
    }

    #[test]
    fn buffer_is_row_major_and_single() {
        let b = ResourceDesc::buffer(4096, flags::UNORDERED_ACCESS);
        assert_eq!(
            (b.dimension, b.width, b.height, b.layout),
            (dimension::BUFFER, 4096, 1, ROW_MAJOR)
        );
        assert_eq!(
            (b.mip_levels, b.depth_or_array_size, b.sample_count),
            (1, 1, 1)
        );
    }
}
