//! The D3D12 side of the host seam ([`victory_host::Device`]).
//!
//! Descriptions and states are D3D12's already, so this is pass-through:
//! the console-side code calls the trait and stays free of the D3D12
//! device API, which a Metal backend then replaces.

use victory_host::{Allocation, ClearValue, Descriptors, Device, HeapUse, Memory, ResourceDesc};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_HEAP_TYPE_CUSTOM, D3D12_HEAP_TYPE_GPU_UPLOAD,
    D3D12_MEMORY_POOL_L0,
    D3D12_DESCRIPTOR_HEAP_DESC, D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
    D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, D3D12_DESCRIPTOR_HEAP_TYPE,
    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
    D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_HEAP_FLAG_NONE,
    D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_TYPE_READBACK,
    D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_DESC, D3D12_RESOURCE_STATES, ID3D12DescriptorHeap,
    ID3D12Device, ID3D12Heap, ID3D12Resource, D3D12_HEAP_DESC, D3D12_HEAP_FLAGS,
    D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
    D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
};

/// A D3D12 device seen through the host seam.
#[derive(Clone, Copy)]
pub(crate) struct D3d12<'a>(pub(crate) &'a ID3D12Device);

/// The console description as D3D12's: same layout (checked in
/// victory-host's tests).
fn host_desc(desc: &ResourceDesc) -> &D3D12_RESOURCE_DESC {
    const _: () = assert!(size_of::<ResourceDesc>() == size_of::<D3D12_RESOURCE_DESC>());
    // SAFETY: both are repr(C) with the same fields at the same offsets.
    unsafe { &*(desc as *const ResourceDesc).cast::<D3D12_RESOURCE_DESC>() }
}

pub(crate) fn heap_properties(memory: Memory) -> D3D12_HEAP_PROPERTIES {
    let kind = match memory {
        Memory::Device => D3D12_HEAP_TYPE_DEFAULT,
        Memory::Upload => D3D12_HEAP_TYPE_UPLOAD,
        Memory::Readback => D3D12_HEAP_TYPE_READBACK,
        Memory::DeviceUpload => D3D12_HEAP_TYPE_GPU_UPLOAD,
        Memory::Shared => {
            return D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
        }
    };
    D3D12_HEAP_PROPERTIES {
        Type: kind,
        ..Default::default()
    }
}

/// The seam's memory kind for a D3D12 heap type (for callers that still
/// carry D3D12's); `None` for custom heaps.
pub(crate) fn memory(kind: D3D12_HEAP_TYPE) -> Option<Memory> {
    Some(match kind {
        D3D12_HEAP_TYPE_DEFAULT => Memory::Device,
        D3D12_HEAP_TYPE_UPLOAD => Memory::Upload,
        D3D12_HEAP_TYPE_READBACK => Memory::Readback,
        D3D12_HEAP_TYPE_GPU_UPLOAD => Memory::DeviceUpload,
        _ => return None,
    })
}

pub(crate) fn heap_flags(use_: HeapUse) -> D3D12_HEAP_FLAGS {
    match use_ {
        HeapUse::Buffers => D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        HeapUse::Textures => D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        HeapUse::Targets => D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
    }
}

pub(crate) fn descriptor_type(kind: Descriptors) -> D3D12_DESCRIPTOR_HEAP_TYPE {
    match kind {
        Descriptors::Views => D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
        Descriptors::Samplers => D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        Descriptors::Targets => D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        Descriptors::Depth => D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
    }
}

/// The seam's descriptor kind for a D3D12 heap type (for callers that
/// still carry D3D12's).
pub(crate) fn descriptors(kind: D3D12_DESCRIPTOR_HEAP_TYPE) -> Descriptors {
    match kind {
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER => Descriptors::Samplers,
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV => Descriptors::Targets,
        D3D12_DESCRIPTOR_HEAP_TYPE_DSV => Descriptors::Depth,
        _ => Descriptors::Views,
    }
}

/// The console description of a D3D12 one (same layout).
pub(crate) fn console_desc(desc: &D3D12_RESOURCE_DESC) -> ResourceDesc {
    // SAFETY: as in `host_desc`.
    unsafe { *(desc as *const D3D12_RESOURCE_DESC).cast::<ResourceDesc>() }
}

impl Device for D3d12<'_> {
    type Resource = ID3D12Resource;
    type Heap = ID3D12Heap;
    type DescriptorHeap = ID3D12DescriptorHeap;
    type Error = windows::core::Error;

    fn committed(
        &self,
        memory: Memory,
        desc: &ResourceDesc,
        state: u32,
        clear: Option<&ClearValue>,
    ) -> windows::core::Result<ID3D12Resource> {
        let heap = heap_properties(memory);
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: the description is a D3D12 one (above); the out pointer
        // is ours.
        unsafe {
            self.0.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                host_desc(desc),
                D3D12_RESOURCE_STATES(state as i32),
                // SAFETY: D3D12_CLEAR_VALUE is the format and 16 bytes.
                clear.map(|v| std::ptr::from_ref(v).cast()),
                &mut out,
            )?;
        }
        out.ok_or_else(windows::core::Error::empty)
    }

    fn reserved(&self, desc: &ResourceDesc, state: u32) -> windows::core::Result<ID3D12Resource> {
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: as for `committed`.
        unsafe {
            self.0.CreateReservedResource(
                host_desc(desc),
                D3D12_RESOURCE_STATES(state as i32),
                None,
                &mut out,
            )?;
        }
        out.ok_or_else(windows::core::Error::empty)
    }

    fn allocation(&self, desc: &ResourceDesc) -> Allocation {
        // SAFETY: a plain query.
        let info = unsafe { self.0.GetResourceAllocationInfo(0, &[*host_desc(desc)]) };
        Allocation {
            size: info.SizeInBytes,
            alignment: info.Alignment,
        }
    }

    fn heap(
        &self,
        memory: Memory,
        bytes: u64,
        alignment: u64,
        use_: HeapUse,
    ) -> windows::core::Result<ID3D12Heap> {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: bytes,
            Alignment: alignment,
            Properties: heap_properties(memory),
            Flags: heap_flags(use_),
        };
        let mut out: Option<ID3D12Heap> = None;
        // SAFETY: a valid description; the out pointer is ours.
        unsafe { self.0.CreateHeap(&desc, &mut out)? };
        out.ok_or_else(windows::core::Error::empty)
    }

    fn placed(
        &self,
        heap: &ID3D12Heap,
        offset: u64,
        desc: &ResourceDesc,
        state: u32,
    ) -> windows::core::Result<ID3D12Resource> {
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: the caller keeps `offset` aligned and in the heap; the out
        // pointer is ours.
        unsafe {
            self.0.CreatePlacedResource(
                heap,
                offset,
                host_desc(desc),
                D3D12_RESOURCE_STATES(state as i32),
                None,
                &mut out,
            )?;
        }
        out.ok_or_else(windows::core::Error::empty)
    }

    fn copyable_bytes(&self, desc: &ResourceDesc, first: u32, count: u32) -> u64 {
        let mut bytes = 0;
        // SAFETY: a plain query; the out pointer is ours.
        unsafe {
            self.0.GetCopyableFootprints(
                host_desc(desc),
                first,
                count,
                0,
                None,
                None,
                None,
                Some(&mut bytes),
            );
        }
        bytes
    }

    fn descriptor_heap(
        &self,
        kind: Descriptors,
        count: u32,
        shader_visible: bool,
    ) -> windows::core::Result<ID3D12DescriptorHeap> {
        let desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: descriptor_type(kind),
            NumDescriptors: count,
            Flags: if shader_visible {
                D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
            } else {
                D3D12_DESCRIPTOR_HEAP_FLAG_NONE
            },
            NodeMask: 0,
        };
        // SAFETY: a valid description.
        unsafe { self.0.CreateDescriptorHeap(&desc) }
    }

    fn descriptor_step(&self, kind: Descriptors) -> u32 {
        // SAFETY: a plain query.
        unsafe { self.0.GetDescriptorHandleIncrementSize(descriptor_type(kind)) }
    }
}
