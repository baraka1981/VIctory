//! The D3D12 side of the host seam ([`victory_host::Device`]).
//!
//! Descriptions and states are D3D12's already, so this is pass-through:
//! the console-side code calls the trait and stays free of the D3D12
//! device API, which a Metal backend then replaces.

use victory_host::{
    Arguments, Binding, ComputeDesc, Draw, Geometry, GraphicsDesc, Pass, ShaderCode, Table, Vertices,
    Barrier, Footprint, Location, Recorder, Region,
    Allocation, ClearValue, DepthViewDesc, Descriptors, Device, HeapUse, Memory, ResourceDesc,
    SamplerDesc, ShaderViewDesc, TargetViewDesc, UnorderedViewDesc,
};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_CONSTANT_BUFFER_VIEW_DESC, D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DEPTH_STENCIL_VIEW_DESC,
    D3D12_RENDER_TARGET_VIEW_DESC, D3D12_SAMPLER_DESC, D3D12_SHADER_RESOURCE_VIEW_DESC,
    D3D12_UNORDERED_ACCESS_VIEW_DESC, D3D12_BOX, D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    D3D12_RESOURCE_ALIASING_BARRIER, D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
    D3D12_RESOURCE_BARRIER_FLAG_NONE, D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
    D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_BARRIER_TYPE_UAV,
    D3D12_RESOURCE_TRANSITION_BARRIER, D3D12_RESOURCE_UAV_BARRIER, D3D12_TEXTURE_COPY_LOCATION,
    D3D12_TEXTURE_COPY_LOCATION_0, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
    D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, ID3D12GraphicsCommandList,
    D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_SHADER_BYTECODE, ID3D12PipelineState,
    ID3D12RootSignature,
};
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

macro_rules! same_layout {
    ($ours:ty, $d3d12:ty, $to:ident, $from:ident) => {
        const _: () = assert!(size_of::<$ours>() == size_of::<$d3d12>());
        const _: () = assert!(align_of::<$ours>() <= align_of::<$d3d12>());
        /// The seam's description of a D3D12 one (same layout).
        #[allow(dead_code)]
        pub(crate) fn $from(desc: &$d3d12) -> &$ours {
            // SAFETY: repr(C) with D3D12's fields at D3D12's offsets.
            unsafe { &*(desc as *const $d3d12).cast::<$ours>() }
        }
        fn $to(desc: &$ours) -> &$d3d12 {
            // SAFETY: as above; D3D12's alignment is at least ours and our
            // values came from D3D12-shaped data.
            unsafe { &*(desc as *const $ours).cast::<$d3d12>() }
        }
    };
}

same_layout!(ShaderViewDesc, D3D12_SHADER_RESOURCE_VIEW_DESC, to_srv, srv);
same_layout!(UnorderedViewDesc, D3D12_UNORDERED_ACCESS_VIEW_DESC, to_uav, uav);
same_layout!(TargetViewDesc, D3D12_RENDER_TARGET_VIEW_DESC, to_rtv, rtv);
same_layout!(DepthViewDesc, D3D12_DEPTH_STENCIL_VIEW_DESC, to_dsv, dsv);
same_layout!(SamplerDesc, D3D12_SAMPLER_DESC, to_sampler, sampler);
same_layout!(Footprint, D3D12_PLACED_SUBRESOURCE_FOOTPRINT, to_footprint, footprint);
same_layout!(Region, D3D12_BOX, to_box, region);

/// A borrowed COM pointer for a D3D12 struct field that never releases it.
fn borrowed(resource: &ID3D12Resource) -> std::mem::ManuallyDrop<Option<ID3D12Resource>> {
    // SAFETY: the copy is never dropped (ManuallyDrop), so the reference
    // count is untouched; the caller keeps the resource alive.
    std::mem::ManuallyDrop::new(Some(unsafe { std::mem::transmute_copy(resource) }))
}

fn barrier(b: &Barrier<'_, ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    let (kind, anonymous) = match *b {
        Barrier::Transition { resource, subresource, before, after } => (
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: borrowed(resource),
                    Subresource: subresource,
                    StateBefore: D3D12_RESOURCE_STATES(before as i32),
                    StateAfter: D3D12_RESOURCE_STATES(after as i32),
                }),
            },
        ),
        Barrier::Unordered(resource) => (
            D3D12_RESOURCE_BARRIER_TYPE_UAV,
            D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: resource.map_or(std::mem::ManuallyDrop::new(None), borrowed),
                }),
            },
        ),
        Barrier::Aliasing(before, after) => (
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: before.map_or(std::mem::ManuallyDrop::new(None), borrowed),
                    pResourceAfter: after.map_or(std::mem::ManuallyDrop::new(None), borrowed),
                }),
            },
        ),
    };
    D3D12_RESOURCE_BARRIER {
        Type: kind,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: anonymous,
    }
}

fn location(l: &Location<'_, ID3D12Resource>) -> D3D12_TEXTURE_COPY_LOCATION {
    match *l {
        Location::Subresource(r, index) => D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrowed(r),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index },
        },
        Location::Footprint(r, f) => D3D12_TEXTURE_COPY_LOCATION {
            pResource: borrowed(r),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *to_footprint(&f) },
        },
    }
}

/// The seam's barrier for a D3D12 one (callers that still build D3D12's).
fn barrier_of(b: &D3D12_RESOURCE_BARRIER) -> Barrier<'_, ID3D12Resource> {
    // SAFETY: each arm reads the union member its Type names.
    unsafe {
        match b.Type {
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => {
                let t = &b.Anonymous.Transition;
                match t.pResource.as_ref() {
                    Some(resource) => Barrier::Transition {
                        resource,
                        subresource: t.Subresource,
                        before: t.StateBefore.0 as u32,
                        after: t.StateAfter.0 as u32,
                    },
                    None => Barrier::Unordered(None),
                }
            }
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING => {
                let a = &b.Anonymous.Aliasing;
                Barrier::Aliasing(a.pResourceBefore.as_ref(), a.pResourceAfter.as_ref())
            }
            _ => Barrier::Unordered(b.Anonymous.UAV.pResource.as_ref()),
        }
    }
}

/// Records D3D12-built barriers through the seam, 16 at a time on the
/// stack (no allocation).
///
/// # Safety
/// As for [`Recorder::barriers`].
pub(crate) unsafe fn record_barriers(list: &ID3D12GraphicsCommandList, barriers: &[D3D12_RESOURCE_BARRIER]) {
    let rec = D3d12List(list);
    for chunk in barriers.chunks(16) {
        let mut seam = [Barrier::Unordered(None); 16];
        for (to, from) in seam.iter_mut().zip(chunk) {
            *to = barrier_of(from);
        }
        // SAFETY: the caller's list and resources.
        unsafe { rec.barriers(&seam[..chunk.len()]) };
    }
}

/// The seam's copy location for a D3D12 one.
pub(crate) fn location_of(l: &D3D12_TEXTURE_COPY_LOCATION) -> Location<'_, ID3D12Resource> {
    let r = l.pResource.as_ref().expect("a copy location names its resource");
    // SAFETY: each arm reads the union member its Type names.
    unsafe {
        if l.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT {
            Location::Footprint(r, *footprint(&l.Anonymous.PlacedFootprint))
        } else {
            Location::Subresource(r, l.Anonymous.SubresourceIndex)
        }
    }
}

/// The recorder for a D3D12 list.
pub(crate) fn rec(list: &ID3D12GraphicsCommandList) -> D3d12List<'_> {
    D3d12List(list)
}

/// A compute pipeline: its root signature, state, and the root
/// signature's parameters (where dispatch arguments go).
pub(crate) struct D3d12Pipeline {
    pub(crate) root: ID3D12RootSignature,
    pub(crate) state: ID3D12PipelineState,
    pub(crate) parameters: Vec<victory_host::root::Kind>,
}

/// A D3D12 command list seen through the host seam.
#[derive(Clone, Copy)]
pub(crate) struct D3d12List<'a>(pub(crate) &'a ID3D12GraphicsCommandList);

impl D3d12List<'_> {
    /// Sets the root signature's arguments as `ComputeDesc` says: walking
    /// its parameters, constants take the next constants, root
    /// descriptors the next binding, tables the next table (their heaps
    /// set first). `graphics`: on the graphics root signature.
    ///
    /// # Safety
    /// The open list, live resources and heaps, ranges inside them.
    unsafe fn root_arguments(
        &self,
        pipeline: &D3d12Pipeline,
        constants: &[u32],
        bindings: &[Binding<'_, ID3D12Resource>],
        tables: &[Table<'_, ID3D12DescriptorHeap>],
        graphics: bool,
    ) {
        use victory_host::root::{Kind, parameter};
        let l = self.0;
        // SAFETY: the caller's.
        unsafe {
            // The tables' heaps (one view heap, one sampler heap at most).
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = Vec::new();
            for t in tables {
                if !heaps.iter().flatten().any(|h| h == t.heap) {
                    heaps.push(Some(t.heap.clone()));
                }
            }
            if !heaps.is_empty() {
                l.SetDescriptorHeaps(&heaps);
            }
            let (mut constants, mut bindings, mut tables) = (constants, bindings.iter(), tables.iter());
            for (i, p) in pipeline.parameters.iter().enumerate() {
                let i = i as u32;
                match *p {
                    Kind::Constants { count, .. } => {
                        let n = (count as usize).min(constants.len());
                        if n > 0 {
                            let data = constants.as_ptr().cast();
                            if graphics {
                                l.SetGraphicsRoot32BitConstants(i, n as u32, data, 0);
                            } else {
                                l.SetComputeRoot32BitConstants(i, n as u32, data, 0);
                            }
                        }
                        constants = &constants[n..];
                    }
                    Kind::Descriptor { kind, .. } => {
                        let Some(&(Binding::Read(r, offset) | Binding::Write(r, offset))) = bindings.next() else {
                            continue;
                        };
                        let address = r.GetGPUVirtualAddress() + offset;
                        match (kind, graphics) {
                            (parameter::CBV, false) => l.SetComputeRootConstantBufferView(i, address),
                            (parameter::SRV, false) => l.SetComputeRootShaderResourceView(i, address),
                            (_, false) => l.SetComputeRootUnorderedAccessView(i, address),
                            (parameter::CBV, true) => l.SetGraphicsRootConstantBufferView(i, address),
                            (parameter::SRV, true) => l.SetGraphicsRootShaderResourceView(i, address),
                            (_, true) => l.SetGraphicsRootUnorderedAccessView(i, address),
                        }
                    }
                    Kind::Table(_) => {
                        let Some(t) = tables.next() else { continue };
                        let mut device: Option<ID3D12Device> = None;
                        if t.heap.GetDevice(&mut device).is_err() {
                            continue;
                        }
                        let Some(device) = device else { continue };
                        let step = device.GetDescriptorHandleIncrementSize(t.heap.GetDesc().Type);
                        let mut handle = t.heap.GetGPUDescriptorHandleForHeapStart();
                        handle.ptr += u64::from(t.first) * u64::from(step);
                        if graphics {
                            l.SetGraphicsRootDescriptorTable(i, handle);
                        } else {
                            l.SetComputeRootDescriptorTable(i, handle);
                        }
                    }
                }
            }
        }
    }

    /// Targets (cleared first when asked), viewport and scissor.
    ///
    /// # Safety
    /// The open list; the slots hold live views.
    unsafe fn pass(&self, pass: &Pass<'_>) {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_CLEAR_FLAG_DEPTH, D3D12_CLEAR_FLAG_STENCIL, D3D12_VIEWPORT,
        };
        let l = self.0;
        let rtvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> =
            pass.targets.iter().map(|t| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: t.at }).collect();
        let dsv = pass.depth.map(|d| D3D12_CPU_DESCRIPTOR_HANDLE { ptr: d.at });
        // SAFETY: the caller's.
        unsafe {
            l.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(|d| d as *const _),
            );
            for (t, h) in pass.targets.iter().zip(&rtvs) {
                if let Some(c) = t.clear {
                    l.ClearRenderTargetView(*h, &c, None);
                }
            }
            if let (Some(d), Some(h)) = (pass.depth, dsv)
                && let Some((depth, stencil)) = d.clear
            {
                l.ClearDepthStencilView(h, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, depth, stencil, None);
            }
            let v = pass.viewport;
            l.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: v.x,
                TopLeftY: v.y,
                Width: v.width,
                Height: v.height,
                MinDepth: v.min_depth,
                MaxDepth: v.max_depth,
            }]);
            let [left, top, right, bottom] = pass.scissor.unwrap_or([
                v.x.max(0.0) as u32,
                v.y.max(0.0) as u32,
                (v.x + v.width).max(0.0) as u32,
                (v.y + v.height).max(0.0) as u32,
            ]);
            l.RSSetScissorRects(&[RECT { left: left as i32, top: top as i32, right: right as i32, bottom: bottom as i32 }]);
        }
    }
}

impl Recorder for D3d12List<'_> {
    type Resource = ID3D12Resource;
    type Pipeline = D3d12Pipeline;
    type GraphicsPipeline = D3d12Pipeline;
    type DescriptorHeap = ID3D12DescriptorHeap;

    unsafe fn dispatch(
        &self,
        pipeline: &D3d12Pipeline,
        constants: &[u32],
        bindings: &[Binding<'_, ID3D12Resource>],
        tables: &[Table<'_, ID3D12DescriptorHeap>],
        groups: [u32; 3],
    ) {
        // SAFETY: the caller's open list, live resources and ranges.
        unsafe {
            self.0.SetComputeRootSignature(&pipeline.root);
            self.0.SetPipelineState(&pipeline.state);
            self.root_arguments(pipeline, constants, bindings, tables, false);
            self.0.Dispatch(groups[0], groups[1], groups[2]);
        }
    }

    unsafe fn draw(
        &self,
        pass: &Pass<'_>,
        pipeline: &D3d12Pipeline,
        arguments: &Arguments<'_, ID3D12Resource, ID3D12DescriptorHeap>,
        vertices: &Vertices<'_, ID3D12Resource>,
        draw: &Draw,
    ) {
        use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
        use windows::Win32::Graphics::Direct3D12::{D3D12_INDEX_BUFFER_VIEW, D3D12_VERTEX_BUFFER_VIEW};
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        let l = self.0;
        // SAFETY: the caller's open list, live resources, views and ranges.
        unsafe {
            l.SetGraphicsRootSignature(&pipeline.root);
            l.SetPipelineState(&pipeline.state);
            self.root_arguments(pipeline, arguments.constants, arguments.bindings, arguments.tables, true);
            self.pass(pass);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(vertices.primitive as i32));
            let size = |r: &ID3D12Resource, offset: u64| r.GetDesc().Width.saturating_sub(offset) as u32;
            let views: Vec<D3D12_VERTEX_BUFFER_VIEW> = vertices
                .buffers
                .iter()
                .map(|b| D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: b.resource.GetGPUVirtualAddress() + b.offset,
                    SizeInBytes: size(b.resource, b.offset),
                    StrideInBytes: b.stride,
                })
                .collect();
            if !views.is_empty() {
                l.IASetVertexBuffers(0, Some(&views));
            }
            match vertices.index {
                Some(i) => {
                    l.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                        BufferLocation: i.resource.GetGPUVirtualAddress() + i.offset,
                        SizeInBytes: size(i.resource, i.offset),
                        Format: DXGI_FORMAT(i.format as i32),
                    }));
                    l.DrawIndexedInstanced(draw.count, draw.instances, draw.first, draw.base_vertex, draw.first_instance);
                }
                None => l.DrawInstanced(draw.count, draw.instances, draw.first, draw.first_instance),
            }
        }
    }

    unsafe fn draw_mesh(
        &self,
        pass: &Pass<'_>,
        pipeline: &D3d12Pipeline,
        arguments: &Arguments<'_, ID3D12Resource, ID3D12DescriptorHeap>,
        groups: [u32; 3],
    ) {
        use windows::core::Interface;
        use windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList6;
        // SAFETY: as for draw.
        unsafe {
            let Ok(l6) = self.0.cast::<ID3D12GraphicsCommandList6>() else { return };
            self.0.SetGraphicsRootSignature(&pipeline.root);
            self.0.SetPipelineState(&pipeline.state);
            self.root_arguments(pipeline, arguments.constants, arguments.bindings, arguments.tables, true);
            self.pass(pass);
            l6.DispatchMesh(groups[0], groups[1], groups[2]);
        }
    }

    unsafe fn barriers(&self, barriers: &[Barrier<'_, ID3D12Resource>]) {
        if barriers.is_empty() {
            return;
        }
        let made: Vec<_> = barriers.iter().map(barrier).collect();
        // SAFETY: the caller's open list and live resources.
        unsafe { self.0.ResourceBarrier(&made) };
    }

    unsafe fn copy_buffer(&self, to: &ID3D12Resource, to_offset: u64, from: &ID3D12Resource, from_offset: u64, bytes: u64) {
        // SAFETY: as above; the caller keeps the ranges inside the buffers.
        unsafe { self.0.CopyBufferRegion(to, to_offset, from, from_offset, bytes) };
    }

    unsafe fn copy_resource(&self, to: &ID3D12Resource, from: &ID3D12Resource) {
        // SAFETY: as above.
        unsafe { self.0.CopyResource(to, from) };
    }

    unsafe fn copy_texture(
        &self,
        to: Location<'_, ID3D12Resource>,
        at: [u32; 3],
        from: Location<'_, ID3D12Resource>,
        region: Option<&Region>,
    ) {
        let (to, from) = (location(&to), location(&from));
        // SAFETY: as above.
        unsafe {
            self.0.CopyTextureRegion(&to, at[0], at[1], at[2], &from, region.map(|r| std::ptr::from_ref(to_box(r))))
        };
    }
}

fn handle(at: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
    D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at }
}

/// The console description of a D3D12 one (same layout).
pub(crate) fn console_desc(desc: &D3D12_RESOURCE_DESC) -> ResourceDesc {
    // SAFETY: as in `host_desc`.
    unsafe { *(desc as *const D3D12_RESOURCE_DESC).cast::<ResourceDesc>() }
}

impl Device for D3d12<'_> {
    type Resource = ID3D12Resource;
    type Heap = ID3D12Heap;
    type Pipeline = D3d12Pipeline;
    type GraphicsPipeline = D3d12Pipeline;
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

    fn compute_pipeline(&self, desc: &ComputeDesc<'_>) -> windows::core::Result<D3d12Pipeline> {
        let ShaderCode::Dxil(code) = desc.code else {
            return Err(windows::core::Error::from_hresult(windows::Win32::Foundation::E_INVALIDARG));
        };
        // No root signature given: the one embedded in the shader.
        let blob = if desc.root_signature.is_empty() { code } else { desc.root_signature };
        let parameters = victory_host::root::parse(blob)
            .map_err(|_| windows::core::Error::from_hresult(windows::Win32::Foundation::E_INVALIDARG))?
            .parameters
            .into_iter()
            .map(|p| p.kind)
            .collect();
        // SAFETY: a serialized root signature (or a DXIL container with
        // one) and a DXIL container; D3D12 validates both.
        unsafe {
            let root: ID3D12RootSignature = self.0.CreateRootSignature(0, blob)?;
            let state: ID3D12PipelineState = self.0.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(std::mem::transmute_copy(&root))),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })?;
            Ok(D3d12Pipeline { root, state, parameters })
        }
    }

    /// Vertex + pixel pipelines; mesh pipelines (a pipeline state stream)
    /// are not done on D3D12 yet (`E_NOTIMPL`).
    fn graphics_pipeline(&self, desc: &GraphicsDesc<'_>) -> windows::core::Result<D3d12Pipeline> {
        use windows::Win32::Foundation::{E_INVALIDARG, E_NOTIMPL};
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_BLEND, D3D12_BLEND_DESC, D3D12_BLEND_OP, D3D12_COMPARISON_FUNC, D3D12_CULL_MODE,
            D3D12_DEPTH_STENCIL_DESC, D3D12_DEPTH_WRITE_MASK, D3D12_FILL_MODE,
            D3D12_GRAPHICS_PIPELINE_STATE_DESC, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
            D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, D3D12_INPUT_ELEMENT_DESC,
            D3D12_INPUT_LAYOUT_DESC, D3D12_PRIMITIVE_TOPOLOGY_TYPE, D3D12_RASTERIZER_DESC,
            D3D12_RENDER_TARGET_BLEND_DESC,
        };
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
        let invalid = || windows::core::Error::from_hresult(E_INVALIDARG);
        let dxil = |code: ShaderCode<'_>| match code {
            ShaderCode::Dxil(d) => Ok(D3D12_SHADER_BYTECODE { pShaderBytecode: d.as_ptr().cast(), BytecodeLength: d.len() }),
            _ => Err(invalid()),
        };
        let Geometry::Vertex { shader, inputs } = desc.geometry else {
            return Err(windows::core::Error::from_hresult(E_NOTIMPL));
        };
        let ShaderCode::Dxil(vs) = shader.code else { return Err(invalid()) };
        // No root signature given: the one embedded in the vertex shader.
        let blob = if desc.root_signature.is_empty() { vs } else { desc.root_signature };
        let parameters = victory_host::root::parse(blob)
            .map_err(|_| invalid())?
            .parameters
            .into_iter()
            .map(|p| p.kind)
            .collect();
        let names: Vec<std::ffi::CString> = inputs
            .iter()
            .map(|e| std::ffi::CString::new(e.semantic).map_err(|_| invalid()))
            .collect::<Result<_, _>>()?;
        let elements: Vec<D3D12_INPUT_ELEMENT_DESC> = inputs
            .iter()
            .zip(&names)
            .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: windows::core::PCSTR(name.as_ptr().cast()),
                SemanticIndex: e.index,
                Format: DXGI_FORMAT(e.format as i32),
                InputSlot: e.slot,
                AlignedByteOffset: e.offset,
                InputSlotClass: if e.per_instance {
                    D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                } else {
                    D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA
                },
                InstanceDataStepRate: if e.per_instance { e.step_rate } else { 0 },
            })
            .collect();
        let blend = |b: &victory_host::Blend| D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: b.enable.into(),
            LogicOpEnable: false.into(),
            SrcBlend: D3D12_BLEND(b.src as i32),
            DestBlend: D3D12_BLEND(b.dst as i32),
            BlendOp: D3D12_BLEND_OP(b.op as i32),
            SrcBlendAlpha: D3D12_BLEND(b.src_alpha as i32),
            DestBlendAlpha: D3D12_BLEND(b.dst_alpha as i32),
            BlendOpAlpha: D3D12_BLEND_OP(b.op_alpha as i32),
            LogicOp: Default::default(),
            RenderTargetWriteMask: b.write_mask,
        };
        let mut formats = [DXGI_FORMAT(0); 8];
        for (f, &t) in formats.iter_mut().zip(desc.targets) {
            *f = DXGI_FORMAT(t as i32);
        }
        let r = desc.raster;
        // SAFETY: a serialized root signature (or a DXIL container with
        // one), DXIL stages and descriptions D3D12 validates; the name
        // strings outlive the call.
        unsafe {
            let root: ID3D12RootSignature = self.0.CreateRootSignature(0, blob)?;
            let pso = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(std::mem::transmute_copy(&root))),
                VS: dxil(shader.code)?,
                PS: match desc.pixel {
                    Some(p) => dxil(p.code)?,
                    None => D3D12_SHADER_BYTECODE::default(),
                },
                BlendState: D3D12_BLEND_DESC {
                    AlphaToCoverageEnable: false.into(),
                    IndependentBlendEnable: true.into(),
                    RenderTarget: desc.blend.map(|b| blend(&b)),
                },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE(r.fill as i32),
                    CullMode: D3D12_CULL_MODE(r.cull as i32),
                    FrontCounterClockwise: r.front_ccw.into(),
                    DepthBias: r.depth_bias,
                    DepthBiasClamp: r.depth_bias_clamp,
                    SlopeScaledDepthBias: r.slope_scaled_depth_bias,
                    DepthClipEnable: r.depth_clip.into(),
                    ..Default::default()
                },
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: desc.depth.enable.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK(i32::from(desc.depth.write)),
                    DepthFunc: D3D12_COMPARISON_FUNC(desc.depth.func as i32),
                    ..Default::default()
                },
                InputLayout: D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: elements.as_ptr(),
                    NumElements: elements.len() as u32,
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE(desc.topology as i32),
                NumRenderTargets: desc.targets.len().min(8) as u32,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT(desc.depth_format as i32),
                SampleDesc: DXGI_SAMPLE_DESC { Count: desc.samples.max(1), Quality: 0 },
                ..Default::default()
            };
            let state: ID3D12PipelineState = self.0.CreateGraphicsPipelineState(&pso)?;
            Ok(D3d12Pipeline { root, state, parameters })
        }
    }

    fn copy_layout(&self, desc: &ResourceDesc, first: u32, out: &mut [Footprint]) -> u64 {
        let mut total = 0;
        // SAFETY: a plain query writing `out.len()` footprints, laid out as
        // D3D12's (same_layout above).
        unsafe {
            self.0.GetCopyableFootprints(
                host_desc(desc),
                first,
                out.len() as u32,
                0,
                Some(out.as_mut_ptr().cast()),
                None,
                None,
                Some(&mut total),
            );
        }
        total
    }

    fn descriptor_start(&self, heap: &ID3D12DescriptorHeap) -> usize {
        // SAFETY: a live heap.
        unsafe { heap.GetCPUDescriptorHandleForHeapStart() }.ptr
    }

    unsafe fn shader_view(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&ShaderViewDesc>,
        at: usize,
    ) {
        // SAFETY: the caller's slot; D3D12 checks the description.
        unsafe {
            self.0.CreateShaderResourceView(resource, desc.map(|d| std::ptr::from_ref(to_srv(d))), handle(at))
        }
    }

    unsafe fn unordered_view(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&UnorderedViewDesc>,
        at: usize,
    ) {
        // SAFETY: as above.
        unsafe {
            self.0.CreateUnorderedAccessView(
                resource,
                None::<&ID3D12Resource>,
                desc.map(|d| std::ptr::from_ref(to_uav(d))),
                handle(at),
            )
        }
    }

    unsafe fn constant_view(&self, address: u64, bytes: u32, at: usize) {
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: address,
            SizeInBytes: bytes,
        };
        // SAFETY: as above.
        unsafe { self.0.CreateConstantBufferView(Some(&desc), handle(at)) }
    }

    unsafe fn sampler(&self, desc: &SamplerDesc, at: usize) {
        // SAFETY: as above.
        unsafe { self.0.CreateSampler(to_sampler(desc), handle(at)) }
    }

    unsafe fn target_view(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&TargetViewDesc>,
        at: usize,
    ) {
        // SAFETY: as above.
        unsafe {
            self.0.CreateRenderTargetView(resource, desc.map(|d| std::ptr::from_ref(to_rtv(d))), handle(at))
        }
    }

    unsafe fn depth_view(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&DepthViewDesc>,
        at: usize,
    ) {
        // SAFETY: as above.
        unsafe {
            self.0.CreateDepthStencilView(resource, desc.map(|d| std::ptr::from_ref(to_dsv(d))), handle(at))
        }
    }

    unsafe fn copy_descriptors(&self, count: u32, to: usize, from: usize, kind: Descriptors) {
        // SAFETY: the caller's ranges.
        unsafe { self.0.CopyDescriptorsSimple(count, handle(to), handle(from), descriptor_type(kind)) }
    }
}
