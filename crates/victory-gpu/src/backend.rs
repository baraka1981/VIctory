//! The D3D12 side of the host seam ([`victory_host::Device`]).
//!
//! Descriptions and states are D3D12's already, so this is pass-through:
//! the console-side code calls the trait and stays free of the D3D12
//! device API, which a Metal backend then replaces.

use victory_host::{
    Arguments, Binding, ComputeDesc, Draw, GraphicsDesc, Pass, Point, ShaderCode, Vertices,
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
        arguments: &Arguments<'_, ID3D12Resource, ID3D12DescriptorHeap>,
        graphics: bool,
    ) {
        use victory_host::root::{Kind, parameter};
        let Arguments { constants, bindings, tables, views, samplers } = *arguments;
        let l = self.0;
        // SAFETY: the caller's.
        unsafe {
            // The tables' and the directly indexed heaps (one view heap,
            // one sampler heap at most).
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = Vec::new();
            for h in tables.iter().map(|t| t.heap).chain(views).chain(samplers) {
                if !heaps.iter().flatten().any(|x| x == h) {
                    heaps.push(Some(h.clone()));
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
            l.OMSetStencilRef(pass.stencil_ref);
            l.OMSetBlendFactor(Some(&pass.blend_factor));
        }
    }
}

/// The command signature for one indirect argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once per process.
pub(crate) fn signature(kind: u32, stride: u32) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12CommandSignature> {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_COMMAND_SIGNATURE_DESC, D3D12_INDIRECT_ARGUMENT_DESC, D3D12_INDIRECT_ARGUMENT_TYPE,
        ID3D12CommandSignature,
    };
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> = std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut signature)
    };
    if let Err(e) = result {
        tracing::warn!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

fn rects_of(rects: &[victory_host::Rect]) -> Vec<windows::Win32::Foundation::RECT> {
    rects
        .iter()
        .map(|&[left, top, right, bottom]| windows::Win32::Foundation::RECT { left, top, right, bottom })
        .collect()
}

impl Recorder for D3d12List<'_> {
    type Resource = ID3D12Resource;
    type Pipeline = D3d12Pipeline;
    type GraphicsPipeline = D3d12Pipeline;
    type DescriptorHeap = ID3D12DescriptorHeap;

    unsafe fn dispatch(
        &self,
        pipeline: &D3d12Pipeline,
        arguments: &Arguments<'_, ID3D12Resource, ID3D12DescriptorHeap>,
        groups: [u32; 3],
    ) {
        // SAFETY: the caller's open list, live resources and ranges.
        unsafe {
            self.0.SetComputeRootSignature(&pipeline.root);
            self.0.SetPipelineState(&pipeline.state);
            self.root_arguments(pipeline, arguments, false);
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
            self.root_arguments(pipeline, arguments, true);
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
            self.root_arguments(pipeline, arguments, true);
            self.pass(pass);
            l6.DispatchMesh(groups[0], groups[1], groups[2]);
        }
    }

    unsafe fn set_heaps(&self, heaps: &[&ID3D12DescriptorHeap]) {
        let heaps: Vec<Option<ID3D12DescriptorHeap>> = heaps.iter().map(|h| Some((*h).clone())).collect();
        // SAFETY: the caller's open list and shader-visible heaps.
        unsafe { self.0.SetDescriptorHeaps(&heaps) };
    }

    unsafe fn set_compute_root(&self, pipeline: &D3d12Pipeline) {
        // SAFETY: the caller's.
        unsafe { self.0.SetComputeRootSignature(&pipeline.root) };
    }

    unsafe fn set_graphics_root(&self, pipeline: &D3d12Pipeline) {
        // SAFETY: the caller's.
        unsafe { self.0.SetGraphicsRootSignature(&pipeline.root) };
    }

    unsafe fn set_compute_pipeline(&self, pipeline: &D3d12Pipeline) {
        // SAFETY: the caller's.
        unsafe { self.0.SetPipelineState(&pipeline.state) };
    }

    unsafe fn set_graphics_pipeline(&self, pipeline: &D3d12Pipeline) {
        // SAFETY: the caller's.
        unsafe { self.0.SetPipelineState(&pipeline.state) };
    }

    unsafe fn set_root_constants(&self, point: Point, param: u32, offset: u32, values: &[u32]) {
        let (n, data) = (values.len() as u32, values.as_ptr().cast());
        // SAFETY: the caller's; `values` covers `n` words.
        unsafe {
            match point {
                Point::Compute => self.0.SetComputeRoot32BitConstants(param, n, data, offset),
                Point::Graphics => self.0.SetGraphicsRoot32BitConstants(param, n, data, offset),
            }
        }
    }

    unsafe fn set_root_view(&self, point: Point, param: u32, kind: u32, address: u64) {
        use victory_host::root::parameter;
        // SAFETY: the caller's.
        unsafe {
            match (point, kind) {
                (Point::Compute, parameter::CBV) => self.0.SetComputeRootConstantBufferView(param, address),
                (Point::Compute, parameter::UAV) => self.0.SetComputeRootUnorderedAccessView(param, address),
                (Point::Compute, _) => self.0.SetComputeRootShaderResourceView(param, address),
                (Point::Graphics, parameter::CBV) => self.0.SetGraphicsRootConstantBufferView(param, address),
                (Point::Graphics, parameter::UAV) => self.0.SetGraphicsRootUnorderedAccessView(param, address),
                (Point::Graphics, _) => self.0.SetGraphicsRootShaderResourceView(param, address),
            }
        }
    }

    unsafe fn set_root_table(&self, point: Point, param: u32, handle: u64) {
        let handle = windows::Win32::Graphics::Direct3D12::D3D12_GPU_DESCRIPTOR_HANDLE { ptr: handle };
        // SAFETY: the caller's.
        unsafe {
            match point {
                Point::Compute => self.0.SetComputeRootDescriptorTable(param, handle),
                Point::Graphics => self.0.SetGraphicsRootDescriptorTable(param, handle),
            }
        }
    }

    unsafe fn set_targets(&self, targets: &[usize], depth: Option<usize>) {
        let rtvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = targets.iter().map(|&ptr| D3D12_CPU_DESCRIPTOR_HANDLE { ptr }).collect();
        let dsv = depth.map(|ptr| D3D12_CPU_DESCRIPTOR_HANDLE { ptr });
        // SAFETY: the caller's.
        unsafe {
            self.0.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    unsafe fn clear_target(&self, at: usize, rgba: [f32; 4], rects: &[victory_host::Rect]) {
        let rects = rects_of(rects);
        // SAFETY: the caller's.
        unsafe {
            self.0.ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at }, &rgba, (!rects.is_empty()).then_some(&rects[..]))
        };
    }

    unsafe fn clear_depth(&self, at: usize, depth: Option<f32>, stencil: Option<u8>, rects: &[victory_host::Rect]) {
        use windows::Win32::Graphics::Direct3D12::{D3D12_CLEAR_FLAGS, D3D12_CLEAR_FLAG_DEPTH, D3D12_CLEAR_FLAG_STENCIL};
        let mut flags = D3D12_CLEAR_FLAGS(0);
        if depth.is_some() {
            flags |= D3D12_CLEAR_FLAG_DEPTH;
        }
        if stencil.is_some() {
            flags |= D3D12_CLEAR_FLAG_STENCIL;
        }
        let rects = rects_of(rects);
        // SAFETY: the caller's.
        unsafe {
            self.0.ClearDepthStencilView(
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at },
                flags,
                depth.unwrap_or(0.0),
                stencil.unwrap_or(0),
                (!rects.is_empty()).then_some(&rects[..]),
            )
        };
    }

    unsafe fn set_viewports(&self, viewports: &[victory_host::Viewport]) {
        use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;
        let v: Vec<D3D12_VIEWPORT> = viewports
            .iter()
            .map(|v| D3D12_VIEWPORT { TopLeftX: v.x, TopLeftY: v.y, Width: v.width, Height: v.height, MinDepth: v.min_depth, MaxDepth: v.max_depth })
            .collect();
        // SAFETY: the caller's.
        unsafe { self.0.RSSetViewports(&v) };
    }

    unsafe fn set_scissors(&self, rects: &[victory_host::Rect]) {
        // SAFETY: the caller's.
        unsafe { self.0.RSSetScissorRects(&rects_of(rects)) };
    }

    unsafe fn set_blend_factor(&self, rgba: [f32; 4]) {
        // SAFETY: the caller's.
        unsafe { self.0.OMSetBlendFactor(Some(&rgba)) };
    }

    unsafe fn set_stencil_ref(&self, reference: u32) {
        // SAFETY: the caller's.
        unsafe { self.0.OMSetStencilRef(reference) };
    }

    unsafe fn set_depth_bounds(&self, min: f32, max: f32) {
        use windows::core::Interface;
        if let Ok(l1) = self.0.cast::<windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList1>() {
            // SAFETY: the caller's (a device with depth bounds).
            unsafe { l1.OMSetDepthBounds(min, max) };
        }
    }

    unsafe fn set_topology(&self, topology: u32) {
        // SAFETY: the caller's.
        unsafe {
            self.0.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(topology as i32))
        };
    }

    unsafe fn set_vertex_buffers(&self, first: u32, views: &[victory_host::VertexView]) {
        use windows::Win32::Graphics::Direct3D12::D3D12_VERTEX_BUFFER_VIEW;
        let v: Vec<D3D12_VERTEX_BUFFER_VIEW> = views
            .iter()
            .map(|v| D3D12_VERTEX_BUFFER_VIEW { BufferLocation: v.address, SizeInBytes: v.size, StrideInBytes: v.stride })
            .collect();
        // SAFETY: the caller's.
        unsafe { self.0.IASetVertexBuffers(first, Some(&v)) };
    }

    unsafe fn set_index_buffer(&self, view: Option<victory_host::IndexView>) {
        use windows::Win32::Graphics::Direct3D12::D3D12_INDEX_BUFFER_VIEW;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        let v = view.map(|v| D3D12_INDEX_BUFFER_VIEW { BufferLocation: v.address, SizeInBytes: v.size, Format: DXGI_FORMAT(v.format as i32) });
        // SAFETY: the caller's.
        unsafe { self.0.IASetIndexBuffer(v.as_ref().map(std::ptr::from_ref)) };
    }

    unsafe fn draw_instanced(&self, count: u32, instances: u32, first: u32, first_instance: u32) {
        // SAFETY: the caller's.
        unsafe { self.0.DrawInstanced(count, instances, first, first_instance) };
    }

    unsafe fn draw_indexed_instanced(&self, count: u32, instances: u32, first: u32, base_vertex: i32, first_instance: u32) {
        // SAFETY: the caller's.
        unsafe { self.0.DrawIndexedInstanced(count, instances, first, base_vertex, first_instance) };
    }

    unsafe fn dispatch_groups(&self, groups: [u32; 3]) {
        // SAFETY: the caller's.
        unsafe { self.0.Dispatch(groups[0], groups[1], groups[2]) };
    }

    unsafe fn dispatch_mesh_groups(&self, groups: [u32; 3]) {
        use windows::core::Interface;
        if let Ok(l6) = self.0.cast::<windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList6>() {
            // SAFETY: the caller's.
            unsafe { l6.DispatchMesh(groups[0], groups[1], groups[2]) };
        }
    }

    unsafe fn execute_indirect(
        &self,
        kind: u32,
        stride: u32,
        max: u32,
        args: &ID3D12Resource,
        offset: u64,
        count: Option<(&ID3D12Resource, u64)>,
    ) {
        let Some(signature) = signature(kind, stride) else { return };
        // SAFETY: the caller's; the signature is cached for the process.
        unsafe {
            self.0.ExecuteIndirect(&signature, max, args, offset, count.map(|(c, _)| c), count.map_or(0, |(_, at)| at))
        };
    }

    unsafe fn resolve(&self, dst: &ID3D12Resource, dst_sub: u32, src: &ID3D12Resource, src_sub: u32, format: u32) {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1};
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        let format = DXGI_FORMAT(format as i32);
        // SAFETY: the caller's live textures in their resolve states.
        unsafe {
            let (a, b) = (dst.GetDesc(), src.GetDesc());
            if (a.Width, a.Height) == (b.Width, b.Height) {
                self.0.ResolveSubresource(dst, dst_sub, src, src_sub, format);
            } else if let Ok(list1) = self.0.cast::<ID3D12GraphicsCommandList1>() {
                let rect = RECT { left: 0, top: 0, right: a.Width.min(b.Width) as i32, bottom: a.Height.min(b.Height) as i32 };
                list1.ResolveSubresourceRegion(dst, dst_sub, 0, 0, src, src_sub, Some(&rect), format, D3D12_RESOLVE_MODE_AVERAGE);
            }
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
            let pso = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // A borrowed pointer the description never releases.
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            };
            let create = || self.0.CreateComputePipelineState(&pso);
            let state: ID3D12PipelineState = match desc.cache {
                Some(name) => crate::pipeline::library::pipeline(
                    name,
                    |library, name| library.LoadComputePipeline(name, &pso),
                    create,
                )?,
                None => create()?,
            };
            Ok(D3d12Pipeline { root, state, parameters })
        }
    }

    /// The descriptions built here are the ones victory-gpu made before
    /// the seam (plain description; a stream with `DEPTH_STENCIL1` for
    /// the depth bounds test; a stream for mesh pipelines, naming a
    /// triangle topology), so pipeline library entries keep loading.
    fn graphics_pipeline(&self, desc: &GraphicsDesc<'_>) -> windows::core::Result<D3d12Pipeline> {
        graphics::pipeline(self.0, desc)
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

    fn descriptor_gpu_start(&self, heap: &ID3D12DescriptorHeap) -> u64 {
        // SAFETY: a live shader-visible heap.
        unsafe { heap.GetGPUDescriptorHandleForHeapStart() }.ptr
    }

    fn gpu_address(&self, buffer: &ID3D12Resource) -> u64 {
        // SAFETY: a live buffer.
        unsafe { buffer.GetGPUVirtualAddress() }
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

/// Graphics pipelines on D3D12, from the seam's description.
mod graphics {
    use std::ffi::CString;

    use victory_host::{Blend, Geometry, GraphicsDesc, ShaderCode, StencilFace};
    use windows::Win32::Foundation::E_INVALIDARG;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
    use windows::core::Interface;

    use super::D3d12Pipeline;

    fn invalid() -> windows::core::Error {
        windows::core::Error::from_hresult(E_INVALIDARG)
    }

    fn bytecode(code: Option<ShaderCode<'_>>) -> windows::core::Result<D3D12_SHADER_BYTECODE> {
        match code {
            None => Ok(D3D12_SHADER_BYTECODE::default()),
            Some(ShaderCode::Dxil(d)) => Ok(D3D12_SHADER_BYTECODE {
                pShaderBytecode: d.as_ptr().cast(),
                BytecodeLength: d.len(),
            }),
            Some(_) => Err(invalid()),
        }
    }

    fn blend(b: &Blend) -> D3D12_RENDER_TARGET_BLEND_DESC {
        D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: b.enable.into(),
            LogicOpEnable: b.logic_op.is_some().into(),
            SrcBlend: D3D12_BLEND(b.src as i32),
            DestBlend: D3D12_BLEND(b.dst as i32),
            BlendOp: D3D12_BLEND_OP(b.op as i32),
            SrcBlendAlpha: D3D12_BLEND(b.src_alpha as i32),
            DestBlendAlpha: D3D12_BLEND(b.dst_alpha as i32),
            BlendOpAlpha: D3D12_BLEND_OP(b.op_alpha as i32),
            LogicOp: D3D12_LOGIC_OP(b.logic_op.unwrap_or(D3D12_LOGIC_OP_NOOP.0 as u32) as i32),
            RenderTargetWriteMask: b.write_mask,
        }
    }

    fn face(f: &StencilFace) -> D3D12_DEPTH_STENCILOP_DESC {
        D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP(f.fail as i32),
            StencilDepthFailOp: D3D12_STENCIL_OP(f.depth_fail as i32),
            StencilPassOp: D3D12_STENCIL_OP(f.pass as i32),
            StencilFunc: D3D12_COMPARISON_FUNC(f.func as i32),
        }
    }

    /// A stream subobject: its type, then its description, pointer-aligned.
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);

    fn stream<S>(
        device: &ID3D12Device,
        stream: &mut S,
        name: Option<&str>,
    ) -> windows::core::Result<ID3D12PipelineState> {
        let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
            SizeInBytes: size_of::<S>(),
            pPipelineStateSubobjectStream: std::ptr::from_mut(stream).cast(),
        };
        let device = device.cast::<ID3D12Device2>()?;
        // SAFETY: the stream holds whole subobjects pointing at live data.
        let create = || unsafe { device.CreatePipelineState(&desc) };
        match name {
            // SAFETY: as above.
            Some(name) => crate::pipeline::library::pipeline(
                name,
                |library, name| unsafe { library.LoadPipeline(name, &desc) },
                create,
            ),
            None => create(),
        }
    }

    pub(super) fn pipeline(
        device: &ID3D12Device,
        desc: &GraphicsDesc<'_>,
    ) -> windows::core::Result<D3d12Pipeline> {
        let first = match desc.geometry {
            Geometry::Vertex { shader, .. } | Geometry::Mesh { shader, .. } => shader,
        };
        let ShaderCode::Dxil(first_code) = first.code else {
            return Err(invalid());
        };
        // No root signature given: the one embedded in the first stage.
        let blob = if desc.root_signature.is_empty() {
            first_code
        } else {
            desc.root_signature
        };
        let parameters = victory_host::root::parse(blob)
            .map_err(|_| invalid())?
            .parameters
            .into_iter()
            .map(|p| p.kind)
            .collect();
        // SAFETY: a serialized root signature; D3D12 validates it.
        let root: ID3D12RootSignature = unsafe { device.CreateRootSignature(0, blob) }?;
        let mut formats = [DXGI_FORMAT(0); 8];
        for (f, &t) in formats.iter_mut().zip(desc.targets) {
            *f = DXGI_FORMAT(t as i32);
        }
        let targets = D3D12_RT_FORMAT_ARRAY {
            RTFormats: formats,
            NumRenderTargets: desc.targets.len().min(8) as u32,
        };
        let blend_state = D3D12_BLEND_DESC {
            AlphaToCoverageEnable: desc.alpha_to_coverage.into(),
            IndependentBlendEnable: desc.independent_blend.into(),
            RenderTarget: desc.blend.map(|b| blend(&b)),
        };
        let r = desc.raster;
        let raster = D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE(r.fill as i32),
            CullMode: D3D12_CULL_MODE(r.cull as i32),
            FrontCounterClockwise: r.front_ccw.into(),
            DepthBias: r.depth_bias,
            DepthBiasClamp: r.depth_bias_clamp,
            SlopeScaledDepthBias: r.slope_scaled_depth_bias,
            DepthClipEnable: r.depth_clip.into(),
            MultisampleEnable: r.multisample.into(),
            AntialiasedLineEnable: r.antialiased_lines.into(),
            ForcedSampleCount: r.forced_samples,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE(i32::from(r.conservative)),
        };
        let s = desc.stencil;
        let depth = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: desc.depth.enable.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK(i32::from(desc.depth.write)),
            DepthFunc: D3D12_COMPARISON_FUNC(desc.depth.func as i32),
            StencilEnable: s.enable.into(),
            StencilReadMask: s.read_mask,
            StencilWriteMask: s.write_mask,
            FrontFace: face(&s.front),
            BackFace: face(&s.back),
        };
        let depth1 = D3D12_DEPTH_STENCIL_DESC1 {
            DepthEnable: depth.DepthEnable,
            DepthWriteMask: depth.DepthWriteMask,
            DepthFunc: depth.DepthFunc,
            StencilEnable: depth.StencilEnable,
            StencilReadMask: depth.StencilReadMask,
            StencilWriteMask: depth.StencilWriteMask,
            FrontFace: depth.FrontFace,
            BackFace: depth.BackFace,
            DepthBoundsTestEnable: true.into(),
        };
        let samples = DXGI_SAMPLE_DESC {
            Count: desc.samples.max(1),
            Quality: desc.sample_quality,
        };
        let dsv = DXGI_FORMAT(desc.depth_format as i32);
        let ps = bytecode(desc.pixel.map(|p| p.code))?;
        let state = match desc.geometry {
            Geometry::Mesh {
                amplification,
                shader,
            } => {
                if amplification.is_some() {
                    // Not made by victory-gpu yet: no stream for it.
                    return Err(windows::core::Error::from_hresult(
                        windows::Win32::Foundation::E_NOTIMPL,
                    ));
                }
                let ms = bytecode(Some(shader.code))?;
                // The stream names a triangle topology (see victory-gpu's
                // pipeline/host.rs: the library refuses to load without).
                #[repr(C)]
                struct Stream<D> {
                    root: Sub<*mut std::ffi::c_void>,
                    ms: Sub<D3D12_SHADER_BYTECODE>,
                    ps: Sub<D3D12_SHADER_BYTECODE>,
                    blend: Sub<D3D12_BLEND_DESC>,
                    mask: Sub<u32>,
                    raster: Sub<D3D12_RASTERIZER_DESC>,
                    depth: Sub<D>,
                    targets: Sub<D3D12_RT_FORMAT_ARRAY>,
                    dsv: Sub<DXGI_FORMAT>,
                    samples: Sub<DXGI_SAMPLE_DESC>,
                    topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
                }
                macro_rules! mesh {
                    ($kind:expr, $depth:expr) => {
                        stream(
                            device,
                            &mut Stream {
                                root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
                                ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
                                ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
                                blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend_state),
                                mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, desc.sample_mask),
                                raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, raster),
                                depth: Sub($kind, $depth),
                                targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, targets),
                                dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, dsv),
                                samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, samples),
                                topology: Sub(
                                    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                                    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                                ),
                            },
                            desc.cache,
                        )
                    };
                }
                if desc.depth_bounds {
                    mesh!(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth1)?
                } else {
                    mesh!(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, depth)?
                }
            }
            Geometry::Vertex { shader, inputs } => {
                let vs = bytecode(Some(shader.code))?;
                let gs = bytecode(desc.geometry_shader.map(|g| g.code))?;
                let names: Vec<CString> = inputs
                    .iter()
                    .map(|e| CString::new(e.semantic).map_err(|_| invalid()))
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
                        InstanceDataStepRate: e.step_rate,
                    })
                    .collect();
                let input = D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: if elements.is_empty() {
                        std::ptr::null()
                    } else {
                        elements.as_ptr()
                    },
                    NumElements: elements.len() as u32,
                };
                let strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(desc.strip_cut as i32);
                let topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE(desc.topology as i32);
                if desc.depth_bounds {
                    #[repr(C)]
                    struct Stream {
                        root: Sub<*mut std::ffi::c_void>,
                        vs: Sub<D3D12_SHADER_BYTECODE>,
                        ps: Sub<D3D12_SHADER_BYTECODE>,
                        blend: Sub<D3D12_BLEND_DESC>,
                        mask: Sub<u32>,
                        raster: Sub<D3D12_RASTERIZER_DESC>,
                        depth: Sub<D3D12_DEPTH_STENCIL_DESC1>,
                        input: Sub<D3D12_INPUT_LAYOUT_DESC>,
                        strip_cut: Sub<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>,
                        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
                        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
                        dsv: Sub<DXGI_FORMAT>,
                        samples: Sub<DXGI_SAMPLE_DESC>,
                    }
                    #[repr(C)]
                    struct StreamGs {
                        base: Stream,
                        gs: Sub<D3D12_SHADER_BYTECODE>,
                    }
                    let base = Stream {
                        root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.as_raw()),
                        vs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, vs),
                        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
                        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend_state),
                        mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, desc.sample_mask),
                        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, raster),
                        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1, depth1),
                        input: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, input),
                        strip_cut: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE, strip_cut),
                        topology: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, topology),
                        targets: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, targets),
                        dsv: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, dsv),
                        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, samples),
                    };
                    if gs.pShaderBytecode.is_null() {
                        stream(device, &mut { base }, desc.cache)?
                    } else {
                        stream(
                            device,
                            &mut StreamGs {
                                base,
                                gs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, gs),
                            },
                            desc.cache,
                        )?
                    }
                } else {
                    let pso = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                        // SAFETY: a borrowed pointer the description never
                        // releases.
                        pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                        VS: vs,
                        PS: ps,
                        GS: gs,
                        BlendState: blend_state,
                        SampleMask: desc.sample_mask,
                        RasterizerState: raster,
                        DepthStencilState: depth,
                        InputLayout: input,
                        IBStripCutValue: strip_cut,
                        PrimitiveTopologyType: topology,
                        NumRenderTargets: targets.NumRenderTargets,
                        RTVFormats: targets.RTFormats,
                        DSVFormat: dsv,
                        SampleDesc: samples,
                        ..Default::default()
                    };
                    // SAFETY: the description points at live data.
                    let create = || unsafe { device.CreateGraphicsPipelineState(&pso) };
                    match desc.cache {
                        Some(name) => crate::pipeline::library::pipeline(
                            name,
                            // SAFETY: as above.
                            |library, name| unsafe { library.LoadGraphicsPipeline(name, &pso) },
                            create,
                        )?,
                        None => create()?,
                    }
                }
            }
        };
        Ok(D3d12Pipeline {
            root,
            state,
            parameters,
        })
    }
}
