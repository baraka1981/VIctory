//! [`victory_host::Recorder`] on Metal: one command buffer, copies in blit
//! encoders.
//!
//! Barriers: Metal tracks hazards on the resources made here (default
//! hazard tracking), so transitions record nothing; an unordered-access
//! barrier ends the open encoder, which orders what came before it.

use std::cell::RefCell;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLCullMode, MTLDepthClipMode, MTLHeap, MTLIndexType,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder,
    MTLRenderPassAttachmentDescriptor, MTLRenderPassDescriptor, MTLRenderStages, MTLResource,
    MTLResourceUsage, MTLScissorRect, MTLSize, MTLStoreAction, MTLTexture, MTLTextureType,
    MTLTriangleFillMode, MTLViewport, MTLWinding,
};
use victory_host::{Arguments, Barrier, Binding, Draw, Location, Pass, Recorder, Region, Vertices};

use crate::device::{Abi, MetalDescriptorHeap, MetalPipeline, MetalResource, StaticSamplers};
use crate::render::{Kind, MetalGraphicsPipeline};
use crate::toplevel::TopLevel;
use crate::views::{Resident, Slots, Target};
use crate::{DESCRIPTOR_BYTES, MetalError};
use crate::format::console_block;

type Encoder = Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>;

/// MSC's buffer indices for directly indexed heaps
/// (`kIRDescriptorHeapBindPoint`, `kIRSamplerHeapBindPoint`).
const VIEW_HEAP_INDEX: usize = 0;
const SAMPLER_HEAP_INDEX: usize = 1;
type RenderEncoder = Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>;

/// What one dispatch or draw makes resident.
#[derive(Default)]
struct Residency {
    heaps: Vec<Retained<ProtocolObject<dyn MTLHeap>>>,
    /// Resources outside heaps, and whether any use writes them.
    loose: Vec<(Retained<ProtocolObject<dyn MTLResource>>, bool)>,
}

/// An open command buffer.
pub struct MetalRecorder {
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    blit: RefCell<Option<Encoder>>,
    /// The device's descriptor slots, for residency.
    slots: Arc<Slots>,
}

impl MetalRecorder {
    pub(crate) fn new(
        buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        slots: Arc<Slots>,
    ) -> MetalRecorder {
        MetalRecorder {
            buffer,
            blit: RefCell::new(None),
            slots,
        }
    }

    /// Resources a converted shader reaches only through addresses: root
    /// descriptors' buffers, the tables' and static samplers' descriptor
    /// buffers, and what the
    /// descriptors the root signature's ranges cover name. Those placed in
    /// a heap go resident a heap at a time (`useHeap`), the rest one by one
    /// (`useResource`, read-write for unordered access views).
    fn residency(
        &self,
        top: &TopLevel,
        arguments: &Arguments<'_, MetalResource, MetalDescriptorHeap>,
        statics: Option<&StaticSamplers>,
    ) -> Residency {
        let Arguments {
            bindings,
            tables,
            views,
            samplers,
            ..
        } = *arguments;
        let mut out = Residency::default();
        let mut add = |resource: Retained<ProtocolObject<dyn MTLResource>>, writes: bool| {
            if let Some(h) = resource.heap() {
                if !out.heaps.iter().any(|x| Retained::as_ptr(x) == Retained::as_ptr(&h)) {
                    out.heaps.push(h);
                }
                return;
            }
            match out
                .loose
                .iter_mut()
                .find(|(x, _)| Retained::as_ptr(x) == Retained::as_ptr(&resource))
            {
                Some((_, w)) => *w |= writes,
                None => out.loose.push((resource, writes)),
            }
        };
        for b in bindings {
            let (writes, r) = match *b {
                Binding::Read(r, _) => (false, r),
                Binding::Write(r, _) => (true, r),
            };
            if let MetalResource::Buffer(buf) = r {
                add(ProtocolObject::from_retained(buf.clone()), writes);
            }
        }
        if let Some(s) = statics {
            add(ProtocolObject::from_retained(s.table.clone()), false);
        }
        let mut named = Vec::new();
        for (k, t) in tables.iter().enumerate() {
            add(ProtocolObject::from_retained(t.heap.table.clone()), false);
            let base = t.heap.table.contents().as_ptr() as usize;
            let step = DESCRIPTOR_BYTES as usize;
            for run in top.reached(k, t.first, t.heap.count) {
                self.slots.resident(
                    base + run.start as usize * step..base + run.end as usize * step,
                    &mut named,
                );
            }
        }
        // A directly indexed heap: any slot may be read, so all of them.
        for h in views.into_iter().chain(samplers) {
            add(ProtocolObject::from_retained(h.table.clone()), false);
            let base = h.table.contents().as_ptr() as usize;
            self.slots.resident(
                base..base + h.count as usize * DESCRIPTOR_BYTES as usize,
                &mut named,
            );
        }
        for (r, writes) in named {
            let resource: Retained<ProtocolObject<dyn MTLResource>> = match r {
                Resident::Buffer(b) => ProtocolObject::from_retained(b),
                // A view's memory is its parent's.
                Resident::Texture(t) => ProtocolObject::from_retained(t.parentTexture().unwrap_or(t)),
            };
            add(resource, writes);
        }
        out
    }

    /// A render encoder for `pass` (its targets loaded or cleared, then
    /// stored), and the first target's size for clamping the scissor.
    fn render_encoder(&self, pass: &Pass<'_>) -> Option<(RenderEncoder, (usize, usize))> {
        self.end();
        let d = MTLRenderPassDescriptor::new();
        let mut size = (usize::MAX, usize::MAX);
        let mut place = |a: &MTLRenderPassAttachmentDescriptor, t: &Target| {
            a.setTexture(Some(&t.texture));
            a.setLevel(t.level);
            if t.texture.textureType() == MTLTextureType::Type3D {
                a.setDepthPlane(t.slice);
            } else {
                a.setSlice(t.slice);
            }
            a.setStoreAction(MTLStoreAction::Store);
            size = (
                size.0.min((t.texture.width() >> t.level).max(1)),
                size.1.min((t.texture.height() >> t.level).max(1)),
            );
        };
        for (i, at) in pass.targets.iter().enumerate() {
            let Some(t) = self.slots.target(at.at) else { continue };
            // SAFETY: at most 8 targets.
            let a = unsafe { d.colorAttachments().objectAtIndexedSubscript(i) };
            place(&a, &t);
            match at.clear {
                Some([r, g, b, alpha]) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearColor(MTLClearColor {
                        red: f64::from(r),
                        green: f64::from(g),
                        blue: f64::from(b),
                        alpha: f64::from(alpha),
                    });
                }
                None => a.setLoadAction(MTLLoadAction::Load),
            }
        }
        if let Some(at) = pass.depth
            && let Some(t) = self.slots.target(at.at)
        {
            let a = d.depthAttachment();
            place(&a, &t);
            let stencil = matches!(
                t.texture.pixelFormat(),
                MTLPixelFormat::Depth32Float_Stencil8 | MTLPixelFormat::X32_Stencil8
            );
            let s = d.stencilAttachment();
            if stencil {
                place(&s, &t);
            }
            match at.clear {
                Some((depth, st)) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearDepth(f64::from(depth));
                    s.setLoadAction(MTLLoadAction::Clear);
                    s.setClearStencil(u32::from(st));
                }
                None => {
                    a.setLoadAction(MTLLoadAction::Load);
                    s.setLoadAction(MTLLoadAction::Load);
                }
            }
        }
        let enc = self.buffer.renderCommandEncoderWithDescriptor(&d)?;
        Some((enc, size))
    }

    /// Pipeline, fixed-function state, the top-level argument buffer on
    /// the pipeline's stages, and residency.
    fn render_state(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        size: (usize, usize),
        pass: &Pass<'_>,
        pipeline: &MetalGraphicsPipeline,
        args: &Arguments<'_, MetalResource, MetalDescriptorHeap>,
    ) {
        enc.setRenderPipelineState(&pipeline.state);
        enc.setDepthStencilState(Some(&pipeline.depth));
        let r = &pipeline.raster;
        enc.setCullMode(match r.cull {
            2 => MTLCullMode::Front,
            3 => MTLCullMode::Back,
            _ => MTLCullMode::None,
        });
        enc.setFrontFacingWinding(if r.front_ccw {
            MTLWinding::CounterClockwise
        } else {
            MTLWinding::Clockwise
        });
        enc.setTriangleFillMode(if r.fill == 2 {
            MTLTriangleFillMode::Lines
        } else {
            MTLTriangleFillMode::Fill
        });
        enc.setDepthClipMode(if r.depth_clip {
            MTLDepthClipMode::Clip
        } else {
            MTLDepthClipMode::Clamp
        });
        enc.setDepthBias_slopeScale_clamp(
            r.depth_bias as f32,
            r.slope_scaled_depth_bias,
            r.depth_bias_clamp,
        );
        let v = pass.viewport;
        enc.setViewport(MTLViewport {
            originX: f64::from(v.x),
            originY: f64::from(v.y),
            width: f64::from(v.width),
            height: f64::from(v.height),
            znear: f64::from(v.min_depth),
            zfar: f64::from(v.max_depth),
        });
        let [left, top, right, bottom] = pass.scissor.unwrap_or([
            v.x.max(0.0) as u32,
            v.y.max(0.0) as u32,
            (v.x + v.width).max(0.0) as u32,
            (v.y + v.height).max(0.0) as u32,
        ]);
        // Metal refuses scissors outside the attachments.
        let (w, h) = size;
        let x = (left as usize).min(w);
        let y = (top as usize).min(h);
        enc.setScissorRect(MTLScissorRect {
            x,
            y,
            width: (right as usize).min(w).saturating_sub(x),
            height: (bottom as usize).min(h).saturating_sub(y),
        });
        let addresses: Vec<u64> = args
            .bindings
            .iter()
            .map(|b| {
                let (Binding::Read(r, offset) | Binding::Write(r, offset)) = *b;
                buffer(r).map_or(0, |b| b.gpuAddress() + offset)
            })
            .collect();
        let starts: Vec<u64> = args
            .tables
            .iter()
            .map(|t| t.heap.table.gpuAddress() + u64::from(t.first) * u64::from(DESCRIPTOR_BYTES))
            .collect();
        let statics = pipeline.statics.as_ref();
        let top = pipeline.top.fill(
            args.constants,
            &addresses,
            &starts,
            StaticSamplers::address(statics),
        );
        let stages = pipeline.stages;
        let bytes = std::ptr::NonNull::from(&top[0]).cast();
        let index = crate::irconverter::ARGUMENT_BUFFER_INDEX;
        // SAFETY: Metal copies the bytes.
        unsafe {
            if stages.contains(MTLRenderStages::Vertex) {
                enc.setVertexBytes_length_atIndex(bytes, top.len(), index);
            }
            if stages.contains(MTLRenderStages::Object) {
                enc.setObjectBytes_length_atIndex(bytes, top.len(), index);
            }
            if stages.contains(MTLRenderStages::Mesh) {
                enc.setMeshBytes_length_atIndex(bytes, top.len(), index);
            }
            if stages.contains(MTLRenderStages::Fragment) {
                enc.setFragmentBytes_length_atIndex(bytes, top.len(), index);
            }
        }
        // Directly indexed heaps at MSC's descriptor heap bind points, on
        // each stage.
        for (heap, index) in [
            (args.views, VIEW_HEAP_INDEX),
            (args.samplers, SAMPLER_HEAP_INDEX),
        ] {
            let Some(h) = heap else { continue };
            let table = Some(&*h.table);
            // SAFETY: a live heap's table.
            unsafe {
                if stages.contains(MTLRenderStages::Vertex) {
                    enc.setVertexBuffer_offset_atIndex(table, 0, index);
                }
                if stages.contains(MTLRenderStages::Object) {
                    enc.setObjectBuffer_offset_atIndex(table, 0, index);
                }
                if stages.contains(MTLRenderStages::Mesh) {
                    enc.setMeshBuffer_offset_atIndex(table, 0, index);
                }
                if stages.contains(MTLRenderStages::Fragment) {
                    enc.setFragmentBuffer_offset_atIndex(table, 0, index);
                }
            }
        }
        let residency = self.residency(&pipeline.top, args, statics);
        for h in &residency.heaps {
            enc.useHeap_stages(h, stages);
        }
        for (r, writes) in &residency.loose {
            let usage = if *writes {
                MTLResourceUsage::Read | MTLResourceUsage::Write
            } else {
                MTLResourceUsage::Read
            };
            enc.useResource_usage_stages(r, usage, stages);
        }
    }

    fn end(&self) {
        if let Some(e) = self.blit.borrow_mut().take() {
            e.endEncoding();
        }
    }

    fn blit(&self) -> Option<Encoder> {
        let mut open = self.blit.borrow_mut();
        if open.is_none() {
            *open = self.buffer.blitCommandEncoder();
        }
        open.clone()
    }

    /// Ends recording and submits.
    pub fn commit(self) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.end();
        self.buffer.commit();
        self.buffer
    }

    /// Submits and waits for the GPU to finish.
    pub fn commit_and_wait(self) -> Result<(), MetalError> {
        let buffer = self.commit();
        buffer.waitUntilCompleted();
        match buffer.error() {
            None => Ok(()),
            Some(_) => Err(MetalError::Failed("a command buffer run")),
        }
    }
}

fn buffer(r: &MetalResource) -> Option<&ProtocolObject<dyn MTLBuffer>> {
    match r {
        MetalResource::Buffer(b) => Some(b),
        MetalResource::Texture(_) => None,
    }
}

fn texture(r: &MetalResource) -> Option<&ProtocolObject<dyn MTLTexture>> {
    match r {
        MetalResource::Texture(t) => Some(t),
        MetalResource::Buffer(_) => None,
    }
}

/// Subresource index to (slice, level); planes are not separate in Metal.
fn place(t: &ProtocolObject<dyn MTLTexture>, index: u32) -> (usize, usize) {
    let mips = t.mipmapLevelCount().max(1);
    let index = index as usize;
    ((index / mips) % t.arrayLength().max(1), index % mips)
}

fn level_size(t: &ProtocolObject<dyn MTLTexture>, level: usize) -> MTLSize {
    MTLSize {
        width: (t.width() >> level).max(1),
        height: (t.height() >> level).max(1),
        depth: (t.depth() >> level).max(1),
    }
}

fn span(region: Option<&Region>, whole: MTLSize) -> (MTLOrigin, MTLSize) {
    match region {
        Some(r) => (
            MTLOrigin {
                x: r.left as usize,
                y: r.top as usize,
                z: r.front as usize,
            },
            MTLSize {
                width: r.right.saturating_sub(r.left) as usize,
                height: r.bottom.saturating_sub(r.top) as usize,
                depth: r.back.saturating_sub(r.front) as usize,
            },
        ),
        None => (MTLOrigin { x: 0, y: 0, z: 0 }, whole),
    }
}

impl Recorder for MetalRecorder {
    type Resource = MetalResource;
    type Pipeline = MetalPipeline;
    type GraphicsPipeline = MetalGraphicsPipeline;
    type DescriptorHeap = MetalDescriptorHeap;

    /// One compute encoder per dispatch. Converted DXIL: the top-level
    /// argument buffer its root signature lays out, with everything it
    /// reaches made resident. Our own shaders: constants at buffer index
    /// 0, bindings from 1 (no tables).
    unsafe fn dispatch(
        &self,
        pipeline: &MetalPipeline,
        arguments: &Arguments<'_, MetalResource, MetalDescriptorHeap>,
        groups: [u32; 3],
    ) {
        let Arguments {
            constants,
            bindings,
            tables,
            views,
            samplers,
        } = *arguments;
        self.end();
        let Some(enc) = self.buffer.computeCommandEncoder() else {
            return;
        };
        enc.setComputePipelineState(&pipeline.state);
        // SAFETY: the constants outlive the call (Metal copies them); the
        // caller keeps bindings inside their buffers and tables inside
        // their heaps.
        if pipeline.abi == Abi::Converted {
            let addresses: Vec<u64> = bindings
                .iter()
                .map(|b| {
                    let (Binding::Read(r, offset) | Binding::Write(r, offset)) = *b;
                    buffer(r).map_or(0, |b| b.gpuAddress() + offset)
                })
                .collect();
            let starts: Vec<u64> = tables
                .iter()
                .map(|t| t.heap.table.gpuAddress() + u64::from(t.first) * u64::from(DESCRIPTOR_BYTES))
                .collect();
            let statics = pipeline.statics.as_ref();
            let top = pipeline.top.fill(
                constants,
                &addresses,
                &starts,
                StaticSamplers::address(statics),
            );
            unsafe {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&top[0]).cast(),
                    top.len(),
                    crate::irconverter::ARGUMENT_BUFFER_INDEX,
                );
            }
            // Directly indexed heaps at MSC's descriptor heap bind points.
            for (heap, index) in [(views, VIEW_HEAP_INDEX), (samplers, SAMPLER_HEAP_INDEX)] {
                if let Some(h) = heap {
                    // SAFETY: a live heap's table.
                    unsafe { enc.setBuffer_offset_atIndex(Some(&h.table), 0, index) };
                }
            }
            let residency = self.residency(&pipeline.top, arguments, statics);
            for h in &residency.heaps {
                enc.useHeap(h);
            }
            for (r, writes) in &residency.loose {
                let usage = if *writes {
                    MTLResourceUsage::Read | MTLResourceUsage::Write
                } else {
                    MTLResourceUsage::Read
                };
                enc.useResource_usage(r, usage);
            }
        } else {
            unsafe {
                if let Some(first) = constants.first() {
                    enc.setBytes_length_atIndex(
                        std::ptr::NonNull::from(first).cast(),
                        constants.len() * 4,
                        0,
                    );
                }
                for (i, b) in bindings.iter().enumerate() {
                    let (Binding::Read(r, offset) | Binding::Write(r, offset)) = *b;
                    enc.setBuffer_offset_atIndex(buffer(r), offset as usize, i + 1);
                }
            }
        }
        let size = |v: [u32; 3]| MTLSize {
            width: v[0] as usize,
            height: v[1] as usize,
            depth: v[2] as usize,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(pipeline.threads));
        enc.endEncoding();
    }

    /// One render encoder per draw (targets loaded or cleared, stored).
    unsafe fn draw(
        &self,
        pass: &Pass<'_>,
        pipeline: &MetalGraphicsPipeline,
        arguments: &Arguments<'_, MetalResource, MetalDescriptorHeap>,
        vertices: &Vertices<'_, MetalResource>,
        draw: &Draw,
    ) {
        let Some((enc, size)) = self.render_encoder(pass) else {
            return;
        };
        self.render_state(&enc, size, pass, pipeline, arguments);
        // SAFETY: the caller keeps buffers live and covering the draw.
        unsafe {
            for (slot, vb) in vertices.buffers.iter().enumerate() {
                enc.setVertexBuffer_offset_attributeStride_atIndex(
                    buffer(vb.resource),
                    vb.offset as usize,
                    vb.stride as usize,
                    crate::irconverter::VERTEX_BUFFER_INDEX + slot,
                );
            }
        }
        let primitive = match vertices.primitive {
            1 => MTLPrimitiveType::Point,
            2 => MTLPrimitiveType::Line,
            3 => MTLPrimitiveType::LineStrip,
            5 => MTLPrimitiveType::TriangleStrip,
            _ => MTLPrimitiveType::Triangle,
        };
        let draw_params = matches!(pipeline.kind, Kind::Vertex { draw_params: true });
        let index = vertices
            .index
            .and_then(|i| buffer(i.resource).map(|b| (b, i.offset, i.format)));
        // SAFETY: as above; MSC's draw parameter layouts (runtime header):
        // IRRuntimeDrawArgument / IRRuntimeDrawIndexedArgument at the draw
        // argument index, the index type (0: none, else MTLIndexType + 1)
        // as a u16 at the uniforms index.
        unsafe {
            match index {
                None => {
                    if draw_params {
                        let args = [draw.count, draw.instances, draw.first, draw.first_instance];
                        enc.setVertexBytes_length_atIndex(
                            std::ptr::NonNull::from(&args).cast(),
                            size_of_val(&args),
                            crate::irconverter::DRAW_ARGUMENTS_INDEX,
                        );
                        let none = 0u16;
                        enc.setVertexBytes_length_atIndex(
                            std::ptr::NonNull::from(&none).cast(),
                            2,
                            crate::irconverter::UNIFORMS_INDEX,
                        );
                    }
                    enc.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                        primitive,
                        draw.first as usize,
                        draw.count as usize,
                        draw.instances as usize,
                        draw.first_instance as usize,
                    );
                }
                Some((ib, offset, format)) => {
                    let (kind, bytes) = if format == 57 {
                        (MTLIndexType::UInt16, 2)
                    } else {
                        (MTLIndexType::UInt32, 4)
                    };
                    if draw_params {
                        let args: [u32; 5] = [
                            draw.count,
                            draw.instances,
                            draw.first,
                            draw.base_vertex as u32,
                            draw.first_instance,
                        ];
                        enc.setVertexBytes_length_atIndex(
                            std::ptr::NonNull::from(&args).cast(),
                            size_of_val(&args),
                            crate::irconverter::DRAW_ARGUMENTS_INDEX,
                        );
                        let ir = kind.0 as u16 + 1;
                        enc.setVertexBytes_length_atIndex(
                            std::ptr::NonNull::from(&ir).cast(),
                            2,
                            crate::irconverter::UNIFORMS_INDEX,
                        );
                    }
                    enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                        primitive,
                        draw.count as usize,
                        kind,
                        ib,
                        offset as usize + draw.first as usize * bytes,
                        draw.instances as usize,
                        draw.base_vertex as isize,
                        draw.first_instance as usize,
                    );
                }
            }
        }
        enc.endEncoding();
    }

    unsafe fn draw_mesh(
        &self,
        pass: &Pass<'_>,
        pipeline: &MetalGraphicsPipeline,
        arguments: &Arguments<'_, MetalResource, MetalDescriptorHeap>,
        groups: [u32; 3],
    ) {
        let Kind::Mesh { object, mesh, .. } = pipeline.kind else {
            return;
        };
        let Some((enc, size)) = self.render_encoder(pass) else {
            return;
        };
        self.render_state(&enc, size, pass, pipeline, arguments);
        let size = |v: [u32; 3]| MTLSize {
            width: v[0] as usize,
            height: v[1] as usize,
            depth: v[2] as usize,
        };
        enc.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
            size(groups),
            size(object),
            size(mesh),
        );
        enc.endEncoding();
    }

    unsafe fn barriers(&self, barriers: &[Barrier<'_, MetalResource>]) {
        if barriers.iter().any(|b| matches!(b, Barrier::Unordered(_))) {
            self.end();
        }
    }

    unsafe fn copy_buffer(
        &self,
        to: &MetalResource,
        to_offset: u64,
        from: &MetalResource,
        from_offset: u64,
        bytes: u64,
    ) {
        let (Some(to), Some(from), Some(blit)) = (buffer(to), buffer(from), self.blit()) else {
            return;
        };
        // SAFETY: the caller keeps both ranges inside the buffers.
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                from,
                from_offset as usize,
                to,
                to_offset as usize,
                bytes as usize,
            )
        };
    }

    unsafe fn copy_resource(&self, to: &MetalResource, from: &MetalResource) {
        let Some(blit) = self.blit() else { return };
        match (to, from) {
            (MetalResource::Buffer(t), MetalResource::Buffer(f)) => {
                // SAFETY: same-shaped buffers (the caller's).
                unsafe {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        f,
                        0,
                        t,
                        0,
                        f.length().min(t.length()),
                    )
                }
            }
            // SAFETY: same-shaped textures (the caller's).
            (MetalResource::Texture(t), MetalResource::Texture(f)) => unsafe {
                blit.copyFromTexture_toTexture(f, t)
            },
            _ => {}
        }
    }

    unsafe fn copy_texture(
        &self,
        to: Location<'_, MetalResource>,
        at: [u32; 3],
        from: Location<'_, MetalResource>,
        region: Option<&Region>,
    ) {
        let Some(blit) = self.blit() else { return };
        let origin = MTLOrigin {
            x: at[0] as usize,
            y: at[1] as usize,
            z: at[2] as usize,
        };
        match (to, from) {
            // Buffer footprint -> texture subresource.
            (Location::Subresource(t, index), Location::Footprint(b, f)) => {
                let (Some(t), Some(b)) = (texture(t), buffer(b)) else {
                    return;
                };
                let (slice, level) = place(t, index);
                let whole = MTLSize {
                    width: f.width as usize,
                    height: f.height as usize,
                    depth: f.depth as usize,
                };
                let (start, size) = span(region, whole);
                let block = console_block(f.format).map_or(1, |b| b.height) as usize;
                let image = f.row_pitch as usize * (f.height as usize).div_ceil(block);
                let offset =
                    f.offset as usize + start.z * image + start.y / block * f.row_pitch as usize;
                // SAFETY: the caller's footprint lies inside the buffer.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        b, offset, f.row_pitch as usize, image, size, t, slice, level, origin,
                    )
                };
            }
            // Texture subresource -> buffer footprint.
            (Location::Footprint(b, f), Location::Subresource(t, index)) => {
                let (Some(t), Some(b)) = (texture(t), buffer(b)) else {
                    return;
                };
                let (slice, level) = place(t, index);
                let (start, size) = span(region, level_size(t, level));
                let block = console_block(f.format).map_or(1, |b| b.height) as usize;
                let image = f.row_pitch as usize * (f.height as usize).div_ceil(block);
                let offset =
                    f.offset as usize + origin.z * image + origin.y / block * f.row_pitch as usize;
                // SAFETY: as above.
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                        t, slice, level, start, size, b, offset, f.row_pitch as usize, image,
                    )
                };
            }
            // Texture -> texture.
            (Location::Subresource(t, to_index), Location::Subresource(f, from_index)) => {
                let (Some(t), Some(f)) = (texture(t), texture(f)) else {
                    return;
                };
                let (to_slice, to_level) = place(t, to_index);
                let (from_slice, from_level) = place(f, from_index);
                let (start, size) = span(region, level_size(f, from_level));
                // SAFETY: as above.
                unsafe {
                    blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        f, from_slice, from_level, start, size, t, to_slice, to_level, origin,
                    )
                };
            }
            // Footprint to footprint is a buffer copy, not a texture one.
            (Location::Footprint(..), Location::Footprint(..)) => {}
        }
    }
}
