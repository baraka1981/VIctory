//! [`victory_host::Recorder`] on Metal: one command buffer and a state
//! machine over it, the way D3D12 lists record (slice 2b).
//!
//! State-setting calls only note state; a draw or dispatch then brings the
//! open encoder up to date with it. One encoder is open at a time: a
//! render encoder stays open across draws while the targets stay the same,
//! a compute encoder across dispatches, a blit encoder across copies; any
//! other kind of command ends it.
//!
//! Root arguments are D3D12's: numbers (GPU addresses, GPU descriptor
//! handles) per root parameter, written into the top-level argument
//! buffer at each draw or dispatch ([`TopLevel::fill_args`]). Resources
//! reached only through addresses are made resident there too: root views'
//! buffers (found by address among the device's buffers), the tables'
//! descriptors that the root signature's ranges reach, the whole of a
//! directly indexed heap, static sampler tables.
//!
//! Clears are recorded at once, as render passes of their own that clear
//! and store (rectangles are not done yet: the whole view is cleared).
//!
//! The one-shot [`Recorder::dispatch`] / [`Recorder::draw`] are made of
//! the state-setting calls (our own MSL compute shaders apart), so tests of
//! them test the state machine.
//!
//! Barriers: Metal tracks hazards on the resources made here (default
//! hazard tracking), so transitions record nothing; an unordered-access
//! barrier ends the open encoder, which orders what came before it.

use std::cell::RefCell;
use std::ptr::NonNull;
use std::sync::Arc;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCullMode, MTLDepthClipMode,
    MTLDepthStencilState, MTLHeap, MTLIndexType, MTLLoadAction, MTLOrigin, MTLPixelFormat,
    MTLPrimitiveType, MTLRenderCommandEncoder, MTLRenderPassAttachmentDescriptor,
    MTLRenderPassDescriptor, MTLRenderPipelineState, MTLRenderStages, MTLResource,
    MTLResourceUsage, MTLScissorRect, MTLSize, MTLStoreAction, MTLTexture, MTLTextureType,
    MTLTriangleFillMode, MTLViewport, MTLWinding,
};
use victory_host::root::parameter;
use victory_host::{
    Arguments, Barrier, Binding, Descriptors, Draw, IndexView, Location, Pass, Point, Raster,
    Recorder, Rect, Region, VertexView, Vertices, Viewport,
};

use crate::device::{Abi, MetalDescriptorHeap, MetalPipeline, MetalResource, StaticSamplers};
use crate::format::console_block;
use crate::render::{Kind, MetalGraphicsPipeline};
use crate::toplevel::{Arg, SAMPLERS_DIRECTLY_INDEXED, Slot, TopLevel, VIEWS_DIRECTLY_INDEXED};
use crate::views::{Buffers, Resident, Slots, Target};
use crate::{DESCRIPTOR_BYTES, MetalError};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type BlitEncoder = Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>;
type ComputeEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
type RenderEncoder = Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>;

/// MSC's buffer indices for directly indexed heaps
/// (`kIRDescriptorHeapBindPoint`, `kIRSamplerHeapBindPoint`).
const VIEW_HEAP_INDEX: usize = 0;
const SAMPLER_HEAP_INDEX: usize = 1;

/// The open encoder.
enum Open {
    None,
    Blit(BlitEncoder),
    Compute(ComputeEncoder),
    /// With the target and depth slots it renders to, and the size of the
    /// smallest attachment (for scissors).
    Render {
        encoder: RenderEncoder,
        targets: Vec<usize>,
        depth: Option<usize>,
        size: (usize, usize),
    },
}

/// A heap set with `set_heaps`.
#[derive(Clone)]
struct HeapRef {
    table: Buffer,
    count: u32,
}

impl HeapRef {
    fn start(&self) -> u64 {
        self.table.gpuAddress()
    }

    /// The descriptor index of GPU handle `h`, if it is in this heap.
    fn index_of(&self, h: u64) -> Option<u32> {
        let off = h.checked_sub(self.start())?;
        let i = off / u64::from(DESCRIPTOR_BYTES);
        (i < u64::from(self.count)).then_some(i as u32)
    }
}

/// A root signature bound at one point, and its arguments by parameter.
#[derive(Default)]
struct Bound {
    top: TopLevel,
    statics: Option<Buffer>,
    args: Vec<Arg>,
}

impl Bound {
    fn set(&mut self, param: u32, arg: Arg) {
        let i = param as usize;
        if self.args.len() <= i {
            self.args.resize(i + 1, Arg::None);
        }
        self.args[i] = arg;
    }
}

struct ComputePso {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    threads: [u32; 3],
}

struct GraphicsPso {
    state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    depth: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    raster: Raster,
    kind: Kind,
    stages: MTLRenderStages,
}

/// What state-setting calls left.
struct State {
    views: Option<HeapRef>,
    samplers: Option<HeapRef>,
    compute: Bound,
    graphics: Bound,
    compute_pso: Option<ComputePso>,
    graphics_pso: Option<GraphicsPso>,
    targets: Vec<usize>,
    depth: Option<usize>,
    viewports: Vec<Viewport>,
    scissors: Vec<Rect>,
    blend_factor: [f32; 4],
    stencil_ref: u32,
    topology: u32,
    vertex: Vec<Option<VertexView>>,
    index: Option<IndexView>,
}

impl Default for State {
    fn default() -> Self {
        State {
            views: None,
            samplers: None,
            compute: Bound::default(),
            graphics: Bound::default(),
            compute_pso: None,
            graphics_pso: None,
            targets: Vec::new(),
            depth: None,
            viewports: Vec::new(),
            scissors: Vec::new(),
            blend_factor: [0.0; 4],
            stencil_ref: 0,
            topology: 4,
            vertex: Vec::new(),
            index: None,
        }
    }
}

impl State {
    fn bound(&mut self, point: Point) -> &mut Bound {
        match point {
            Point::Compute => &mut self.compute,
            Point::Graphics => &mut self.graphics,
        }
    }
}

/// What one dispatch or draw makes resident.
#[derive(Default)]
struct Residency {
    heaps: Vec<Retained<ProtocolObject<dyn MTLHeap>>>,
    /// Resources outside heaps, and whether any use writes them.
    loose: Vec<(Retained<ProtocolObject<dyn MTLResource>>, bool)>,
}

impl Residency {
    fn add(&mut self, resource: Retained<ProtocolObject<dyn MTLResource>>, writes: bool) {
        if let Some(h) = resource.heap() {
            if !self
                .heaps
                .iter()
                .any(|x| Retained::as_ptr(x) == Retained::as_ptr(&h))
            {
                self.heaps.push(h);
            }
            return;
        }
        match self
            .loose
            .iter_mut()
            .find(|(x, _)| Retained::as_ptr(x) == Retained::as_ptr(&resource))
        {
            Some((_, w)) => *w |= writes,
            None => self.loose.push((resource, writes)),
        }
    }
}

fn usage(writes: bool) -> MTLResourceUsage {
    if writes {
        MTLResourceUsage::Read | MTLResourceUsage::Write
    } else {
        MTLResourceUsage::Read
    }
}

/// An open command buffer.
pub struct MetalRecorder {
    buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    open: RefCell<Open>,
    state: RefCell<State>,
    /// The device's descriptor slots, for residency.
    slots: Arc<Slots>,
    /// The device's buffers by GPU address, for addresses given as numbers.
    buffers: Arc<Buffers>,
}

impl MetalRecorder {
    pub(crate) fn new(
        buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        slots: Arc<Slots>,
        buffers: Arc<Buffers>,
    ) -> MetalRecorder {
        MetalRecorder {
            buffer,
            open: RefCell::new(Open::None),
            state: RefCell::new(State::default()),
            slots,
            buffers,
        }
    }

    /// Ends the open encoder, if any.
    fn end(&self) {
        match std::mem::replace(&mut *self.open.borrow_mut(), Open::None) {
            Open::None => {}
            Open::Blit(e) => e.endEncoding(),
            Open::Compute(e) => e.endEncoding(),
            Open::Render { encoder, .. } => encoder.endEncoding(),
        }
    }

    fn blit(&self) -> Option<BlitEncoder> {
        if let Open::Blit(e) = &*self.open.borrow() {
            return Some(e.clone());
        }
        self.end();
        let e = self.buffer.blitCommandEncoder()?;
        *self.open.borrow_mut() = Open::Blit(e.clone());
        Some(e)
    }

    fn compute(&self) -> Option<ComputeEncoder> {
        if let Open::Compute(e) = &*self.open.borrow() {
            return Some(e.clone());
        }
        self.end();
        let e = self.buffer.computeCommandEncoder()?;
        *self.open.borrow_mut() = Open::Compute(e.clone());
        Some(e)
    }

    /// Points an attachment at a target view, stored at the end.
    fn attach(a: &MTLRenderPassAttachmentDescriptor, t: &Target) {
        a.setTexture(Some(&t.texture));
        a.setLevel(t.level);
        if t.texture.textureType() == MTLTextureType::Type3D {
            a.setDepthPlane(t.slice);
        } else {
            a.setSlice(t.slice);
        }
        a.setStoreAction(MTLStoreAction::Store);
    }

    fn has_stencil(t: &Target) -> bool {
        matches!(
            t.texture.pixelFormat(),
            MTLPixelFormat::Depth32Float_Stencil8 | MTLPixelFormat::X32_Stencil8
        )
    }

    /// A render encoder for the set targets: the open one if it renders to
    /// them, else a new one loading them.
    fn render(
        &self,
        targets: &[usize],
        depth: Option<usize>,
    ) -> Option<(RenderEncoder, (usize, usize))> {
        if let Open::Render {
            encoder,
            targets: t,
            depth: d,
            size,
        } = &*self.open.borrow()
            && t == targets
            && *d == depth
        {
            return Some((encoder.clone(), *size));
        }
        self.end();
        let d = MTLRenderPassDescriptor::new();
        let mut size = (usize::MAX, usize::MAX);
        let mut grow = |t: &Target| {
            size = (
                size.0.min((t.texture.width() >> t.level).max(1)),
                size.1.min((t.texture.height() >> t.level).max(1)),
            );
        };
        for (i, &at) in targets.iter().enumerate() {
            let Some(t) = self.slots.target(at) else {
                continue;
            };
            // SAFETY: at most 8 targets.
            let a = unsafe { d.colorAttachments().objectAtIndexedSubscript(i) };
            Self::attach(&a, &t);
            a.setLoadAction(MTLLoadAction::Load);
            grow(&t);
        }
        if let Some(at) = depth
            && let Some(t) = self.slots.target(at)
        {
            let a = d.depthAttachment();
            Self::attach(&a, &t);
            a.setLoadAction(MTLLoadAction::Load);
            if Self::has_stencil(&t) {
                let s = d.stencilAttachment();
                Self::attach(&s, &t);
                s.setLoadAction(MTLLoadAction::Load);
            }
            grow(&t);
        }
        let encoder = self.buffer.renderCommandEncoderWithDescriptor(&d)?;
        *self.open.borrow_mut() = Open::Render {
            encoder: encoder.clone(),
            targets: targets.to_vec(),
            depth,
            size,
        };
        Some((encoder, size))
    }

    /// Resources the bound root signature's arguments reach only through
    /// addresses (see the module notes).
    fn residency(&self, st: &State, bound: &Bound) -> Residency {
        let mut out = Residency::default();
        let heaps = [st.views.as_ref(), st.samplers.as_ref()];
        let mut named = Vec::new();
        let mut table = 0;
        for (e, arg) in bound.top.entries.iter().zip(&bound.args) {
            let &Arg::Address(a) = arg else {
                if e.slot == Slot::Table {
                    table += 1;
                }
                continue;
            };
            match e.slot {
                Slot::Descriptor(kind) => {
                    if let Some(b) = self.buffers.at(a) {
                        out.add(ProtocolObject::from_retained(b), kind == parameter::UAV);
                    }
                }
                Slot::Table => {
                    if let Some((h, first)) = heaps
                        .iter()
                        .flatten()
                        .find_map(|h| Some((*h, h.index_of(a)?)))
                    {
                        let base = h.table.contents().as_ptr() as usize;
                        let step = DESCRIPTOR_BYTES as usize;
                        for run in bound.top.reached(table, first, h.count) {
                            self.slots.resident(
                                base + run.start as usize * step..base + run.end as usize * step,
                                &mut named,
                            );
                        }
                    }
                    table += 1;
                }
                Slot::Constants(_) => {}
            }
        }
        for h in heaps.iter().flatten() {
            out.add(ProtocolObject::from_retained(h.table.clone()), false);
        }
        // A directly indexed heap: any slot may be read, so all of them.
        for (h, flag) in [
            (&st.views, VIEWS_DIRECTLY_INDEXED),
            (&st.samplers, SAMPLERS_DIRECTLY_INDEXED),
        ] {
            if let Some(h) = h
                && bound.top.flags & flag != 0
            {
                let base = h.table.contents().as_ptr() as usize;
                self.slots.resident(
                    base..base + h.count as usize * DESCRIPTOR_BYTES as usize,
                    &mut named,
                );
            }
        }
        if let Some(s) = &bound.statics {
            out.add(ProtocolObject::from_retained(s.clone()), false);
        }
        for (r, writes) in named {
            let resource: Retained<ProtocolObject<dyn MTLResource>> = match r {
                Resident::Buffer(b) => ProtocolObject::from_retained(b),
                // A view's memory is its parent's.
                Resident::Texture(t) => {
                    ProtocolObject::from_retained(t.parentTexture().unwrap_or(t))
                }
            };
            out.add(resource, writes);
        }
        out
    }

    /// The bound graphics state onto `enc` for a draw.
    fn render_state(
        &self,
        enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        size: (usize, usize),
        st: &State,
        pso: &GraphicsPso,
    ) {
        enc.setRenderPipelineState(&pso.state);
        enc.setDepthStencilState(Some(&pso.depth));
        enc.setStencilReferenceValue(st.stencil_ref);
        let [red, green, blue, alpha] = st.blend_factor;
        enc.setBlendColorRed_green_blue_alpha(red, green, blue, alpha);
        let r = &pso.raster;
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
        let (w, h) = size;
        let viewports: Vec<MTLViewport> = if st.viewports.is_empty() {
            vec![MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: w as f64,
                height: h as f64,
                znear: 0.0,
                zfar: 1.0,
            }]
        } else {
            st.viewports
                .iter()
                .map(|v| MTLViewport {
                    originX: f64::from(v.x),
                    originY: f64::from(v.y),
                    width: f64::from(v.width),
                    height: f64::from(v.height),
                    znear: f64::from(v.min_depth),
                    zfar: f64::from(v.max_depth),
                })
                .collect()
        };
        // Metal refuses scissors outside the attachments.
        let clamp = |[left, top, right, bottom]: Rect| {
            let x = (left.max(0) as usize).min(w);
            let y = (top.max(0) as usize).min(h);
            MTLScissorRect {
                x,
                y,
                width: (right.max(0) as usize).min(w).saturating_sub(x),
                height: (bottom.max(0) as usize).min(h).saturating_sub(y),
            }
        };
        let scissors: Vec<MTLScissorRect> = if st.scissors.is_empty() {
            vec![clamp([0, 0, w as i32, h as i32])]
        } else {
            st.scissors.iter().map(|&r| clamp(r)).collect()
        };
        // SAFETY: the arrays live through the calls.
        unsafe {
            enc.setViewports_count(NonNull::from(&viewports[0]), viewports.len());
            enc.setScissorRects_count(NonNull::from(&scissors[0]), scissors.len());
        }
        let bound = &st.graphics;
        let top = bound.top.fill_args(
            &bound.args,
            bound.statics.as_ref().map_or(0, |s| s.gpuAddress()),
        );
        let stages = pso.stages;
        let bytes = NonNull::from(&top[0]).cast();
        let index = crate::irconverter::ARGUMENT_BUFFER_INDEX;
        // SAFETY: Metal copies the bytes; the heaps' tables are live.
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
            for (heap, index) in [
                (&st.views, VIEW_HEAP_INDEX),
                (&st.samplers, SAMPLER_HEAP_INDEX),
            ] {
                let Some(h) = heap else { continue };
                let table = Some(&*h.table);
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
        let residency = self.residency(st, bound);
        for h in &residency.heaps {
            enc.useHeap_stages(h, stages);
        }
        for (r, writes) in &residency.loose {
            enc.useResource_usage_stages(r, usage(*writes), stages);
        }
        // Vertex buffers by address, found among the device's buffers.
        if matches!(pso.kind, Kind::Vertex { .. }) {
            for (slot, v) in st.vertex.iter().enumerate() {
                let Some(v) = v else { continue };
                let Some(b) = self.buffers.at(v.address) else {
                    continue;
                };
                // SAFETY: the address is in `b`.
                unsafe {
                    enc.setVertexBuffer_offset_attributeStride_atIndex(
                        Some(&b),
                        (v.address - b.gpuAddress()) as usize,
                        v.stride as usize,
                        crate::irconverter::VERTEX_BUFFER_INDEX + slot,
                    );
                }
            }
        }
    }

    /// The bound compute state onto `enc` for a dispatch.
    fn compute_state(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        st: &State,
        pso: &ComputePso,
    ) {
        enc.setComputePipelineState(&pso.state);
        let bound = &st.compute;
        let top = bound.top.fill_args(
            &bound.args,
            bound.statics.as_ref().map_or(0, |s| s.gpuAddress()),
        );
        // SAFETY: Metal copies the bytes; the heaps' tables are live.
        unsafe {
            enc.setBytes_length_atIndex(
                NonNull::from(&top[0]).cast(),
                top.len(),
                crate::irconverter::ARGUMENT_BUFFER_INDEX,
            );
            for (heap, index) in [
                (&st.views, VIEW_HEAP_INDEX),
                (&st.samplers, SAMPLER_HEAP_INDEX),
            ] {
                if let Some(h) = heap {
                    enc.setBuffer_offset_atIndex(Some(&h.table), 0, index);
                }
            }
        }
        let residency = self.residency(st, bound);
        for h in &residency.heaps {
            enc.useHeap(h);
        }
        for (r, writes) in &residency.loose {
            enc.useResource_usage(r, usage(*writes));
        }
    }

    /// Clears `at`'s view: a render pass of its own that clears and stores.
    fn clear(
        &self,
        at: usize,
        colour: Option<[f32; 4]>,
        depth: Option<f32>,
        stencil: Option<u8>,
        rects: &[Rect],
    ) {
        if !rects.is_empty() {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    "Metal clears by rectangle not done: the whole view is cleared (logged once)"
                )
            });
        }
        let Some(t) = self.slots.target(at) else {
            return;
        };
        self.end();
        let d = MTLRenderPassDescriptor::new();
        if let Some([r, g, b, a]) = colour {
            // SAFETY: attachment 0.
            let c = unsafe { d.colorAttachments().objectAtIndexedSubscript(0) };
            Self::attach(&c, &t);
            c.setLoadAction(MTLLoadAction::Clear);
            c.setClearColor(MTLClearColor {
                red: f64::from(r),
                green: f64::from(g),
                blue: f64::from(b),
                alpha: f64::from(a),
            });
        } else {
            let a = d.depthAttachment();
            Self::attach(&a, &t);
            match depth {
                Some(v) => {
                    a.setLoadAction(MTLLoadAction::Clear);
                    a.setClearDepth(f64::from(v));
                }
                None => a.setLoadAction(MTLLoadAction::Load),
            }
            if Self::has_stencil(&t) {
                let s = d.stencilAttachment();
                Self::attach(&s, &t);
                match stencil {
                    Some(v) => {
                        s.setLoadAction(MTLLoadAction::Clear);
                        s.setClearStencil(u32::from(v));
                    }
                    None => s.setLoadAction(MTLLoadAction::Load),
                }
            }
        }
        if let Some(e) = self.buffer.renderCommandEncoderWithDescriptor(&d) {
            e.endEncoding();
        }
    }

    /// Sets `arguments` on the root signature at `point` (as
    /// `ComputeDesc` says) and its heaps, for the one-shot calls.
    fn set_arguments(
        &self,
        point: Point,
        top: &TopLevel,
        arguments: &Arguments<'_, MetalResource, MetalDescriptorHeap>,
    ) {
        let Arguments {
            constants,
            bindings,
            tables,
            views,
            samplers,
        } = *arguments;
        let heaps: Vec<&MetalDescriptorHeap> = views
            .into_iter()
            .chain(samplers)
            .chain(tables.iter().map(|t| t.heap))
            .collect();
        // SAFETY (all below): the caller's live resources and heaps.
        unsafe { self.set_heaps(&heaps) };
        let (mut constants, mut bindings, mut tables) = (constants, bindings.iter(), tables.iter());
        for (i, e) in top.entries.iter().enumerate() {
            let i = i as u32;
            match e.slot {
                Slot::Constants(n) => {
                    let n = (n as usize).min(constants.len());
                    unsafe { self.set_root_constants(point, i, 0, &constants[..n]) };
                    constants = &constants[n..];
                }
                Slot::Descriptor(kind) => {
                    let Some(&(Binding::Read(r, offset) | Binding::Write(r, offset))) =
                        bindings.next()
                    else {
                        continue;
                    };
                    let address = buffer(r).map_or(0, |b| b.gpuAddress() + offset);
                    unsafe { self.set_root_view(point, i, kind, address) };
                }
                Slot::Table => {
                    let Some(t) = tables.next() else { continue };
                    let handle = t.heap.table.gpuAddress()
                        + u64::from(t.first) * u64::from(DESCRIPTOR_BYTES);
                    unsafe { self.set_root_table(point, i, handle) };
                }
            }
        }
    }

    /// The pass's targets, clears, viewport, scissor, blend factor and
    /// stencil reference, for the one-shot calls.
    fn set_pass(&self, pass: &Pass<'_>) {
        let targets: Vec<usize> = pass.targets.iter().map(|t| t.at).collect();
        let v = pass.viewport;
        let scissor = pass.scissor.map_or(
            [
                v.x as i32,
                v.y as i32,
                (v.x + v.width) as i32,
                (v.y + v.height) as i32,
            ],
            |[l, t, r, b]| [l as i32, t as i32, r as i32, b as i32],
        );
        // SAFETY (all below): the caller's live views.
        unsafe {
            for t in pass.targets {
                if let Some(c) = t.clear {
                    self.clear_target(t.at, c, &[]);
                }
            }
            if let Some(d) = pass.depth
                && let Some((depth, stencil)) = d.clear
            {
                self.clear_depth(d.at, Some(depth), Some(stencil), &[]);
            }
            self.set_targets(&targets, pass.depth.map(|d| d.at));
            self.set_viewports(&[v]);
            self.set_scissors(&[scissor]);
            self.set_blend_factor(pass.blend_factor);
            self.set_stencil_ref(pass.stencil_ref);
        }
    }

    /// Ends recording and submits.
    pub fn commit(self) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
        self.end();
        self.buffer.commit();
        self.buffer.clone()
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

fn size3(v: [u32; 3]) -> MTLSize {
    MTLSize {
        width: v[0] as usize,
        height: v[1] as usize,
        depth: v[2] as usize,
    }
}

/// `D3D_PRIMITIVE_TOPOLOGY` as Metal's primitive type.
fn primitive(topology: u32) -> MTLPrimitiveType {
    match topology {
        1 => MTLPrimitiveType::Point,
        2 => MTLPrimitiveType::Line,
        3 => MTLPrimitiveType::LineStrip,
        5 => MTLPrimitiveType::TriangleStrip,
        _ => MTLPrimitiveType::Triangle,
    }
}

impl Recorder for MetalRecorder {
    type Resource = MetalResource;
    type Pipeline = MetalPipeline;
    type GraphicsPipeline = MetalGraphicsPipeline;
    type DescriptorHeap = MetalDescriptorHeap;

    /// Converted DXIL: through the state machine. Our own shaders: their
    /// own binding (constants at buffer index 0, bindings from 1).
    unsafe fn dispatch(
        &self,
        pipeline: &MetalPipeline,
        arguments: &Arguments<'_, MetalResource, MetalDescriptorHeap>,
        groups: [u32; 3],
    ) {
        if pipeline.abi == Abi::Converted {
            // SAFETY (all below): the caller's.
            unsafe {
                self.set_compute_root(pipeline);
                self.set_compute_pipeline(pipeline);
            }
            self.set_arguments(Point::Compute, &pipeline.top, arguments);
            unsafe { self.dispatch_groups(groups) };
            return;
        }
        let Some(enc) = self.compute() else { return };
        enc.setComputePipelineState(&pipeline.state);
        // SAFETY: Metal copies the constants; the caller keeps bindings
        // inside their buffers.
        unsafe {
            if let Some(first) = arguments.constants.first() {
                enc.setBytes_length_atIndex(
                    NonNull::from(first).cast(),
                    arguments.constants.len() * 4,
                    0,
                );
            }
            for (i, b) in arguments.bindings.iter().enumerate() {
                let (Binding::Read(r, offset) | Binding::Write(r, offset)) = *b;
                enc.setBuffer_offset_atIndex(buffer(r), offset as usize, i + 1);
            }
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(size3(groups), size3(pipeline.threads));
    }

    unsafe fn draw(
        &self,
        pass: &Pass<'_>,
        pipeline: &MetalGraphicsPipeline,
        arguments: &Arguments<'_, MetalResource, MetalDescriptorHeap>,
        vertices: &Vertices<'_, MetalResource>,
        draw: &Draw,
    ) {
        self.set_pass(pass);
        // SAFETY (all below): the caller's.
        unsafe {
            self.set_graphics_root(pipeline);
            self.set_graphics_pipeline(pipeline);
        }
        self.set_arguments(Point::Graphics, &pipeline.top, arguments);
        let views: Vec<VertexView> = vertices
            .buffers
            .iter()
            .map(|v| {
                let b = buffer(v.resource);
                VertexView {
                    address: b.map_or(0, |b| b.gpuAddress() + v.offset),
                    size: b.map_or(0, |b| (b.length() as u64).saturating_sub(v.offset) as u32),
                    stride: v.stride,
                }
            })
            .collect();
        let index = vertices.index.and_then(|i| {
            let b = buffer(i.resource)?;
            Some(IndexView {
                address: b.gpuAddress() + i.offset,
                size: (b.length() as u64).saturating_sub(i.offset) as u32,
                format: i.format,
            })
        });
        unsafe {
            self.set_vertex_buffers(0, &views);
            self.set_index_buffer(index);
            self.set_topology(vertices.primitive);
            if index.is_some() {
                self.draw_indexed_instanced(
                    draw.count,
                    draw.instances,
                    draw.first,
                    draw.base_vertex,
                    draw.first_instance,
                );
            } else {
                self.draw_instanced(draw.count, draw.instances, draw.first, draw.first_instance);
            }
        }
    }

    unsafe fn draw_mesh(
        &self,
        pass: &Pass<'_>,
        pipeline: &MetalGraphicsPipeline,
        arguments: &Arguments<'_, MetalResource, MetalDescriptorHeap>,
        groups: [u32; 3],
    ) {
        self.set_pass(pass);
        // SAFETY (all below): the caller's.
        unsafe {
            self.set_graphics_root(pipeline);
            self.set_graphics_pipeline(pipeline);
        }
        self.set_arguments(Point::Graphics, &pipeline.top, arguments);
        unsafe { self.dispatch_mesh_groups(groups) };
    }

    unsafe fn set_heaps(&self, heaps: &[&MetalDescriptorHeap]) {
        let mut st = self.state.borrow_mut();
        st.views = None;
        st.samplers = None;
        for h in heaps {
            let r = Some(HeapRef {
                table: h.table.clone(),
                count: h.count,
            });
            match h.kind {
                Descriptors::Samplers => st.samplers = r,
                _ => st.views = r,
            }
        }
    }

    unsafe fn set_compute_root(&self, pipeline: &MetalPipeline) {
        self.state.borrow_mut().compute = Bound {
            top: pipeline.top.clone(),
            statics: pipeline.statics.as_ref().map(|s| s.table.clone()),
            args: Vec::new(),
        };
    }

    unsafe fn set_graphics_root(&self, pipeline: &MetalGraphicsPipeline) {
        self.state.borrow_mut().graphics = Bound {
            top: pipeline.top.clone(),
            statics: pipeline
                .statics
                .as_ref()
                .map(|s: &StaticSamplers| s.table.clone()),
            args: Vec::new(),
        };
    }

    unsafe fn set_compute_pipeline(&self, pipeline: &MetalPipeline) {
        self.state.borrow_mut().compute_pso = Some(ComputePso {
            state: pipeline.state.clone(),
            threads: pipeline.threads,
        });
    }

    unsafe fn set_graphics_pipeline(&self, pipeline: &MetalGraphicsPipeline) {
        self.state.borrow_mut().graphics_pso = Some(GraphicsPso {
            state: pipeline.state.clone(),
            depth: pipeline.depth.clone(),
            raster: pipeline.raster,
            kind: pipeline.kind,
            stages: pipeline.stages,
        });
    }

    unsafe fn set_root_constants(&self, point: Point, param: u32, offset: u32, values: &[u32]) {
        let mut st = self.state.borrow_mut();
        let bound = st.bound(point);
        let mut now = match bound.args.get(param as usize) {
            Some(Arg::Constants(v)) => v.clone(),
            _ => Vec::new(),
        };
        let end = offset as usize + values.len();
        if now.len() < end {
            now.resize(end, 0);
        }
        now[offset as usize..end].copy_from_slice(values);
        bound.set(param, Arg::Constants(now));
    }

    unsafe fn set_root_view(&self, point: Point, param: u32, _kind: u32, address: u64) {
        self.state
            .borrow_mut()
            .bound(point)
            .set(param, Arg::Address(address));
    }

    unsafe fn set_root_table(&self, point: Point, param: u32, handle: u64) {
        self.state
            .borrow_mut()
            .bound(point)
            .set(param, Arg::Address(handle));
    }

    unsafe fn set_targets(&self, targets: &[usize], depth: Option<usize>) {
        let mut st = self.state.borrow_mut();
        st.targets = targets.to_vec();
        st.depth = depth;
    }

    unsafe fn clear_target(&self, at: usize, rgba: [f32; 4], rects: &[Rect]) {
        self.clear(at, Some(rgba), None, None, rects);
    }

    unsafe fn clear_depth(
        &self,
        at: usize,
        depth: Option<f32>,
        stencil: Option<u8>,
        rects: &[Rect],
    ) {
        self.clear(at, None, depth, stencil, rects);
    }

    unsafe fn set_viewports(&self, viewports: &[Viewport]) {
        self.state.borrow_mut().viewports = viewports.to_vec();
    }

    unsafe fn set_scissors(&self, rects: &[Rect]) {
        self.state.borrow_mut().scissors = rects.to_vec();
    }

    unsafe fn set_blend_factor(&self, rgba: [f32; 4]) {
        self.state.borrow_mut().blend_factor = rgba;
    }

    unsafe fn set_stencil_ref(&self, reference: u32) {
        self.state.borrow_mut().stencil_ref = reference;
    }

    /// Metal has no depth bounds test (pipelines that ask are warned
    /// about when made).
    unsafe fn set_depth_bounds(&self, _min: f32, _max: f32) {}

    unsafe fn set_topology(&self, topology: u32) {
        self.state.borrow_mut().topology = topology;
    }

    unsafe fn set_vertex_buffers(&self, first: u32, views: &[VertexView]) {
        let mut st = self.state.borrow_mut();
        let end = first as usize + views.len();
        if st.vertex.len() < end {
            st.vertex.resize(end, None);
        }
        for (slot, v) in st.vertex[first as usize..end].iter_mut().zip(views) {
            *slot = Some(*v);
        }
    }

    unsafe fn set_index_buffer(&self, view: Option<IndexView>) {
        self.state.borrow_mut().index = view;
    }

    unsafe fn draw_instanced(&self, count: u32, instances: u32, first: u32, first_instance: u32) {
        let st = self.state.borrow();
        let Some(pso) = &st.graphics_pso else { return };
        let Some((enc, size)) = self.render(&st.targets, st.depth) else {
            return;
        };
        self.render_state(&enc, size, &st, pso);
        // SAFETY: MSC's draw parameter layouts (runtime header):
        // IRRuntimeDrawArgument at the draw argument index, the index type
        // (0: none) as a u16 at the uniforms index; the caller's buffers
        // cover the draw.
        unsafe {
            if matches!(pso.kind, Kind::Vertex { draw_params: true }) {
                let args = [count, instances, first, first_instance];
                enc.setVertexBytes_length_atIndex(
                    NonNull::from(&args).cast(),
                    size_of_val(&args),
                    crate::irconverter::DRAW_ARGUMENTS_INDEX,
                );
                let none = 0u16;
                enc.setVertexBytes_length_atIndex(
                    NonNull::from(&none).cast(),
                    2,
                    crate::irconverter::UNIFORMS_INDEX,
                );
            }
            enc.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                primitive(st.topology),
                first as usize,
                count as usize,
                instances as usize,
                first_instance as usize,
            );
        }
    }

    unsafe fn draw_indexed_instanced(
        &self,
        count: u32,
        instances: u32,
        first: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        let st = self.state.borrow();
        let Some(pso) = &st.graphics_pso else { return };
        let Some(view) = st.index else { return };
        let Some(ib) = self.buffers.at(view.address) else {
            return;
        };
        let Some((enc, size)) = self.render(&st.targets, st.depth) else {
            return;
        };
        self.render_state(&enc, size, &st, pso);
        let (kind, bytes) = if view.format == 57 {
            (MTLIndexType::UInt16, 2)
        } else {
            (MTLIndexType::UInt32, 4)
        };
        let offset = (view.address - ib.gpuAddress()) as usize + first as usize * bytes;
        // SAFETY: as for draw_instanced; IRRuntimeDrawIndexedArgument and
        // the index type as MTLIndexType + 1.
        unsafe {
            if matches!(pso.kind, Kind::Vertex { draw_params: true }) {
                let args: [u32; 5] = [count, instances, first, base_vertex as u32, first_instance];
                enc.setVertexBytes_length_atIndex(
                    NonNull::from(&args).cast(),
                    size_of_val(&args),
                    crate::irconverter::DRAW_ARGUMENTS_INDEX,
                );
                let ir = kind.0 as u16 + 1;
                enc.setVertexBytes_length_atIndex(
                    NonNull::from(&ir).cast(),
                    2,
                    crate::irconverter::UNIFORMS_INDEX,
                );
            }
            enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                primitive(st.topology),
                count as usize,
                kind,
                &ib,
                offset,
                instances as usize,
                base_vertex as isize,
                first_instance as usize,
            );
        }
    }

    unsafe fn dispatch_groups(&self, groups: [u32; 3]) {
        let st = self.state.borrow();
        let Some(pso) = &st.compute_pso else { return };
        let Some(enc) = self.compute() else { return };
        self.compute_state(&enc, &st, pso);
        enc.dispatchThreadgroups_threadsPerThreadgroup(size3(groups), size3(pso.threads));
    }

    unsafe fn dispatch_mesh_groups(&self, groups: [u32; 3]) {
        let st = self.state.borrow();
        let Some(pso) = &st.graphics_pso else { return };
        let Kind::Mesh { object, mesh, .. } = pso.kind else {
            return;
        };
        let Some((enc, size)) = self.render(&st.targets, st.depth) else {
            return;
        };
        self.render_state(&enc, size, &st, pso);
        enc.drawMeshThreadgroups_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
            size3(groups),
            size3(object),
            size3(mesh),
        );
    }

    /// D3D12's record layouts are Metal's indirect argument layouts
    /// (`MTLDrawPrimitivesIndirectArguments` and the rest), so each record
    /// is one Metal indirect command. The count is not read: all `max`
    /// records run (see the trait). Draw parameters for vertex shaders
    /// that read them are the record itself (MSC's
    /// `IRRuntimeDrawArgument` / `IndexedArgument` share the layout).
    unsafe fn execute_indirect(
        &self,
        kind: u32,
        stride: u32,
        max: u32,
        args: &MetalResource,
        offset: u64,
        _count: Option<(&MetalResource, u64)>,
    ) {
        use victory_host::indirect;
        let Some(args) = buffer(args) else { return };
        let at = |i: u32| offset as usize + i as usize * stride as usize;
        let st = self.state.borrow();
        if kind == indirect::DISPATCH {
            let Some(pso) = &st.compute_pso else { return };
            let Some(enc) = self.compute() else { return };
            self.compute_state(&enc, &st, pso);
            for i in 0..max {
                // SAFETY: the caller's records, inside `args`.
                unsafe {
                    enc.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                        args,
                        at(i),
                        size3(pso.threads),
                    )
                };
            }
            return;
        }
        let Some(pso) = &st.graphics_pso else { return };
        let Some((enc, size)) = self.render(&st.targets, st.depth) else {
            return;
        };
        self.render_state(&enc, size, &st, pso);
        let draw_params = matches!(pso.kind, Kind::Vertex { draw_params: true });
        let ty = primitive(st.topology);
        // SAFETY (all below): the caller's records, inside `args`; MSC's
        // draw parameter buffers as for the direct draws.
        match kind {
            indirect::DRAW => {
                if draw_params {
                    let none = 0u16;
                    unsafe {
                        enc.setVertexBytes_length_atIndex(
                            NonNull::from(&none).cast(),
                            2,
                            crate::irconverter::UNIFORMS_INDEX,
                        )
                    };
                }
                for i in 0..max {
                    unsafe {
                        if draw_params {
                            enc.setVertexBuffer_offset_atIndex(
                                Some(args),
                                at(i),
                                crate::irconverter::DRAW_ARGUMENTS_INDEX,
                            );
                        }
                        enc.drawPrimitives_indirectBuffer_indirectBufferOffset(ty, args, at(i));
                    }
                }
            }
            indirect::INDEXED => {
                let Some(view) = st.index else { return };
                let Some(ib) = self.buffers.at(view.address) else {
                    return;
                };
                let kind = if view.format == 57 {
                    MTLIndexType::UInt16
                } else {
                    MTLIndexType::UInt32
                };
                if draw_params {
                    let ir = kind.0 as u16 + 1;
                    unsafe {
                        enc.setVertexBytes_length_atIndex(
                            NonNull::from(&ir).cast(),
                            2,
                            crate::irconverter::UNIFORMS_INDEX,
                        )
                    };
                }
                let ib_offset = (view.address - ib.gpuAddress()) as usize;
                for i in 0..max {
                    unsafe {
                        if draw_params {
                            enc.setVertexBuffer_offset_atIndex(
                                Some(args),
                                at(i),
                                crate::irconverter::DRAW_ARGUMENTS_INDEX,
                            );
                        }
                        enc.drawIndexedPrimitives_indexType_indexBuffer_indexBufferOffset_indirectBuffer_indirectBufferOffset(
                            ty, kind, &ib, ib_offset, args, at(i),
                        );
                    }
                }
            }
            indirect::MESH => {
                let Kind::Mesh { object, mesh, .. } = pso.kind else {
                    return;
                };
                for i in 0..max {
                    unsafe {
                        enc.drawMeshThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerObjectThreadgroup_threadsPerMeshThreadgroup(
                            args,
                            at(i),
                            size3(object),
                            size3(mesh),
                        )
                    };
                }
            }
            _ => {}
        }
    }

    /// A render pass that loads the multisampled source and stores it
    /// resolved into the destination. Metal resolves whole attachments of
    /// one size: a size mismatch is warned about and skipped.
    unsafe fn resolve(
        &self,
        dst: &MetalResource,
        dst_sub: u32,
        src: &MetalResource,
        src_sub: u32,
        _format: u32,
    ) {
        let (Some(to), Some(from)) = (texture(dst), texture(src)) else {
            return;
        };
        let (to_slice, to_level) = place(to, dst_sub);
        let (from_slice, from_level) = place(from, src_sub);
        let size = |t: &ProtocolObject<dyn MTLTexture>, l: usize| {
            ((t.width() >> l).max(1), (t.height() >> l).max(1))
        };
        if size(to, to_level) != size(from, from_level) {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!("Metal resolve of unlike sizes not done: skipped (logged once)")
            });
            return;
        }
        self.end();
        let d = MTLRenderPassDescriptor::new();
        // SAFETY: attachment 0.
        let a = unsafe { d.colorAttachments().objectAtIndexedSubscript(0) };
        a.setTexture(Some(from));
        a.setLevel(from_level);
        a.setSlice(from_slice);
        a.setLoadAction(MTLLoadAction::Load);
        a.setStoreAction(MTLStoreAction::MultisampleResolve);
        a.setResolveTexture(Some(to));
        a.setResolveLevel(to_level);
        if to.textureType() == MTLTextureType::Type3D {
            a.setResolveDepthPlane(to_slice);
        } else {
            a.setResolveSlice(to_slice);
        }
        if let Some(e) = self.buffer.renderCommandEncoderWithDescriptor(&d) {
            e.endEncoding();
        }
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
