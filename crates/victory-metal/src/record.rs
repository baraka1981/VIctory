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
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
    MTLComputeCommandEncoder, MTLHeap, MTLOrigin, MTLResource, MTLResourceUsage, MTLSize,
    MTLTexture,
};
use victory_host::{Barrier, Binding, Location, Recorder, Region, Table};

use crate::device::{Abi, MetalDescriptorHeap, MetalPipeline, MetalResource};
use crate::views::{Resident, Slots};
use crate::{DESCRIPTOR_BYTES, MetalError};
use crate::format::console_block;

type Encoder = Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>;

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
    /// descriptors' buffers, the tables' descriptor buffers, and what the
    /// descriptors the root signature's ranges cover name. Those placed in
    /// a heap go resident a heap at a time (`useHeap`), the rest one by one
    /// (`useResource`, read-write for unordered access views).
    fn make_resident(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pipeline: &MetalPipeline,
        bindings: &[Binding<'_, MetalResource>],
        tables: &[Table<'_, MetalDescriptorHeap>],
    ) {
        let read = MTLResourceUsage::Read;
        let write = MTLResourceUsage::Read | MTLResourceUsage::Write;
        for b in bindings {
            let (usage, r) = match *b {
                Binding::Read(r, _) => (read, r),
                Binding::Write(r, _) => (write, r),
            };
            if let Some(buf) = buffer(r) {
                enc.useResource_usage(ProtocolObject::from_ref(buf), usage);
            }
        }
        let mut named = Vec::new();
        for (k, t) in tables.iter().enumerate() {
            enc.useResource_usage(ProtocolObject::from_ref(&*t.heap.table), read);
            let base = t.heap.table.contents().as_ptr() as usize;
            let step = DESCRIPTOR_BYTES as usize;
            for run in pipeline.top.reached(k, t.first, t.heap.count) {
                self.slots.resident(
                    base + run.start as usize * step..base + run.end as usize * step,
                    &mut named,
                );
            }
        }
        // One call per heap, and per resource outside heaps (the
        // strongest usage it is named with).
        let mut heaps: Vec<Retained<ProtocolObject<dyn MTLHeap>>> = Vec::new();
        let mut loose: Vec<(Retained<ProtocolObject<dyn MTLResource>>, bool)> = Vec::new();
        for (r, writes) in named {
            let resource: Retained<ProtocolObject<dyn MTLResource>> = match r {
                Resident::Buffer(b) => ProtocolObject::from_retained(b),
                Resident::Texture(t) => {
                    // A view's memory is its parent's.
                    let t = t.parentTexture().unwrap_or(t);
                    ProtocolObject::from_retained(t)
                }
            };
            if let Some(h) = resource.heap() {
                if !heaps.iter().any(|x| Retained::as_ptr(x) == Retained::as_ptr(&h)) {
                    heaps.push(h);
                }
                continue;
            }
            match loose
                .iter_mut()
                .find(|(x, _)| Retained::as_ptr(x) == Retained::as_ptr(&resource))
            {
                Some((_, w)) => *w |= writes,
                None => loose.push((resource, writes)),
            }
        }
        for h in &heaps {
            enc.useHeap(h);
        }
        for (r, writes) in &loose {
            enc.useResource_usage(r, if *writes { write } else { read });
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
    type DescriptorHeap = MetalDescriptorHeap;

    /// One compute encoder per dispatch. Converted DXIL: the top-level
    /// argument buffer its root signature lays out, with everything it
    /// reaches made resident. Our own shaders: constants at buffer index
    /// 0, bindings from 1 (no tables).
    unsafe fn dispatch(
        &self,
        pipeline: &MetalPipeline,
        constants: &[u32],
        bindings: &[Binding<'_, MetalResource>],
        tables: &[Table<'_, MetalDescriptorHeap>],
        groups: [u32; 3],
    ) {
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
            let top = pipeline.top.fill(constants, &addresses, &starts);
            unsafe {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&top[0]).cast(),
                    top.len(),
                    crate::irconverter::ARGUMENT_BUFFER_INDEX,
                );
            }
            self.make_resident(&enc, pipeline, bindings, tables);
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
