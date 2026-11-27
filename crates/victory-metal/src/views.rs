//! Views and samplers as descriptor table entries.
//!
//! An entry is Metal Shader Converter's `IRDescriptorTableEntry`: three
//! words, written the way its runtime header's helpers write them
//! (`IRDescriptorTableSetBuffer` / `SetTexture` / `SetSampler`):
//!
//! - buffer:  GPU address, 0, view length in bytes
//! - texture: 0, texture resource id, 0
//! - sampler: sampler resource id, 0, LOD bias (f32 bits)
//!
//! TODO(device): check the metadata words against metal_irconverter_runtime.h
//! once shaders run (min LOD clamp for textures, typed buffer views need a
//! texture-buffer view rather than a raw address).
//!
//! Metal objects a slot needs (texture views, sampler states, target
//! views) are kept alive per slot address until the slot is rewritten,
//! and so are the buffers buffer views name: a dispatch makes what a
//! table's slots name resident (`Slots::resident`), so it has to know them.
//! (A buffer thus lives until the last slot naming it is rewritten.)

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBuffer, MTLCompareFunction, MTLDevice, MTLPixelFormat, MTLResourceID, MTLSamplerAddressMode,
    MTLSamplerBorderColor, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerMipFilter,
    MTLSamplerReductionMode, MTLSamplerState, MTLTexture, MTLTextureType,
};
use victory_host::{
    DepthViewDesc, SamplerDesc, ShaderViewDesc, TargetViewDesc, UnorderedViewDesc, view,
};

use crate::format;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
type Sampler = Retained<ProtocolObject<dyn MTLSamplerState>>;

/// A render target or depth view: what a render pass attachment needs.
#[derive(Clone)]
pub struct Target {
    pub texture: Texture,
    pub level: usize,
    pub slice: usize,
}

/// What a slot names: kept alive while it does.
#[derive(Clone)]
pub(crate) enum Kept {
    Buffer(Buffer),
    Texture(Texture),
    Sampler(#[allow(dead_code)] Sampler),
    Target(Target),
}

#[derive(Clone)]
struct Slot {
    kept: Kept,
    /// An unordered access view: the shader may write it.
    write: bool,
}

/// A resource a dispatch makes resident.
pub(crate) enum Resident {
    Buffer(Buffer),
    Texture(Texture),
}

/// Objects slots refer to, by slot address.
#[derive(Default)]
pub(crate) struct Slots(Mutex<BTreeMap<usize, Slot>>);

impl Slots {
    pub(crate) fn set(&self, at: usize, kept: Option<Kept>, write: bool) {
        let mut slots = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match kept {
            Some(kept) => slots.insert(at, Slot { kept, write }),
            None => slots.remove(&at),
        };
    }

    pub(crate) fn target(&self, at: usize) -> Option<Target> {
        match self.0.lock().unwrap_or_else(|e| e.into_inner()).get(&at) {
            Some(Slot {
                kept: Kept::Target(t),
                ..
            }) => Some(t.clone()),
            _ => None,
        }
    }

    /// The buffers and textures (with whether they may be written) named
    /// by slots at addresses in `range`.
    pub(crate) fn resident(&self, range: Range<usize>, out: &mut Vec<(Resident, bool)>) {
        let slots = self.0.lock().unwrap_or_else(|e| e.into_inner());
        for slot in slots.range(range).map(|(_, s)| s) {
            let r = match &slot.kept {
                Kept::Buffer(b) => Resident::Buffer(b.clone()),
                Kept::Texture(t) => Resident::Texture(t.clone()),
                Kept::Sampler(_) | Kept::Target(_) => continue,
            };
            out.push((r, slot.write));
        }
    }

    pub(crate) fn copy(&self, count: u32, to: usize, from: usize, step: usize) {
        let mut slots = self.0.lock().unwrap_or_else(|e| e.into_inner());
        for i in 0..count as usize {
            match slots.get(&(from + i * step)).cloned() {
                Some(k) => slots.insert(to + i * step, k),
                None => slots.remove(&(to + i * step)),
            };
        }
    }
}

pub(crate) fn resource_id(id: MTLResourceID) -> u64 {
    const _: () = assert!(size_of::<MTLResourceID>() == size_of::<u64>());
    // SAFETY: MTLResourceID is a repr(C) wrapper of one u64.
    unsafe { core::mem::transmute::<MTLResourceID, u64>(id) }
}

/// Writes one table entry.
///
/// # Safety
/// `at` is the address of an entry in a live descriptor table.
pub(crate) unsafe fn write(at: usize, words: [u64; 3]) {
    // SAFETY: the caller's entry; tables are 8-byte aligned shared buffers.
    unsafe { (at as *mut [u64; 3]).write(words) }
}

fn is_depth(format: MTLPixelFormat) -> bool {
    (250..=261).contains(&format.0)
}

fn view_type(dimension: u32) -> Option<MTLTextureType> {
    Some(match dimension {
        view::TEXTURE_1D => MTLTextureType::Type1D,
        view::TEXTURE_1D_ARRAY => MTLTextureType::Type1DArray,
        view::TEXTURE_2D => MTLTextureType::Type2D,
        view::TEXTURE_2D_ARRAY => MTLTextureType::Type2DArray,
        view::TEXTURE_2D_MS => MTLTextureType::Type2DMultisample,
        view::TEXTURE_2D_MS_ARRAY => MTLTextureType::Type2DMultisampleArray,
        view::TEXTURE_3D => MTLTextureType::Type3D,
        view::TEXTURE_CUBE => MTLTextureType::TypeCube,
        view::TEXTURE_CUBE_ARRAY => MTLTextureType::TypeCubeArray,
        _ => return None,
    })
}

/// A view of `texture` as `format` (D3D12.X number) and `kind`, over mips
/// `mips` and slices `slices` (`u32::MAX` counts: to the end). The texture
/// itself when that is what it already is. Depth textures keep their own
/// format: Metal has no colour views of depth.
pub(crate) fn texture_view(
    texture: &Texture,
    format: u32,
    kind: MTLTextureType,
    (first_mip, mips): (u32, u32),
    (first_slice, slices): (u32, u32),
) -> Option<Texture> {
    let own = texture.pixelFormat();
    let pixel = if is_depth(own) || format == 0 {
        own
    } else {
        MTLPixelFormat(format::map(format, false)?.pixel_format)
    };
    let all_mips = texture.mipmapLevelCount();
    let all_slices = match texture.textureType() {
        MTLTextureType::Type3D => 1,
        MTLTextureType::TypeCube | MTLTextureType::TypeCubeArray => texture.arrayLength() * 6,
        _ => texture.arrayLength(),
    };
    let first_mip = first_mip as usize;
    let first_slice = first_slice as usize;
    let mips = if mips == u32::MAX {
        all_mips.saturating_sub(first_mip)
    } else {
        mips as usize
    };
    let slices = match kind {
        MTLTextureType::TypeCube => 6,
        MTLTextureType::TypeCubeArray => slices as usize * 6,
        _ if slices == u32::MAX => all_slices.saturating_sub(first_slice),
        _ => slices as usize,
    };
    let whole = pixel == own
        && kind == texture.textureType()
        && first_mip == 0
        && mips == all_mips
        && first_slice == 0
        && slices == all_slices;
    if whole {
        return Some(texture.clone());
    }
    // SAFETY: ranges are clamped to the texture by Metal's validation; an
    // out-of-range request returns nil, which reads as a null view.
    unsafe {
        texture.newTextureViewWithPixelFormat_textureType_levels_slices(
            pixel,
            kind,
            NSRange::new(first_mip, mips.max(1)),
            NSRange::new(first_slice, slices.max(1)),
        )
    }
}

/// A shader resource view of a texture.
pub(crate) fn shader_texture(texture: &Texture, desc: Option<&ShaderViewDesc>) -> Option<Texture> {
    let Some(d) = desc else {
        return Some(texture.clone());
    };
    let (mip, mips, slice, slices) = d.texture();
    texture_view(
        texture,
        d.format,
        view_type(d.dimension)?,
        (mip, mips),
        (slice, slices),
    )
}

/// An unordered access view of a texture: one mip.
pub(crate) fn unordered_texture(
    texture: &Texture,
    desc: Option<&UnorderedViewDesc>,
) -> Option<Texture> {
    let Some(d) = desc else {
        return Some(texture.clone());
    };
    let w = |i: usize| (d.fields[i / 2] >> (32 * (i % 2))) as u32;
    let (mip, slice, slices) = match d.dimension {
        view::TEXTURE_1D_ARRAY | view::TEXTURE_2D_ARRAY => (w(0), w(1), w(2)),
        view::TEXTURE_2D_MS_ARRAY => (0, w(0), w(1)),
        // 3D: the whole depth of the mip (W slices are not views in Metal).
        view::TEXTURE_3D => (w(0), 0, 1),
        view::TEXTURE_2D_MS => (0, 0, 1),
        _ => (w(0), 0, 1),
    };
    texture_view(
        texture,
        d.format,
        view_type(d.dimension)?,
        (mip, 1),
        (slice, slices),
    )
}

/// Bytes per element of a buffer view.
pub(crate) fn element_bytes(format: u32, stride: u32, flags: u32) -> u64 {
    if flags & view::BUFFER_RAW != 0 {
        4
    } else if stride != 0 {
        u64::from(stride)
    } else {
        format::console_block(format).map_or(4, |b| u64::from(b.bytes))
    }
}

/// The entry for a buffer view of `buffer`.
pub(crate) fn buffer_entry(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    (first, count, stride, flags): (u64, u32, u32, u32),
    format: u32,
) -> [u64; 3] {
    let element = element_bytes(format, stride, flags);
    let offset = first * element;
    let length = (u64::from(count) * element).min((buffer.length() as u64).saturating_sub(offset));
    [buffer.gpuAddress() + offset, 0, length]
}

fn address_mode(d3d12: u32) -> MTLSamplerAddressMode {
    match d3d12 {
        1 => MTLSamplerAddressMode::Repeat,
        2 => MTLSamplerAddressMode::MirrorRepeat,
        4 => MTLSamplerAddressMode::ClampToBorderColor,
        5 => MTLSamplerAddressMode::MirrorClampToEdge,
        _ => MTLSamplerAddressMode::ClampToEdge,
    }
}

/// Metal has three border colours: the nearest of them.
fn border(rgba: [f32; 4]) -> MTLSamplerBorderColor {
    if rgba[3] < 0.5 {
        MTLSamplerBorderColor::TransparentBlack
    } else if rgba[0] + rgba[1] + rgba[2] > 1.5 {
        MTLSamplerBorderColor::OpaqueWhite
    } else {
        MTLSamplerBorderColor::OpaqueBlack
    }
}

/// A sampler state for a D3D12.X sampler description.
pub(crate) fn sampler(device: &ProtocolObject<dyn MTLDevice>, d: &SamplerDesc) -> Option<Sampler> {
    let s = MTLSamplerDescriptor::new();
    let linear = |bit: u32| d.filter & bit != 0;
    let filter = |bit| {
        if linear(bit) {
            MTLSamplerMinMagFilter::Linear
        } else {
            MTLSamplerMinMagFilter::Nearest
        }
    };
    s.setMinFilter(filter(0x10));
    s.setMagFilter(filter(0x4));
    s.setMipFilter(if linear(0x1) {
        MTLSamplerMipFilter::Linear
    } else {
        MTLSamplerMipFilter::Nearest
    });
    if d.filter & 0x40 != 0 {
        s.setMaxAnisotropy(d.max_anisotropy.clamp(1, 16) as usize);
    }
    s.setSAddressMode(address_mode(d.address[0]));
    s.setTAddressMode(address_mode(d.address[1]));
    s.setRAddressMode(address_mode(d.address[2]));
    s.setBorderColor(border(d.border));
    s.setLodMinClamp(d.min_lod);
    s.setLodMaxClamp(d.max_lod);
    // Filter reduction: bits 7-8 (standard, comparison, minimum, maximum).
    match (d.filter >> 7) & 3 {
        1 if (1..=8).contains(&d.comparison) => {
            s.setCompareFunction(MTLCompareFunction(d.comparison as usize - 1));
        }
        2 => s.setReductionMode(MTLSamplerReductionMode::Minimum),
        3 => s.setReductionMode(MTLSamplerReductionMode::Maximum),
        _ => {}
    }
    s.setSupportArgumentBuffers(true);
    device.newSamplerStateWithDescriptor(&s)
}

/// A render target view: texture, mip and slice.
pub(crate) fn target(texture: &Texture, desc: Option<&TargetViewDesc>) -> Option<Target> {
    let Some(d) = desc else {
        return Some(Target {
            texture: texture.clone(),
            level: 0,
            slice: 0,
        });
    };
    let w = |i: usize| (d.fields[i / 2] >> (32 * (i % 2))) as usize;
    // RTV dimensions: 1D=2, 1D array=3, 2D=4, 2D array=5, MS=6, MS array=7, 3D=8.
    let (level, slice) = match d.dimension {
        3 | 5 => (w(0), w(1)),
        7 => (0, w(0)),
        8 => (w(0), w(1)),
        6 => (0, 0),
        _ => (w(0), 0),
    };
    let texture = texture_view(
        texture,
        d.format,
        texture.textureType(),
        (0, u32::MAX),
        (0, u32::MAX),
    )?;
    Some(Target {
        texture,
        level,
        slice,
    })
}

/// A depth-stencil view: texture, mip and slice.
pub(crate) fn depth(texture: &Texture, desc: Option<&DepthViewDesc>) -> Target {
    let Some(d) = desc else {
        return Target {
            texture: texture.clone(),
            level: 0,
            slice: 0,
        };
    };
    // DSV dimensions: 1D=1, 1D array=2, 2D=3, 2D array=4, MS=5, MS array=6.
    let (level, slice) = match d.dimension {
        2 | 4 => (d.fields[0], d.fields[1]),
        6 => (0, d.fields[0]),
        5 => (0, 0),
        _ => (d.fields[0], 0),
    };
    Target {
        texture: texture.clone(),
        level: level as usize,
        slice: slice as usize,
    }
}
