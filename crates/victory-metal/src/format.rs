//! D3D12.X format numbers (DXGI plus the console's) to Metal pixel formats.
//!
//! Portable: the numbers are `MTLPixelFormat` raw values, so the table is
//! built and tested off macOS; [`crate::device`] wraps them.

/// How a format's texels are stored: bytes per block and the block's size
/// in texels (4x4 for block compression).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    pub bytes: u32,
    pub width: u32,
    pub height: u32,
}

const fn texel(bytes: u32) -> Block {
    Block { bytes, width: 1, height: 1 }
}

const fn bc(bytes: u32) -> Block {
    Block { bytes, width: 4, height: 4 }
}

/// A format as Metal has it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mapped {
    /// `MTLPixelFormat` raw value.
    pub pixel_format: usize,
    pub block: Block,
    /// Typeless on the console: views reinterpret it, so the texture needs
    /// `MTLTextureUsagePixelFormatView`.
    pub typeless: bool,
}

/// `MTLPixelFormat` raw values used here (MTLPixelFormat.h).
pub mod mtl {
    pub const A8_UNORM: usize = 1;
    pub const R8_UNORM: usize = 10;
    pub const R8_SNORM: usize = 12;
    pub const R8_UINT: usize = 13;
    pub const R8_SINT: usize = 14;
    pub const R16_UNORM: usize = 20;
    pub const R16_SNORM: usize = 22;
    pub const R16_UINT: usize = 23;
    pub const R16_SINT: usize = 24;
    pub const R16_FLOAT: usize = 25;
    pub const RG8_UNORM: usize = 30;
    pub const RG8_SNORM: usize = 32;
    pub const RG8_UINT: usize = 33;
    pub const RG8_SINT: usize = 34;
    pub const B5G6R5_UNORM: usize = 40;
    pub const BGR5A1_UNORM: usize = 43;
    pub const ABGR4_UNORM: usize = 42;
    pub const R32_UINT: usize = 53;
    pub const R32_SINT: usize = 54;
    pub const R32_FLOAT: usize = 55;
    pub const RG16_UNORM: usize = 60;
    pub const RG16_SNORM: usize = 62;
    pub const RG16_UINT: usize = 63;
    pub const RG16_SINT: usize = 64;
    pub const RG16_FLOAT: usize = 65;
    pub const RGBA8_UNORM: usize = 70;
    pub const RGBA8_UNORM_SRGB: usize = 71;
    pub const RGBA8_SNORM: usize = 72;
    pub const RGBA8_UINT: usize = 73;
    pub const RGBA8_SINT: usize = 74;
    pub const BGRA8_UNORM: usize = 80;
    pub const BGRA8_UNORM_SRGB: usize = 81;
    pub const RGB10A2_UNORM: usize = 90;
    pub const RGB10A2_UINT: usize = 91;
    pub const RG11B10_FLOAT: usize = 92;
    pub const RGB9E5_FLOAT: usize = 93;
    pub const RG32_UINT: usize = 103;
    pub const RG32_SINT: usize = 104;
    pub const RG32_FLOAT: usize = 105;
    pub const RGBA16_UNORM: usize = 110;
    pub const RGBA16_SNORM: usize = 112;
    pub const RGBA16_UINT: usize = 113;
    pub const RGBA16_SINT: usize = 114;
    pub const RGBA16_FLOAT: usize = 115;
    pub const RGBA32_UINT: usize = 123;
    pub const RGBA32_SINT: usize = 124;
    pub const RGBA32_FLOAT: usize = 125;
    pub const BC1_RGBA: usize = 130;
    pub const BC1_RGBA_SRGB: usize = 131;
    pub const BC2_RGBA: usize = 132;
    pub const BC2_RGBA_SRGB: usize = 133;
    pub const BC3_RGBA: usize = 134;
    pub const BC3_RGBA_SRGB: usize = 135;
    pub const BC4_RUNORM: usize = 140;
    pub const BC4_RSNORM: usize = 141;
    pub const BC5_RGUNORM: usize = 142;
    pub const BC5_RGSNORM: usize = 143;
    pub const BC6H_RGB_FLOAT: usize = 150;
    pub const BC6H_RGB_UFLOAT: usize = 151;
    pub const BC7_RGBA_UNORM: usize = 152;
    pub const BC7_RGBA_UNORM_SRGB: usize = 153;
    pub const DEPTH16_UNORM: usize = 250;
    pub const DEPTH32_FLOAT: usize = 252;
    pub const DEPTH32_FLOAT_STENCIL8: usize = 260;
}

/// The Metal format for D3D12.X `format`; `depth` when the resource is a
/// depth-stencil target (typeless depth families then map to depth
/// formats). `None` where Metal has no match (96-bit RGB, 1-bit, packed
/// video formats, and console formats not translated yet).
///
/// Apple silicon has no 24-bit depth: D24S8 becomes D32FS8, as MoltenVK
/// and D3DMetal do; texel bytes then differ from the console's, which only
/// matters for host-side copies of depth (they go through shaders).
pub fn map(format: u32, depth: bool) -> Option<Mapped> {
    use mtl::*;
    let (pixel_format, block, typeless) = match format {
        1 => (RGBA32_FLOAT, texel(16), true),
        2 => (RGBA32_FLOAT, texel(16), false),
        3 => (RGBA32_UINT, texel(16), false),
        4 => (RGBA32_SINT, texel(16), false),
        9 => (RGBA16_FLOAT, texel(8), true),
        10 => (RGBA16_FLOAT, texel(8), false),
        11 => (RGBA16_UNORM, texel(8), false),
        12 => (RGBA16_UINT, texel(8), false),
        13 => (RGBA16_SNORM, texel(8), false),
        14 => (RGBA16_SINT, texel(8), false),
        15 => (RG32_FLOAT, texel(8), true),
        16 => (RG32_FLOAT, texel(8), false),
        17 => (RG32_UINT, texel(8), false),
        18 => (RG32_SINT, texel(8), false),
        // D32 + S8 (+24 unused): typeless family, depth, and its plane views.
        19..=22 => (DEPTH32_FLOAT_STENCIL8, texel(8), format == 19),
        23 => (RGB10A2_UNORM, texel(4), true),
        24 => (RGB10A2_UNORM, texel(4), false),
        25 => (RGB10A2_UINT, texel(4), false),
        26 => (RG11B10_FLOAT, texel(4), false),
        27 => (RGBA8_UNORM, texel(4), true),
        28 => (RGBA8_UNORM, texel(4), false),
        29 => (RGBA8_UNORM_SRGB, texel(4), false),
        30 => (RGBA8_UINT, texel(4), false),
        31 => (RGBA8_SNORM, texel(4), false),
        32 => (RGBA8_SINT, texel(4), false),
        33 => (RG16_FLOAT, texel(4), true),
        34 => (RG16_FLOAT, texel(4), false),
        35 => (RG16_UNORM, texel(4), false),
        36 => (RG16_UINT, texel(4), false),
        37 => (RG16_SNORM, texel(4), false),
        38 => (RG16_SINT, texel(4), false),
        39 if depth => (DEPTH32_FLOAT, texel(4), true),
        39 => (R32_FLOAT, texel(4), true),
        40 => (DEPTH32_FLOAT, texel(4), false),
        41 => (R32_FLOAT, texel(4), false),
        42 => (R32_UINT, texel(4), false),
        43 => (R32_SINT, texel(4), false),
        // D24 + S8: no 24-bit depth on Apple silicon (see above).
        44..=47 => (DEPTH32_FLOAT_STENCIL8, texel(8), format == 44),
        48 => (RG8_UNORM, texel(2), true),
        49 => (RG8_UNORM, texel(2), false),
        50 => (RG8_UINT, texel(2), false),
        51 => (RG8_SNORM, texel(2), false),
        52 => (RG8_SINT, texel(2), false),
        53 if depth => (DEPTH16_UNORM, texel(2), true),
        53 => (R16_UNORM, texel(2), true),
        54 => (R16_FLOAT, texel(2), false),
        55 => (DEPTH16_UNORM, texel(2), false),
        56 => (R16_UNORM, texel(2), false),
        57 => (R16_UINT, texel(2), false),
        58 => (R16_SNORM, texel(2), false),
        59 => (R16_SINT, texel(2), false),
        60 => (R8_UNORM, texel(1), true),
        61 => (R8_UNORM, texel(1), false),
        62 => (R8_UINT, texel(1), false),
        63 => (R8_SNORM, texel(1), false),
        64 => (R8_SINT, texel(1), false),
        65 => (A8_UNORM, texel(1), false),
        67 => (RGB9E5_FLOAT, texel(4), false),
        70 => (BC1_RGBA, bc(8), true),
        71 => (BC1_RGBA, bc(8), false),
        72 => (BC1_RGBA_SRGB, bc(8), false),
        73 => (BC2_RGBA, bc(16), true),
        74 => (BC2_RGBA, bc(16), false),
        75 => (BC2_RGBA_SRGB, bc(16), false),
        76 => (BC3_RGBA, bc(16), true),
        77 => (BC3_RGBA, bc(16), false),
        78 => (BC3_RGBA_SRGB, bc(16), false),
        79 => (BC4_RUNORM, bc(8), true),
        80 => (BC4_RUNORM, bc(8), false),
        81 => (BC4_RSNORM, bc(8), false),
        82 => (BC5_RGUNORM, bc(16), true),
        83 => (BC5_RGUNORM, bc(16), false),
        84 => (BC5_RGSNORM, bc(16), false),
        85 => (B5G6R5_UNORM, texel(2), false),
        86 => (BGR5A1_UNORM, texel(2), false),
        87 => (BGRA8_UNORM, texel(4), false),
        // X8 is ignored: plain BGRA8 (alpha is never read as such).
        88 => (BGRA8_UNORM, texel(4), false),
        90 => (BGRA8_UNORM, texel(4), true),
        91 => (BGRA8_UNORM_SRGB, texel(4), false),
        92 => (BGRA8_UNORM, texel(4), true),
        93 => (BGRA8_UNORM_SRGB, texel(4), false),
        94 => (BC6H_RGB_UFLOAT, bc(16), true),
        95 => (BC6H_RGB_UFLOAT, bc(16), false),
        96 => (BC6H_RGB_FLOAT, bc(16), false),
        97 => (BC7_RGBA_UNORM, bc(16), true),
        98 => (BC7_RGBA_UNORM, bc(16), false),
        99 => (BC7_RGBA_UNORM_SRGB, bc(16), false),
        // B4G4R4A4. Packed 16-bit formats (85, 86, 115): channel bit order
        // vs DXGI not yet checked on a device (TODO: draw-and-read test).
        115 => (ABGR4_UNORM, texel(2), false),
        // The console's 16-bit depth with 8-bit stencil (two planes).
        118..=120 => (DEPTH32_FLOAT_STENCIL8, texel(8), format == 118),
        _ => return None,
    };
    Some(Mapped { pixel_format, block, typeless })
}

/// Bytes and block shape of a D3D12.X format as the *console* stores it,
/// for copy layouts (independent of what Metal maps it to). `None` for
/// formats without whole-byte texels.
pub fn console_block(format: u32) -> Option<Block> {
    Some(match format {
        1..=4 => texel(16),
        5..=8 => texel(12),
        9..=22 => texel(8),
        23..=47 | 67 | 87..=93 | 116 | 117 | 189 | 191 => texel(4),
        118..=120 => texel(3),
        48..=59 | 68 | 69 | 85 | 86 | 115 => texel(2),
        60..=65 | 190 => texel(1),
        70..=72 | 79..=81 => bc(8),
        73..=78 | 82..=84 | 94..=99 => bc(16),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_formats_map() {
        assert_eq!(map(28, false).unwrap().pixel_format, mtl::RGBA8_UNORM);
        assert_eq!(map(87, false).unwrap().pixel_format, mtl::BGRA8_UNORM);
        assert_eq!(map(98, false).unwrap().block, bc(16));
        assert_eq!(map(71, false).unwrap().block, bc(8));
    }

    #[test]
    fn typeless_depth_families_follow_use() {
        assert_eq!(map(39, true).unwrap().pixel_format, mtl::DEPTH32_FLOAT);
        assert_eq!(map(39, false).unwrap().pixel_format, mtl::R32_FLOAT);
        assert_eq!(map(53, true).unwrap().pixel_format, mtl::DEPTH16_UNORM);
        assert!(map(39, false).unwrap().typeless);
        assert_eq!(map(45, true).unwrap().pixel_format, mtl::DEPTH32_FLOAT_STENCIL8);
    }

    #[test]
    fn unmappable_formats_say_so() {
        assert!(map(6, false).is_none()); // R32G32B32_FLOAT
        assert!(map(66, false).is_none()); // R1_UNORM
        assert!(map(0, false).is_none());
    }

    #[test]
    fn mapped_blocks_match_console_where_bytes_agree() {
        // Every mapped colour format stores texels as the console does;
        // only depth (D24S8, D16S8 -> D32FS8) is allowed to differ.
        for f in 1..=200 {
            let (Some(m), Some(c)) = (map(f, false), console_block(f)) else { continue };
            let depth = m.pixel_format == mtl::DEPTH32_FLOAT_STENCIL8;
            assert!(depth || m.block == c, "format {f}: {:?} vs console {:?}", m.block, c);
        }
    }
}
