//! Copy layouts by D3D12's rules, which the console side sizes its
//! staging buffers with: rows padded to 256 bytes, subresources placed at
//! 512-byte offsets. Metal's own rules (row pitch a multiple of the
//! texel/block size) are looser, so a D3D12 layout is always a valid
//! Metal one.

use victory_host::{Footprint as Placed, ResourceDesc, dimension};

use crate::format::console_block;

pub const ROW_PITCH_ALIGNMENT: u64 = 256;
pub const PLACEMENT_ALIGNMENT: u64 = 512;

/// One subresource laid out for a copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Footprint {
    pub offset: u64,
    pub row_pitch: u64,
    /// Rows of blocks (texel rows / block height).
    pub rows: u32,
    /// Bytes of one row without padding.
    pub row_bytes: u64,
    pub depth: u32,
}

pub(crate) fn mips(desc: &ResourceDesc) -> u32 {
    match desc.mip_levels {
        0 => {
            let largest = desc
                .width
                .max(u64::from(desc.height))
                .max(match desc.dimension {
                    dimension::TEXTURE_3D => u64::from(desc.depth_or_array_size),
                    _ => 1,
                });
            64 - largest.max(1).leading_zeros()
        }
        n => u32::from(n),
    }
}

/// Planes of a format: depth-stencil formats have a stencil plane.
fn planes(format: u32) -> u32 {
    match format {
        19..=22 | 44..=47 | 118..=120 => 2,
        _ => 1,
    }
}

/// The footprint of subresource `index` of `desc`, placed at `offset`
/// (aligned up to [`PLACEMENT_ALIGNMENT`]). `None` for formats without a
/// byte layout.
pub fn footprint(desc: &ResourceDesc, index: u32, offset: u64) -> Option<Footprint> {
    let offset = offset.next_multiple_of(PLACEMENT_ALIGNMENT);
    if desc.dimension == dimension::BUFFER {
        return Some(Footprint {
            offset,
            row_pitch: desc.width,
            rows: 1,
            row_bytes: desc.width,
            depth: 1,
        });
    }
    let mips = mips(desc);
    let slices = match desc.dimension {
        dimension::TEXTURE_3D => 1,
        _ => u32::from(desc.depth_or_array_size).max(1),
    };
    let mip = index % mips;
    let plane = index / (mips * slices);
    let mut block = console_block(desc.format)?;
    if planes(desc.format) == 2 {
        // D3D12's planes: depth as 32 bits (16 for the 16-bit console
        // formats), stencil as 8.
        block.bytes = match (plane, desc.format) {
            (0, 118..=120) => 2,
            (0, _) => 4,
            _ => 1,
        };
    }
    let width = (desc.width >> mip).max(1);
    let height = (desc.height >> mip).max(1);
    let depth = match desc.dimension {
        dimension::TEXTURE_3D => (u32::from(desc.depth_or_array_size) >> mip).max(1),
        _ => 1,
    };
    let row_bytes = width.div_ceil(u64::from(block.width)) * u64::from(block.bytes);
    let rows = height.div_ceil(block.height);
    Some(Footprint {
        offset,
        row_pitch: row_bytes.next_multiple_of(ROW_PITCH_ALIGNMENT),
        rows,
        row_bytes,
        depth,
    })
}

/// Bytes a buffer needs for subresources `first..first + count` laid out
/// one after another (D3D12's `GetCopyableFootprints` total: the last
/// row of the last subresource is not padded).
pub fn copyable_bytes(desc: &ResourceDesc, first: u32, count: u32) -> Option<u64> {
    let mut next = 0;
    let mut total = 0;
    for index in first..first + count {
        let f = footprint(desc, index, next)?;
        let rows = u64::from(f.rows) * u64::from(f.depth);
        total = f.offset + f.row_pitch * (rows - 1) + f.row_bytes;
        next = f.offset + f.row_pitch * rows;
    }
    Some(total)
}

/// The seam's footprints of subresources `first..first + out.len()`,
/// placed from offset 0, and the total bytes; `None` for formats without a
/// byte layout. Widths and heights are in texels, as D3D12 reports them.
pub fn layout(desc: &ResourceDesc, first: u32, out: &mut [Placed]) -> Option<u64> {
    let mut next = 0;
    let mut total = 0;
    for (i, slot) in out.iter_mut().enumerate() {
        let index = first + i as u32;
        let f = footprint(desc, index, next)?;
        let mip = index % mips(desc);
        let block = console_block(desc.format).unwrap_or(crate::format::Block {
            bytes: 1,
            width: 1,
            height: 1,
        });
        *slot = Placed {
            offset: f.offset,
            format: desc.format,
            width: if desc.dimension == dimension::BUFFER {
                desc.width as u32
            } else {
                ((desc.width >> mip).max(1) as u32).next_multiple_of(block.width)
            },
            height: if desc.dimension == dimension::BUFFER {
                1
            } else {
                f.rows * block.height
            },
            depth: f.depth,
            row_pitch: f.row_pitch as u32,
        };
        let rows = u64::from(f.rows) * u64::from(f.depth);
        total = f.offset + f.row_pitch * (rows - 1) + f.row_bytes;
        next = f.offset + f.row_pitch * rows;
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(width: u64, height: u32, format: u32, mips: u16) -> ResourceDesc {
        ResourceDesc {
            dimension: dimension::TEXTURE_2D,
            alignment: 0,
            width,
            height,
            depth_or_array_size: 1,
            mip_levels: mips,
            format,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: 0,
        }
    }

    #[test]
    fn rgba8_4x4_is_three_padded_rows_and_one_tight() {
        // 16 bytes a row, padded to 256: 3 * 256 + 16.
        assert_eq!(copyable_bytes(&texture(4, 4, 28, 1), 0, 1), Some(784));
    }

    #[test]
    fn mips_are_placed_at_512() {
        // 256x256 RGBA8: mip 0 is 256 rows of 1024 bytes.
        let d = texture(256, 256, 28, 2);
        let m1 = footprint(&d, 1, 256 * 1024).unwrap();
        assert_eq!((m1.offset, m1.row_bytes, m1.rows), (262_144, 512, 128));
        assert_eq!(copyable_bytes(&d, 0, 2), Some(262_144 + 512 * 127 + 512));
    }

    #[test]
    fn block_compressed_rows_are_block_rows() {
        // BC1 8x8: 2 block rows of 2 blocks * 8 bytes.
        let f = footprint(&texture(8, 8, 71, 1), 0, 0).unwrap();
        assert_eq!((f.rows, f.row_bytes, f.row_pitch), (2, 16, 256));
        // Smaller than a block still takes one.
        let f = footprint(&texture(2, 2, 98, 1), 0, 0).unwrap();
        assert_eq!((f.rows, f.row_bytes), (1, 16));
    }

    #[test]
    fn full_chain_when_mips_is_zero() {
        assert_eq!(mips(&texture(256, 64, 28, 0)), 9);
        assert_eq!(mips(&texture(1, 1, 28, 0)), 1);
    }

    #[test]
    fn stencil_plane_follows_depth() {
        // D32S8 (20) 4x4, one mip: plane 1 is 1 byte a texel.
        let d = texture(4, 4, 20, 1);
        assert_eq!(footprint(&d, 0, 0).unwrap().row_bytes, 16);
        assert_eq!(footprint(&d, 1, 0).unwrap().row_bytes, 4);
    }

    #[test]
    fn layout_places_each_mip() {
        let d = texture(256, 256, 28, 2);
        let mut out = [Placed::default(); 2];
        assert_eq!(layout(&d, 0, &mut out), copyable_bytes(&d, 0, 2));
        assert_eq!(
            (out[0].offset, out[0].width, out[0].height, out[0].row_pitch),
            (0, 256, 256, 1024)
        );
        assert_eq!(
            (out[1].offset, out[1].width, out[1].height, out[1].row_pitch),
            (262_144, 128, 128, 512)
        );
        // BC: widths and heights round up to whole blocks.
        let mut bc = [Placed::default(); 1];
        layout(&texture(6, 6, 71, 1), 0, &mut bc);
        assert_eq!((bc[0].width, bc[0].height), (8, 8));
    }

    #[test]
    fn buffers_are_one_row() {
        let b = ResourceDesc::buffer(1000, 0);
        assert_eq!(copyable_bytes(&b, 0, 1), Some(1000));
    }
}
