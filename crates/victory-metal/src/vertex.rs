//! Vertex input for converted vertex shaders: DXGI formats as
//! `MTLVertexFormat`, D3D12's append-aligned offsets, and which stage-in
//! attribute each input element feeds (Metal Shader Converter names its
//! attributes after the HLSL semantics; reflection gives the pairs).
//!
//! Portable; the `MTLVertexFormat` numbers are checked against
//! objc2-metal's on the Mac (`device::tests::vertex_formats_match_metal`).

use victory_host::{APPEND_ALIGNED, InputElement};

/// `MTLVertexFormat` raw value and bytes per element for a DXGI format.
pub fn format(dxgi: u32) -> Option<(usize, u32)> {
    Some(match dxgi {
        2 => (31, 16), // R32G32B32A32_FLOAT -> Float4
        3 => (39, 16), // R32G32B32A32_UINT -> UInt4
        4 => (35, 16), // R32G32B32A32_SINT -> Int4
        6 => (30, 12), // R32G32B32_FLOAT -> Float3
        7 => (38, 12), // R32G32B32_UINT -> UInt3
        8 => (34, 12), // R32G32B32_SINT -> Int3
        10 => (27, 8), // R16G16B16A16_FLOAT -> Half4
        11 => (21, 8), // R16G16B16A16_UNORM -> UShort4Normalized
        12 => (15, 8), // R16G16B16A16_UINT -> UShort4
        13 => (24, 8), // R16G16B16A16_SNORM -> Short4Normalized
        14 => (18, 8), // R16G16B16A16_SINT -> Short4
        16 => (29, 8), // R32G32_FLOAT -> Float2
        17 => (37, 8), // R32G32_UINT -> UInt2
        18 => (33, 8), // R32G32_SINT -> Int2
        24 => (41, 4), // R10G10B10A2_UNORM -> UInt1010102Normalized
        26 => (54, 4), // R11G11B10_FLOAT -> FloatRG11B10
        28 => (9, 4),  // R8G8B8A8_UNORM -> UChar4Normalized
        30 => (3, 4),  // R8G8B8A8_UINT -> UChar4
        31 => (12, 4), // R8G8B8A8_SNORM -> Char4Normalized
        32 => (6, 4),  // R8G8B8A8_SINT -> Char4
        34 => (25, 4), // R16G16_FLOAT -> Half2
        35 => (19, 4), // R16G16_UNORM -> UShort2Normalized
        36 => (13, 4), // R16G16_UINT -> UShort2
        37 => (22, 4), // R16G16_SNORM -> Short2Normalized
        38 => (16, 4), // R16G16_SINT -> Short2
        41 => (28, 4), // R32_FLOAT -> Float
        42 => (36, 4), // R32_UINT -> UInt
        43 => (32, 4), // R32_SINT -> Int
        49 => (7, 2),  // R8G8_UNORM -> UChar2Normalized
        50 => (1, 2),  // R8G8_UINT -> UChar2
        51 => (10, 2), // R8G8_SNORM -> Char2Normalized
        52 => (4, 2),  // R8G8_SINT -> Char2
        54 => (53, 2), // R16_FLOAT -> Half
        56 => (51, 2), // R16_UNORM -> UShortNormalized
        57 => (49, 2), // R16_UINT -> UShort
        58 => (52, 2), // R16_SNORM -> ShortNormalized
        59 => (50, 2), // R16_SINT -> Short
        61 => (47, 1), // R8_UNORM -> UCharNormalized
        62 => (45, 1), // R8_UINT -> UChar
        63 => (48, 1), // R8_SNORM -> CharNormalized
        64 => (46, 1), // R8_SINT -> Char
        67 => (55, 4), // R9G9B9E5_SHAREDEXP -> FloatRGB9E5
        87 => (42, 4), // B8G8R8A8_UNORM -> UChar4Normalized_BGRA
        _ => return None,
    })
}

/// Each element's byte offset in its slot, with [`APPEND_ALIGNED`]
/// resolved (right after the slot's previous element).
pub fn offsets(inputs: &[InputElement<'_>]) -> Option<Vec<u32>> {
    let mut next = std::collections::HashMap::<u32, u32>::new();
    inputs
        .iter()
        .map(|e| {
            let (_, bytes) = format(e.format)?;
            let end = next.entry(e.slot).or_insert(0);
            let at = if e.offset == APPEND_ALIGNED {
                *end
            } else {
                e.offset
            };
            *end = at + bytes;
            Some(at)
        })
        .collect()
}

/// Splits a reflected attribute name into semantic and index:
/// "TEXCOORD3" -> ("TEXCOORD", 3), "POSITION" -> ("POSITION", 0).
fn semantic(name: &str) -> (String, u32) {
    let digits = name.len() - name.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    let (base, index) = name.split_at(name.len() - digits);
    (base.to_ascii_uppercase(), index.parse().unwrap_or(0))
}

/// The attribute index each element feeds, from reflection's
/// (name, attribute index) pairs; `None` for an element the shader does
/// not read.
pub fn attributes(inputs: &[InputElement<'_>], reflected: &[(String, u32)]) -> Vec<Option<u32>> {
    inputs
        .iter()
        .map(|e| {
            let want = (e.semantic.to_ascii_uppercase(), e.index);
            reflected
                .iter()
                .find(|(name, _)| semantic(name) == want)
                .map(|&(_, at)| at)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn element(
        semantic: &'static str,
        index: u32,
        format: u32,
        slot: u32,
        offset: u32,
    ) -> InputElement<'static> {
        InputElement {
            semantic,
            index,
            format,
            slot,
            offset,
            per_instance: false,
            step_rate: 0,
        }
    }

    #[test]
    fn append_aligned_follows_each_slot() {
        let inputs = [
            element("POSITION", 0, 6, 0, APPEND_ALIGNED), // 12 bytes
            element("NORMAL", 0, 6, 0, APPEND_ALIGNED),   // 12
            element("COLOR", 0, 28, 1, APPEND_ALIGNED),   // slot 1
            element("TEXCOORD", 0, 16, 0, 32),            // explicit
            element("TEXCOORD", 1, 34, 0, APPEND_ALIGNED),
        ];
        assert_eq!(offsets(&inputs).unwrap(), [0, 12, 0, 32, 40]);
        assert_eq!(
            offsets(&[element("X", 0, 999, 0, 0)]),
            None,
            "no such format"
        );
    }

    #[test]
    fn reflected_names_find_their_elements() {
        let inputs = [
            element("POSITION", 0, 16, 0, 0),
            element("texcoord", 2, 16, 0, 8),
            element("COLOR", 0, 28, 0, 16),
        ];
        let reflected = [("TEXCOORD2".to_owned(), 13), ("POSITION".to_owned(), 11)];
        assert_eq!(attributes(&inputs, &reflected), [Some(11), Some(13), None]);
        assert_eq!(semantic("SV_Position0"), ("SV_POSITION".to_owned(), 0));
        assert_eq!(semantic("TEXCOORD12"), ("TEXCOORD".to_owned(), 12));
    }

    #[test]
    fn formats_have_their_sizes() {
        assert_eq!(format(2), Some((31, 16)));
        assert_eq!(format(16), Some((29, 8)));
        assert_eq!(format(87), Some((42, 4)));
        assert_eq!(format(0), None);
    }
}
