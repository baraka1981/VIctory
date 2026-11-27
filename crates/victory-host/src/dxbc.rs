//! The DXBC container checksum (bytes 4..20 of a container): MD5's
//! compression over everything after the checksum, with D3D's own
//! padding (bit count first in the last block, `size * 2 | 1` last;
//! DirectXShaderCompiler's `ComputeHashRetail`). Checked against dxc's
//! containers in the tests.

const S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

/// MD5's sine table: floor(abs(sin(i + 1)) * 2^32).
fn k() -> [u32; 64] {
    let mut k = [0; 64];
    for (i, v) in k.iter_mut().enumerate() {
        *v = ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32;
    }
    k
}

fn compress(state: &mut [u32; 4], block: &[u32; 16], k: &[u32; 64]) {
    let [mut a, mut b, mut c, mut d] = *state;
    for i in 0..64 {
        let (f, g) = match i / 16 {
            0 => ((b & c) | (!b & d), i),
            1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            2 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let f = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(block[g]);
        a = d;
        d = c;
        c = b;
        b = b.wrapping_add(f.rotate_left(S[i]));
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d]) {
        *s = s.wrapping_add(v);
    }
}

fn words(bytes: &[u8; 64]) -> [u32; 16] {
    let mut w = [0; 16];
    for (i, v) in w.iter_mut().enumerate() {
        *v = u32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    w
}

/// The checksum of a whole container (its bytes 4..20 are not read).
pub fn checksum(container: &[u8]) -> [u8; 16] {
    let data = container.get(20..).unwrap_or(&[]);
    let n = data.len();
    let k = k();
    let mut state = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    let full = n / 64 * 64;
    for chunk in data[..full].chunks_exact(64) {
        compress(&mut state, &words(chunk.try_into().expect("64 bytes")), &k);
    }
    let rest = &data[full..];
    let bits = (n as u32).wrapping_mul(8);
    let last = (n as u32).wrapping_mul(2) | 1;
    let mut block = [0u8; 64];
    if rest.len() < 56 {
        // [bit count][rest][0x80][zeros][size * 2 | 1]
        block[..4].copy_from_slice(&bits.to_le_bytes());
        block[4..4 + rest.len()].copy_from_slice(rest);
        block[4 + rest.len()] = 0x80;
    } else {
        // [rest][0x80][zeros], then [bit count][zeros][size * 2 | 1]
        block[..rest.len()].copy_from_slice(rest);
        if rest.len() < 64 {
            block[rest.len()] = 0x80;
        }
        compress(&mut state, &words(&block), &k);
        block = [0; 64];
        block[..4].copy_from_slice(&bits.to_le_bytes());
    }
    block[60..].copy_from_slice(&last.to_le_bytes());
    compress(&mut state, &words(&block), &k);
    let mut out = [0; 16];
    for (i, s) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&s.to_le_bytes());
    }
    out
}

/// Writes `container`'s checksum into its bytes 4..20.
pub fn sign(container: &mut [u8]) {
    if container.len() >= 20 {
        let sum = checksum(container);
        container[4..20].copy_from_slice(&sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::root::samples;

    #[test]
    fn matches_dxc_on_root_signatures() {
        // RS10's tail is under 56 bytes, RS11's over: both paddings.
        assert!((samples::RS10.len() - 20) % 64 < 56);
        assert!((samples::RS11.len() - 20) % 64 >= 56);
        for blob in [&samples::RS10[..], &samples::RS11[..]] {
            assert_eq!(checksum(blob), blob[4..20], "{} bytes", blob.len());
        }
    }

    #[test]
    fn matches_dxc_on_signed_shaders() {
        // victory-metal's fixtures, built and signed by dxc v1.8.2505.
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../victory-metal/tests/fixtures"
        );
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut seen = 0;
        for e in entries.flatten() {
            let path = e.path();
            if path.extension().is_some_and(|x| x == "dxil") {
                let bytes = std::fs::read(&path).unwrap();
                assert_eq!(checksum(&bytes), bytes[4..20], "{}", path.display());
                seen += 1;
            }
        }
        assert!(seen > 0, "no fixtures in {dir}");
    }

    #[test]
    fn md5_core_is_md5() {
        // RFC 1321's empty-message digest, with MD5's own padding, checks
        // the compression and the constants.
        let mut state = [0x6745_2301, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
        let mut block = [0u8; 64];
        block[0] = 0x80;
        compress(&mut state, &words(&block), &k());
        let mut out = Vec::new();
        for s in state {
            out.extend_from_slice(&s.to_le_bytes());
        }
        let hex: String = out.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, "d41d8cd98f00b204e9800998ecf8427e");
    }
}
