//! Serialized D3D12 root signatures (versions 1.0 and 1.1), read without
//! D3D12, so a backend that is not D3D12 can lay out what a root
//! signature describes.
//!
//! The input is what `D3D12SerializeRootSignature` and dxc's `rootsig_1_x`
//! targets produce: a DXBC container with an `RTS0` part (a DXIL shader
//! with an embedded root signature has one too), or the bare part.
//! Numbers stay D3D12's: parameter types, visibilities, range types,
//! flags, sampler enums.

use core::fmt;

/// Root parameter types (`D3D12_ROOT_PARAMETER_TYPE`).
pub mod parameter {
    pub const TABLE: u32 = 0;
    pub const CONSTANTS: u32 = 1;
    pub const CBV: u32 = 2;
    pub const SRV: u32 = 3;
    pub const UAV: u32 = 4;
}

/// Descriptor range types (`D3D12_DESCRIPTOR_RANGE_TYPE`).
pub mod range {
    pub const SRV: u32 = 0;
    pub const UAV: u32 = 1;
    pub const CBV: u32 = 2;
    pub const SAMPLER: u32 = 3;
}

/// `OffsetInDescriptorsFromTableStart`: right after the range before.
pub const APPEND: u32 = u32::MAX;
/// `NumDescriptors` of a range that runs to the end of the heap.
pub const UNBOUNDED: u32 = u32::MAX;

/// Highest `D3D12_SHADER_VISIBILITY` (mesh).
const MAX_VISIBILITY: u32 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    /// `D3D_ROOT_SIGNATURE_VERSION_1_0`: no flags on descriptors or ranges.
    V1_0 = 1,
    /// `D3D_ROOT_SIGNATURE_VERSION_1_1`.
    V1_1 = 2,
}

/// A descriptor range of a table (`D3D12_DESCRIPTOR_RANGE1`; `flags` is 0
/// in version 1.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    /// [`range`] type.
    pub kind: u32,
    /// Descriptors, or [`UNBOUNDED`].
    pub count: u32,
    pub register: u32,
    pub space: u32,
    pub flags: u32,
    /// First descriptor from the table's start, or [`APPEND`].
    pub offset: u32,
}

/// What one root parameter holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// A descriptor table: a GPU address of descriptors, read by ranges.
    Table(Vec<Range>),
    /// 32-bit values inline.
    Constants {
        register: u32,
        space: u32,
        count: u32,
    },
    /// A root CBV, SRV or UAV (`kind` is the [`parameter`] type): one GPU
    /// address. `flags` is 0 in version 1.0.
    Descriptor {
        kind: u32,
        register: u32,
        space: u32,
        flags: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parameter {
    pub kind: Kind,
    /// `D3D12_SHADER_VISIBILITY`.
    pub visibility: u32,
}

/// `D3D12_STATIC_SAMPLER_DESC`, D3D12's numbers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticSampler {
    pub filter: u32,
    pub address: [u32; 3],
    pub mip_lod_bias: f32,
    pub max_anisotropy: u32,
    pub comparison: u32,
    pub border: u32,
    pub min_lod: f32,
    pub max_lod: f32,
    pub register: u32,
    pub space: u32,
    pub visibility: u32,
}

/// A whole root signature.
#[derive(Debug, Clone, PartialEq)]
pub struct RootSignature {
    pub version: Version,
    /// `D3D12_ROOT_SIGNATURE_FLAGS`.
    pub flags: u32,
    pub parameters: Vec<Parameter>,
    pub samplers: Vec<StaticSampler>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootError {
    /// An offset or count runs past the end.
    Truncated,
    /// Neither an `RTS0` part nor a container with one.
    NotRootSignature,
    /// A version other than 1.0 and 1.1.
    Version(u32),
    /// A field with a value D3D12 does not have.
    Value(&'static str, u32),
    /// An [`APPEND`] range after an unbounded one.
    AppendAfterUnbounded,
}

impl fmt::Display for RootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RootError::Truncated => write!(f, "root signature runs past its end"),
            RootError::NotRootSignature => write!(f, "no root signature (RTS0) in the blob"),
            RootError::Version(v) => write!(f, "root signature version {v} not supported"),
            RootError::Value(what, v) => write!(f, "root signature {what} {v} not valid"),
            RootError::AppendAfterUnbounded => {
                write!(f, "descriptor range appended after an unbounded one")
            }
        }
    }
}

impl std::error::Error for RootError {}

const DXBC: &[u8; 4] = b"DXBC";
const RTS0: &[u8; 4] = b"RTS0";

/// The bytes of the part `fourcc` of a DXBC container (DXIL or root
/// signature), if it has one.
pub fn part<'a>(container: &'a [u8], fourcc: &[u8; 4]) -> Option<&'a [u8]> {
    let word = |at: usize| -> Option<u32> {
        let b = container.get(at..at.checked_add(4)?)?;
        Some(u32::from_le_bytes(b.try_into().ok()?))
    };
    if container.get(..4)? != DXBC {
        return None;
    }
    let count = word(28)? as usize;
    (0..count).find_map(|i| {
        let at = word(32 + i * 4)? as usize;
        let size = word(at + 4)? as usize;
        if container.get(at..at + 4)? != fourcc {
            return None;
        }
        container.get(at + 8..(at + 8).checked_add(size)?)
    })
}

/// The root signature embedded in a DXIL shader, as a bare `RTS0` part.
pub fn embedded(dxil: &[u8]) -> Option<&[u8]> {
    part(dxil, RTS0)
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn u32(&self, at: u32) -> Result<u32, RootError> {
        let at = at as usize;
        let b = self
            .0
            .get(at..at.checked_add(4).ok_or(RootError::Truncated)?)
            .ok_or(RootError::Truncated)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn words<const N: usize>(&self, at: u32) -> Result<[u32; N], RootError> {
        let mut out = [0; N];
        for (i, w) in out.iter_mut().enumerate() {
            *w = self.u32(at.checked_add(i as u32 * 4).ok_or(RootError::Truncated)?)?;
        }
        Ok(out)
    }

    /// `count` records of `size` bytes from `at` fit in the part.
    fn fits(&self, at: u32, count: u32, size: u32) -> Result<(), RootError> {
        let end = u64::from(at) + u64::from(count) * u64::from(size);
        if end > self.0.len() as u64 {
            return Err(RootError::Truncated);
        }
        Ok(())
    }
}

fn check(what: &'static str, value: u32, max: u32) -> Result<u32, RootError> {
    if value > max {
        return Err(RootError::Value(what, value));
    }
    Ok(value)
}

/// Parses a serialized root signature (a container or a bare `RTS0`).
pub fn parse(blob: &[u8]) -> Result<RootSignature, RootError> {
    let rts0 = if blob.starts_with(DXBC) {
        part(blob, RTS0).ok_or(RootError::NotRootSignature)?
    } else {
        blob
    };
    let r = Reader(rts0);
    let [version, count, at, sampler_count, samplers_at, flags] = r.words::<6>(0)?;
    let version = match version {
        1 => Version::V1_0,
        2 => Version::V1_1,
        v => return Err(RootError::Version(v)),
    };
    let v11 = version == Version::V1_1;
    r.fits(at, count, 12)?;
    let mut parameters = Vec::with_capacity(count as usize);
    for i in 0..count {
        let [kind, visibility, payload] = r.words::<3>(at + i * 12)?;
        let visibility = check("shader visibility", visibility, MAX_VISIBILITY)?;
        let kind = match kind {
            parameter::TABLE => {
                let [n, ranges_at] = r.words::<2>(payload)?;
                let size = if v11 { 24 } else { 20 };
                r.fits(ranges_at, n, size)?;
                let ranges = (0..n)
                    .map(|j| {
                        let at = ranges_at + j * size;
                        let range = if v11 {
                            let [kind, count, register, space, flags, offset] = r.words::<6>(at)?;
                            Range {
                                kind,
                                count,
                                register,
                                space,
                                flags,
                                offset,
                            }
                        } else {
                            let [kind, count, register, space, offset] = r.words::<5>(at)?;
                            Range {
                                kind,
                                count,
                                register,
                                space,
                                flags: 0,
                                offset,
                            }
                        };
                        check("descriptor range type", range.kind, range::SAMPLER)?;
                        Ok(range)
                    })
                    .collect::<Result<Vec<_>, RootError>>()?;
                Kind::Table(ranges)
            }
            parameter::CONSTANTS => {
                let [register, space, count] = r.words::<3>(payload)?;
                Kind::Constants {
                    register,
                    space,
                    count,
                }
            }
            parameter::CBV | parameter::SRV | parameter::UAV => {
                let [register, space] = r.words::<2>(payload)?;
                let flags = if v11 { r.u32(payload + 8)? } else { 0 };
                Kind::Descriptor {
                    kind,
                    register,
                    space,
                    flags,
                }
            }
            other => return Err(RootError::Value("root parameter type", other)),
        };
        parameters.push(Parameter { kind, visibility });
    }
    r.fits(samplers_at, sampler_count, 52)?;
    let samplers = (0..sampler_count)
        .map(|i| {
            let w = r.words::<13>(samplers_at + i * 52)?;
            Ok(StaticSampler {
                filter: w[0],
                address: [w[1], w[2], w[3]],
                mip_lod_bias: f32::from_bits(w[4]),
                max_anisotropy: w[5],
                comparison: w[6],
                border: w[7],
                min_lod: f32::from_bits(w[8]),
                max_lod: f32::from_bits(w[9]),
                register: w[10],
                space: w[11],
                visibility: check("shader visibility", w[12], MAX_VISIBILITY)?,
            })
        })
        .collect::<Result<Vec<_>, RootError>>()?;
    Ok(RootSignature {
        version,
        flags,
        parameters,
        samplers,
    })
}

/// Where each range of a table starts, in descriptors from the table's
/// start, with [`APPEND`] resolved.
pub fn table_starts(ranges: &[Range]) -> Result<Vec<u32>, RootError> {
    let mut next = 0u32;
    ranges
        .iter()
        .map(|r| {
            let start = if r.offset == APPEND {
                if next == UNBOUNDED {
                    return Err(RootError::AppendAfterUnbounded);
                }
                next
            } else {
                r.offset
            };
            next = if r.count == UNBOUNDED {
                UNBOUNDED
            } else {
                start.saturating_add(r.count)
            };
            Ok(start)
        })
        .collect()
}

impl RootSignature {
    /// As version 1.1, the way D3D12 reads a 1.0 root signature: root
    /// descriptors `DATA_VOLATILE`, ranges `DESCRIPTORS_VOLATILE |
    /// DATA_VOLATILE` (sampler ranges only `DESCRIPTORS_VOLATILE`). A 1.1
    /// one comes back as it is.
    pub fn to_v1_1(&self) -> RootSignature {
        const DESCRIPTORS_VOLATILE: u32 = 1;
        const DATA_VOLATILE: u32 = 2;
        let mut out = self.clone();
        if self.version == Version::V1_1 {
            return out;
        }
        out.version = Version::V1_1;
        for p in &mut out.parameters {
            match &mut p.kind {
                Kind::Descriptor { flags, .. } => *flags = DATA_VOLATILE,
                Kind::Table(ranges) => {
                    for r in ranges {
                        r.flags = if r.kind == range::SAMPLER {
                            DESCRIPTORS_VOLATILE
                        } else {
                            DESCRIPTORS_VOLATILE | DATA_VOLATILE
                        };
                    }
                }
                Kind::Constants { .. } => {}
            }
        }
        out
    }

    /// Serialized as D3D12 does (dxc's byte order), in a DXBC container.
    /// The container hash is left zero: this is for tests and for the
    /// backends that read the blob themselves (Metal), not for D3D12.
    pub fn serialize(&self) -> Vec<u8> {
        let v11 = self.version == Version::V1_1;
        let mut part: Vec<u32> = vec![
            self.version as u32,
            self.parameters.len() as u32,
            24,
            self.samplers.len() as u32,
            0, // static samplers' offset, set below
            self.flags,
        ];
        // Parameter headers, payloads after them (each table's ranges
        // right after its own header), static samplers last.
        let headers = part.len();
        part.resize(headers + self.parameters.len() * 3, 0);
        for (i, p) in self.parameters.iter().enumerate() {
            let payload = part.len() as u32 * 4;
            let kind = match &p.kind {
                Kind::Table(ranges) => {
                    part.extend([ranges.len() as u32, payload + 8]);
                    for r in ranges {
                        if v11 {
                            part.extend([r.kind, r.count, r.register, r.space, r.flags, r.offset]);
                        } else {
                            part.extend([r.kind, r.count, r.register, r.space, r.offset]);
                        }
                    }
                    parameter::TABLE
                }
                &Kind::Constants {
                    register,
                    space,
                    count,
                } => {
                    part.extend([register, space, count]);
                    parameter::CONSTANTS
                }
                &Kind::Descriptor {
                    kind,
                    register,
                    space,
                    flags,
                } => {
                    part.extend([register, space]);
                    if v11 {
                        part.push(flags);
                    }
                    kind
                }
            };
            part[headers + i * 3..headers + i * 3 + 3].copy_from_slice(&[
                kind,
                p.visibility,
                payload,
            ]);
        }
        part[4] = part.len() as u32 * 4;
        for s in &self.samplers {
            part.extend([
                s.filter,
                s.address[0],
                s.address[1],
                s.address[2],
                s.mip_lod_bias.to_bits(),
                s.max_anisotropy,
                s.comparison,
                s.border,
                s.min_lod.to_bits(),
                s.max_lod.to_bits(),
                s.register,
                s.space,
                s.visibility,
            ]);
        }
        let part_bytes = part.len() as u32 * 4;
        let total = 32 + 4 + 8 + part_bytes;
        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(DXBC);
        out.extend_from_slice(&[0; 16]);
        // Container version 1.0, size, one part, at 36.
        for w in [1u32, total, 1, 36] {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(RTS0);
        out.extend_from_slice(&part_bytes.to_le_bytes());
        for w in part {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }
}

/// Root signatures serialized by Microsoft's dxc (v1.8.2505) from
/// `tests/fixtures/root_signatures.hlsl`, for tests in this and other
/// crates.
#[doc(hidden)]
pub mod samples {
    /// Version 1.0: flags, constants (vertex), CBV/SRV/UAV, a table of
    /// SRV + CBV + unbounded UAV, a sampler table (pixel), a static sampler.
    pub const RS10: [u8; 324] = [
        0x44, 0x58, 0x42, 0x43, 0x46, 0x4a, 0xf7, 0xc4, 0xa4, 0x02, 0x4d, 0x92, 0x61, 0xa3, 0x97,
        0x58, 0x2c, 0x16, 0xd3, 0x2a, 0x01, 0x00, 0x00, 0x00, 0x44, 0x01, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x52, 0x54, 0x53, 0x30, 0x18, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0xe4, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x00, 0x60, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6c, 0x00,
        0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x74, 0x00, 0x00, 0x00, 0x04,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00,
        0x00, 0xc8, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x8c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xd0, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff,
        0xff, 0xff, 0x15, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x41, 0x04, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    /// Version 1.1: constants, flagged SRV and UAV, a table of SRV + UAV.
    pub const RS11: [u8; 208] = [
        0x44, 0x58, 0x42, 0x43, 0x4a, 0x2b, 0x46, 0xe1, 0x5a, 0x4b, 0x67, 0x62, 0xaf, 0xf0, 0x8d,
        0x75, 0xa4, 0x01, 0xe2, 0xa8, 0x01, 0x00, 0x00, 0x00, 0xd0, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x52, 0x54, 0x53, 0x30, 0xa4, 0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xa4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x48, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x54, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
        0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0xff, 0xff, 0xff, 0xff, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_version_1_0_root_signature() {
        let rs = parse(&samples::RS10).unwrap();
        assert_eq!(rs.version, Version::V1_0);
        assert_eq!(rs.flags, 1, "ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT");
        let kinds: Vec<_> = rs.parameters.iter().map(|p| &p.kind).collect();
        assert_eq!(
            kinds[0],
            &Kind::Constants {
                register: 1,
                space: 2,
                count: 3
            }
        );
        assert_eq!(rs.parameters[0].visibility, 1, "vertex");
        assert_eq!(
            kinds[1..4],
            [
                &Kind::Descriptor {
                    kind: parameter::CBV,
                    register: 0,
                    space: 0,
                    flags: 0
                },
                &Kind::Descriptor {
                    kind: parameter::SRV,
                    register: 3,
                    space: 1,
                    flags: 0
                },
                &Kind::Descriptor {
                    kind: parameter::UAV,
                    register: 0,
                    space: 0,
                    flags: 0
                },
            ]
        );
        let Kind::Table(ranges) = kinds[4] else {
            panic!("a table")
        };
        assert_eq!(
            ranges,
            &[
                Range {
                    kind: range::SRV,
                    count: 4,
                    register: 0,
                    space: 0,
                    flags: 0,
                    offset: APPEND
                },
                Range {
                    kind: range::CBV,
                    count: 1,
                    register: 2,
                    space: 0,
                    flags: 0,
                    offset: 4
                },
                Range {
                    kind: range::UAV,
                    count: UNBOUNDED,
                    register: 1,
                    space: 0,
                    flags: 0,
                    offset: 8
                },
            ]
        );
        assert_eq!(table_starts(ranges).unwrap(), [0, 4, 8]);
        let Kind::Table(samplers) = kinds[5] else {
            panic!("a table")
        };
        assert_eq!((samplers[0].kind, samplers[0].count), (range::SAMPLER, 2));
        assert_eq!(rs.parameters[5].visibility, 5, "pixel");
        assert_eq!(
            rs.samplers,
            [StaticSampler {
                filter: 0x15,
                address: [3, 1, 1],
                mip_lod_bias: 0.5,
                max_anisotropy: 4,
                comparison: 2,
                border: 2,
                min_lod: 0.0,
                max_lod: 8.0,
                register: 4,
                space: 0,
                visibility: 0,
            }]
        );
    }

    #[test]
    fn reads_a_version_1_1_root_signature() {
        let rs = parse(&samples::RS11).unwrap();
        assert_eq!((rs.version, rs.flags), (Version::V1_1, 0));
        assert!(rs.samplers.is_empty());
        assert_eq!(
            rs.parameters[1].kind,
            Kind::Descriptor {
                kind: parameter::SRV,
                register: 0,
                space: 0,
                flags: 8
            },
            "DATA_STATIC"
        );
        assert_eq!(
            rs.parameters[2].kind,
            Kind::Descriptor {
                kind: parameter::UAV,
                register: 0,
                space: 0,
                flags: 2
            },
            "DATA_VOLATILE"
        );
        let Kind::Table(ranges) = &rs.parameters[3].kind else {
            panic!("a table")
        };
        assert_eq!((ranges[0].count, ranges[0].flags), (2, 1));
        assert_eq!(table_starts(ranges).unwrap(), [0, 2]);
    }

    #[test]
    fn version_1_0_reads_as_volatile_1_1() {
        let rs = parse(&samples::RS10).unwrap().to_v1_1();
        assert_eq!(rs.version, Version::V1_1);
        let Kind::Descriptor { flags, .. } = rs.parameters[1].kind else {
            panic!("a root CBV")
        };
        assert_eq!(flags, 2, "DATA_VOLATILE");
        let Kind::Table(ranges) = &rs.parameters[4].kind else {
            panic!("a table")
        };
        assert!(ranges.iter().all(|r| r.flags == 3));
        let Kind::Table(samplers) = &rs.parameters[5].kind else {
            panic!("a table")
        };
        assert_eq!(samplers[0].flags, 1, "samplers: DESCRIPTORS_VOLATILE only");
        assert_eq!(rs.samplers, parse(&samples::RS10).unwrap().samplers);
        // Offsets, registers and the rest are untouched; 1.1 stays as is.
        assert_eq!(ranges[2].offset, 8);
        let v11 = parse(&samples::RS11).unwrap();
        assert_eq!(v11.to_v1_1(), v11);
    }

    #[test]
    fn serializes_as_dxc_does() {
        // Everything but the container hash, which is left zero.
        for blob in [&samples::RS10[..], &samples::RS11[..]] {
            let made = parse(blob).unwrap().serialize();
            assert_eq!(made.len(), blob.len());
            assert_eq!(made[..4], blob[..4]);
            assert_eq!(made[20..], blob[20..]);
            assert_eq!(parse(&made).unwrap(), parse(blob).unwrap());
        }
    }

    #[test]
    fn reads_the_bare_part_and_finds_it_in_containers() {
        let bare = part(&samples::RS11, b"RTS0").unwrap();
        assert_eq!(bare.len(), 164);
        assert_eq!(parse(bare).unwrap(), parse(&samples::RS11).unwrap());
        assert_eq!(embedded(&samples::RS11), Some(bare));
        assert_eq!(part(&samples::RS11, b"DXIL"), None);
    }

    #[test]
    fn bad_blobs_are_errors() {
        assert_eq!(parse(&[]), Err(RootError::Truncated));
        assert_eq!(
            parse(&samples::RS10[..100]),
            Err(RootError::NotRootSignature)
        );
        let bare = part(&samples::RS10, b"RTS0").unwrap();
        assert_eq!(parse(&bare[..100]), Err(RootError::Truncated));
        let mut v3 = bare.to_vec();
        v3[0] = 3;
        assert_eq!(parse(&v3), Err(RootError::Version(3)));
        let mut bad_type = bare.to_vec();
        bad_type[24] = 9; // first parameter's type
        assert_eq!(
            parse(&bad_type),
            Err(RootError::Value("root parameter type", 9))
        );
        let mut huge = bare.to_vec();
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(parse(&huge), Err(RootError::Truncated));
    }

    #[test]
    fn appending_after_unbounded_is_an_error() {
        let r = |count, offset| Range {
            kind: range::SRV,
            count,
            register: 0,
            space: 0,
            flags: 0,
            offset,
        };
        assert_eq!(
            table_starts(&[r(UNBOUNDED, 0), r(1, APPEND)]),
            Err(RootError::AppendAfterUnbounded)
        );
        assert_eq!(
            table_starts(&[r(2, 3), r(1, APPEND), r(1, 0)]).unwrap(),
            [3, 5, 0]
        );
    }
}
