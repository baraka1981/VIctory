//! The top-level argument buffer Metal Shader Converter expects for a
//! root signature: one entry per root parameter, in order, each at its
//! natural alignment. Root constants are their 32-bit values inline (4-byte
//! aligned); a root CBV / SRV / UAV is one 64-bit GPU address, and a
//! descriptor table the 64-bit GPU address of its first entry (8-byte
//! aligned). Static samplers take no room.
//!
//! Portable (unit-tested anywhere); `irconverter_layout_matches` on the
//! Mac checks it against what MSC reports for the same root signature.

use victory_host::root::{Kind, RootSignature};

/// What one root parameter puts in the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    /// `count` 32-bit constants.
    Constants(u32),
    /// A root descriptor's GPU address (`victory_host::root::parameter`
    /// type: CBV, SRV or UAV).
    Descriptor(u32),
    /// A descriptor table's GPU address.
    Table,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub slot: Slot,
    /// Bytes from the buffer's start.
    pub offset: u32,
}

/// The layout for one root signature.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TopLevel {
    /// One per root parameter, in order.
    pub entries: Vec<Entry>,
    pub bytes: u32,
}

impl TopLevel {
    pub fn new(rs: &RootSignature) -> TopLevel {
        let mut at = 0u32;
        let entries = rs
            .parameters
            .iter()
            .map(|p| {
                let (slot, size, align) = match p.kind {
                    Kind::Constants { count, .. } => (Slot::Constants(count), 4 * count, 4),
                    Kind::Descriptor { kind, .. } => (Slot::Descriptor(kind), 8, 8),
                    Kind::Table(_) => (Slot::Table, 8, 8),
                };
                let offset = at.next_multiple_of(align);
                at = offset + size;
                Entry { slot, offset }
            })
            .collect();
        TopLevel {
            entries,
            bytes: at.next_multiple_of(8),
        }
    }

    /// The buffer for one dispatch: constants fill the constants entries
    /// in order, `addresses` the root descriptors, `tables` the tables
    /// (missing ones read as zero).
    pub fn fill(&self, constants: &[u32], addresses: &[u64], tables: &[u64]) -> Vec<u8> {
        let mut out = vec![0u8; self.bytes.max(8) as usize];
        let (mut c, mut a, mut t) = (constants.iter(), addresses.iter(), tables.iter());
        for e in &self.entries {
            let at = e.offset as usize;
            match e.slot {
                Slot::Constants(n) => {
                    for (i, v) in c.by_ref().take(n as usize).enumerate() {
                        out[at + i * 4..at + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
                Slot::Descriptor(_) => {
                    let v = a.next().copied().unwrap_or(0);
                    out[at..at + 8].copy_from_slice(&v.to_le_bytes());
                }
                Slot::Table => {
                    let v = t.next().copied().unwrap_or(0);
                    out[at..at + 8].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    /// Root descriptors, in order.
    pub fn descriptors(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e.slot, Slot::Descriptor(_)))
            .count()
    }
}

/// `tests/fixtures/swap.hlsl`'s root signature, made here rather than
/// embedded (the fixture is compiled with it stripped):
/// `UAV(u0), RootConstants(num32BitConstants=2, b0), SRV(t0, flags=DATA_STATIC)`,
/// version 1.1.
#[cfg(test)]
pub(crate) fn swap_root_signature() -> RootSignature {
    use victory_host::root::{Parameter, Version, parameter};
    let p = |kind| Parameter {
        kind,
        visibility: 0,
    };
    RootSignature {
        version: Version::V1_1,
        flags: 0,
        parameters: vec![
            p(Kind::Descriptor {
                kind: parameter::UAV,
                register: 0,
                space: 0,
                flags: 0,
            }),
            p(Kind::Constants {
                register: 0,
                space: 0,
                count: 2,
            }),
            p(Kind::Descriptor {
                kind: parameter::SRV,
                register: 0,
                space: 0,
                flags: 8,
            }),
        ],
        samplers: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_host::root::{self, parameter, samples};

    use super::swap_root_signature;

    #[test]
    fn constants_then_aligned_addresses() {
        // add.hlsl's: RootConstants(1, b0), SRV(t0), UAV(u0); the layout
        // confirmed on the GPU in slice 3b's first step.
        let rs = RootSignature {
            version: root::Version::V1_1,
            flags: 0,
            parameters: vec![
                root::Parameter {
                    kind: Kind::Constants {
                        register: 0,
                        space: 0,
                        count: 1,
                    },
                    visibility: 0,
                },
                root::Parameter {
                    kind: Kind::Descriptor {
                        kind: parameter::SRV,
                        register: 0,
                        space: 0,
                        flags: 0,
                    },
                    visibility: 0,
                },
                root::Parameter {
                    kind: Kind::Descriptor {
                        kind: parameter::UAV,
                        register: 0,
                        space: 0,
                        flags: 0,
                    },
                    visibility: 0,
                },
            ],
            samplers: vec![],
        };
        let top = TopLevel::new(&rs);
        let offsets: Vec<u32> = top.entries.iter().map(|e| e.offset).collect();
        assert_eq!(offsets, [0, 8, 16]);
        assert_eq!(top.bytes, 24);
        let ab = top.fill(&[7], &[0x1000, 0x2000], &[]);
        assert_eq!(ab.len(), 24);
        assert_eq!(&ab[0..4], &7u32.to_le_bytes());
        assert_eq!(&ab[8..16], &0x1000u64.to_le_bytes());
        assert_eq!(&ab[16..24], &0x2000u64.to_le_bytes());
    }

    #[test]
    fn every_parameter_kind_in_order() {
        // RS10: constants(3), CBV, SRV, UAV, table, sampler table.
        let top = TopLevel::new(&root::parse(&samples::RS10).unwrap());
        assert_eq!(
            top.entries,
            [
                Entry {
                    slot: Slot::Constants(3),
                    offset: 0
                },
                Entry {
                    slot: Slot::Descriptor(parameter::CBV),
                    offset: 16
                },
                Entry {
                    slot: Slot::Descriptor(parameter::SRV),
                    offset: 24
                },
                Entry {
                    slot: Slot::Descriptor(parameter::UAV),
                    offset: 32
                },
                Entry {
                    slot: Slot::Table,
                    offset: 40
                },
                Entry {
                    slot: Slot::Table,
                    offset: 48
                },
            ]
        );
        assert_eq!((top.bytes, top.descriptors()), (56, 3));
        let ab = top.fill(&[1, 2, 3], &[0xa, 0xb], &[0x7000]);
        assert_eq!(&ab[0..12], &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
        assert_eq!(&ab[16..24], &0xau64.to_le_bytes());
        assert_eq!(&ab[24..32], &0xbu64.to_le_bytes());
        assert_eq!(&ab[32..40], &[0; 8], "a missing binding reads as null");
        assert_eq!(&ab[40..48], &0x7000u64.to_le_bytes());
    }

    #[test]
    fn addresses_before_constants() {
        // UAV, constants(2), SRV: the order the swap.hlsl test uses.
        let rs = root::parse(&swap_root_signature().serialize()).unwrap();
        let top = TopLevel::new(&rs);
        let offsets: Vec<u32> = top.entries.iter().map(|e| e.offset).collect();
        assert_eq!(offsets, [0, 8, 16]);
        let ab = top.fill(&[3, 1000], &[0x1000, 0x2000], &[]);
        assert_eq!(&ab[0..8], &0x1000u64.to_le_bytes());
        assert_eq!(&ab[8..16], &[3, 0, 0, 0, 0xe8, 0x03, 0, 0]);
        assert_eq!(&ab[16..24], &0x2000u64.to_le_bytes());
    }
}
