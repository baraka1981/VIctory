//! The top-level argument buffer Metal Shader Converter expects for a
//! root signature: one entry per root parameter, in order, each at its
//! natural alignment. Root constants are their 32-bit values inline (4-byte
//! aligned); a root CBV / SRV / UAV is one 64-bit GPU address, and a
//! descriptor table the 64-bit GPU address of its first entry (8-byte
//! aligned). Static samplers take no room.
//!
//! A table's entry points at descriptor `first` of a heap; its ranges say
//! which descriptors from there the shader may read ([`TopLevel::reached`]),
//! which is what has to be made resident.
//!
//! Portable (unit-tested anywhere); `irconverter_layout_matches` on the
//! Mac checks it against what MSC reports for the same root signature.

use std::ops::Range;

use victory_host::root::{self, Kind, RootError, RootSignature};

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

/// Descriptors one range of a table covers, from the table's start
/// (`count` [`root::UNBOUNDED`]: to the heap's end).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: u32,
    pub count: u32,
}

/// The layout for one root signature.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TopLevel {
    /// One per root parameter, in order.
    pub entries: Vec<Entry>,
    /// Each table's ranges, in table order.
    pub tables: Vec<Vec<Span>>,
    pub bytes: u32,
}

impl TopLevel {
    pub fn new(rs: &RootSignature) -> Result<TopLevel, RootError> {
        let mut at = 0u32;
        let mut tables = Vec::new();
        let mut entries = Vec::with_capacity(rs.parameters.len());
        for p in &rs.parameters {
            let (slot, size, align) = match &p.kind {
                &Kind::Constants { count, .. } => (Slot::Constants(count), 4 * count, 4),
                &Kind::Descriptor { kind, .. } => (Slot::Descriptor(kind), 8, 8),
                Kind::Table(ranges) => {
                    let starts = root::table_starts(ranges)?;
                    tables.push(
                        ranges
                            .iter()
                            .zip(starts)
                            .map(|(r, start)| Span {
                                start,
                                count: r.count,
                            })
                            .collect(),
                    );
                    (Slot::Table, 8, 8)
                }
            };
            let offset = at.next_multiple_of(align);
            at = offset + size;
            entries.push(Entry { slot, offset });
        }
        Ok(TopLevel {
            entries,
            tables,
            bytes: at.next_multiple_of(8),
        })
    }

    /// Descriptor indices of a heap of `heap_count` that table `table`,
    /// pointed at index `first`, can reach: its ranges, merged, sorted,
    /// clipped to the heap.
    pub fn reached(&self, table: usize, first: u32, heap_count: u32) -> Vec<Range<u32>> {
        let Some(spans) = self.tables.get(table) else {
            return Vec::new();
        };
        let mut out: Vec<Range<u32>> = spans
            .iter()
            .map(|s| {
                let start = first.saturating_add(s.start).min(heap_count);
                let end = if s.count == root::UNBOUNDED {
                    heap_count
                } else {
                    start.saturating_add(s.count).min(heap_count)
                };
                start..end
            })
            .filter(|r| !r.is_empty())
            .collect();
        out.sort_by_key(|r| r.start);
        let mut merged: Vec<Range<u32>> = Vec::with_capacity(out.len());
        for r in out {
            match merged.last_mut() {
                Some(m) if r.start <= m.end => m.end = m.end.max(r.end),
                _ => merged.push(r),
            }
        }
        merged
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
        let top = TopLevel::new(&rs).unwrap();
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
        let top = TopLevel::new(&root::parse(&samples::RS10).unwrap()).unwrap();
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
    #[allow(clippy::single_range_in_vec_init)]
    fn tables_reach_their_ranges() {
        // RS10's first table: SRV x4 (append), CBV at 4, UAV unbounded at
        // 8; its second: two samplers.
        let top = TopLevel::new(&root::parse(&samples::RS10).unwrap()).unwrap();
        assert_eq!(
            top.tables[0],
            [
                Span { start: 0, count: 4 },
                Span { start: 4, count: 1 },
                Span {
                    start: 8,
                    count: root::UNBOUNDED
                },
            ]
        );
        // From index 10 of 32: 10..15 then 18.. to the end.
        assert_eq!(top.reached(0, 10, 32), [10..15, 18..32]);
        // Clipped by a small heap; nothing past its end.
        assert_eq!(top.reached(0, 10, 12), [10..12]);
        assert_eq!(top.reached(0, 40, 32), []);
        assert_eq!(top.reached(1, 3, 16), [3..5]);
        assert_eq!(top.reached(2, 0, 16), [], "no third table");
        // RS11's table: SRV x2, UAV appended after: one run.
        let top = TopLevel::new(&root::parse(&samples::RS11).unwrap()).unwrap();
        assert_eq!(top.reached(0, 5, 16), [5..8]);
    }

    #[test]
    fn addresses_before_constants() {
        // UAV, constants(2), SRV: the order the swap.hlsl test uses.
        let rs = root::parse(&swap_root_signature().serialize()).unwrap();
        let top = TopLevel::new(&rs).unwrap();
        let offsets: Vec<u32> = top.entries.iter().map(|e| e.offset).collect();
        assert_eq!(offsets, [0, 8, 16]);
        let ab = top.fill(&[3, 1000], &[0x1000, 0x2000], &[]);
        assert_eq!(&ab[0..8], &0x1000u64.to_le_bytes());
        assert_eq!(&ab[8..16], &[3, 0, 0, 0, 0xe8, 0x03, 0, 0]);
        assert_eq!(&ab[16..24], &0x2000u64.to_le_bytes());
    }
}
