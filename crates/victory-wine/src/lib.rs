//! The Wine bridge to `victory-metal` (slice 4a).
//!
//! victory runs as a Windows program under Wine (Game Porting Toolkit's
//! Wine 7.7, x86_64 under Rosetta). The Metal backend runs in the same
//! process on the Unix side: a Wine builtin DLL (`victory_metal.dll`,
//! empty but marked builtin) whose Unix library (`victory_metal.so`, the
//! `victory-wine-unix` crate, x86_64 macOS) Wine loads for it. The Windows
//! side gets the library's call table with `NtQueryVirtualMemory(...,
//! MemoryWineUnixFuncs, ...)` on the DLL and calls into it with ntdll's
//! `__wine_unix_call(table, op, args)` ([`client`], Windows only).
//!
//! One address space: pointers in the argument blocks are good on both
//! sides, buffer contents are written straight from the Windows side, and
//! object handles are the Unix side's boxed objects. Argument blocks are
//! `repr(C)` with fixed-width fields only (the two sides are built for
//! different targets). Recording is a byte stream ([`Stream`]) replayed on
//! the Unix side at submission: one call per submission, not per command.
//!
//! This first step bridges what the smoke test needs: a device, committed
//! resources and their contents / addresses, compute pipelines, and a
//! compute recording. 4b grows it to all of `victory_host`.

#[cfg(windows)]
pub mod client;

use victory_host::{Memory, ResourceDesc};

/// Unix call numbers: the index into `victory_metal.so`'s
/// `__wine_unix_call_funcs`.
pub mod op {
    pub const OPEN: u32 = 0;
    pub const RELEASE: u32 = 1;
    pub const COMMITTED: u32 = 2;
    pub const CONTENTS: u32 = 3;
    pub const GPU_ADDRESS: u32 = 4;
    pub const COMPUTE_PIPELINE: u32 = 5;
    pub const SUBMIT: u32 = 6;
    /// How many there are.
    pub const COUNT: usize = 7;
}

/// `NTSTATUS`es the Unix side returns.
pub mod status {
    pub const SUCCESS: u32 = 0;
    /// `STATUS_UNSUCCESSFUL`: see the block's message.
    pub const FAILED: u32 = 0xc000_0001;
    /// `STATUS_INVALID_PARAMETER`.
    pub const INVALID: u32 = 0xc000_000d;
}

/// A failure's text, for the Windows side to report.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Message {
    pub len: u32,
    pub text: [u8; 252],
}

impl Default for Message {
    fn default() -> Self {
        Message {
            len: 0,
            text: [0; 252],
        }
    }
}

impl Message {
    pub fn set(&mut self, s: &str) {
        let n = s.len().min(self.text.len());
        self.text[..n].copy_from_slice(&s.as_bytes()[..n]);
        self.len = n as u32;
    }

    pub fn get(&self) -> String {
        String::from_utf8_lossy(&self.text[..(self.len as usize).min(self.text.len())]).into_owned()
    }
}

/// What [`Release`] frees.
pub mod kind {
    pub const DEVICE: u32 = 0;
    pub const RESOURCE: u32 = 1;
    pub const PIPELINE: u32 = 2;
}

/// [`Memory`] as a number and back.
pub fn memory_code(m: Memory) -> u32 {
    match m {
        Memory::Device => 0,
        Memory::Upload => 1,
        Memory::Readback => 2,
        Memory::DeviceUpload => 3,
        Memory::Shared => 4,
    }
}

pub fn memory_of(code: u32) -> Option<Memory> {
    Some(match code {
        0 => Memory::Device,
        1 => Memory::Upload,
        2 => Memory::Readback,
        3 => Memory::DeviceUpload,
        4 => Memory::Shared,
        _ => return None,
    })
}

/// [`op::OPEN`]: the system's Metal device.
#[repr(C)]
#[derive(Default)]
pub struct Open {
    pub device: u64,
    pub error: Message,
}

/// [`op::RELEASE`].
#[repr(C)]
#[derive(Default)]
pub struct Release {
    pub kind: u32,
    pub handle: u64,
}

/// [`op::COMMITTED`].
#[repr(C)]
pub struct Committed {
    pub device: u64,
    pub memory: u32,
    pub state: u32,
    pub desc: ResourceDesc,
    pub resource: u64,
    pub error: Message,
}

/// [`op::CONTENTS`]: a CPU-visible buffer's contents (0 if none).
#[repr(C)]
#[derive(Default)]
pub struct Contents {
    pub resource: u64,
    pub ptr: u64,
    pub len: u64,
}

/// [`op::GPU_ADDRESS`].
#[repr(C)]
#[derive(Default)]
pub struct GpuAddress {
    pub resource: u64,
    pub address: u64,
}

/// Shader code kinds for [`ComputePipeline`].
pub mod code {
    pub const DXIL: u32 = 0;
    pub const MSL: u32 = 1;
    pub const METALLIB: u32 = 2;
}

/// [`op::COMPUTE_PIPELINE`]: pointers and lengths into the caller's memory.
#[repr(C)]
#[derive(Default)]
pub struct ComputePipeline {
    pub device: u64,
    pub code_kind: u32,
    pub threads: [u32; 3],
    pub code: u64,
    pub code_len: u64,
    pub entry: u64,
    pub entry_len: u64,
    pub root: u64,
    pub root_len: u64,
    pub pipeline: u64,
    pub error: Message,
}

/// [`op::SUBMIT`]: replays a [`Stream`] into one command buffer and
/// commits it (waiting for it when `wait` is non-zero).
#[repr(C)]
#[derive(Default)]
pub struct Submit {
    pub device: u64,
    pub stream: u64,
    pub len: u64,
    pub wait: u32,
    pub error: Message,
}

/// Recorded commands, one byte stream: a 32-bit command then its fields,
/// little-endian. Handles are the Unix side's.
pub mod cmd {
    /// pipeline
    pub const SET_COMPUTE_ROOT: u32 = 1;
    /// pipeline
    pub const SET_COMPUTE_PIPELINE: u32 = 2;
    /// point, param, offset, count, values
    pub const SET_ROOT_CONSTANTS: u32 = 3;
    /// point, param, kind, address (u64)
    pub const SET_ROOT_VIEW: u32 = 4;
    /// point, param, handle (u64)
    pub const SET_ROOT_TABLE: u32 = 5;
    /// x, y, z
    pub const DISPATCH: u32 = 6;
    /// an unordered-access barrier
    pub const BARRIER_UNORDERED: u32 = 7;
    /// to, to offset, from, from offset, bytes (u64 each)
    pub const COPY_BUFFER: u32 = 8;
}

/// A recording being written.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct Stream(pub Vec<u8>);

impl Stream {
    pub fn u32(&mut self, v: u32) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
}

/// One decoded command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    SetComputeRoot(u64),
    SetComputePipeline(u64),
    SetRootConstants {
        point: u32,
        param: u32,
        offset: u32,
        values: Vec<u32>,
    },
    SetRootView {
        point: u32,
        param: u32,
        kind: u32,
        address: u64,
    },
    SetRootTable {
        point: u32,
        param: u32,
        handle: u64,
    },
    Dispatch([u32; 3]),
    BarrierUnordered,
    CopyBuffer {
        to: u64,
        to_offset: u64,
        from: u64,
        from_offset: u64,
        bytes: u64,
    },
}

impl Command {
    /// Appends this command to `s`.
    pub fn write(&self, s: &mut Stream) {
        match self {
            Command::SetComputeRoot(p) => s.u32(cmd::SET_COMPUTE_ROOT).u64(*p),
            Command::SetComputePipeline(p) => s.u32(cmd::SET_COMPUTE_PIPELINE).u64(*p),
            Command::SetRootConstants {
                point,
                param,
                offset,
                values,
            } => {
                s.u32(cmd::SET_ROOT_CONSTANTS)
                    .u32(*point)
                    .u32(*param)
                    .u32(*offset)
                    .u32(values.len() as u32);
                for v in values {
                    s.u32(*v);
                }
                s
            }
            Command::SetRootView {
                point,
                param,
                kind,
                address,
            } => s
                .u32(cmd::SET_ROOT_VIEW)
                .u32(*point)
                .u32(*param)
                .u32(*kind)
                .u64(*address),
            Command::SetRootTable {
                point,
                param,
                handle,
            } => s
                .u32(cmd::SET_ROOT_TABLE)
                .u32(*point)
                .u32(*param)
                .u64(*handle),
            Command::Dispatch([x, y, z]) => s.u32(cmd::DISPATCH).u32(*x).u32(*y).u32(*z),
            Command::BarrierUnordered => s.u32(cmd::BARRIER_UNORDERED),
            Command::CopyBuffer {
                to,
                to_offset,
                from,
                from_offset,
                bytes,
            } => s
                .u32(cmd::COPY_BUFFER)
                .u64(*to)
                .u64(*to_offset)
                .u64(*from)
                .u64(*from_offset)
                .u64(*bytes),
        };
    }
}

/// Reads commands back; `None` at the end, `Some(Err(at))` at a byte the
/// stream can't be read from.
pub struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, at: 0 }
    }

    fn u32(&mut self) -> Option<u32> {
        let b = self.bytes.get(self.at..self.at + 4)?;
        self.at += 4;
        Some(u32::from_le_bytes(b.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        let b = self.bytes.get(self.at..self.at + 8)?;
        self.at += 8;
        Some(u64::from_le_bytes(b.try_into().ok()?))
    }

    fn command(&mut self) -> Option<Command> {
        Some(match self.u32()? {
            cmd::SET_COMPUTE_ROOT => Command::SetComputeRoot(self.u64()?),
            cmd::SET_COMPUTE_PIPELINE => Command::SetComputePipeline(self.u64()?),
            cmd::SET_ROOT_CONSTANTS => {
                let (point, param, offset, n) =
                    (self.u32()?, self.u32()?, self.u32()?, self.u32()?);
                // At most what is left of the stream.
                if n as usize > (self.bytes.len() - self.at) / 4 {
                    return None;
                }
                let values = (0..n).map(|_| self.u32()).collect::<Option<Vec<_>>>()?;
                Command::SetRootConstants {
                    point,
                    param,
                    offset,
                    values,
                }
            }
            cmd::SET_ROOT_VIEW => Command::SetRootView {
                point: self.u32()?,
                param: self.u32()?,
                kind: self.u32()?,
                address: self.u64()?,
            },
            cmd::SET_ROOT_TABLE => Command::SetRootTable {
                point: self.u32()?,
                param: self.u32()?,
                handle: self.u64()?,
            },
            cmd::DISPATCH => Command::Dispatch([self.u32()?, self.u32()?, self.u32()?]),
            cmd::BARRIER_UNORDERED => Command::BarrierUnordered,
            cmd::COPY_BUFFER => Command::CopyBuffer {
                to: self.u64()?,
                to_offset: self.u64()?,
                from: self.u64()?,
                from_offset: self.u64()?,
                bytes: self.u64()?,
            },
            _ => return None,
        })
    }
}

impl Iterator for Reader<'_> {
    type Item = Result<Command, usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.at >= self.bytes.len() {
            return None;
        }
        let start = self.at;
        match self.command() {
            Some(c) => Some(Ok(c)),
            None => {
                // Nothing more after a bad command.
                self.at = self.bytes.len();
                Some(Err(start))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every() -> Vec<Command> {
        vec![
            Command::SetComputeRoot(0x1000),
            Command::SetComputePipeline(0x2000),
            Command::SetRootConstants {
                point: 0,
                param: 0,
                offset: 1,
                values: vec![7, 8, 9],
            },
            Command::SetRootView {
                point: 0,
                param: 1,
                kind: 3,
                address: 0xdead_beef_0000,
            },
            Command::SetRootTable {
                point: 1,
                param: 2,
                handle: 0x4000,
            },
            Command::Dispatch([4, 1, 1]),
            Command::BarrierUnordered,
            Command::CopyBuffer {
                to: 1,
                to_offset: 2,
                from: 3,
                from_offset: 4,
                bytes: 5,
            },
        ]
    }

    #[test]
    fn commands_round_trip() {
        let mut s = Stream::default();
        for c in every() {
            c.write(&mut s);
        }
        let back: Vec<Command> = Reader::new(&s.0).map(|c| c.unwrap()).collect();
        assert_eq!(back, every());
    }

    #[test]
    fn bad_streams_stop_where_they_go_wrong() {
        let mut s = Stream::default();
        Command::Dispatch([1, 2, 3]).write(&mut s);
        let good = s.0.len();
        s.u32(99);
        let got: Vec<_> = Reader::new(&s.0).collect();
        assert_eq!(got, vec![Ok(Command::Dispatch([1, 2, 3])), Err(good)]);
        // Cut short; and a count past the end.
        let mut t = Stream::default();
        Command::SetRootView {
            point: 0,
            param: 0,
            kind: 0,
            address: 1,
        }
        .write(&mut t);
        assert_eq!(
            Reader::new(&t.0[..t.0.len() - 1]).collect::<Vec<_>>(),
            vec![Err(0)]
        );
        let mut u = Stream::default();
        u.u32(cmd::SET_ROOT_CONSTANTS)
            .u32(0)
            .u32(0)
            .u32(0)
            .u32(u32::MAX);
        assert_eq!(Reader::new(&u.0).collect::<Vec<_>>(), vec![Err(0)]);
    }

    #[test]
    fn blocks_have_fixed_layouts() {
        // Both sides read these: fixed-width fields, no padding surprises.
        assert_eq!(size_of::<Message>(), 256);
        assert_eq!(size_of::<Open>(), 264);
        assert_eq!(size_of::<Release>(), 16);
        assert_eq!(size_of::<Contents>(), 24);
        assert_eq!(size_of::<ComputePipeline>(), 8 + 4 + 12 + 8 * 7 + 256);
        assert_eq!(size_of::<Submit>(), 8 * 3 + 4 + 256 + 4);
        assert_eq!(std::mem::offset_of!(Committed, desc), 16);
    }

    #[test]
    fn memory_codes_round_trip() {
        for m in [
            Memory::Device,
            Memory::Upload,
            Memory::Readback,
            Memory::DeviceUpload,
            Memory::Shared,
        ] {
            assert_eq!(memory_of(memory_code(m)), Some(m));
        }
        assert_eq!(memory_of(9), None);
    }

    #[test]
    fn messages_truncate() {
        let mut m = Message::default();
        m.set(&"x".repeat(300));
        assert_eq!(m.get().len(), 252);
        m.set("refused");
        assert_eq!(m.get(), "refused");
    }
}
