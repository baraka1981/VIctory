//! The Metal side of the host seam ([`victory_host::Device`]), for Apple
//! silicon.
//!
//! Slice 1: resources, heaps and descriptor heaps. [`format`] and
//! [`layout`] are portable (built and tested anywhere); the Metal calls
//! are in `device`, macOS only.
//!
//! How this reaches the emulator: victory runs as a Windows program under
//! Wine; this crate builds as a native arm64 library the Windows side
//! calls through Wine's unix-call mechanism (docs/metal-port.md).

use core::fmt;

pub mod format;
pub mod layout;

#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
pub use device::{MetalDescriptorHeap, MetalDevice, MetalResource};

/// Bytes of one entry in a descriptor table: Metal Shader Converter's
/// `IRDescriptorTableEntry` (GPU address, texture view id, metadata), the
/// layout translated shaders read.
pub const DESCRIPTOR_BYTES: u32 = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalError {
    /// No Metal device on this machine.
    NoDevice,
    /// A D3D12.X format Metal has no match for (yet).
    Format(u32),
    /// A resource dimension Metal does not have.
    Dimension(u32),
    /// Metal returned nil: out of memory, or a heap is full.
    Failed(&'static str),
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetalError::NoDevice => write!(f, "no Metal device"),
            MetalError::Format(n) => write!(f, "format {n} has no Metal match"),
            MetalError::Dimension(n) => write!(f, "resource dimension {n} has no Metal match"),
            MetalError::Failed(what) => write!(f, "Metal could not make {what}"),
        }
    }
}

impl std::error::Error for MetalError {}
