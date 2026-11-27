//! DXIL to Metal through Apple's Metal Shader Converter
//! (`libmetalirconverter.dylib`, loaded at run time through saxaboom).
//!
//! Resources reach converted shaders through a top-level argument buffer
//! laid out by the root signature embedded in the DXIL (MSC's default).
//! For [`victory_host::ComputeDesc`]'s model (constants, then one root
//! descriptor per binding) that is: the constants inline, then one 64-bit
//! GPU address per binding, each at its natural alignment.
//!
//! TODO(3b): root signatures handed in separately (the serialized blob in
//! `ComputeDesc::root_signature`) need turning into MSC's descriptor; and
//! the layout rule above wants checking against a real converted shader.

use std::ffi::CStr;
use std::sync::OnceLock;

use saxaboom::{MetalIrConverter, ffi};

use crate::MetalError;

/// Where the installer puts the library; `VICTORY_IRCONVERTER` overrides.
const DEFAULT_PATH: &str = "/usr/local/lib/libmetalirconverter.dylib";

/// The Metal function name converted shaders get.
pub(crate) const ENTRY: &CStr = c"victory_main";

/// MSC's buffer index for the top-level argument buffer
/// (`kIRArgumentBufferBindPoint`).
pub(crate) const ARGUMENT_BUFFER_INDEX: usize = 2;

fn converter() -> Result<&'static MetalIrConverter, MetalError> {
    static LOADED: OnceLock<Result<MetalIrConverter, String>> = OnceLock::new();
    LOADED
        .get_or_init(|| {
            let path =
                std::env::var_os("VICTORY_IRCONVERTER").unwrap_or_else(|| DEFAULT_PATH.into());
            MetalIrConverter::new(&path).map_err(|e| format!("{e} ({})", path.to_string_lossy()))
        })
        .as_ref()
        .map_err(|e| MetalError::Shader(format!("Metal Shader Converter not loaded: {e}")))
}

/// A converted compute shader.
pub(crate) struct Converted {
    pub metallib: Vec<u8>,
    pub threads: [u32; 3],
}

/// Converts a DXIL compute shader whose entry point is `entry`.
pub(crate) fn compute(dxil: &[u8], entry: &CStr) -> Result<Converted, MetalError> {
    let c = converter()?;
    let mut compiler = c.create_compiler();
    compiler.set_entry_point_name(ENTRY);
    let input = c.create_object_from_dxil(dxil);
    let out = compiler
        .alloc_compile_and_link(entry, &input)
        .map_err(|e| MetalError::Shader(format!("Metal Shader Converter: {e:?}")))?;
    let metallib = out
        .metal_lib_binary()
        .ok_or_else(|| MetalError::Shader("no metallib from Metal Shader Converter".into()))?
        .byte_code();
    let threads = out
        .reflection()
        .and_then(|r| r.compute_info(ffi::IRReflectionVersion::_1_0))
        // SAFETY: version 1.0 info, as asked for.
        .map(|info| unsafe { info.u_1.info_1_0.tg_size })
        .unwrap_or([1, 1, 1]);
    Ok(Converted { metallib, threads })
}

/// The top-level argument buffer for `constants` then `addresses` (one
/// 64-bit GPU address per binding), as laid out for ComputeDesc's model.
pub(crate) fn argument_buffer(constants: &[u32], addresses: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(constants.len() * 4 + addresses.len() * 8 + 4);
    for c in constants {
        out.extend_from_slice(&c.to_le_bytes());
    }
    for a in addresses {
        while out.len() % 8 != 0 {
            out.push(0);
        }
        out.extend_from_slice(&a.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_then_aligned_addresses() {
        let ab = argument_buffer(&[7], &[0x1000, 0x2000]);
        assert_eq!(ab.len(), 8 + 16);
        assert_eq!(&ab[0..4], &7u32.to_le_bytes());
        assert_eq!(&ab[8..16], &0x1000u64.to_le_bytes());
        assert_eq!(&ab[16..24], &0x2000u64.to_le_bytes());
    }
}
