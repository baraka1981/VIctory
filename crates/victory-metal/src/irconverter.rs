//! DXIL to Metal through Apple's Metal Shader Converter
//! (`libmetalirconverter.dylib`, loaded at run time through saxaboom).
//!
//! Resources reach converted shaders through a top-level argument buffer
//! laid out by the root signature ([`crate::toplevel`]). A root signature
//! handed in (`ComputeDesc::root_signature`) becomes MSC's global root
//! signature and the embedded one is ignored, as a D3D12 pipeline's root
//! signature wins over the shader's; with none handed in, MSC reads the
//! embedded one and we lay out from the same bytes.

use std::ffi::CStr;
use std::sync::OnceLock;

use saxaboom::{IRRootSignature, MetalIrConverter, ffi};
use victory_host::root::{Kind, RootSignature, Version};

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

/// The MSC enum value with D3D12's number `$value` (they share numbers;
/// looked up rather than transmuted so a bad number is an error).
macro_rules! pick {
    ($ty:ident, $what:literal, $value:expr, [$($v:ident),* $(,)?]) => {{
        let value = $value;
        [$(ffi::$ty::$v),*]
            .into_iter()
            .find(|x| *x as u32 == value)
            .ok_or_else(|| {
                MetalError::Shader(format!(concat!($what, " {} has no Metal Shader Converter match"), value))
            })
    }};
}

fn visibility(v: u32) -> Result<ffi::IRShaderVisibility, MetalError> {
    pick!(
        IRShaderVisibility,
        "shader visibility",
        v,
        [
            All,
            Vertex,
            Hull,
            Domain,
            Geometry,
            Pixel,
            Amplification,
            Mesh,
        ]
    )
}

fn parameter_type(t: u32) -> Result<ffi::IRRootParameterType, MetalError> {
    pick!(
        IRRootParameterType,
        "root parameter type",
        t,
        [DescriptorTable, _32BitConstants, CBV, SRV, UAV,]
    )
}

fn range_type(t: u32) -> Result<ffi::IRDescriptorRangeType, MetalError> {
    pick!(
        IRDescriptorRangeType,
        "descriptor range type",
        t,
        [SRV, UAV, CBV, Sampler]
    )
}

fn sampler(
    s: &victory_host::root::StaticSampler,
) -> Result<ffi::IRStaticSamplerDescriptor, MetalError> {
    let address = |a: u32| {
        pick!(
            IRTextureAddressMode,
            "texture address mode",
            a,
            [Wrap, Mirror, Clamp, Border, MirrorOnce,]
        )
    };
    Ok(ffi::IRStaticSamplerDescriptor {
        Filter: pick!(
            IRFilter,
            "filter",
            s.filter,
            [
                MinMagMipPoint,
                MinMagPointMipLinear,
                MinPointMagLinearMipPoint,
                MinPointMagMipLinear,
                MinLinearMagMipPoint,
                MinLinearMagPointMipLinear,
                MinMagLinearMipPoint,
                MinMagMipLinear,
                Anisotropic,
                ComparisonMinMagMipPoint,
                ComparisonMinMagPointMipLinear,
                ComparisonMinPointMagLinearMipPoint,
                ComparisonMinPointMagMipLinear,
                ComparisonMinLinearMagMipPoint,
                ComparisonMinLinearMagPointMipLinear,
                ComparisonMinMagLinearMipPoint,
                ComparisonMinMagMipLinear,
                ComparisonAnisotropic,
                MinimumMinMagMipPoint,
                MinimumMinMagPointMipLinear,
                MinimumMinPointMagLinearMipPoint,
                MinimumMinPointMagMipLinear,
                MinimumMinLinearMagMipPoint,
                MinimumMinLinearMagPointMipLinear,
                MinimumMinMagLinearMipPoint,
                MinimumMinMagMipLinear,
                MinimumAnisotropic,
                MaximumMinMagMipPoint,
                MaximumMinMagPointMipLinear,
                MaximumMinPointMagLinearMipPoint,
                MaximumMinPointMagMipLinear,
                MaximumMinLinearMagMipPoint,
                MaximumMinLinearMagPointMipLinear,
                MaximumMinMagLinearMipPoint,
                MaximumMinMagMipLinear,
                MaximumAnisotropic,
            ]
        )?,
        AddressU: address(s.address[0])?,
        AddressV: address(s.address[1])?,
        AddressW: address(s.address[2])?,
        MipLODBias: s.mip_lod_bias,
        MaxAnisotropy: s.max_anisotropy,
        ComparisonFunc: pick!(
            IRComparisonFunction,
            "comparison function",
            s.comparison,
            [
                Never,
                Less,
                Equal,
                LessEqual,
                Greater,
                NotEqual,
                GreaterEqual,
                Always,
            ]
        )?,
        BorderColor: pick!(
            IRStaticBorderColor,
            "border colour",
            s.border,
            [TransparentBlack, OpaqueBlack, OpaqueWhite,]
        )?,
        MinLOD: s.min_lod,
        MaxLOD: s.max_lod,
        ShaderRegister: s.register,
        RegisterSpace: s.space,
        ShaderVisibility: visibility(s.visibility)?,
    })
}

/// `rs` as MSC's root signature object. The descriptor's arrays live on
/// this stack frame; MSC copies what it keeps.
pub(crate) fn root_signature(rs: &RootSignature) -> Result<IRRootSignature, MetalError> {
    let c = converter()?;
    let mut samplers = rs
        .samplers
        .iter()
        .map(sampler)
        .collect::<Result<Vec<_>, _>>()?;
    let flags = ffi::IRRootSignatureFlags(rs.flags);
    let made = match rs.version {
        Version::V1_0 => {
            let ranges = rs
                .parameters
                .iter()
                .map(|p| match &p.kind {
                    Kind::Table(ranges) => ranges
                        .iter()
                        .map(|r| {
                            Ok(ffi::IRDescriptorRange {
                                RangeType: range_type(r.kind)?,
                                NumDescriptors: r.count,
                                BaseShaderRegister: r.register,
                                RegisterSpace: r.space,
                                OffsetInDescriptorsFromTableStart: r.offset,
                            })
                        })
                        .collect::<Result<Vec<_>, MetalError>>(),
                    _ => Ok(Vec::new()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let parameters = rs
                .parameters
                .iter()
                .zip(&ranges)
                .map(|(p, ranges)| {
                    let (t, u_1) = match p.kind {
                        Kind::Table(_) => (
                            0,
                            ffi::IRRootParameter_u {
                                DescriptorTable: ffi::IRRootDescriptorTable {
                                    NumDescriptorRanges: ranges.len() as u32,
                                    pDescriptorRanges: ranges.as_ptr(),
                                },
                            },
                        ),
                        Kind::Constants {
                            register,
                            space,
                            count,
                        } => (
                            1,
                            ffi::IRRootParameter_u {
                                Constants: ffi::IRRootConstants {
                                    ShaderRegister: register,
                                    RegisterSpace: space,
                                    Num32BitValues: count,
                                },
                            },
                        ),
                        Kind::Descriptor {
                            kind,
                            register,
                            space,
                            ..
                        } => (
                            kind,
                            ffi::IRRootParameter_u {
                                Descriptor: ffi::IRRootDescriptor {
                                    ShaderRegister: register,
                                    RegisterSpace: space,
                                },
                            },
                        ),
                    };
                    Ok(ffi::IRRootParameter {
                        ParameterType: parameter_type(t)?,
                        u_1,
                        ShaderVisibility: visibility(p.visibility)?,
                    })
                })
                .collect::<Result<Vec<_>, MetalError>>()?;
            let desc = ffi::IRVersionedRootSignatureDescriptor {
                version: ffi::IRRootSignatureVersion::_1,
                u_1: ffi::IRVersionedRootSignatureDescriptor_u {
                    desc_1_0: ffi::IRRootSignatureDescriptor {
                        NumParameters: parameters.len() as u32,
                        pParameters: parameters.as_ptr(),
                        NumStaticSamplers: samplers.len() as u32,
                        pStaticSamplers: samplers.as_ptr(),
                        Flags: flags,
                    },
                },
            };
            c.create_root_signature_from_descriptor(&desc)
        }
        Version::V1_1 => {
            let mut ranges = rs
                .parameters
                .iter()
                .map(|p| match &p.kind {
                    Kind::Table(ranges) => ranges
                        .iter()
                        .map(|r| {
                            Ok(ffi::IRDescriptorRange1 {
                                RangeType: range_type(r.kind)?,
                                NumDescriptors: r.count,
                                BaseShaderRegister: r.register,
                                RegisterSpace: r.space,
                                Flags: ffi::IRDescriptorRangeFlags(r.flags),
                                OffsetInDescriptorsFromTableStart: r.offset,
                            })
                        })
                        .collect::<Result<Vec<_>, MetalError>>(),
                    _ => Ok(Vec::new()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut parameters = rs
                .parameters
                .iter()
                .zip(&mut ranges)
                .map(|(p, ranges)| {
                    let (t, u_1) = match p.kind {
                        Kind::Table(_) => (
                            0,
                            ffi::IRRootParameter1_u {
                                DescriptorTable: ffi::IRRootDescriptorTable1 {
                                    NumDescriptorRanges: ranges.len() as u32,
                                    pDescriptorRanges: ranges.as_mut_ptr(),
                                },
                            },
                        ),
                        Kind::Constants {
                            register,
                            space,
                            count,
                        } => (
                            1,
                            ffi::IRRootParameter1_u {
                                Constants: ffi::IRRootConstants {
                                    ShaderRegister: register,
                                    RegisterSpace: space,
                                    Num32BitValues: count,
                                },
                            },
                        ),
                        Kind::Descriptor {
                            kind,
                            register,
                            space,
                            flags,
                        } => (
                            kind,
                            ffi::IRRootParameter1_u {
                                Descriptor: ffi::IRRootDescriptor1 {
                                    ShaderRegister: register,
                                    RegisterSpace: space,
                                    Flags: ffi::IRRootDescriptorFlags(flags),
                                },
                            },
                        ),
                    };
                    Ok(ffi::IRRootParameter1 {
                        ParameterType: parameter_type(t)?,
                        u_1,
                        ShaderVisibility: visibility(p.visibility)?,
                    })
                })
                .collect::<Result<Vec<_>, MetalError>>()?;
            let desc = ffi::IRVersionedRootSignatureDescriptor {
                version: ffi::IRRootSignatureVersion::_1_1,
                u_1: ffi::IRVersionedRootSignatureDescriptor_u {
                    desc_1_1: ffi::IRRootSignatureDescriptor1 {
                        NumParameters: parameters.len() as u32,
                        pParameters: parameters.as_mut_ptr(),
                        NumStaticSamplers: samplers.len() as u32,
                        pStaticSamplers: samplers.as_mut_ptr(),
                        Flags: flags,
                    },
                },
            };
            c.create_root_signature_from_descriptor(&desc)
        }
    };
    made.map_err(|e| MetalError::Shader(format!("Metal Shader Converter: {e:?}")))
}

/// Where MSC puts each resource of `rs` in the top-level argument buffer
/// (type, space, register, byte offset, size), for checking
/// [`crate::toplevel`] against it.
#[cfg(test)]
pub(crate) fn locations(rs: &RootSignature) -> Result<Vec<ffi::IRResourceLocation>, MetalError> {
    Ok(root_signature(rs)?.resource_locations())
}

/// A converted compute shader.
pub(crate) struct Converted {
    pub metallib: Vec<u8>,
    pub threads: [u32; 3],
}

/// Converts a DXIL compute shader whose entry point is `entry`, with
/// `root` as its root signature (the embedded one when `None`).
pub(crate) fn compute(
    dxil: &[u8],
    entry: &CStr,
    root: Option<&RootSignature>,
) -> Result<Converted, MetalError> {
    let c = converter()?;
    let mut compiler = c.create_compiler();
    compiler.set_entry_point_name(ENTRY);
    // Kept alive until the compile below is done.
    let global = root.map(root_signature).transpose()?;
    if let Some(global) = &global {
        compiler.set_global_root_signature(global);
        compiler.ignore_root_signature(true);
    }
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
