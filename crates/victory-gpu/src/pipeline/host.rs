//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, USER_DATA_SPACE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::{Shader, Translated};
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    depth_bounds: bool,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            depth_bounds: false,
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

impl GraphicsState {
    /// Blending per target in the host seam's terms (D3D12's numbers).
    fn seam_blend(&self) -> [victory_host::Blend; 8] {
        self.blend.RenderTarget.map(|t| victory_host::Blend {
            enable: t.BlendEnable.as_bool(),
            src: t.SrcBlend.0 as u32,
            dst: t.DestBlend.0 as u32,
            op: t.BlendOp.0 as u32,
            src_alpha: t.SrcBlendAlpha.0 as u32,
            dst_alpha: t.DestBlendAlpha.0 as u32,
            op_alpha: t.BlendOpAlpha.0 as u32,
            write_mask: t.RenderTargetWriteMask,
            logic_op: t.LogicOpEnable.as_bool().then_some(t.LogicOp.0 as u32),
        })
    }

    fn seam_raster(&self) -> victory_host::Raster {
        let r = &self.rasterizer;
        victory_host::Raster {
            fill: r.FillMode.0 as u32,
            cull: r.CullMode.0 as u32,
            front_ccw: r.FrontCounterClockwise.as_bool(),
            depth_bias: r.DepthBias,
            depth_bias_clamp: r.DepthBiasClamp,
            slope_scaled_depth_bias: r.SlopeScaledDepthBias,
            depth_clip: r.DepthClipEnable.as_bool(),
            multisample: r.MultisampleEnable.as_bool(),
            antialiased_lines: r.AntialiasedLineEnable.as_bool(),
            forced_samples: r.ForcedSampleCount,
            conservative: r.ConservativeRaster.0 != 0,
        }
    }

    fn seam_depth(&self) -> (victory_host::DepthTest, victory_host::Stencil) {
        let d = &self.depth;
        let face = |f: &D3D12_DEPTH_STENCILOP_DESC| victory_host::StencilFace {
            fail: f.StencilFailOp.0 as u32,
            depth_fail: f.StencilDepthFailOp.0 as u32,
            pass: f.StencilPassOp.0 as u32,
            func: f.StencilFunc.0 as u32,
        };
        (
            victory_host::DepthTest {
                enable: d.DepthEnable.as_bool(),
                write: d.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL,
                func: d.DepthFunc.0 as u32,
            },
            victory_host::Stencil {
                enable: d.StencilEnable.as_bool(),
                read_mask: d.StencilReadMask,
                write_mask: d.StencilWriteMask,
                front: face(&d.FrontFace),
                back: face(&d.BackFace),
            },
        )
    }

    /// The input layout in the seam's terms, names borrowed from `self`.
    fn seam_inputs(&self) -> Vec<victory_host::InputElement<'_>> {
        self.inputs
            .iter()
            .map(|e| victory_host::InputElement {
                // SAFETY: the names point into `self.names`, NUL-ended.
                semantic: unsafe { CStr::from_ptr(e.SemanticName.0.cast()) }
                    .to_str()
                    .unwrap_or(""),
                index: e.SemanticIndex,
                format: e.Format.0 as u32,
                slot: e.InputSlot,
                offset: e.AlignedByteOffset,
                per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                step_rate: e.InstanceDataStepRate,
            })
            .collect()
    }

    pub(super) fn writes_depth_only(&self) -> bool {
        self.targets.NumRenderTargets == 0
            && self.depth_format != DXGI_FORMAT_UNKNOWN
            && self.depth.DepthEnable.as_bool()
            && self.depth.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL
    }

    /// The guest color buffer bypasses its blender for integer surfaces,
    /// even when the draw asks for blending. D3D12 validates blend enable
    /// against every active target format when the pipeline is created.
    pub(crate) fn apply_integer_blend_bypass(&mut self) {
        let integer = |f: &DXGI_FORMAT| matches!(
            *f,
            DXGI_FORMAT_R32G32B32A32_UINT | DXGI_FORMAT_R32G32B32A32_SINT
                | DXGI_FORMAT_R32G32B32_UINT | DXGI_FORMAT_R32G32B32_SINT
                | DXGI_FORMAT_R16G16B16A16_UINT | DXGI_FORMAT_R16G16B16A16_SINT
                | DXGI_FORMAT_R32G32_UINT | DXGI_FORMAT_R32G32_SINT
                | DXGI_FORMAT_R10G10B10A2_UINT
                | DXGI_FORMAT_R8G8B8A8_UINT | DXGI_FORMAT_R8G8B8A8_SINT
                | DXGI_FORMAT_R16G16_UINT | DXGI_FORMAT_R16G16_SINT
                | DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R32_SINT
                | DXGI_FORMAT_R8G8_UINT | DXGI_FORMAT_R8G8_SINT
                | DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16_SINT
                | DXGI_FORMAT_R8_UINT | DXGI_FORMAT_R8_SINT
        );
        let formats = &self.targets.RTFormats[..(self.targets.NumRenderTargets as usize).min(8)];
        if !self.blend.IndependentBlendEnable.as_bool() {
            let shared = self.blend.RenderTarget[0];
            if !shared.BlendEnable.as_bool() || !formats.iter().any(integer) {
                return;
            }
            // With shared blending D3D12 reads slot zero for every target.
            // Expand it before overriding individual targets, including
            // their shared write mask.
            self.blend.RenderTarget[..formats.len()].fill(shared);
            self.blend.IndependentBlendEnable = (formats.len() > 1).into();
        }
        for (format, target) in formats.iter().zip(&mut self.blend.RenderTarget) {
            if integer(format) {
                target.BlendEnable = false.into();
            }
        }
    }

    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => {
                    self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned();
                    self.depth_bounds = false;
                }
                21 => {
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth_bounds = d.DepthBoundsTestEnable.as_bool();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::target_format(crate::resource::pc_format(
                            f.0 as u32,
                        )) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex, e.Format.0, e.InputSlot, e.AlignedByteOffset, e.InputSlotClass.0, e.InstanceDataStepRate
                )
            })
            .collect();
        let bounds = if self.depth_bounds {
            " depth-bounds"
        } else {
            ""
        };
        format!(
            "{:?} {} {:?} {:?}{bounds} {inputs:?} {} {} {:?} {} {:?}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                } else {
                    e.AlignedByteOffset
                };
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                next[slot] = offset + bytes;
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// Whether the title's root SRVs are bound as one-view tables of the
/// buffer's host copy, whose reads past its end return zeros, rather than
/// as root descriptors, which have no size: a mesh shader's vertex fetch
/// has no record limit, and a read past the copy reaches whatever the host
/// placed after it (memory it has freed: the device is removed, upstream
/// PR #493). `VICTORY_ROOT_SRV_DIRECT=1` goes back to root descriptors.
pub(crate) fn bounded_root_srvs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ROOT_SRV_DIRECT").is_none())
}

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    /// Root UAVs: one-view tables like the SRVs (or root descriptors with
    /// `VICTORY_ROOT_SRV_DIRECT=1`).
    pub(crate) root_uavs: Vec<(u8, u32)>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated()
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    /// The host pipeline and its root signature, as the backend made them.
    pub(crate) host: crate::backend::D3d12Pipeline,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Has the geometry stage that completes the console's rectangles
    /// (`victory_shader::emit::rect_list_gs`): its rectangle draws go out
    /// as the title gives them, three vertices a rectangle.
    pub(crate) rects: bool,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
    /// Its depth, stencil, cull and blend state in short, for the frame
    /// listing (empty for compute).
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    pub(crate) state: String,
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

impl HostPipeline {
    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        // For the report when the host device is removed: what this thread
        // has in the library from here on.
        super::library::label(format!("{shaders:?}"));
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh"))
        {
            return refuse(
                "only vertex, mesh, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse("the host GPU has no mesh shaders (or no shader model 6.6)", shaders);
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        let host = host()?;

        // The root signature the translated shaders' bindings define,
        // made without D3D12 (victory_host::root), so the Metal backend
        // takes the same bytes.
        use victory_host::root::{self as rs, Kind, Parameter, Range};
        let mut params: Vec<Parameter> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for &(shader, visibility) in &stages {
            let visibility = visibility.0 as u32;
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(Parameter {
                    kind: Kind::Constants {
                        register: 0,
                        space: USER_DATA_SPACE,
                        count: dwords,
                    },
                    visibility,
                });
            }
            for (slots, kind, out) in [
                (&t.root_cbvs, rs::parameter::CBV, &mut stage.root_cbvs),
                (&t.root_srvs, rs::parameter::SRV, &mut stage.root_srvs),
                (&t.root_uavs, rs::parameter::UAV, &mut stage.root_uavs),
            ] {
                for &slot in slots {
                    let uav = kind == rs::parameter::UAV;
                    // Ours, not the title's, stay root descriptors: the
                    // table of strides, the directory of title memory by
                    // address and its fault notes.
                    let ours = if uav {
                        slot == victory_shader::emit::SLOT_FAULTS
                    } else {
                        slot == victory_shader::emit::SLOT_STRIDES
                            || slot == victory_shader::emit::SLOT_MEMORY
                            || slot == victory_shader::emit::SLOT_VIEW_ADDRESSES
                    };
                    out.push((slot, params.len() as u32));
                    if (uav || kind == rs::parameter::SRV) && !ours && bounded_root_srvs() {
                        // A one-view table: the view's size bounds the
                        // reads (and a root UAV's writes), which a root
                        // descriptor leaves unbounded.
                        params.push(Parameter {
                            kind: Kind::Table(vec![Range {
                                kind: if uav { rs::range::UAV } else { rs::range::SRV },
                                count: 1,
                                register: u32::from(slot),
                                space: ROOT_CBV_SPACE,
                                flags: (D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                                    | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE)
                                    .0 as u32,
                                offset: 0,
                            }]),
                            visibility,
                        });
                        continue;
                    }
                    params.push(Parameter {
                        kind: Kind::Descriptor {
                            kind,
                            register: u32::from(slot),
                            space: ROOT_CBV_SPACE,
                            flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE.0 as u32,
                        },
                        visibility,
                    });
                }
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(rs::StaticSampler {
                    filter: s.filter,
                    address: s.address,
                    mip_lod_bias: s.mip_lod_bias,
                    max_anisotropy: s.max_anisotropy,
                    comparison: s.comparison,
                    border: s.border,
                    min_lod: s.min_lod,
                    max_lod: s.max_lod,
                    register: n as u32,
                    space: SAMPLER_SPACE,
                    visibility,
                });
            }
            bound.push(stage);
        }
        let root_blob = root_blob(params, samplers, compute.is_some());
        let dxil = |i: usize| stages.get(i).and_then(|(s, _)| s.translated()).map(|t| &t.host.dxil[..]);
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        if compute.is_some() {
            // Through the host seam: the backend makes the root signature
            // from the same bytes and caches the pipeline under the name.
            use victory_host::Device as _;
            let dxil = &stages[0].0.translated()?.host.dxil;
            let name = super::library::name("compute", &root_blob, &[&dxil[..]], "");
            let made = crate::backend::D3d12(&host.device).compute_pipeline(&victory_host::ComputeDesc {
                code: victory_host::ShaderCode::Dxil(dxil),
                entry: "main",
                root_signature: &root_blob,
                threads: [0; 3],
                cache: Some(&name),
            });
            let made = match made {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&made.state, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                host: made,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                rects: false,
                state: String::new(),
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let depth_bounds = if g.depth_bounds {
            if host.depth_bounds() {
                true
            } else {
                static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 4 {
                    tracing::warn!(
                        "a pipeline enables the depth bounds test, which the host GPU does not support: its draws pass at every depth"
                    );
                }
                false
            }
        } else {
            false
        };
        let state = g.describe();
        if mesh.is_some() {
            // "triangles": the stream names its topology since it did not
            // load from the library without one (see `mesh_pipeline`); a
            // new name keeps such entries, stored before, from being
            // looked up again.
            let name = super::library::name(
                "mesh triangles",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let made = graphics_pipeline(
                &host.device,
                g,
                &root_blob,
                victory_host::Geometry::Mesh {
                    amplification: None,
                    shader: stage(dxil(0)?),
                },
                dxil(1).map(stage),
                None,
                depth_bounds,
                &name,
            );
            let made = match made {
                Ok(p) => p,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&made.state, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                host: made,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                rects: false,
                state: String::new(),
            });
        }

        // A pipeline the title makes for rectangles gets a geometry stage
        // that completes each one from the three vertices the vertex
        // shader made. Without it (refused here or by the host) the draws
        // fall back on what `draw` does for rectangles.
        let mut rect_gs = None;
        if g.topology.0 == RECTANGLES {
            let ps = stages.get(1).and_then(|(s, _)| s.translated());
            let vs = &stages[0].0.translated()?.host.dxil;
            match victory_shader::emit::rect_list_gs(vs, ps.map(|t| &t.host.dxil[..])) {
                Ok(gs) => rect_gs = Some(gs),
                Err(e) => tracing::warn!(error = %e, ?shaders, "no rectangle geometry stage"),
            }
        }
        // A pipeline with the geometry stage has a name of its own in the
        // library; the others keep theirs.
        let name = |gs: Option<&Vec<u8>>| match gs {
            Some(gs) => super::library::name(
                "graphics rectangles",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1)), &gs[..]],
                &state,
            ),
            None => super::library::name(
                "graphics",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            ),
        };
        let inputs = g.seam_inputs();
        let make = |gs: Option<&Vec<u8>>| {
            graphics_pipeline(
                &host.device,
                g,
                &root_blob,
                victory_host::Geometry::Vertex {
                    shader: stage(dxil(0)?),
                    inputs: &inputs,
                },
                dxil(1).map(stage),
                gs.map(|gs| stage(&gs[..])),
                depth_bounds,
                &name(gs),
            )
            .ok()
        };
        let mut made = make(rect_gs.as_ref());
        if made.is_none() && rect_gs.is_some() {
            // The host refused the geometry stage: once more without it.
            tracing::warn!(?shaders, "host pipeline refused with the rectangle geometry stage");
            host.log_messages();
            rect_gs = None;
            made = make(None);
        }
        let made = match made {
            Some(p) => p,
            None => {
                tracing::warn!(?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(?shaders, "host pipeline");
        crate::host::name(&made.state, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            depth_bounds,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            compute: false,
            mesh: false,
            host: made,
            stages: bound,
            name: format!("{shaders:?}"),
            slots: g.slots(),
            inputs: g.elements(),
            rects: rect_gs.is_some(),
            state: {
                let (d, r, b) = (&g.depth, &g.rasterizer, &g.blend.RenderTarget[0]);
                let face = |f: &D3D12_DEPTH_STENCILOP_DESC| {
                    format!(
                        "fail {} depth-fail {} pass {} func {}",
                        f.StencilFailOp.0, f.StencilDepthFailOp.0, f.StencilPassOp.0, f.StencilFunc.0
                    )
                };
                let bounds = if depth_bounds { " bounds true" } else { "" };
                format!(
                    "depth {} write {} func {}{bounds}; stencil {} read {:#x} write {:#x} front [{}] back [{}]; cull {} front-ccw {}; blend {} src {} dst {} op {} mask {:#x}",
                    d.DepthEnable.as_bool(),
                    d.DepthWriteMask.0,
                    d.DepthFunc.0,
                    d.StencilEnable.as_bool(),
                    d.StencilReadMask,
                    d.StencilWriteMask,
                    face(&d.FrontFace),
                    face(&d.BackFace),
                    r.CullMode.0,
                    r.FrontCounterClockwise.as_bool(),
                    b.BlendEnable.as_bool(),
                    b.SrcBlend.0,
                    b.DestBlend.0,
                    b.BlendOp.0,
                    b.RenderTargetWriteMask
                )
            },
        })
    }
}

/// A host graphics or mesh pipeline for the title's state `g` through the
/// host seam (`victory_host::Device::graphics_pipeline`): the D3D12
/// backend builds the same descriptions as before (plain, depth bounds
/// stream, mesh stream), stored in the pipeline library under `name`.
#[allow(clippy::too_many_arguments)]
fn graphics_pipeline(
    device: &ID3D12Device,
    g: &GraphicsState,
    root_blob: &[u8],
    geometry: victory_host::Geometry<'_>,
    pixel: Option<victory_host::Stage<'_>>,
    geometry_shader: Option<victory_host::Stage<'_>>,
    depth_bounds: bool,
    name: &str,
) -> windows::core::Result<crate::backend::D3d12Pipeline> {
    use victory_host::Device as _;
    let targets: Vec<u32> = g.targets.RTFormats[..(g.targets.NumRenderTargets as usize).min(8)]
        .iter()
        .map(|f| f.0 as u32)
        .collect();
    let (depth, stencil) = g.seam_depth();
    crate::backend::D3d12(device).graphics_pipeline(&victory_host::GraphicsDesc {
        geometry,
        pixel,
        root_signature: root_blob,
        targets: &targets,
        depth_format: g.depth_format.0 as u32,
        samples: g.samples.Count,
        sample_quality: g.samples.Quality,
        sample_mask: g.sample_mask,
        // The console's rectangle and quad types draw as triangles on
        // the host (see `draw`).
        topology: match g.topology.0 {
            RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0 as u32,
            t => t as u32,
        },
        strip_cut: g.strip_cut.0 as u32,
        blend: g.seam_blend(),
        independent_blend: g.blend.IndependentBlendEnable.as_bool(),
        alpha_to_coverage: g.blend.AlphaToCoverageEnable.as_bool(),
        raster: g.seam_raster(),
        depth,
        stencil,
        depth_bounds,
        geometry_shader,
        cache: Some(name),
    })
}

/// A translated stage for the host seam (victory-shader's entry is
/// `main`).
fn stage(code: &[u8]) -> victory_host::Stage<'_> {
    victory_host::Stage {
        code: victory_host::ShaderCode::Dxil(code),
        entry: "main",
    }
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }.to_vec()
}

/// The host root signature of `params` and `samplers`, serialized by
/// `victory_host::root` (byte for byte what D3D12's serializer returns,
/// the container checksum included): what backends make it from.
fn root_blob(
    params: Vec<victory_host::root::Parameter>,
    samplers: Vec<victory_host::root::StaticSampler>,
    compute: bool,
) -> Vec<u8> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = victory_host::root::RootSignature {
        version: victory_host::root::Version::V1_1,
        flags: (input
            | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
            | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED)
            .0 as u32,
        parameters: params,
        samplers,
    };
    desc.serialize()
}

/// [`root_blob`], made on `device`.
#[cfg(test)]
fn root_signature(
    device: &ID3D12Device,
    params: Vec<victory_host::root::Parameter>,
    samplers: Vec<victory_host::root::StaticSampler>,
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let bytes = root_blob(params, samplers, compute);
    // SAFETY: a serialized root signature; D3D12 validates it.
    let root = unsafe { device.CreateRootSignature(0, &bytes) }?;
    Ok((root, bytes))
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;

    use super::*;

    #[test]
    fn integer_target_bypasses_blending_and_preserves_mixed_target_masks() {
        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 3;
        g.targets.RTFormats[..3].copy_from_slice(&[
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_B4G4R4A4_UNORM,
            DXGI_FORMAT_R8_UINT,
        ]);
        g.blend.IndependentBlendEnable = true.into();
        for (i, target) in g.blend.RenderTarget[..3].iter_mut().enumerate() {
            target.BlendEnable = true.into();
            target.BlendOp = D3D12_BLEND_OP_MAX;
            target.RenderTargetWriteMask = [15, 15, 1][i];
        }
        let before = g.blend;
        g.apply_integer_blend_bypass();
        assert_eq!(g.blend.RenderTarget[0..2], before.RenderTarget[0..2]);
        let mut expected = before.RenderTarget[2];
        expected.BlendEnable = false.into();
        assert_eq!(g.blend.RenderTarget[2], expected);

        g.blend = before;
        g.blend.IndependentBlendEnable = false.into();
        g.blend.RenderTarget[0].RenderTargetWriteMask = 3;
        g.targets.RTFormats.swap(0, 2);
        g.apply_integer_blend_bypass();
        assert!(g.blend.IndependentBlendEnable.as_bool());
        assert!(!g.blend.RenderTarget[0].BlendEnable.as_bool());
        for target in &g.blend.RenderTarget[1..3] {
            assert!(target.BlendEnable.as_bool());
            assert_eq!(target.BlendOp, D3D12_BLEND_OP_MAX);
            assert_eq!(target.RenderTargetWriteMask, 3);
        }
    }

    /// A triangle mesh shader of our own and a pixel shader for it.
    const MS: &str = "struct V { float4 p : SV_Position; };
        [outputtopology(\"triangle\")] [numthreads(1, 1, 1)]
        void main(out vertices V v[3], out indices uint3 t[1]) {
            SetMeshOutputCounts(3, 1);
            v[0].p = float4(0, 0, 0, 1);
            v[1].p = float4(1, 0, 0, 1);
            v[2].p = float4(0, 1, 0, 1);
            t[0] = uint3(0, 1, 2);
        }";
    const PS: &str = "float4 main() : SV_Target { return 1; }";

    /// The PC `dxc.exe` (`VICTORY_PC_DXC`, else the newest Windows SDK's).
    fn dxc() -> Option<PathBuf> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => PathBuf::from(p),
            None => victory_shader::emit::validator::Validator::default_path()?
                .with_file_name("dxc.exe"),
        };
        dxc.exists().then_some(dxc)
    }

    fn compile(dxc: &Path, dir: &Path, src: &str, profile: &str) -> Vec<u8> {
        let (hlsl, out) = (
            dir.join(format!("{profile}.hlsl")),
            dir.join(format!("{profile}.dxil")),
        );
        std::fs::write(&hlsl, src).unwrap();
        let status = std::process::Command::new(dxc)
            .args(["-nologo", "-T", profile, "-E", "main", "-Fo"])
            .arg(&out)
            .arg(&hlsl)
            .status()
            .unwrap();
        assert!(status.success(), "dxc failed for {profile}");
        std::fs::read(out).unwrap()
    }

    /// A mesh pipeline made once is loaded from the library the next time,
    /// not compiled again (a stream without a topology never loads).
    #[test]
    fn a_mesh_pipeline_loads_from_the_library() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !host.mesh_shaders() {
            eprintln!("no host mesh shader support; skipped");
            return;
        }
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let ms = compile(&dxc, dir.path(), MS, "ms_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (_root, root_blob) = root_signature(&host.device, Vec::new(), Vec::new(), false).unwrap();
        let mut g = GraphicsState::default();
        // This pipeline has only a color target, with no depth buffer.
        g.depth.DepthEnable = false.into();
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let name = super::super::library::name("mesh test", &root_blob, &[&ms, &ps], &g.describe());
        let mesh = || {
            let geometry = victory_host::Geometry::Mesh {
                amplification: None,
                shader: stage(&ms),
            };
            graphics_pipeline(&host.device, &g, &root_blob, geometry, Some(stage(&ps)), None, false, &name)
        };
        mesh().expect("made");
        let loaded = super::super::library::LOADED.load(Ordering::Relaxed);
        mesh().expect("made again");
        assert!(
            super::super::library::LOADED.load(Ordering::Relaxed) > loaded,
            "the mesh pipeline was compiled again instead of loaded from the library"
        );
    }

    const VS: &str = "float4 main(uint i : SV_VertexID) : SV_Position {
        float2 p = float2((i << 1) & 2, i & 2);
        return float4(p * 2.0 - 1.0, 0.5, 1.0);
    }";

    /// A graphics pipeline with `DepthBoundsTestEnable` loads back from the
    /// pipeline library and culls or passes fragments according to
    /// `OMSetDepthBounds` (including across a command-list cut).
    #[test]
    fn depth_bounds_pipeline_loads_and_filters_draws() {
        let Some(host) = host() else {
            return;
        };
        if !host.depth_bounds() {
            eprintln!("no host depth bounds support; skipped");
            return;
        }
        // The software adapter says it takes the test and draws through it
        // all the same (depth 0.5 passes bounds of [0.6, 1.0]): the test is
        // for a card. A run that asked for the software adapter skips it, as
        // before. A runner whose only device is software skips it when the
        // CI-only variable asks; a desk that did not ask for the software
        // adapter and has no card still fails here, as it should.
        if crate::host::on_warp() {
            eprintln!("the software adapter does not apply depth bounds; skipped");
            return;
        }
        if host.is_software()
            && std::env::var("VICTORY_TESTS_ALLOW_SOFTWARE_SKIP").is_ok_and(|v| v == "1")
        {
            eprintln!("the software adapter does not apply depth bounds; skipped (VICTORY_TESTS_ALLOW_SOFTWARE_SKIP=1)");
            return;
        }
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let vs = compile(&dxc, dir.path(), VS, "vs_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (root, root_blob) = root_signature(&host.device, Vec::new(), Vec::new(), false).unwrap();

        let mut g = GraphicsState::default();
        // Subobject 21 (`DEPTH_STENCIL1`) with `DepthBoundsTestEnable = true`,
        // depth test itself off so only depth bounds gates the draw.
        #[repr(C, align(8))]
        struct Sub21(u32, D3D12_DEPTH_STENCIL_DESC1);
        let sub = Sub21(
            21,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthEnable: false.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                DepthBoundsTestEnable: true.into(),
                ..Default::default()
            },
        );
        // SAFETY: `sub` is a complete subobject 21.
        unsafe { g.read(21, (&raw const sub).cast()) };
        assert!(g.depth_bounds);
        assert!(g.describe().contains("depth-bounds"));
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        g.depth_format = DXGI_FORMAT_D32_FLOAT;

        g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        let name = super::super::library::name(
            "graphics depth-bounds test",
            &root_blob,
            &[&vs, &ps],
            &g.describe(),
        );
        let make = || {
            let geometry = victory_host::Geometry::Vertex {
                shader: stage(&vs),
                inputs: &[],
            };
            graphics_pipeline(&host.device, &g, &root_blob, geometry, Some(stage(&ps)), None, true, &name)
        };
        let pso = make().expect("depth-bounds PSO made").state;
        let loaded = super::super::library::LOADED.load(Ordering::Relaxed);
        make().expect("depth-bounds PSO made again");
        assert!(
            super::super::library::LOADED.load(Ordering::Relaxed) > loaded,
            "the depth-bounds pipeline was compiled again instead of loaded from the library"
        );

        // Execute two draws against a depth buffer cleared to 0.5:
        // 1) bounds [0.6, 1.0] -> rejected (target stays 0)
        // 2) bounds [0.4, 0.6] replayed onto a fresh HostList -> passes (target becomes 255)
        let make_tex = |format, flags, state, clear: D3D12_CLEAR_VALUE| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: flags,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: valid heap and resource descriptions.
            unsafe {
                host.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, state, Some(&clear), &mut out)
                    .unwrap();
            }
            out.unwrap()
        };
        let color = make_tex(
            DXGI_FORMAT_R8G8B8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
            },
        );
        let depth = make_tex(
            DXGI_FORMAT_D32_FLOAT,
            D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_D32_FLOAT,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                        Depth: 0.5,
                        Stencil: 0,
                    },
                },
            },
        );
        let readback = {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 1024,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            // SAFETY: valid readback buffer description.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &rd,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut out,
                    )
                    .unwrap();
            }
            out.unwrap()
        };

        let run_draw = |min: f32, max: f32| -> u8 {
            let mut state = crate::draw::State::default();
            let first_list = host.new_list().unwrap();
            state.set_depth_bounds(&first_list, min, max);
            first_list.close();
            // Replay onto a second host list, exercising `State::replay`.
            let list = host.new_list().unwrap();
            state.replay(&list);
            let rtv = host.rtv(Some(&color), None);
            let dsv = host.dsv(&depth, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: 4.0,
                Height: 4.0,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            let rect = windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: 4,
                bottom: 4,
            };
            // SAFETY: open command list, live resources and views.
            unsafe {
                list.list.ClearRenderTargetView(rtv, &[0.0; 4], None);
                list.list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 0.5, 0, None);
                list.list.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
                list.list.RSSetViewports(&[vp]);
                list.list.RSSetScissorRects(&[rect]);
                list.list.SetGraphicsRootSignature(&root);
                list.list.SetPipelineState(&pso);
                list.list
                    .IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                list.list.DrawInstanced(3, 1, 0, 0);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &color,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&readback),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: 4,
                                Height: 4,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&color),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.list.ResourceBarrier(&[crate::host::transition(
                    &color,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let mut ptr = std::ptr::null_mut();
            // SAFETY: `execute` waited for the GPU; readback holds 1024 bytes.
            unsafe {
                readback.Map(0, None, Some(&mut ptr)).unwrap();
                let pixel = *ptr.cast::<u8>();
                readback.Unmap(0, None);
                pixel
            }
        };

        assert_eq!(run_draw(0.6, 1.0), 0, "depth 0.5 outside [0.6, 1.0] must be rejected");
        assert_eq!(run_draw(0.4, 0.6), 255, "depth 0.5 inside [0.4, 0.6] must pass");
    }
}
