//! Graphics pipelines on Metal, from DXIL through Metal Shader Converter:
//! vertex + pixel shaders (Metal vertex fetch from the input layout), or
//! mesh (+ amplification as Metal's object stage) + pixel.
//!
//! Every stage is converted with the same global root signature, so all
//! of them read one top-level argument buffer, bound at
//! `irconverter::ARGUMENT_BUFFER_INDEX` on each stage at draw time.
//! Vertex buffers bind at `irconverter::VERTEX_BUFFER_INDEX + slot` with
//! dynamic strides (given per draw, as D3D12 does).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLColorWriteMask, MTLCompareFunction,
    MTLDepthStencilDescriptor, MTLDepthStencilState, MTLDevice, MTLFunction,
    MTLMeshRenderPipelineDescriptor, MTLPipelineOption, MTLPixelFormat, MTLPrimitiveTopologyClass,
    MTLRenderPipelineColorAttachmentDescriptor, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLRenderStages, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};
use saxaboom::ffi::IRShaderStage;
use victory_host::{Blend, DepthTest, Geometry, GraphicsDesc, Raster, ShaderCode, Stage};

use crate::MetalError;
use crate::device::{MetalDevice, StaticSamplers, c_entry, converted_entry, root_signature_for};
use crate::irconverter::{self, Converted, VERTEX_BUFFER_INDEX};
use crate::toplevel::TopLevel;
use crate::{format, vertex};

/// How a graphics pipeline draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Vertex + pixel; `draw_params`: the vertex shader reads draw
    /// parameters, given at MSC's draw-argument and uniform buffers.
    Vertex { draw_params: bool },
    /// Mesh (+ amplification): threads per object (amplification) and
    /// mesh group; `amplified` when there is an amplification stage.
    Mesh {
        object: [u32; 3],
        mesh: [u32; 3],
        amplified: bool,
    },
}

pub struct MetalGraphicsPipeline {
    pub state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub depth: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    pub raster: Raster,
    pub top: TopLevel,
    pub kind: Kind,
    /// The stages a draw binds the top-level argument buffer on, and
    /// makes resources resident for.
    pub stages: MTLRenderStages,
    /// The root signature's static samplers, if it has any.
    pub statics: Option<StaticSamplers>,
}

fn dxil<'a>(stage: &Stage<'a>) -> Result<&'a [u8], MetalError> {
    match stage.code {
        ShaderCode::Dxil(d) => Ok(d),
        _ => Err(MetalError::Shader(
            "graphics pipelines take DXIL (converted with Metal Shader Converter)".into(),
        )),
    }
}

fn convert(
    stage: &Stage<'_>,
    rs: &victory_host::root::RootSignature,
    want: IRShaderStage,
) -> Result<Converted, MetalError> {
    let c = irconverter::convert(dxil(stage)?, &c_entry(stage.entry)?, Some(rs))?;
    if c.stage != want {
        return Err(MetalError::Shader(format!(
            "{} is a {:?} shader, not {want:?}",
            stage.entry, c.stage
        )));
    }
    Ok(c)
}

fn blend_factor(d3d12: u32) -> MTLBlendFactor {
    match d3d12 {
        1 => MTLBlendFactor::Zero,
        3 => MTLBlendFactor::SourceColor,
        4 => MTLBlendFactor::OneMinusSourceColor,
        5 => MTLBlendFactor::SourceAlpha,
        6 => MTLBlendFactor::OneMinusSourceAlpha,
        7 => MTLBlendFactor::DestinationAlpha,
        8 => MTLBlendFactor::OneMinusDestinationAlpha,
        9 => MTLBlendFactor::DestinationColor,
        10 => MTLBlendFactor::OneMinusDestinationColor,
        11 => MTLBlendFactor::SourceAlphaSaturated,
        14 => MTLBlendFactor::BlendColor,
        15 => MTLBlendFactor::OneMinusBlendColor,
        16 => MTLBlendFactor::Source1Color,
        17 => MTLBlendFactor::OneMinusSource1Color,
        18 => MTLBlendFactor::Source1Alpha,
        19 => MTLBlendFactor::OneMinusSource1Alpha,
        20 => MTLBlendFactor::BlendAlpha,
        21 => MTLBlendFactor::OneMinusBlendAlpha,
        _ => MTLBlendFactor::One,
    }
}

fn blend_op(d3d12: u32) -> MTLBlendOperation {
    match d3d12 {
        2 => MTLBlendOperation::Subtract,
        3 => MTLBlendOperation::ReverseSubtract,
        4 => MTLBlendOperation::Min,
        5 => MTLBlendOperation::Max,
        _ => MTLBlendOperation::Add,
    }
}

/// D3D12's write mask bits (red 1 .. alpha 8) as Metal's (red 8 .. alpha 1).
fn write_mask(d3d12: u8) -> MTLColorWriteMask {
    let mut m = MTLColorWriteMask::empty();
    for (bit, mtl) in [
        (1, MTLColorWriteMask::Red),
        (2, MTLColorWriteMask::Green),
        (4, MTLColorWriteMask::Blue),
        (8, MTLColorWriteMask::Alpha),
    ] {
        if d3d12 & bit != 0 {
            m |= mtl;
        }
    }
    m
}

pub(crate) fn compare(d3d12: u32) -> MTLCompareFunction {
    match d3d12 {
        1 => MTLCompareFunction::Never,
        2 => MTLCompareFunction::Less,
        3 => MTLCompareFunction::Equal,
        4 => MTLCompareFunction::LessEqual,
        5 => MTLCompareFunction::Greater,
        6 => MTLCompareFunction::NotEqual,
        7 => MTLCompareFunction::GreaterEqual,
        _ => MTLCompareFunction::Always,
    }
}

fn set_blend(a: &MTLRenderPipelineColorAttachmentDescriptor, pixel: MTLPixelFormat, b: &Blend) {
    a.setPixelFormat(pixel);
    a.setBlendingEnabled(b.enable);
    a.setSourceRGBBlendFactor(blend_factor(b.src));
    a.setDestinationRGBBlendFactor(blend_factor(b.dst));
    a.setRgbBlendOperation(blend_op(b.op));
    a.setSourceAlphaBlendFactor(blend_factor(b.src_alpha));
    a.setDestinationAlphaBlendFactor(blend_factor(b.dst_alpha));
    a.setAlphaBlendOperation(blend_op(b.op_alpha));
    a.setWriteMask(write_mask(b.write_mask));
}

fn pixel_format(dxgi: u32, depth: bool) -> Result<MTLPixelFormat, MetalError> {
    format::map(dxgi, depth)
        .map(|m| MTLPixelFormat(m.pixel_format))
        .ok_or(MetalError::Format(dxgi))
}

fn has_stencil(f: MTLPixelFormat) -> bool {
    f == MTLPixelFormat::Depth32Float_Stencil8
        || f == MTLPixelFormat::Stencil8
        || f == MTLPixelFormat::X32_Stencil8
}

fn depth_state(
    device: &ProtocolObject<dyn MTLDevice>,
    d: &DepthTest,
) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>, MetalError> {
    let s = MTLDepthStencilDescriptor::new();
    s.setDepthCompareFunction(if d.enable {
        compare(d.func)
    } else {
        MTLCompareFunction::Always
    });
    s.setDepthWriteEnabled(d.enable && d.write);
    device
        .newDepthStencilStateWithDescriptor(&s)
        .ok_or(MetalError::Failed("a depth-stencil state"))
}

/// The vertex descriptor for `inputs`, fed to the attributes `converted`
/// reflects. MSC's stage-in attributes start at
/// `kIRStageInAttributeStartIndex` (11); reflection's indices are taken as
/// is when the function itself uses them, else shifted by 11.
fn vertex_descriptor(
    inputs: &[victory_host::InputElement<'_>],
    converted: &Converted,
    function: &ProtocolObject<dyn MTLFunction>,
) -> Result<Retained<MTLVertexDescriptor>, MetalError> {
    const STAGE_IN_START: u32 = 11;
    let offsets = vertex::offsets(inputs).ok_or_else(|| {
        MetalError::Shader("an input element's format has no Metal vertex format".into())
    })?;
    let used: Vec<u32> = function
        .vertexAttributes()
        .map(|a| {
            a.iter()
                .filter(|a| a.isActive())
                .map(|a| a.attributeIndex() as u32)
                .collect()
        })
        .unwrap_or_default();
    let reflected: Vec<(String, u32)> = converted.vertex_inputs.clone();
    let as_is = reflected
        .iter()
        .all(|(_, i)| used.is_empty() || used.contains(i));
    let shift = if as_is { 0 } else { STAGE_IN_START };
    let d = MTLVertexDescriptor::new();
    let mut slots = Vec::new();
    for ((e, at), attribute) in inputs
        .iter()
        .zip(&offsets)
        .zip(vertex::attributes(inputs, &reflected))
    {
        let Some(attribute) = attribute else { continue };
        let (fmt, _) = vertex::format(e.format).ok_or(MetalError::Format(e.format))?;
        // SAFETY: attribute and buffer indices are below Metal's limits
        // (31 attributes from MSC, slots below 31 - 6 by D3D12's 16).
        unsafe {
            let a = d
                .attributes()
                .objectAtIndexedSubscript((attribute + shift) as usize);
            a.setFormat(MTLVertexFormat(fmt));
            a.setOffset(*at as usize);
            a.setBufferIndex(VERTEX_BUFFER_INDEX + e.slot as usize);
        }
        if !slots.contains(&(e.slot, e.per_instance, e.step_rate)) {
            slots.push((e.slot, e.per_instance, e.step_rate));
        }
    }
    for (slot, per_instance, rate) in slots {
        // SAFETY: as above; a dynamic stride is set per draw.
        unsafe {
            let l = d
                .layouts()
                .objectAtIndexedSubscript(VERTEX_BUFFER_INDEX + slot as usize);
            l.setStride(objc2_metal::MTLBufferLayoutStrideDynamic);
            if per_instance {
                l.setStepFunction(MTLVertexStepFunction::PerInstance);
                l.setStepRate(rate.max(1) as usize);
            } else {
                l.setStepFunction(MTLVertexStepFunction::PerVertex);
                l.setStepRate(1);
            }
        }
    }
    Ok(d)
}

pub(crate) fn pipeline(
    dev: &MetalDevice,
    desc: &GraphicsDesc<'_>,
) -> Result<MetalGraphicsPipeline, MetalError> {
    let shader = |e: Retained<objc2_foundation::NSError>| {
        MetalError::Shader(e.localizedDescription().to_string())
    };
    let first = match &desc.geometry {
        Geometry::Vertex { shader, .. } | Geometry::Mesh { shader, .. } => shader,
    };
    let (rs, _) = root_signature_for(desc.root_signature, dxil(first)?)?;
    let top = TopLevel::new(&rs).map_err(|e| MetalError::Shader(e.to_string()))?;
    let statics = dev.static_samplers(&rs)?;
    let function =
        |c: &Converted| dev.function(ShaderCode::MetalLib(&c.metallib), converted_entry());
    let fragment = desc
        .pixel
        .as_ref()
        .map(|p| convert(p, &rs, IRShaderStage::Fragment))
        .transpose()?;
    let fragment_fn = fragment.as_ref().map(&function).transpose()?;
    let targets: Vec<MTLPixelFormat> = desc
        .targets
        .iter()
        .map(|&f| pixel_format(f, false))
        .collect::<Result<_, _>>()?;
    let depth_format = if desc.depth_format == 0 {
        MTLPixelFormat::Invalid
    } else {
        pixel_format(desc.depth_format, true)?
    };
    let stencil_format = if has_stencil(depth_format) {
        depth_format
    } else {
        MTLPixelFormat::Invalid
    };
    let samples = desc.samples.max(1) as usize;
    let device = dev.raw();
    let (state, kind, mut stages) = match &desc.geometry {
        Geometry::Vertex { shader: vs, inputs } => {
            let v = convert(vs, &rs, IRShaderStage::Vertex)?;
            let vertex_fn = function(&v)?;
            let d = MTLRenderPipelineDescriptor::new();
            d.setVertexFunction(Some(&vertex_fn));
            d.setFragmentFunction(fragment_fn.as_deref());
            if !inputs.is_empty() {
                let layout = vertex_descriptor(inputs, &v, &vertex_fn)?;
                d.setVertexDescriptor(Some(&layout));
            }
            for (i, (f, b)) in targets.iter().zip(&desc.blend).enumerate() {
                // SAFETY: at most 8 targets (GraphicsDesc::blend's length).
                let a = unsafe { d.colorAttachments().objectAtIndexedSubscript(i) };
                set_blend(&a, *f, b);
            }
            d.setDepthAttachmentPixelFormat(depth_format);
            d.setStencilAttachmentPixelFormat(stencil_format);
            d.setRasterSampleCount(samples);
            let class = match desc.topology {
                1 => MTLPrimitiveTopologyClass::Point,
                2 => MTLPrimitiveTopologyClass::Line,
                3 => MTLPrimitiveTopologyClass::Triangle,
                _ => MTLPrimitiveTopologyClass::Unspecified,
            };
            // SAFETY: a valid class.
            unsafe { d.setInputPrimitiveTopology(class) };
            let state = device
                .newRenderPipelineStateWithDescriptor_error(&d)
                .map_err(shader)?;
            (
                state,
                Kind::Vertex {
                    draw_params: v.needs_draw_params,
                },
                MTLRenderStages::Vertex,
            )
        }
        Geometry::Mesh {
            amplification,
            shader: ms,
        } => {
            let m = convert(ms, &rs, IRShaderStage::Mesh)?;
            let a = amplification
                .as_ref()
                .map(|a| convert(a, &rs, IRShaderStage::Amplification))
                .transpose()?;
            let mesh_fn = function(&m)?;
            let object_fn = a.as_ref().map(&function).transpose()?;
            let d = MTLMeshRenderPipelineDescriptor::new();
            // SAFETY: functions of the right stages (checked above).
            unsafe {
                d.setObjectFunction(object_fn.as_deref());
                d.setMeshFunction(Some(&mesh_fn));
                d.setFragmentFunction(fragment_fn.as_deref());
                d.setRasterSampleCount(samples);
            }
            if let Some(a) = &a {
                d.setPayloadMemoryLength(a.payload as usize);
            }
            for (i, (f, b)) in targets.iter().zip(&desc.blend).enumerate() {
                // SAFETY: as above.
                let a = unsafe { d.colorAttachments().objectAtIndexedSubscript(i) };
                set_blend(&a, *f, b);
            }
            d.setDepthAttachmentPixelFormat(depth_format);
            d.setStencilAttachmentPixelFormat(stencil_format);
            let state = device
                .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                    &d,
                    MTLPipelineOption::None,
                    None,
                )
                .map_err(shader)?;
            let stages = if a.is_some() {
                MTLRenderStages::Object | MTLRenderStages::Mesh
            } else {
                MTLRenderStages::Mesh
            };
            (
                state,
                Kind::Mesh {
                    object: a.as_ref().map_or([1, 1, 1], |a| a.threads),
                    mesh: m.threads,
                    amplified: a.is_some(),
                },
                stages,
            )
        }
    };
    if fragment.is_some() {
        stages |= MTLRenderStages::Fragment;
    }
    Ok(MetalGraphicsPipeline {
        state,
        depth: depth_state(device, &desc.depth)?,
        raster: desc.raster,
        top,
        kind,
        stages,
        statics,
    })
}
