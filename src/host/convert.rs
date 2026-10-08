//! Conversions between the C API's enums and structs and wgpu's.
//!
//! `webgpu.h` mirrors the WebGPU JavaScript API, and so does wgpu, so most of
//! this is one-to-one. The interesting parts are the "undefined" values the C
//! API uses for defaults, and the handful of things wgpu spells differently.

use crate::abi::enums::{self as c, FeatureName};
use crate::abi::structs as s;

// -- features -------------------------------------------------------------------

/// The wgpu feature bits behind a WebGPU feature name. `Some(empty)` means the
/// feature is always available; `None` that this backend cannot provide it.
pub(crate) fn feature_to_wgpu(feature: FeatureName) -> Option<wgpu::Features> {
    use wgpu::Features as F;
    Some(match feature {
        FeatureName::CoreFeaturesAndLimits => F::empty(),
        FeatureName::DepthClipControl => F::DEPTH_CLIP_CONTROL,
        FeatureName::Depth32FloatStencil8 => F::DEPTH32FLOAT_STENCIL8,
        FeatureName::TextureCompressionBC => F::TEXTURE_COMPRESSION_BC,
        FeatureName::TextureCompressionBCSliced3D => F::TEXTURE_COMPRESSION_BC_SLICED_3D,
        FeatureName::TextureCompressionETC2 => F::TEXTURE_COMPRESSION_ETC2,
        FeatureName::TextureCompressionASTC => F::TEXTURE_COMPRESSION_ASTC,
        FeatureName::TextureCompressionASTCSliced3D => F::TEXTURE_COMPRESSION_ASTC_SLICED_3D,
        FeatureName::TimestampQuery => F::TIMESTAMP_QUERY,
        FeatureName::IndirectFirstInstance => F::INDIRECT_FIRST_INSTANCE,
        FeatureName::ShaderF16 => F::SHADER_F16,
        FeatureName::RG11B10UfloatRenderable => F::RG11B10UFLOAT_RENDERABLE,
        FeatureName::BGRA8UnormStorage => F::BGRA8UNORM_STORAGE,
        FeatureName::Float32Filterable => F::FLOAT32_FILTERABLE,
        FeatureName::Float32Blendable => F::FLOAT32_BLENDABLE,
        FeatureName::ClipDistances => F::CLIP_DISTANCES,
        FeatureName::DualSourceBlending => F::DUAL_SOURCE_BLENDING,
        FeatureName::Subgroups => F::SUBGROUP,
        FeatureName::PrimitiveIndex => F::PRIMITIVE_INDEX,
        // wgpu has no equivalent for these yet.
        FeatureName::TextureFormatsTier1
        | FeatureName::TextureFormatsTier2
        | FeatureName::TextureComponentSwizzle
        | FeatureName::SubgroupSizeControl
        | FeatureName::TextureCompressionUnaligned
        | FeatureName::AtomicVec2uMinMax => return None,
    })
}

const ALL_FEATURES: &[FeatureName] = &[
    FeatureName::CoreFeaturesAndLimits,
    FeatureName::DepthClipControl,
    FeatureName::Depth32FloatStencil8,
    FeatureName::TextureCompressionBC,
    FeatureName::TextureCompressionBCSliced3D,
    FeatureName::TextureCompressionETC2,
    FeatureName::TextureCompressionASTC,
    FeatureName::TextureCompressionASTCSliced3D,
    FeatureName::TimestampQuery,
    FeatureName::IndirectFirstInstance,
    FeatureName::ShaderF16,
    FeatureName::RG11B10UfloatRenderable,
    FeatureName::BGRA8UnormStorage,
    FeatureName::Float32Filterable,
    FeatureName::Float32Blendable,
    FeatureName::ClipDistances,
    FeatureName::DualSourceBlending,
    FeatureName::Subgroups,
    FeatureName::PrimitiveIndex,
];

/// The WebGPU feature names a set of wgpu features amounts to.
pub(crate) fn features_to_abi(features: wgpu::Features) -> Vec<FeatureName> {
    ALL_FEATURES
        .iter()
        .copied()
        .filter(|&name| feature_to_wgpu(name).is_some_and(|bits| features.contains(bits)))
        .collect()
}

/// Features the host turns on by itself whenever the adapter has them,
/// because `webgpu.h` exposes what they gate as core API.
pub(crate) fn implicit_features() -> wgpu::Features {
    wgpu::Features::IMMEDIATES
}

// -- limits ---------------------------------------------------------------------

macro_rules! standard_limits {
    ($apply:ident) => {
        $apply!(
            max_texture_dimension_1d,
            max_texture_dimension_2d,
            max_texture_dimension_3d,
            max_texture_array_layers,
            max_bind_groups,
            max_bindings_per_bind_group,
            max_dynamic_uniform_buffers_per_pipeline_layout,
            max_dynamic_storage_buffers_per_pipeline_layout,
            max_sampled_textures_per_shader_stage,
            max_samplers_per_shader_stage,
            max_storage_buffers_per_shader_stage,
            max_storage_textures_per_shader_stage,
            max_uniform_buffers_per_shader_stage,
            max_uniform_buffer_binding_size,
            max_storage_buffer_binding_size,
            min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment,
            max_vertex_buffers,
            max_buffer_size,
            max_vertex_attributes,
            max_vertex_buffer_array_stride,
            max_inter_stage_shader_variables,
            max_color_attachments,
            max_color_attachment_bytes_per_sample,
            max_compute_workgroup_storage_size,
            max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x,
            max_compute_workgroup_size_y,
            max_compute_workgroup_size_z,
            max_compute_workgroups_per_dimension,
            max_immediate_size
        )
    };
}

pub(crate) fn limits_to_abi(limits: &wgpu::Limits) -> s::Limits {
    macro_rules! copy {
        ($($field:ident),*) => {
            s::Limits {
                chain: s::Chain::default(),
                // wgpu has no combined limit: each kind is only bounded by its
                // own. The specification's default (24) is the floor.
                max_bind_groups_plus_vertex_buffers: (limits.max_bind_groups + limits.max_vertex_buffers)
                    .max(24),
                $( $field: limits.$field as _, )*
            }
        };
    }
    standard_limits!(copy)
}

trait Undefined: Copy {
    fn is_undefined(self) -> bool;
}
impl Undefined for u32 {
    fn is_undefined(self) -> bool {
        self == c::LIMIT_U32_UNDEFINED
    }
}
impl Undefined for u64 {
    fn is_undefined(self) -> bool {
        self == c::LIMIT_U64_UNDEFINED
    }
}

/// Applies the limits a guest requires on top of `base`. A member left at its
/// "undefined" sentinel keeps the default, as in the JavaScript API.
pub(crate) fn limits_from_abi(required: &s::Limits, base: wgpu::Limits) -> wgpu::Limits {
    let mut limits = base;
    macro_rules! apply {
        ($($field:ident),*) => {
            $(
                if !required.$field.is_undefined() {
                    limits.$field = required.$field as _;
                }
            )*
        };
    }
    standard_limits!(apply);
    limits
}

// -- adapter info ---------------------------------------------------------------

pub(crate) fn backend_type(backend: wgpu::Backend) -> c::BackendType {
    match backend {
        wgpu::Backend::Noop => c::BackendType::Null,
        wgpu::Backend::Vulkan => c::BackendType::Vulkan,
        wgpu::Backend::Metal => c::BackendType::Metal,
        wgpu::Backend::Dx12 => c::BackendType::D3D12,
        wgpu::Backend::Gl => c::BackendType::OpenGL,
        wgpu::Backend::BrowserWebGpu => c::BackendType::WebGPU,
    }
}

pub(crate) fn backends_of(backend: c::BackendType) -> Option<wgpu::Backends> {
    Some(match backend {
        c::BackendType::Undefined => return None,
        c::BackendType::Null => wgpu::Backends::NOOP,
        c::BackendType::Vulkan => wgpu::Backends::VULKAN,
        c::BackendType::Metal => wgpu::Backends::METAL,
        c::BackendType::D3D12 => wgpu::Backends::DX12,
        c::BackendType::OpenGL | c::BackendType::OpenGLES => wgpu::Backends::GL,
        c::BackendType::WebGPU => wgpu::Backends::BROWSER_WEBGPU,
        c::BackendType::D3D11 => wgpu::Backends::empty(),
    })
}

pub(crate) fn adapter_type(ty: wgpu::DeviceType) -> c::AdapterType {
    match ty {
        wgpu::DeviceType::DiscreteGpu => c::AdapterType::DiscreteGPU,
        wgpu::DeviceType::IntegratedGpu => c::AdapterType::IntegratedGPU,
        wgpu::DeviceType::Cpu => c::AdapterType::CPU,
        wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => c::AdapterType::Unknown,
    }
}

/// The lowercase vendor name WebGPU reports, from a PCI vendor id.
pub(crate) fn vendor_name(vendor: u32) -> &'static str {
    match vendor {
        0x1002 | 0x1022 => "amd",
        0x106B => "apple",
        0x13B5 => "arm",
        0x1AE0 => "google",
        0x1010 => "img-tec",
        0x8086 => "intel",
        0x10005 => "mesa",
        0x1414 => "microsoft",
        0x10DE => "nvidia",
        0x5143 => "qualcomm",
        0x144D => "samsung",
        _ => "",
    }
}

// -- plain enums ----------------------------------------------------------------

pub(crate) fn power_preference(value: c::PowerPreference) -> wgpu::PowerPreference {
    match value {
        c::PowerPreference::Undefined => wgpu::PowerPreference::None,
        c::PowerPreference::LowPower => wgpu::PowerPreference::LowPower,
        c::PowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
    }
}

pub(crate) fn texture_format(format: c::TextureFormat) -> Option<wgpu::TextureFormat> {
    use c::TextureFormat as C;
    use wgpu::TextureFormat as W;
    use wgpu::{AstcBlock as B, AstcChannel as Ch};
    let astc = |block, channel| W::Astc { block, channel };
    Some(match format {
        C::Undefined => return None,
        C::R8Unorm => W::R8Unorm,
        C::R8Snorm => W::R8Snorm,
        C::R8Uint => W::R8Uint,
        C::R8Sint => W::R8Sint,
        C::R16Unorm => W::R16Unorm,
        C::R16Snorm => W::R16Snorm,
        C::R16Uint => W::R16Uint,
        C::R16Sint => W::R16Sint,
        C::R16Float => W::R16Float,
        C::RG8Unorm => W::Rg8Unorm,
        C::RG8Snorm => W::Rg8Snorm,
        C::RG8Uint => W::Rg8Uint,
        C::RG8Sint => W::Rg8Sint,
        C::R32Float => W::R32Float,
        C::R32Uint => W::R32Uint,
        C::R32Sint => W::R32Sint,
        C::RG16Unorm => W::Rg16Unorm,
        C::RG16Snorm => W::Rg16Snorm,
        C::RG16Uint => W::Rg16Uint,
        C::RG16Sint => W::Rg16Sint,
        C::RG16Float => W::Rg16Float,
        C::RGBA8Unorm => W::Rgba8Unorm,
        C::RGBA8UnormSrgb => W::Rgba8UnormSrgb,
        C::RGBA8Snorm => W::Rgba8Snorm,
        C::RGBA8Uint => W::Rgba8Uint,
        C::RGBA8Sint => W::Rgba8Sint,
        C::BGRA8Unorm => W::Bgra8Unorm,
        C::BGRA8UnormSrgb => W::Bgra8UnormSrgb,
        C::RGB10A2Uint => W::Rgb10a2Uint,
        C::RGB10A2Unorm => W::Rgb10a2Unorm,
        C::RG11B10Ufloat => W::Rg11b10Ufloat,
        C::RGB9E5Ufloat => W::Rgb9e5Ufloat,
        C::RG32Float => W::Rg32Float,
        C::RG32Uint => W::Rg32Uint,
        C::RG32Sint => W::Rg32Sint,
        C::RGBA16Unorm => W::Rgba16Unorm,
        C::RGBA16Snorm => W::Rgba16Snorm,
        C::RGBA16Uint => W::Rgba16Uint,
        C::RGBA16Sint => W::Rgba16Sint,
        C::RGBA16Float => W::Rgba16Float,
        C::RGBA32Float => W::Rgba32Float,
        C::RGBA32Uint => W::Rgba32Uint,
        C::RGBA32Sint => W::Rgba32Sint,
        C::Stencil8 => W::Stencil8,
        C::Depth16Unorm => W::Depth16Unorm,
        C::Depth24Plus => W::Depth24Plus,
        C::Depth24PlusStencil8 => W::Depth24PlusStencil8,
        C::Depth32Float => W::Depth32Float,
        C::Depth32FloatStencil8 => W::Depth32FloatStencil8,
        C::BC1RGBAUnorm => W::Bc1RgbaUnorm,
        C::BC1RGBAUnormSrgb => W::Bc1RgbaUnormSrgb,
        C::BC2RGBAUnorm => W::Bc2RgbaUnorm,
        C::BC2RGBAUnormSrgb => W::Bc2RgbaUnormSrgb,
        C::BC3RGBAUnorm => W::Bc3RgbaUnorm,
        C::BC3RGBAUnormSrgb => W::Bc3RgbaUnormSrgb,
        C::BC4RUnorm => W::Bc4RUnorm,
        C::BC4RSnorm => W::Bc4RSnorm,
        C::BC5RGUnorm => W::Bc5RgUnorm,
        C::BC5RGSnorm => W::Bc5RgSnorm,
        C::BC6HRGBUfloat => W::Bc6hRgbUfloat,
        C::BC6HRGBFloat => W::Bc6hRgbFloat,
        C::BC7RGBAUnorm => W::Bc7RgbaUnorm,
        C::BC7RGBAUnormSrgb => W::Bc7RgbaUnormSrgb,
        C::ETC2RGB8Unorm => W::Etc2Rgb8Unorm,
        C::ETC2RGB8UnormSrgb => W::Etc2Rgb8UnormSrgb,
        C::ETC2RGB8A1Unorm => W::Etc2Rgb8A1Unorm,
        C::ETC2RGB8A1UnormSrgb => W::Etc2Rgb8A1UnormSrgb,
        C::ETC2RGBA8Unorm => W::Etc2Rgba8Unorm,
        C::ETC2RGBA8UnormSrgb => W::Etc2Rgba8UnormSrgb,
        C::EACR11Unorm => W::EacR11Unorm,
        C::EACR11Snorm => W::EacR11Snorm,
        C::EACRG11Unorm => W::EacRg11Unorm,
        C::EACRG11Snorm => W::EacRg11Snorm,
        C::ASTC4x4Unorm => astc(B::B4x4, Ch::Unorm),
        C::ASTC4x4UnormSrgb => astc(B::B4x4, Ch::UnormSrgb),
        C::ASTC5x4Unorm => astc(B::B5x4, Ch::Unorm),
        C::ASTC5x4UnormSrgb => astc(B::B5x4, Ch::UnormSrgb),
        C::ASTC5x5Unorm => astc(B::B5x5, Ch::Unorm),
        C::ASTC5x5UnormSrgb => astc(B::B5x5, Ch::UnormSrgb),
        C::ASTC6x5Unorm => astc(B::B6x5, Ch::Unorm),
        C::ASTC6x5UnormSrgb => astc(B::B6x5, Ch::UnormSrgb),
        C::ASTC6x6Unorm => astc(B::B6x6, Ch::Unorm),
        C::ASTC6x6UnormSrgb => astc(B::B6x6, Ch::UnormSrgb),
        C::ASTC8x5Unorm => astc(B::B8x5, Ch::Unorm),
        C::ASTC8x5UnormSrgb => astc(B::B8x5, Ch::UnormSrgb),
        C::ASTC8x6Unorm => astc(B::B8x6, Ch::Unorm),
        C::ASTC8x6UnormSrgb => astc(B::B8x6, Ch::UnormSrgb),
        C::ASTC8x8Unorm => astc(B::B8x8, Ch::Unorm),
        C::ASTC8x8UnormSrgb => astc(B::B8x8, Ch::UnormSrgb),
        C::ASTC10x5Unorm => astc(B::B10x5, Ch::Unorm),
        C::ASTC10x5UnormSrgb => astc(B::B10x5, Ch::UnormSrgb),
        C::ASTC10x6Unorm => astc(B::B10x6, Ch::Unorm),
        C::ASTC10x6UnormSrgb => astc(B::B10x6, Ch::UnormSrgb),
        C::ASTC10x8Unorm => astc(B::B10x8, Ch::Unorm),
        C::ASTC10x8UnormSrgb => astc(B::B10x8, Ch::UnormSrgb),
        C::ASTC10x10Unorm => astc(B::B10x10, Ch::Unorm),
        C::ASTC10x10UnormSrgb => astc(B::B10x10, Ch::UnormSrgb),
        C::ASTC12x10Unorm => astc(B::B12x10, Ch::Unorm),
        C::ASTC12x10UnormSrgb => astc(B::B12x10, Ch::UnormSrgb),
        C::ASTC12x12Unorm => astc(B::B12x12, Ch::Unorm),
        C::ASTC12x12UnormSrgb => astc(B::B12x12, Ch::UnormSrgb),
    })
}

/// The C enum value of a wgpu format, or `Undefined` for formats `webgpu.h`
/// has no name for.
pub(crate) fn texture_format_to_abi(format: wgpu::TextureFormat) -> c::TextureFormat {
    use crate::abi::AbiEnum;
    // The forward table is the single source of truth; formats are few enough
    // that a reverse scan over the C enum's value range is fine.
    (1..=c::TextureFormat::ASTC12x12UnormSrgb.as_u32())
        .filter_map(c::TextureFormat::from_u32)
        .find(|&candidate| texture_format(candidate) == Some(format))
        .unwrap_or(c::TextureFormat::Undefined)
}

pub(crate) fn texture_dimension(value: c::TextureDimension) -> wgpu::TextureDimension {
    match value {
        c::TextureDimension::D1 => wgpu::TextureDimension::D1,
        c::TextureDimension::Undefined | c::TextureDimension::D2 => wgpu::TextureDimension::D2,
        c::TextureDimension::D3 => wgpu::TextureDimension::D3,
    }
}

pub(crate) fn texture_dimension_to_abi(value: wgpu::TextureDimension) -> c::TextureDimension {
    match value {
        wgpu::TextureDimension::D1 => c::TextureDimension::D1,
        wgpu::TextureDimension::D2 => c::TextureDimension::D2,
        wgpu::TextureDimension::D3 => c::TextureDimension::D3,
    }
}

pub(crate) fn view_dimension(value: c::TextureViewDimension) -> Option<wgpu::TextureViewDimension> {
    Some(match value {
        c::TextureViewDimension::Undefined => return None,
        c::TextureViewDimension::D1 => wgpu::TextureViewDimension::D1,
        c::TextureViewDimension::D2 => wgpu::TextureViewDimension::D2,
        c::TextureViewDimension::D2Array => wgpu::TextureViewDimension::D2Array,
        c::TextureViewDimension::Cube => wgpu::TextureViewDimension::Cube,
        c::TextureViewDimension::CubeArray => wgpu::TextureViewDimension::CubeArray,
        c::TextureViewDimension::D3 => wgpu::TextureViewDimension::D3,
    })
}

pub(crate) fn texture_aspect(value: c::TextureAspect) -> wgpu::TextureAspect {
    match value {
        c::TextureAspect::Undefined | c::TextureAspect::All => wgpu::TextureAspect::All,
        c::TextureAspect::StencilOnly => wgpu::TextureAspect::StencilOnly,
        c::TextureAspect::DepthOnly => wgpu::TextureAspect::DepthOnly,
    }
}

pub(crate) fn address_mode(value: c::AddressMode) -> wgpu::AddressMode {
    match value {
        c::AddressMode::Undefined | c::AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        c::AddressMode::Repeat => wgpu::AddressMode::Repeat,
        c::AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
    }
}

pub(crate) fn filter_mode(value: c::FilterMode) -> wgpu::FilterMode {
    match value {
        c::FilterMode::Undefined | c::FilterMode::Nearest => wgpu::FilterMode::Nearest,
        c::FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

pub(crate) fn mipmap_filter_mode(value: c::MipmapFilterMode) -> wgpu::MipmapFilterMode {
    match value {
        c::MipmapFilterMode::Undefined | c::MipmapFilterMode::Nearest => {
            wgpu::MipmapFilterMode::Nearest
        }
        c::MipmapFilterMode::Linear => wgpu::MipmapFilterMode::Linear,
    }
}

pub(crate) fn compare_function(value: c::CompareFunction) -> Option<wgpu::CompareFunction> {
    Some(match value {
        c::CompareFunction::Undefined => return None,
        c::CompareFunction::Never => wgpu::CompareFunction::Never,
        c::CompareFunction::Less => wgpu::CompareFunction::Less,
        c::CompareFunction::Equal => wgpu::CompareFunction::Equal,
        c::CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
        c::CompareFunction::Greater => wgpu::CompareFunction::Greater,
        c::CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
        c::CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        c::CompareFunction::Always => wgpu::CompareFunction::Always,
    })
}

pub(crate) fn blend_factor(value: c::BlendFactor, default: wgpu::BlendFactor) -> wgpu::BlendFactor {
    use wgpu::BlendFactor as W;
    match value {
        c::BlendFactor::Undefined => default,
        c::BlendFactor::Zero => W::Zero,
        c::BlendFactor::One => W::One,
        c::BlendFactor::Src => W::Src,
        c::BlendFactor::OneMinusSrc => W::OneMinusSrc,
        c::BlendFactor::SrcAlpha => W::SrcAlpha,
        c::BlendFactor::OneMinusSrcAlpha => W::OneMinusSrcAlpha,
        c::BlendFactor::Dst => W::Dst,
        c::BlendFactor::OneMinusDst => W::OneMinusDst,
        c::BlendFactor::DstAlpha => W::DstAlpha,
        c::BlendFactor::OneMinusDstAlpha => W::OneMinusDstAlpha,
        c::BlendFactor::SrcAlphaSaturated => W::SrcAlphaSaturated,
        c::BlendFactor::Constant => W::Constant,
        c::BlendFactor::OneMinusConstant => W::OneMinusConstant,
        c::BlendFactor::Src1 => W::Src1,
        c::BlendFactor::OneMinusSrc1 => W::OneMinusSrc1,
        c::BlendFactor::Src1Alpha => W::Src1Alpha,
        c::BlendFactor::OneMinusSrc1Alpha => W::OneMinusSrc1Alpha,
    }
}

pub(crate) fn blend_operation(value: c::BlendOperation) -> wgpu::BlendOperation {
    match value {
        c::BlendOperation::Undefined | c::BlendOperation::Add => wgpu::BlendOperation::Add,
        c::BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
        c::BlendOperation::ReverseSubtract => wgpu::BlendOperation::ReverseSubtract,
        c::BlendOperation::Min => wgpu::BlendOperation::Min,
        c::BlendOperation::Max => wgpu::BlendOperation::Max,
    }
}

pub(crate) fn blend_component(value: &s::BlendComponent) -> wgpu::BlendComponent {
    wgpu::BlendComponent {
        src_factor: blend_factor(value.src_factor, wgpu::BlendFactor::One),
        dst_factor: blend_factor(value.dst_factor, wgpu::BlendFactor::Zero),
        operation: blend_operation(value.operation),
    }
}

pub(crate) fn stencil_operation(value: c::StencilOperation) -> wgpu::StencilOperation {
    use wgpu::StencilOperation as W;
    match value {
        c::StencilOperation::Undefined | c::StencilOperation::Keep => W::Keep,
        c::StencilOperation::Zero => W::Zero,
        c::StencilOperation::Replace => W::Replace,
        c::StencilOperation::Invert => W::Invert,
        c::StencilOperation::IncrementClamp => W::IncrementClamp,
        c::StencilOperation::DecrementClamp => W::DecrementClamp,
        c::StencilOperation::IncrementWrap => W::IncrementWrap,
        c::StencilOperation::DecrementWrap => W::DecrementWrap,
    }
}

pub(crate) fn stencil_face(value: &s::StencilFaceState) -> wgpu::StencilFaceState {
    wgpu::StencilFaceState {
        compare: compare_function(value.compare).unwrap_or(wgpu::CompareFunction::Always),
        fail_op: stencil_operation(value.fail_op),
        depth_fail_op: stencil_operation(value.depth_fail_op),
        pass_op: stencil_operation(value.pass_op),
    }
}

pub(crate) fn primitive_topology(value: c::PrimitiveTopology) -> wgpu::PrimitiveTopology {
    use wgpu::PrimitiveTopology as W;
    match value {
        c::PrimitiveTopology::PointList => W::PointList,
        c::PrimitiveTopology::LineList => W::LineList,
        c::PrimitiveTopology::LineStrip => W::LineStrip,
        c::PrimitiveTopology::Undefined | c::PrimitiveTopology::TriangleList => W::TriangleList,
        c::PrimitiveTopology::TriangleStrip => W::TriangleStrip,
    }
}

pub(crate) fn front_face(value: c::FrontFace) -> wgpu::FrontFace {
    match value {
        c::FrontFace::Undefined | c::FrontFace::CCW => wgpu::FrontFace::Ccw,
        c::FrontFace::CW => wgpu::FrontFace::Cw,
    }
}

pub(crate) fn cull_mode(value: c::CullMode) -> Option<wgpu::Face> {
    match value {
        c::CullMode::Undefined | c::CullMode::None => None,
        c::CullMode::Front => Some(wgpu::Face::Front),
        c::CullMode::Back => Some(wgpu::Face::Back),
    }
}

pub(crate) fn index_format(value: c::IndexFormat) -> Option<wgpu::IndexFormat> {
    match value {
        c::IndexFormat::Undefined => None,
        c::IndexFormat::Uint16 => Some(wgpu::IndexFormat::Uint16),
        c::IndexFormat::Uint32 => Some(wgpu::IndexFormat::Uint32),
    }
}

pub(crate) fn vertex_format(value: c::VertexFormat) -> Option<wgpu::VertexFormat> {
    use c::VertexFormat as C;
    use wgpu::VertexFormat as W;
    Some(match value {
        C::Uint8 => W::Uint8,
        C::Uint8x2 => W::Uint8x2,
        C::Uint8x4 => W::Uint8x4,
        C::Sint8 => W::Sint8,
        C::Sint8x2 => W::Sint8x2,
        C::Sint8x4 => W::Sint8x4,
        C::Unorm8 => W::Unorm8,
        C::Unorm8x2 => W::Unorm8x2,
        C::Unorm8x4 => W::Unorm8x4,
        C::Snorm8 => W::Snorm8,
        C::Snorm8x2 => W::Snorm8x2,
        C::Snorm8x4 => W::Snorm8x4,
        C::Uint16 => W::Uint16,
        C::Uint16x2 => W::Uint16x2,
        C::Uint16x4 => W::Uint16x4,
        C::Sint16 => W::Sint16,
        C::Sint16x2 => W::Sint16x2,
        C::Sint16x4 => W::Sint16x4,
        C::Unorm16 => W::Unorm16,
        C::Unorm16x2 => W::Unorm16x2,
        C::Unorm16x4 => W::Unorm16x4,
        C::Snorm16 => W::Snorm16,
        C::Snorm16x2 => W::Snorm16x2,
        C::Snorm16x4 => W::Snorm16x4,
        C::Float16 => W::Float16,
        C::Float16x2 => W::Float16x2,
        C::Float16x4 => W::Float16x4,
        C::Float32 => W::Float32,
        C::Float32x2 => W::Float32x2,
        C::Float32x3 => W::Float32x3,
        C::Float32x4 => W::Float32x4,
        C::Uint32 => W::Uint32,
        C::Uint32x2 => W::Uint32x2,
        C::Uint32x3 => W::Uint32x3,
        C::Uint32x4 => W::Uint32x4,
        C::Sint32 => W::Sint32,
        C::Sint32x2 => W::Sint32x2,
        C::Sint32x3 => W::Sint32x3,
        C::Sint32x4 => W::Sint32x4,
        C::Unorm10_10_10_2 => W::Unorm10_10_10_2,
        C::Unorm8x4BGRA => W::Unorm8x4Bgra,
        // wgpu has no such vertex format.
        C::Snorm10_10_10_2 => return None,
    })
}

pub(crate) fn vertex_step_mode(value: c::VertexStepMode) -> wgpu::VertexStepMode {
    match value {
        c::VertexStepMode::Undefined | c::VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
        c::VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
    }
}

pub(crate) fn store_op(value: c::StoreOp) -> wgpu::StoreOp {
    match value {
        c::StoreOp::Undefined | c::StoreOp::Store => wgpu::StoreOp::Store,
        c::StoreOp::Discard => wgpu::StoreOp::Discard,
    }
}

pub(crate) fn present_mode(value: c::PresentMode) -> wgpu::PresentMode {
    match value {
        c::PresentMode::Undefined | c::PresentMode::Fifo => wgpu::PresentMode::Fifo,
        c::PresentMode::FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
        c::PresentMode::Immediate => wgpu::PresentMode::Immediate,
        c::PresentMode::Mailbox => wgpu::PresentMode::Mailbox,
    }
}

pub(crate) fn present_mode_to_abi(value: wgpu::PresentMode) -> Option<c::PresentMode> {
    Some(match value {
        wgpu::PresentMode::Fifo => c::PresentMode::Fifo,
        wgpu::PresentMode::FifoRelaxed => c::PresentMode::FifoRelaxed,
        wgpu::PresentMode::Immediate => c::PresentMode::Immediate,
        wgpu::PresentMode::Mailbox => c::PresentMode::Mailbox,
        wgpu::PresentMode::AutoVsync | wgpu::PresentMode::AutoNoVsync => return None,
    })
}

pub(crate) fn alpha_mode(value: c::CompositeAlphaMode) -> wgpu::CompositeAlphaMode {
    match value {
        c::CompositeAlphaMode::Auto => wgpu::CompositeAlphaMode::Auto,
        c::CompositeAlphaMode::Opaque => wgpu::CompositeAlphaMode::Opaque,
        c::CompositeAlphaMode::Premultiplied => wgpu::CompositeAlphaMode::PreMultiplied,
        c::CompositeAlphaMode::Unpremultiplied => wgpu::CompositeAlphaMode::PostMultiplied,
        c::CompositeAlphaMode::Inherit => wgpu::CompositeAlphaMode::Inherit,
    }
}

pub(crate) fn alpha_mode_to_abi(value: wgpu::CompositeAlphaMode) -> c::CompositeAlphaMode {
    match value {
        wgpu::CompositeAlphaMode::Auto => c::CompositeAlphaMode::Auto,
        wgpu::CompositeAlphaMode::Opaque => c::CompositeAlphaMode::Opaque,
        wgpu::CompositeAlphaMode::PreMultiplied => c::CompositeAlphaMode::Premultiplied,
        wgpu::CompositeAlphaMode::PostMultiplied => c::CompositeAlphaMode::Unpremultiplied,
        wgpu::CompositeAlphaMode::Inherit => c::CompositeAlphaMode::Inherit,
    }
}

// -- small structs --------------------------------------------------------------

pub(crate) fn extent3d(value: &s::Extent3D) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: value.width,
        height: value.height,
        depth_or_array_layers: value.depth_or_array_layers,
    }
}

pub(crate) fn origin3d(value: &s::Origin3D) -> wgpu::Origin3d {
    wgpu::Origin3d {
        x: value.x,
        y: value.y,
        z: value.z,
    }
}

pub(crate) fn color(value: &s::Color) -> wgpu::Color {
    wgpu::Color {
        r: value.r,
        g: value.g,
        b: value.b,
        a: value.a,
    }
}

/// `WGPU_COPY_STRIDE_UNDEFINED` means "not specified".
pub(crate) fn copy_stride(value: u32) -> Option<u32> {
    (value != c::COPY_STRIDE_UNDEFINED).then_some(value)
}

pub(crate) fn texel_copy_buffer_layout(
    value: &s::TexelCopyBufferLayout,
) -> wgpu::TexelCopyBufferLayout {
    wgpu::TexelCopyBufferLayout {
        offset: value.offset,
        bytes_per_row: copy_stride(value.bytes_per_row),
        rows_per_image: copy_stride(value.rows_per_image),
    }
}

/// Usage bits are numbered identically by `webgpu.h` and wgpu; anything else
/// the guest sets is not a WebGPU usage.
pub(crate) fn buffer_usages(bits: u64) -> Option<wgpu::BufferUsages> {
    u32::try_from(bits)
        .ok()
        .and_then(wgpu::BufferUsages::from_bits)
}

pub(crate) fn texture_usages(bits: u64) -> Option<wgpu::TextureUsages> {
    use crate::abi::enums::texture_usage as usage;
    // The first five bits are the same on both sides. The transient
    // attachment has a bit of its own in wgpu, among its native usages.
    let known = [
        (usage::COPY_SRC, wgpu::TextureUsages::COPY_SRC),
        (usage::COPY_DST, wgpu::TextureUsages::COPY_DST),
        (usage::TEXTURE_BINDING, wgpu::TextureUsages::TEXTURE_BINDING),
        (usage::STORAGE_BINDING, wgpu::TextureUsages::STORAGE_BINDING),
        (
            usage::RENDER_ATTACHMENT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ),
        (usage::TRANSIENT_ATTACHMENT, wgpu::TextureUsages::TRANSIENT),
    ];
    let mut rest = bits;
    let mut usages = wgpu::TextureUsages::empty();
    for (bit, flag) in known {
        if rest & bit != 0 {
            rest &= !bit;
            usages |= flag;
        }
    }
    (rest == 0).then_some(usages)
}

pub(crate) fn shader_stages(bits: u64) -> Option<wgpu::ShaderStages> {
    let known =
        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
    let stages = u32::try_from(bits)
        .ok()
        .and_then(wgpu::ShaderStages::from_bits)?;
    known.contains(stages).then_some(stages)
}

pub(crate) fn color_writes(bits: u64) -> Option<wgpu::ColorWrites> {
    u32::try_from(bits)
        .ok()
        .and_then(wgpu::ColorWrites::from_bits)
}
