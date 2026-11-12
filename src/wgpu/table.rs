//! What the wgpu host keeps behind a handle (see `crate::native::table`).

use std::sync::Arc;

use super::bind::{BindGroupState, PipelineLayoutState};
use super::encoder::{CommandBufferState, EncoderState};
use super::pass::{BundleEncoderState, ComputePassState, RenderBundleState, RenderPassState};
use super::pipeline::{ComputePipelineState, RenderPipelineState};
use super::shader::ShaderState;
use super::texture::{QuerySetState, TextureViewState};
use super::{
    AdapterState, BufferState, DeviceState, Host, InstanceState, SurfaceState, TextureState,
};
use crate::abi::Handle;

macro_rules! objects {
    ($( $variant:ident($state:ty) => $getter:ident ),* $(,)?) => {
        /// What a handle refers to. Cloning is a reference-count bump.
        #[derive(Clone)]
        #[allow(dead_code)] // `ExternalTexture`, see below
        pub(crate) enum Object {
            $( $variant(Arc<$state>), )*
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum ObjectKind {
            $( $variant, )*
        }

        impl Object {
            pub(crate) fn kind(&self) -> ObjectKind {
                match self {
                    $( Object::$variant(_) => ObjectKind::$variant, )*
                }
            }
        }

        impl Host {
            $(
                /// The object behind `handle`, if it is live and of this type.
                pub(crate) fn $getter(&self, handle: Handle) -> Option<Arc<$state>> {
                    match self.object(handle) {
                        Some(Object::$variant(state)) => Some(state),
                        other => {
                            super::log(|| format!(
                                "expected a {} handle, got {:#x} ({})",
                                stringify!($variant),
                                handle,
                                match &other {
                                    Some(object) => format!("a {:?}", object.kind()),
                                    None if handle == 0 => "null".to_owned(),
                                    None => "not a live handle".to_owned(),
                                },
                            ));
                            None
                        }
                    }
                }
            )*
        }
    };
}

objects! {
    Instance(InstanceState) => instance,
    Adapter(AdapterState) => adapter,
    Device(DeviceState) => device,
    // A queue is its device's queue: same state, separate handle.
    Queue(DeviceState) => queue,
    Buffer(BufferState) => buffer,
    Texture(TextureState) => texture,
    TextureView(TextureViewState) => texture_view,
    Sampler(wgpu::Sampler) => sampler,
    BindGroupLayout(wgpu::BindGroupLayout) => bind_group_layout,
    BindGroup(BindGroupState) => bind_group,
    PipelineLayout(PipelineLayoutState) => pipeline_layout,
    ShaderModule(ShaderState) => shader_module,
    ComputePipeline(ComputePipelineState) => compute_pipeline,
    RenderPipeline(RenderPipelineState) => render_pipeline,
    CommandEncoder(EncoderState) => command_encoder,
    CommandBuffer(CommandBufferState) => command_buffer,
    ComputePassEncoder(ComputePassState) => compute_pass,
    RenderPassEncoder(RenderPassState) => render_pass,
    RenderBundleEncoder(BundleEncoderState) => render_bundle_encoder,
    RenderBundle(RenderBundleState) => render_bundle,
    QuerySet(QuerySetState) => query_set,
    Surface(SurfaceState) => surface,
    // webgpu.h has the type but no way to create one, so nothing ever
    // constructs this variant; it exists for the type's AddRef/Release.
    ExternalTexture(()) => external_texture,
}

impl crate::native::table::TableObject for Object {
    type Kind = ObjectKind;
    fn kind(&self) -> ObjectKind {
        Object::kind(self)
    }
    fn is_instance(&self) -> bool {
        matches!(self, Object::Instance(_))
    }
}

pub(super) type Table = crate::native::table::Table<Object>;
