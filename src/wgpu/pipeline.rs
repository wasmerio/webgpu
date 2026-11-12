//! Compute and render pipelines.

use std::sync::Arc;

use super::bind::PipelineLayoutState;
use super::buffer::label;
use super::shader::ShaderState;
use super::{DeviceState, Host, Object, ObjectKind, Record, convert};
use crate::abi::api::{ComputePipelineApi, RenderPipelineApi};
use crate::abi::enums::{CreatePipelineAsyncStatus, OptionalBool, TextureFormat};
use crate::abi::structs::{
    ComputePipelineDescriptor, ConstantEntry, DepthStencilState, RenderPipelineDescriptor,
};
use crate::abi::{AbiEnum, Cx, Handle};

pub(crate) struct ComputePipelineState {
    pub(crate) pipeline: wgpu::ComputePipeline,
}

pub(crate) struct RenderPipelineState {
    pub(crate) pipeline: wgpu::RenderPipeline,
}

fn constants(entries: &[ConstantEntry]) -> Vec<(&str, f64)> {
    entries
        .iter()
        .map(|entry| (entry.key.as_str(), entry.value))
        .collect()
}

fn compilation_options<'a>(
    constants: &'a [(&'a str, f64)],
) -> wgpu::PipelineCompilationOptions<'a> {
    wgpu::PipelineCompilationOptions {
        constants,
        zero_initialize_workgroup_memory: true,
    }
}

fn depth_stencil(state: &DepthStencilState) -> Option<wgpu::DepthStencilState> {
    Some(wgpu::DepthStencilState {
        format: convert::texture_format(state.format)?,
        depth_write_enabled: match state.depth_write_enabled {
            OptionalBool::False => Some(false),
            OptionalBool::True => Some(true),
            OptionalBool::Undefined => None,
        },
        depth_compare: convert::compare_function(state.depth_compare),
        stencil: wgpu::StencilState {
            front: convert::stencil_face(&state.stencil_front),
            back: convert::stencil_face(&state.stencil_back),
            read_mask: state.stencil_read_mask,
            write_mask: state.stencil_write_mask,
        },
        bias: wgpu::DepthBiasState {
            constant: state.depth_bias,
            slope_scale: state.depth_bias_slope_scale,
            clamp: state.depth_bias_clamp,
        },
    })
}

/// A compute pipeline's descriptor with every object it names resolved. It
/// needs no guest memory, store or host, so it can be compiled on any thread.
pub(crate) struct ComputeJob {
    descriptor: ComputePipelineDescriptor,
    device: Arc<DeviceState>,
    module: Arc<ShaderState>,
    layout: Option<Arc<PipelineLayoutState>>,
}

/// A render pipeline's descriptor with every object it names resolved.
pub(crate) struct RenderJob {
    descriptor: RenderPipelineDescriptor,
    device: Arc<DeviceState>,
    vertex: Arc<ShaderState>,
    fragment: Option<Arc<ShaderState>>,
    layout: Option<Arc<PipelineLayoutState>>,
}

impl ComputeJob {
    /// Compiles the pipeline. wgpu reports its own errors through the device's
    /// error scopes; `Err` is a descriptor this host cannot translate.
    pub(crate) fn create(self) -> Result<Object, String> {
        let constants = constants(&self.descriptor.compute.constants);
        let pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: label(&self.descriptor.label),
                    layout: self.layout.as_ref().map(|layout| &layout.layout),
                    module: &self.module.module,
                    entry_point: self.descriptor.compute.entry_point.as_deref(),
                    compilation_options: compilation_options(&constants),
                    cache: None,
                });
        Ok(Object::ComputePipeline(Arc::new(ComputePipelineState {
            pipeline,
        })))
    }
}

impl RenderJob {
    /// Compiles the pipeline, as [`ComputeJob::create`].
    pub(crate) fn create(self) -> Result<Object, String> {
        let descriptor = &self.descriptor;

        // Vertex buffers.
        let mut attributes = Vec::with_capacity(descriptor.vertex.buffers.len());
        for buffer in &descriptor.vertex.buffers {
            let mut converted = Vec::with_capacity(buffer.attributes.len());
            for attribute in &buffer.attributes {
                let format =
                    convert::vertex_format(attribute.format).ok_or("unsupported vertex format.")?;
                converted.push(wgpu::VertexAttribute {
                    format,
                    offset: attribute.offset,
                    shader_location: attribute.shader_location,
                });
            }
            attributes.push(converted);
        }
        // A hole is a slot with no step mode and no attributes; to wgpu, which
        // has no holes, a layout without attributes says the same thing.
        let buffers: Vec<wgpu::VertexBufferLayout<'_>> = descriptor
            .vertex
            .buffers
            .iter()
            .zip(&attributes)
            .map(|(buffer, attributes)| wgpu::VertexBufferLayout {
                array_stride: buffer.array_stride,
                step_mode: convert::vertex_step_mode(buffer.step_mode),
                attributes,
            })
            .collect();

        let vertex_constants = constants(&descriptor.vertex.constants);

        // Fragment stage. A target with an undefined format is a hole.
        let mut targets = Vec::new();
        let mut fragment_constants = Vec::new();
        if let Some(fragment) = &descriptor.fragment {
            fragment_constants = constants(&fragment.constants);
            for target in &fragment.targets {
                if target.format == TextureFormat::Undefined {
                    targets.push(None);
                    continue;
                }
                let format = convert::texture_format(target.format)
                    .ok_or("unsupported color target format.")?;
                let write_mask =
                    convert::color_writes(target.write_mask).ok_or("invalid color write mask.")?;
                targets.push(Some(wgpu::ColorTargetState {
                    format,
                    blend: target.blend.as_ref().map(|blend| wgpu::BlendState {
                        color: convert::blend_component(&blend.color),
                        alpha: convert::blend_component(&blend.alpha),
                    }),
                    write_mask,
                }));
            }
        }

        let depth_stencil = match &descriptor.depth_stencil {
            Some(state) => {
                Some(depth_stencil(state).ok_or("a depth-stencil state needs a format.")?)
            }
            None => None,
        };

        let pipeline = self
            .device
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: label(&descriptor.label),
                layout: self.layout.as_ref().map(|layout| &layout.layout),
                vertex: wgpu::VertexState {
                    module: &self.vertex.module,
                    entry_point: descriptor.vertex.entry_point.as_deref(),
                    compilation_options: compilation_options(&vertex_constants),
                    buffers: &buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology: convert::primitive_topology(descriptor.primitive.topology),
                    strip_index_format: convert::index_format(
                        descriptor.primitive.strip_index_format,
                    ),
                    front_face: convert::front_face(descriptor.primitive.front_face),
                    cull_mode: convert::cull_mode(descriptor.primitive.cull_mode),
                    unclipped_depth: descriptor.primitive.unclipped_depth,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    // Zero is what a zero-initialized struct says; it means one.
                    count: descriptor.multisample.count.max(1),
                    mask: u64::from(descriptor.multisample.mask),
                    alpha_to_coverage_enabled: descriptor.multisample.alpha_to_coverage_enabled,
                },
                fragment: self
                    .fragment
                    .as_ref()
                    .zip(descriptor.fragment.as_ref())
                    .map(|(module, fragment)| wgpu::FragmentState {
                        module: &module.module,
                        entry_point: fragment.entry_point.as_deref(),
                        compilation_options: compilation_options(&fragment_constants),
                        targets: &targets,
                    }),
                multiview_mask: None,
                cache: None,
            });
        Ok(Object::RenderPipeline(Arc::new(RenderPipelineState {
            pipeline,
        })))
    }
}

impl Host {
    /// A null layout asks for the layout to be derived from the shader.
    fn job_layout(&self, handle: Handle) -> Result<Option<Arc<PipelineLayoutState>>, String> {
        match handle {
            0 => Ok(None),
            handle => self
                .pipeline_layout(handle)
                .map(Some)
                .ok_or_else(|| "invalid layout.".into()),
        }
    }

    /// Resolves the objects a compute pipeline names.
    pub(crate) fn compute_job(
        &self,
        device: Arc<DeviceState>,
        descriptor: ComputePipelineDescriptor,
    ) -> Result<ComputeJob, String> {
        let module = self
            .shader_module(descriptor.compute.module)
            .ok_or("invalid shader module.")?;
        let layout = self.job_layout(descriptor.layout)?;
        Ok(ComputeJob {
            descriptor,
            device,
            module,
            layout,
        })
    }

    /// Resolves the objects a render pipeline names.
    pub(crate) fn render_job(
        &self,
        device: Arc<DeviceState>,
        descriptor: RenderPipelineDescriptor,
    ) -> Result<RenderJob, String> {
        let vertex = self
            .shader_module(descriptor.vertex.module)
            .ok_or("invalid vertex shader module.")?;
        let layout = self.job_layout(descriptor.layout)?;
        let fragment = match &descriptor.fragment {
            Some(fragment) => Some(
                self.shader_module(fragment.module)
                    .ok_or("invalid fragment shader module.")?,
            ),
            None => None,
        };
        Ok(RenderJob {
            descriptor,
            device,
            vertex,
            fragment,
            layout,
        })
    }

    pub(crate) fn create_compute_pipeline(
        &self,
        device: Handle,
        descriptor: &ComputePipelineDescriptor,
    ) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        let created = self
            .compute_job(Arc::clone(&device_state), descriptor.clone())
            .and_then(ComputeJob::create);
        self.insert_pipeline(
            device,
            &device_state,
            "wgpuDeviceCreateComputePipeline",
            created,
        )
    }

    pub(crate) fn create_render_pipeline(
        &self,
        device: Handle,
        descriptor: &RenderPipelineDescriptor,
    ) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        let created = self
            .render_job(Arc::clone(&device_state), descriptor.clone())
            .and_then(RenderJob::create);
        self.insert_pipeline(
            device,
            &device_state,
            "wgpuDeviceCreateRenderPipeline",
            created,
        )
    }

    fn insert_pipeline(
        &self,
        device: Handle,
        device_state: &DeviceState,
        function: &str,
        created: Result<Object, String>,
    ) -> Handle {
        match created {
            Ok(pipeline) => self.insert(pipeline, self.instance_of(device)),
            Err(message) => {
                self.validation(device_state, format!("{function}: {message}"));
                0
            }
        }
    }

    /// Completes the future of an asynchronous pipeline creation that could
    /// not even be started. As for every failure of these calls, the callback
    /// is what reports it: no device error is raised.
    pub(crate) fn fail_pipeline_async(&self, kind: u32, future: u64, message: String) {
        self.events.push(Record {
            future,
            kind,
            status: CreatePipelineAsyncStatus::ValidationError.as_u32(),
            message,
            ..Record::default()
        });
    }
}

impl ComputePipelineApi for Host {
    fn compute_pipeline_get_bind_group_layout(
        &self,
        _cx: &mut Cx<'_>,
        compute_pipeline: Handle,
        group_index: u32,
    ) -> Handle {
        let Some(state) = self.compute_pipeline(compute_pipeline) else {
            return 0;
        };
        let layout = state.pipeline.get_bind_group_layout(group_index);
        self.insert(
            Object::BindGroupLayout(Arc::new(layout)),
            self.instance_of(compute_pipeline),
        )
    }

    fn compute_pipeline_set_label(&self, _cx: &mut Cx<'_>, compute_pipeline: Handle, _label: &str) {
        let _ = self.compute_pipeline(compute_pipeline);
    }

    fn compute_pipeline_add_ref(&self, _cx: &mut Cx<'_>, compute_pipeline: Handle) {
        self.add_ref(compute_pipeline, ObjectKind::ComputePipeline);
    }

    fn compute_pipeline_release(&self, _cx: &mut Cx<'_>, compute_pipeline: Handle) {
        self.release(compute_pipeline, ObjectKind::ComputePipeline);
    }
}

impl RenderPipelineApi for Host {
    fn render_pipeline_get_bind_group_layout(
        &self,
        _cx: &mut Cx<'_>,
        render_pipeline: Handle,
        group_index: u32,
    ) -> Handle {
        let Some(state) = self.render_pipeline(render_pipeline) else {
            return 0;
        };
        let layout = state.pipeline.get_bind_group_layout(group_index);
        self.insert(
            Object::BindGroupLayout(Arc::new(layout)),
            self.instance_of(render_pipeline),
        )
    }

    fn render_pipeline_set_label(&self, _cx: &mut Cx<'_>, render_pipeline: Handle, _label: &str) {
        let _ = self.render_pipeline(render_pipeline);
    }

    fn render_pipeline_add_ref(&self, _cx: &mut Cx<'_>, render_pipeline: Handle) {
        self.add_ref(render_pipeline, ObjectKind::RenderPipeline);
    }

    fn render_pipeline_release(&self, _cx: &mut Cx<'_>, render_pipeline: Handle) {
        self.release(render_pipeline, ObjectKind::RenderPipeline);
    }
}
