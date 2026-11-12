//! Bind group layouts, bind groups, pipeline layouts and samplers.

use std::num::{NonZeroU32, NonZeroU64};
use std::sync::Arc;

use super::buffer::label;
use super::{Host, Object, ObjectKind, convert};
use crate::abi::api::{BindGroupApi, BindGroupLayoutApi, PipelineLayoutApi, SamplerApi};
use crate::abi::enums::{
    self as c, BufferBindingType, SamplerBindingType, StorageTextureAccess, TextureSampleType,
};
use crate::abi::structs::{
    BindGroupDescriptor, BindGroupLayoutDescriptor, BindGroupLayoutEntry, PipelineLayoutDescriptor,
    SamplerDescriptor,
};
use crate::abi::{Cx, Handle};

pub(crate) struct BindGroupState {
    pub(crate) group: wgpu::BindGroup,
}

pub(crate) struct PipelineLayoutState {
    pub(crate) layout: wgpu::PipelineLayout,
}

/// The wgpu binding type of one layout entry. Each of the entry's four
/// sub-layouts says whether it is used; exactly one must be.
fn binding_type(entry: &BindGroupLayoutEntry) -> Result<wgpu::BindingType, &'static str> {
    let mut found = Vec::with_capacity(1);

    match entry.buffer.type_ {
        BufferBindingType::BindingNotUsed => {}
        ty => found.push(wgpu::BindingType::Buffer {
            ty: match ty {
                BufferBindingType::Storage => wgpu::BufferBindingType::Storage { read_only: false },
                BufferBindingType::ReadOnlyStorage => {
                    wgpu::BufferBindingType::Storage { read_only: true }
                }
                _ => wgpu::BufferBindingType::Uniform,
            },
            has_dynamic_offset: entry.buffer.has_dynamic_offset,
            min_binding_size: NonZeroU64::new(entry.buffer.min_binding_size),
        }),
    }

    match entry.sampler.type_ {
        SamplerBindingType::BindingNotUsed => {}
        ty => found.push(wgpu::BindingType::Sampler(match ty {
            SamplerBindingType::NonFiltering => wgpu::SamplerBindingType::NonFiltering,
            SamplerBindingType::Comparison => wgpu::SamplerBindingType::Comparison,
            _ => wgpu::SamplerBindingType::Filtering,
        })),
    }

    match entry.texture.sample_type {
        TextureSampleType::BindingNotUsed => {}
        ty => found.push(wgpu::BindingType::Texture {
            sample_type: match ty {
                TextureSampleType::UnfilterableFloat => {
                    wgpu::TextureSampleType::Float { filterable: false }
                }
                TextureSampleType::Depth => wgpu::TextureSampleType::Depth,
                TextureSampleType::Sint => wgpu::TextureSampleType::Sint,
                TextureSampleType::Uint => wgpu::TextureSampleType::Uint,
                _ => wgpu::TextureSampleType::Float { filterable: true },
            },
            view_dimension: convert::view_dimension(entry.texture.view_dimension)
                .unwrap_or(wgpu::TextureViewDimension::D2),
            multisampled: entry.texture.multisampled,
        }),
    }

    match entry.storage_texture.access {
        StorageTextureAccess::BindingNotUsed => {}
        access => found.push(wgpu::BindingType::StorageTexture {
            access: match access {
                StorageTextureAccess::ReadOnly => wgpu::StorageTextureAccess::ReadOnly,
                StorageTextureAccess::ReadWrite => wgpu::StorageTextureAccess::ReadWrite,
                _ => wgpu::StorageTextureAccess::WriteOnly,
            },
            format: convert::texture_format(entry.storage_texture.format)
                .ok_or("a storage texture binding needs a format")?,
            view_dimension: convert::view_dimension(entry.storage_texture.view_dimension)
                .unwrap_or(wgpu::TextureViewDimension::D2),
        }),
    }

    if entry.chain.external_texture_binding_layout().is_some() {
        return Err("external texture bindings are not supported");
    }
    match found.len() {
        1 => Ok(found.remove(0)),
        0 => Err(
            "a bind group layout entry must use exactly one of buffer, sampler, texture or storageTexture",
        ),
        _ => Err(
            "a bind group layout entry uses more than one of buffer, sampler, texture and storageTexture",
        ),
    }
}

impl Host {
    pub(crate) fn create_bind_group_layout(
        &self,
        device: Handle,
        descriptor: &BindGroupLayoutDescriptor,
    ) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        let mut entries = Vec::with_capacity(descriptor.entries.len());
        for entry in &descriptor.entries {
            let ty = match binding_type(entry) {
                Ok(ty) => ty,
                Err(message) => {
                    self.validation(
                        &device_state,
                        format!("binding {}: {message}", entry.binding),
                    );
                    return 0;
                }
            };
            let Some(visibility) = convert::shader_stages(entry.visibility) else {
                self.validation(
                    &device_state,
                    format!("binding {}: invalid shader stage visibility", entry.binding),
                );
                return 0;
            };
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: entry.binding,
                visibility,
                ty,
                count: NonZeroU32::new(entry.binding_array_size),
            });
        }
        let layout =
            device_state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: label(&descriptor.label),
                    entries: &entries,
                });
        self.insert(
            Object::BindGroupLayout(Arc::new(layout)),
            self.instance_of(device),
        )
    }

    pub(crate) fn create_bind_group(
        &self,
        device: Handle,
        descriptor: &BindGroupDescriptor,
    ) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        let Some(layout) = self.bind_group_layout(descriptor.layout) else {
            self.validation(&device_state, "wgpuDeviceCreateBindGroup: invalid layout.");
            return 0;
        };

        // Resolve every handle first: the wgpu entries borrow from these.
        enum Resource {
            Buffer(Arc<super::BufferState>, u64, u64),
            Sampler(Arc<wgpu::Sampler>),
            View(Arc<super::texture::TextureViewState>),
        }
        let mut resources = Vec::with_capacity(descriptor.entries.len());
        for entry in &descriptor.entries {
            if entry.chain.external_texture_binding_entry().is_some() {
                self.validation(
                    &device_state,
                    "External texture bindings are not supported.",
                );
                return 0;
            }
            let resource = if entry.buffer != 0 {
                self.buffer(entry.buffer)
                    .map(|buffer| Resource::Buffer(buffer, entry.offset, entry.size))
            } else if entry.sampler != 0 {
                self.sampler(entry.sampler).map(Resource::Sampler)
            } else if entry.texture_view != 0 {
                self.texture_view(entry.texture_view).map(Resource::View)
            } else {
                None
            };
            let Some(resource) = resource else {
                self.validation(
                    &device_state,
                    format!(
                        "binding {}: a bind group entry needs a valid buffer, sampler or texture view",
                        entry.binding
                    ),
                );
                return 0;
            };
            resources.push((entry.binding, resource));
        }
        let entries: Vec<_> = resources
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: match resource {
                    Resource::Buffer(buffer, offset, size) => {
                        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffer.buffer,
                            offset: *offset,
                            // WGPU_WHOLE_SIZE: to the end of the buffer. So
                            // is the zero of an entry that was zero-filled
                            // rather than initialized with the header's
                            // macro: implementations disagree about it (an
                            // error in one, the whole buffer in another), and
                            // the reading that keeps both kinds of program
                            // working is the one taken, on both hosts.
                            size: if *size == c::WHOLE_SIZE {
                                None
                            } else {
                                NonZeroU64::new(*size)
                            },
                        })
                    }
                    Resource::Sampler(sampler) => wgpu::BindingResource::Sampler(sampler),
                    Resource::View(view) => wgpu::BindingResource::TextureView(&view.view),
                },
            })
            .collect();
        let group = device_state
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: label(&descriptor.label),
                layout: &layout,
                entries: &entries,
            });
        self.insert(
            Object::BindGroup(Arc::new(BindGroupState { group })),
            self.instance_of(device),
        )
    }

    pub(crate) fn create_pipeline_layout(
        &self,
        device: Handle,
        descriptor: &PipelineLayoutDescriptor,
    ) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        // A null entry is a deliberate hole in the layout.
        let mut layouts = Vec::with_capacity(descriptor.bind_group_layouts.len());
        for &handle in &descriptor.bind_group_layouts {
            if handle == 0 {
                layouts.push(None);
                continue;
            }
            let Some(layout) = self.bind_group_layout(handle) else {
                self.validation(
                    &device_state,
                    "wgpuDeviceCreatePipelineLayout: invalid bind group layout.",
                );
                return 0;
            };
            layouts.push(Some(layout));
        }
        let refs: Vec<Option<&wgpu::BindGroupLayout>> =
            layouts.iter().map(|l| l.as_deref()).collect();
        let layout = device_state
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: label(&descriptor.label),
                bind_group_layouts: &refs,
                immediate_size: descriptor.immediate_size,
            });
        self.insert(
            Object::PipelineLayout(Arc::new(PipelineLayoutState { layout })),
            self.instance_of(device),
        )
    }

    pub(crate) fn create_sampler(
        &self,
        device: Handle,
        descriptor: Option<&SamplerDescriptor>,
    ) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        let default = SamplerDescriptor {
            lod_max_clamp: 32.0,
            max_anisotropy: 1,
            ..SamplerDescriptor::default()
        };
        let d = descriptor.unwrap_or(&default);
        let sampler = device_state
            .device
            .create_sampler(&wgpu::SamplerDescriptor {
                label: label(&d.label),
                address_mode_u: convert::address_mode(d.address_mode_u),
                address_mode_v: convert::address_mode(d.address_mode_v),
                address_mode_w: convert::address_mode(d.address_mode_w),
                mag_filter: convert::filter_mode(d.mag_filter),
                min_filter: convert::filter_mode(d.min_filter),
                mipmap_filter: convert::mipmap_filter_mode(d.mipmap_filter),
                lod_min_clamp: d.lod_min_clamp,
                lod_max_clamp: d.lod_max_clamp,
                compare: convert::compare_function(d.compare),
                anisotropy_clamp: d.max_anisotropy.max(1),
                border_color: None,
            });
        self.insert(Object::Sampler(Arc::new(sampler)), self.instance_of(device))
    }
}

impl BindGroupApi for Host {
    fn bind_group_set_label(&self, _cx: &mut Cx<'_>, bind_group: Handle, _label: &str) {
        let _ = self.bind_group(bind_group);
    }

    fn bind_group_add_ref(&self, _cx: &mut Cx<'_>, bind_group: Handle) {
        self.add_ref(bind_group, ObjectKind::BindGroup);
    }

    fn bind_group_release(&self, _cx: &mut Cx<'_>, bind_group: Handle) {
        self.release(bind_group, ObjectKind::BindGroup);
    }
}

impl BindGroupLayoutApi for Host {
    fn bind_group_layout_set_label(
        &self,
        _cx: &mut Cx<'_>,
        bind_group_layout: Handle,
        _label: &str,
    ) {
        let _ = self.bind_group_layout(bind_group_layout);
    }

    fn bind_group_layout_add_ref(&self, _cx: &mut Cx<'_>, bind_group_layout: Handle) {
        self.add_ref(bind_group_layout, ObjectKind::BindGroupLayout);
    }

    fn bind_group_layout_release(&self, _cx: &mut Cx<'_>, bind_group_layout: Handle) {
        self.release(bind_group_layout, ObjectKind::BindGroupLayout);
    }
}

impl PipelineLayoutApi for Host {
    fn pipeline_layout_set_label(&self, _cx: &mut Cx<'_>, pipeline_layout: Handle, _label: &str) {
        let _ = self.pipeline_layout(pipeline_layout);
    }

    fn pipeline_layout_add_ref(&self, _cx: &mut Cx<'_>, pipeline_layout: Handle) {
        self.add_ref(pipeline_layout, ObjectKind::PipelineLayout);
    }

    fn pipeline_layout_release(&self, _cx: &mut Cx<'_>, pipeline_layout: Handle) {
        self.release(pipeline_layout, ObjectKind::PipelineLayout);
    }
}

impl SamplerApi for Host {
    fn sampler_set_label(&self, _cx: &mut Cx<'_>, sampler: Handle, _label: &str) {
        let _ = self.sampler(sampler);
    }

    fn sampler_add_ref(&self, _cx: &mut Cx<'_>, sampler: Handle) {
        self.add_ref(sampler, ObjectKind::Sampler);
    }

    fn sampler_release(&self, _cx: &mut Cx<'_>, sampler: Handle) {
        self.release(sampler, ObjectKind::Sampler);
    }
}
