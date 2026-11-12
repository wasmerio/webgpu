//! Textures, texture views, query sets and the (uncreatable) external texture.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::buffer::label;
use super::{DeviceState, Host, Object, ObjectKind, convert};
use crate::abi::api::{ExternalTextureApi, QuerySetApi, TextureApi, TextureViewApi};
use crate::abi::enums::{
    self as c, ComponentSwizzle, ErrorType, QueryType, TextureDimension, TextureFormat,
    TextureViewDimension,
};
use crate::abi::structs::{QuerySetDescriptor, TextureDescriptor, TextureViewDescriptor};
use crate::abi::{Cx, Handle};

pub(crate) struct TextureState {
    pub(crate) texture: wgpu::Texture,
    pub(crate) device: Arc<DeviceState>,
    /// `WGPUTextureUsage` bits, as the guest gave them.
    pub(crate) usage: u64,
    /// Bytes this object still counts for against the guest's budget.
    accounted: AtomicU64,
    gpu_memory: Arc<AtomicU64>,
}

impl TextureState {
    /// Wraps a texture the host created itself (a surface's current frame).
    pub(crate) fn from_surface(
        texture: wgpu::Texture,
        device: Arc<DeviceState>,
        usage: u64,
        gpu_memory: Arc<AtomicU64>,
    ) -> Self {
        Self {
            texture,
            device,
            usage,
            accounted: AtomicU64::new(0),
            gpu_memory,
        }
    }
}

impl Drop for TextureState {
    fn drop(&mut self) {
        self.release_storage();
    }
}

impl TextureState {
    /// Gives the object's bytes back to the guest's budget, once: when the
    /// guest destroys it, or else when its last reference goes.
    pub(crate) fn release_storage(&self) {
        let bytes = self.accounted.swap(0, Ordering::Relaxed);
        self.gpu_memory.fetch_sub(bytes, Ordering::Relaxed);
    }
}

pub(crate) struct TextureViewState {
    pub(crate) view: wgpu::TextureView,
    /// `WGPUTextureUsage` bits of the texture this is a view of.
    pub(crate) texture_usage: u64,
}

pub(crate) struct QuerySetState {
    pub(crate) set: wgpu::QuerySet,
    ty: QueryType,
    count: u32,
}

/// A rough size for accounting: every mip of every layer and sample.
fn texture_bytes(descriptor: &wgpu::TextureDescriptor<'_>) -> u64 {
    let (block_width, block_height) = descriptor.format.block_dimensions();
    let block_bytes = u64::from(descriptor.format.block_copy_size(None).unwrap_or(8));
    let blocks = u64::from(descriptor.size.width.div_ceil(block_width))
        * u64::from(descriptor.size.height.div_ceil(block_height))
        * u64::from(descriptor.size.depth_or_array_layers);
    let base = blocks
        .saturating_mul(block_bytes)
        .saturating_mul(u64::from(descriptor.sample_count.max(1)));
    // A full mip chain adds a third.
    if descriptor.mip_level_count > 1 {
        base.saturating_add(base / 3)
    } else {
        base
    }
}

impl Host {
    pub(crate) fn create_texture(&self, device: Handle, descriptor: &TextureDescriptor) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        let fail = |message: &str| {
            self.validation(&device_state, format!("wgpuDeviceCreateTexture: {message}"));
            0
        };
        let Some(format) = convert::texture_format(descriptor.format) else {
            return fail("a texture needs a format.");
        };
        let Some(usage) = convert::texture_usages(descriptor.usage) else {
            return fail("invalid texture usage.");
        };
        let Some(view_formats) = descriptor
            .view_formats
            .iter()
            .map(|&format| convert::texture_format(format))
            .collect::<Option<Vec<_>>>()
        else {
            return fail("invalid view format.");
        };
        let wgpu_descriptor = wgpu::TextureDescriptor {
            label: label(&descriptor.label),
            size: convert::extent3d(&descriptor.size),
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: convert::texture_dimension(descriptor.dimension),
            format,
            usage,
            view_formats: &view_formats,
        };
        let bytes = texture_bytes(&wgpu_descriptor);
        if let Some(budget) = self.ctx.limits.max_gpu_memory_bytes
            && self
                .gpu_memory
                .load(Ordering::Relaxed)
                .saturating_add(bytes)
                > budget
        {
            self.error(
                &device_state,
                ErrorType::OutOfMemory,
                "creating this texture would exceed this guest's GPU memory budget",
            );
            return 0;
        }
        let texture = device_state.device.create_texture(&wgpu_descriptor);
        self.gpu_memory.fetch_add(bytes, Ordering::Relaxed);
        let state = TextureState {
            texture,
            device: device_state,
            usage: descriptor.usage,
            accounted: AtomicU64::new(bytes),
            gpu_memory: Arc::clone(&self.gpu_memory),
        };
        self.insert(Object::Texture(Arc::new(state)), self.instance_of(device))
    }

    pub(crate) fn create_query_set(
        &self,
        device: Handle,
        descriptor: &QuerySetDescriptor,
    ) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        let set = device_state
            .device
            .create_query_set(&wgpu::QuerySetDescriptor {
                label: label(&descriptor.label),
                ty: match descriptor.type_ {
                    QueryType::Occlusion => wgpu::QueryType::Occlusion,
                    QueryType::Timestamp => wgpu::QueryType::Timestamp,
                },
                count: descriptor.count,
            });
        let state = QuerySetState {
            set,
            ty: descriptor.type_,
            count: descriptor.count,
        };
        self.insert(Object::QuerySet(Arc::new(state)), self.instance_of(device))
    }
}

impl TextureApi for Host {
    fn texture_create_view(
        &self,
        _cx: &mut Cx<'_>,
        texture: Handle,
        descriptor: Option<&TextureViewDescriptor>,
    ) -> Handle {
        let Some(state) = self.texture(texture) else {
            return 0;
        };
        let view = match descriptor {
            None => state
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            Some(d) => {
                if let Some(swizzle) = d.chain.texture_component_swizzle_descriptor() {
                    let identity = |value: ComponentSwizzle, same: ComponentSwizzle| {
                        value == ComponentSwizzle::Undefined || value == same
                    };
                    let s = &swizzle.swizzle;
                    if !(identity(s.r, ComponentSwizzle::R)
                        && identity(s.g, ComponentSwizzle::G)
                        && identity(s.b, ComponentSwizzle::B)
                        && identity(s.a, ComponentSwizzle::A))
                    {
                        self.validation(
                            &state.device,
                            "Texture component swizzle is not supported by this backend.",
                        );
                        return 0;
                    }
                }
                let usage = match d.usage {
                    0 => None,
                    bits => match convert::texture_usages(bits) {
                        Some(usage) => Some(usage),
                        None => {
                            self.validation(&state.device, "wgpuTextureCreateView: invalid usage.");
                            return 0;
                        }
                    },
                };
                state.texture.create_view(&wgpu::TextureViewDescriptor {
                    label: label(&d.label),
                    format: convert::texture_format(d.format),
                    dimension: convert::view_dimension(d.dimension),
                    usage,
                    aspect: convert::texture_aspect(d.aspect),
                    base_mip_level: d.base_mip_level,
                    mip_level_count: (d.mip_level_count != c::MIP_LEVEL_COUNT_UNDEFINED)
                        .then_some(d.mip_level_count),
                    base_array_layer: d.base_array_layer,
                    array_layer_count: (d.array_layer_count != c::ARRAY_LAYER_COUNT_UNDEFINED)
                        .then_some(d.array_layer_count),
                })
            }
        };
        self.insert(
            Object::TextureView(Arc::new(TextureViewState {
                view,
                texture_usage: state.usage,
            })),
            self.instance_of(texture),
        )
    }

    fn texture_set_label(&self, _cx: &mut Cx<'_>, texture: Handle, _label: &str) {
        let _ = self.texture(texture);
    }

    fn texture_get_width(&self, _cx: &mut Cx<'_>, texture: Handle) -> u32 {
        self.texture(texture)
            .map_or(0, |state| state.texture.width())
    }

    fn texture_get_height(&self, _cx: &mut Cx<'_>, texture: Handle) -> u32 {
        self.texture(texture)
            .map_or(0, |state| state.texture.height())
    }

    fn texture_get_depth_or_array_layers(&self, _cx: &mut Cx<'_>, texture: Handle) -> u32 {
        self.texture(texture)
            .map_or(0, |state| state.texture.depth_or_array_layers())
    }

    fn texture_get_mip_level_count(&self, _cx: &mut Cx<'_>, texture: Handle) -> u32 {
        self.texture(texture)
            .map_or(0, |state| state.texture.mip_level_count())
    }

    fn texture_get_sample_count(&self, _cx: &mut Cx<'_>, texture: Handle) -> u32 {
        self.texture(texture)
            .map_or(0, |state| state.texture.sample_count())
    }

    fn texture_get_dimension(&self, _cx: &mut Cx<'_>, texture: Handle) -> TextureDimension {
        self.texture(texture)
            .map_or(TextureDimension::Undefined, |state| {
                convert::texture_dimension_to_abi(state.texture.dimension())
            })
    }

    fn texture_get_texture_binding_view_dimension(
        &self,
        _cx: &mut Cx<'_>,
        texture: Handle,
    ) -> TextureViewDimension {
        // Only meaningful in compatibility mode, which this backend does not
        // offer.
        let _ = self.texture(texture);
        TextureViewDimension::Undefined
    }

    fn texture_get_format(&self, _cx: &mut Cx<'_>, texture: Handle) -> TextureFormat {
        self.texture(texture)
            .map_or(TextureFormat::Undefined, |state| {
                convert::texture_format_to_abi(state.texture.format())
            })
    }

    fn texture_get_usage(&self, _cx: &mut Cx<'_>, texture: Handle) -> u64 {
        self.texture(texture).map_or(0, |state| state.usage)
    }

    fn texture_destroy(&self, _cx: &mut Cx<'_>, texture: Handle) {
        if let Some(state) = self.texture(texture) {
            state.texture.destroy();
            state.release_storage();
        }
    }

    fn texture_add_ref(&self, _cx: &mut Cx<'_>, texture: Handle) {
        self.add_ref(texture, ObjectKind::Texture);
    }

    fn texture_release(&self, _cx: &mut Cx<'_>, texture: Handle) {
        self.release(texture, ObjectKind::Texture);
    }
}

impl TextureViewApi for Host {
    fn texture_view_set_label(&self, _cx: &mut Cx<'_>, texture_view: Handle, _label: &str) {
        let _ = self.texture_view(texture_view);
    }

    fn texture_view_add_ref(&self, _cx: &mut Cx<'_>, texture_view: Handle) {
        self.add_ref(texture_view, ObjectKind::TextureView);
    }

    fn texture_view_release(&self, _cx: &mut Cx<'_>, texture_view: Handle) {
        self.release(texture_view, ObjectKind::TextureView);
    }
}

impl QuerySetApi for Host {
    fn query_set_set_label(&self, _cx: &mut Cx<'_>, query_set: Handle, _label: &str) {
        let _ = self.query_set(query_set);
    }

    fn query_set_get_type(&self, _cx: &mut Cx<'_>, query_set: Handle) -> QueryType {
        self.query_set(query_set)
            .map_or(QueryType::Occlusion, |state| state.ty)
    }

    fn query_set_get_count(&self, _cx: &mut Cx<'_>, query_set: Handle) -> u32 {
        self.query_set(query_set).map_or(0, |state| state.count)
    }

    fn query_set_destroy(&self, _cx: &mut Cx<'_>, query_set: Handle) {
        // wgpu frees a query set when its last user is gone; there is nothing
        // to release early.
        let _ = self.query_set(query_set);
    }

    fn query_set_add_ref(&self, _cx: &mut Cx<'_>, query_set: Handle) {
        self.add_ref(query_set, ObjectKind::QuerySet);
    }

    fn query_set_release(&self, _cx: &mut Cx<'_>, query_set: Handle) {
        self.release(query_set, ObjectKind::QuerySet);
    }
}

// webgpu.h declares the type but has no function that creates one, so these
// can only ever see invalid handles.
impl ExternalTextureApi for Host {
    fn external_texture_set_label(&self, _cx: &mut Cx<'_>, external_texture: Handle, _label: &str) {
        let _ = self.external_texture(external_texture);
    }

    fn external_texture_add_ref(&self, _cx: &mut Cx<'_>, external_texture: Handle) {
        self.add_ref(external_texture, ObjectKind::ExternalTexture);
    }

    fn external_texture_release(&self, _cx: &mut Cx<'_>, external_texture: Handle) {
        self.release(external_texture, ObjectKind::ExternalTexture);
    }
}
