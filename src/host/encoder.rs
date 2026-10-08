//! Command encoders and command buffers.

use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};

use super::buffer::label;
use super::pass::{ComputePassState, RenderPassState};
use super::{DeviceState, Host, Object, ObjectKind, convert, lock};
use crate::abi::api::{CommandBufferApi, CommandEncoderApi};
use crate::abi::enums::{self as c, LoadOp, texture_usage};
use crate::abi::structs::{
    CommandBufferDescriptor, CommandEncoderDescriptor, ComputePassDescriptor, Extent3D,
    PassTimestampWrites, RenderPassDescriptor, TexelCopyBufferInfo, TexelCopyTextureInfo,
};
use crate::abi::{Cx, Handle};

pub(crate) struct EncoderState {
    pub(crate) device: Arc<DeviceState>,
    /// `None` once finished.
    pub(crate) encoder: Mutex<Option<wgpu::CommandEncoder>>,
    /// Why finishing has to fail, if something recorded made the encoder
    /// invalid. WebGPU reports such mistakes when the encoder is finished,
    /// not where they were made.
    pub(crate) invalid: Mutex<Option<String>>,
}

impl EncoderState {
    /// Makes the encoder invalid. The first reason is the one reported.
    pub(crate) fn invalidate(&self, reason: String) {
        lock(&self.invalid).get_or_insert(reason);
    }
}

/// `maxDrawCount` of a render pass that does not say: what the WebGPU
/// specification defaults it to.
const DEFAULT_MAX_DRAW_COUNT: u64 = 50_000_000;

/// Whether a transient attachment's operations keep to its one rule: nothing
/// goes in and nothing comes out. Undefined operations belong to an aspect
/// the pass does not touch.
fn transient_ops_are_valid(load: LoadOp, store: c::StoreOp) -> bool {
    matches!(load, LoadOp::Clear | LoadOp::Undefined)
        && matches!(store, c::StoreOp::Discard | c::StoreOp::Undefined)
}

pub(crate) struct CommandBufferState {
    /// `None` once submitted.
    pub(crate) buffer: Mutex<Option<wgpu::CommandBuffer>>,
}

/// `WGPU_QUERY_SET_INDEX_UNDEFINED` means "do not write".
fn query_index(value: u32) -> Option<u32> {
    (value != c::QUERY_SET_INDEX_UNDEFINED).then_some(value)
}

/// `WGPU_WHOLE_SIZE` means "to the end".
pub(crate) fn whole_size(value: u64) -> Option<u64> {
    (value != c::WHOLE_SIZE).then_some(value)
}

impl Host {
    pub(crate) fn create_command_encoder(
        &self,
        device: Handle,
        descriptor: Option<&CommandEncoderDescriptor>,
    ) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        let encoder = device_state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: descriptor.and_then(|d| label(&d.label)),
            });
        let state = EncoderState {
            device: device_state,
            encoder: Mutex::new(Some(encoder)),
            invalid: Mutex::new(None),
        };
        self.insert(
            Object::CommandEncoder(Arc::new(state)),
            self.instance_of(device),
        )
    }

    /// Runs `f` on a live encoder, or reports that it was already finished.
    fn with_encoder<R: Default>(
        &self,
        encoder: Handle,
        f: impl FnOnce(&EncoderState, &mut wgpu::CommandEncoder) -> R,
    ) -> R {
        let Some(state) = self.command_encoder(encoder) else {
            return R::default();
        };
        let mut guard = lock(&state.encoder);
        match guard.as_mut() {
            Some(encoder) => f(&state, encoder),
            None => {
                drop(guard);
                self.validation(&state.device, "The command encoder is already finished.");
                R::default()
            }
        }
    }

    fn texel_copy_texture<'a>(
        info: &TexelCopyTextureInfo,
        texture: &'a wgpu::Texture,
    ) -> wgpu::TexelCopyTextureInfo<'a> {
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: info.mip_level,
            origin: convert::origin3d(&info.origin),
            aspect: convert::texture_aspect(info.aspect),
        }
    }

    fn texel_copy_buffer<'a>(
        info: &TexelCopyBufferInfo,
        buffer: &'a wgpu::Buffer,
    ) -> wgpu::TexelCopyBufferInfo<'a> {
        wgpu::TexelCopyBufferInfo {
            buffer,
            layout: convert::texel_copy_buffer_layout(&info.layout),
        }
    }
}

impl CommandEncoderApi for Host {
    fn command_encoder_finish(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        _descriptor: Option<&CommandBufferDescriptor>,
    ) -> Handle {
        let Some(state) = self.command_encoder(command_encoder) else {
            return 0;
        };
        let Some(encoder) = lock(&state.encoder).take() else {
            self.validation(&state.device, "The command encoder is already finished.");
            return 0;
        };
        if let Some(reason) = lock(&state.invalid).take() {
            self.validation(&state.device, reason);
            return 0;
        }
        let buffer = encoder.finish();
        self.insert(
            Object::CommandBuffer(Arc::new(CommandBufferState {
                buffer: Mutex::new(Some(buffer)),
            })),
            self.instance_of(command_encoder),
        )
    }

    fn command_encoder_begin_compute_pass(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        descriptor: Option<&ComputePassDescriptor>,
    ) -> Handle {
        let timestamps = descriptor.and_then(|d| d.timestamp_writes.as_deref());
        let query_set = match timestamps {
            Some(writes) => match self.query_set(writes.query_set) {
                Some(set) => Some(set),
                None => {
                    if let Some(encoder) = self.command_encoder(command_encoder) {
                        self.validation(
                            &encoder.device,
                            "wgpuCommandEncoderBeginComputePass: invalid timestamp query set.",
                        );
                    }
                    return 0;
                }
            },
            None => None,
        };
        let pass = self.with_encoder(command_encoder, |state, encoder| {
            let pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: descriptor.and_then(|d| label(&d.label)),
                    timestamp_writes: timestamps.zip(query_set.as_ref()).map(
                        |(writes, set): (&PassTimestampWrites, _)| {
                            wgpu::ComputePassTimestampWrites {
                                query_set: &set.set,
                                beginning_of_pass_write_index: query_index(
                                    writes.beginning_of_pass_write_index,
                                ),
                                end_of_pass_write_index: query_index(
                                    writes.end_of_pass_write_index,
                                ),
                            }
                        },
                    ),
                })
                .forget_lifetime();
            Some(ComputePassState {
                device: Arc::clone(&state.device),
                pass: Mutex::new(Some(pass)),
            })
        });
        match pass {
            Some(pass) => self.insert(
                Object::ComputePassEncoder(Arc::new(pass)),
                self.instance_of(command_encoder),
            ),
            None => 0,
        }
    }

    fn command_encoder_begin_render_pass(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        descriptor: &RenderPassDescriptor,
    ) -> Handle {
        let Some(encoder_state) = self.command_encoder(command_encoder) else {
            return 0;
        };
        let fail = |message: &str| {
            self.validation(
                &encoder_state.device,
                format!("wgpuCommandEncoderBeginRenderPass: {message}"),
            );
            0
        };

        // Resolve every view up front; the wgpu descriptor borrows from them.
        struct Color {
            view: Arc<super::texture::TextureViewState>,
            resolve: Option<Arc<super::texture::TextureViewState>>,
        }
        let mut colors: Vec<Option<Color>> = Vec::with_capacity(descriptor.color_attachments.len());
        for attachment in &descriptor.color_attachments {
            // A null view is a hole in the attachment list.
            if attachment.view == 0 {
                colors.push(None);
                continue;
            }
            let Some(view) = self.texture_view(attachment.view) else {
                return fail("invalid color attachment view.");
            };
            let resolve = match attachment.resolve_target {
                0 => None,
                handle => match self.texture_view(handle) {
                    Some(view) => Some(view),
                    None => return fail("invalid resolve target."),
                },
            };
            if attachment.load_op == LoadOp::Undefined {
                return fail("a color attachment needs a load op.");
            }
            if view.texture_usage & texture_usage::TRANSIENT_ATTACHMENT != 0
                && !transient_ops_are_valid(attachment.load_op, attachment.store_op)
            {
                return fail(
                    "a transient color attachment must be cleared when the pass begins \
                     and discarded when it ends.",
                );
            }
            colors.push(Some(Color { view, resolve }));
        }
        let depth_view = match &descriptor.depth_stencil_attachment {
            Some(attachment) => match self.texture_view(attachment.view) {
                Some(view) => {
                    if view.texture_usage & texture_usage::TRANSIENT_ATTACHMENT != 0
                        && !(transient_ops_are_valid(
                            attachment.depth_load_op,
                            attachment.depth_store_op,
                        ) && transient_ops_are_valid(
                            attachment.stencil_load_op,
                            attachment.stencil_store_op,
                        ))
                    {
                        return fail(
                            "a transient depth-stencil attachment must be cleared when the \
                             pass begins and discarded when it ends.",
                        );
                    }
                    Some(view)
                }
                None => return fail("invalid depth-stencil attachment view."),
            },
            None => None,
        };
        let occlusion = match descriptor.occlusion_query_set {
            0 => None,
            handle => match self.query_set(handle) {
                Some(set) => Some(set),
                None => return fail("invalid occlusion query set."),
            },
        };
        let timestamps = descriptor.timestamp_writes.as_deref();
        let timestamp_set = match timestamps {
            Some(writes) => match self.query_set(writes.query_set) {
                Some(set) => Some(set),
                None => return fail("invalid timestamp query set."),
            },
            None => None,
        };

        let color_attachments: Vec<_> = descriptor
            .color_attachments
            .iter()
            .zip(&colors)
            .map(|(attachment, color)| {
                let color = color.as_ref()?;
                Some(wgpu::RenderPassColorAttachment {
                    view: &color.view.view,
                    depth_slice: (attachment.depth_slice != c::DEPTH_SLICE_UNDEFINED)
                        .then_some(attachment.depth_slice),
                    resolve_target: color.resolve.as_ref().map(|view| &view.view),
                    ops: wgpu::Operations {
                        load: match attachment.load_op {
                            LoadOp::Clear => {
                                wgpu::LoadOp::Clear(convert::color(&attachment.clear_value))
                            }
                            _ => wgpu::LoadOp::Load,
                        },
                        store: convert::store_op(attachment.store_op),
                    },
                })
            })
            .collect();
        let depth_stencil_attachment = descriptor
            .depth_stencil_attachment
            .as_deref()
            .zip(depth_view.as_ref())
            .map(|(attachment, view)| {
                // An aspect with neither a load nor a store op is one the
                // pass does not touch (absent from the format, or read-only).
                let depth_ops = (!attachment.depth_read_only
                    && (attachment.depth_load_op != LoadOp::Undefined
                        || attachment.depth_store_op != c::StoreOp::Undefined))
                    .then(|| wgpu::Operations {
                        load: match attachment.depth_load_op {
                            LoadOp::Clear => wgpu::LoadOp::Clear(attachment.depth_clear_value),
                            _ => wgpu::LoadOp::Load,
                        },
                        store: convert::store_op(attachment.depth_store_op),
                    });
                let stencil_ops = (!attachment.stencil_read_only
                    && (attachment.stencil_load_op != LoadOp::Undefined
                        || attachment.stencil_store_op != c::StoreOp::Undefined))
                    .then(|| wgpu::Operations {
                        load: match attachment.stencil_load_op {
                            LoadOp::Clear => wgpu::LoadOp::Clear(attachment.stencil_clear_value),
                            _ => wgpu::LoadOp::Load,
                        },
                        store: convert::store_op(attachment.stencil_store_op),
                    });
                wgpu::RenderPassDepthStencilAttachment {
                    view: &view.view,
                    depth_ops,
                    stencil_ops,
                }
            });

        let mut guard = lock(&encoder_state.encoder);
        let Some(encoder) = guard.as_mut() else {
            drop(guard);
            return fail("the command encoder is already finished.");
        };
        let pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: label(&descriptor.label),
                color_attachments: &color_attachments,
                depth_stencil_attachment,
                timestamp_writes: timestamps.zip(timestamp_set.as_ref()).map(|(writes, set)| {
                    wgpu::RenderPassTimestampWrites {
                        query_set: &set.set,
                        beginning_of_pass_write_index: query_index(
                            writes.beginning_of_pass_write_index,
                        ),
                        end_of_pass_write_index: query_index(writes.end_of_pass_write_index),
                    }
                }),
                occlusion_query_set: occlusion.as_ref().map(|set| &set.set),
                multiview_mask: None,
            })
            .forget_lifetime();
        drop(guard);
        self.insert(
            Object::RenderPassEncoder(Arc::new(RenderPassState {
                device: Arc::clone(&encoder_state.device),
                pass: Mutex::new(Some(pass)),
                encoder: Arc::clone(&encoder_state),
                max_draw_count: descriptor
                    .chain
                    .render_pass_max_draw_count()
                    .map_or(DEFAULT_MAX_DRAW_COUNT, |limit| limit.max_draw_count),
                draw_count: AtomicU64::new(0),
            })),
            self.instance_of(command_encoder),
        )
    }

    fn command_encoder_copy_buffer_to_buffer(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        source: Handle,
        source_offset: u64,
        destination: Handle,
        destination_offset: u64,
        size: u64,
    ) {
        let (Some(source), Some(destination)) = (self.buffer(source), self.buffer(destination))
        else {
            return;
        };
        self.with_encoder(command_encoder, |_, encoder| {
            encoder.copy_buffer_to_buffer(
                &source.buffer,
                source_offset,
                &destination.buffer,
                destination_offset,
                whole_size(size),
            );
        });
    }

    fn command_encoder_copy_buffer_to_texture(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        source: &TexelCopyBufferInfo,
        destination: &TexelCopyTextureInfo,
        copy_size: &Extent3D,
    ) {
        let (Some(buffer), Some(texture)) = (
            self.buffer(source.buffer),
            self.texture(destination.texture),
        ) else {
            return;
        };
        self.with_encoder(command_encoder, |_, encoder| {
            encoder.copy_buffer_to_texture(
                Self::texel_copy_buffer(source, &buffer.buffer),
                Self::texel_copy_texture(destination, &texture.texture),
                convert::extent3d(copy_size),
            );
        });
    }

    fn command_encoder_copy_texture_to_buffer(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        source: &TexelCopyTextureInfo,
        destination: &TexelCopyBufferInfo,
        copy_size: &Extent3D,
    ) {
        let (Some(texture), Some(buffer)) = (
            self.texture(source.texture),
            self.buffer(destination.buffer),
        ) else {
            return;
        };
        self.with_encoder(command_encoder, |_, encoder| {
            encoder.copy_texture_to_buffer(
                Self::texel_copy_texture(source, &texture.texture),
                Self::texel_copy_buffer(destination, &buffer.buffer),
                convert::extent3d(copy_size),
            );
        });
    }

    fn command_encoder_copy_texture_to_texture(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        source: &TexelCopyTextureInfo,
        destination: &TexelCopyTextureInfo,
        copy_size: &Extent3D,
    ) {
        let (Some(from), Some(to)) = (
            self.texture(source.texture),
            self.texture(destination.texture),
        ) else {
            return;
        };
        self.with_encoder(command_encoder, |_, encoder| {
            encoder.copy_texture_to_texture(
                Self::texel_copy_texture(source, &from.texture),
                Self::texel_copy_texture(destination, &to.texture),
                convert::extent3d(copy_size),
            );
        });
    }

    fn command_encoder_clear_buffer(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        buffer: Handle,
        offset: u64,
        size: u64,
    ) {
        let Some(buffer) = self.buffer(buffer) else {
            return;
        };
        self.with_encoder(command_encoder, |_, encoder| {
            encoder.clear_buffer(&buffer.buffer, offset, whole_size(size));
        });
    }

    fn command_encoder_insert_debug_marker(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        marker_label: &str,
    ) {
        self.with_encoder(command_encoder, |_, encoder| {
            encoder.insert_debug_marker(marker_label);
        });
    }

    fn command_encoder_pop_debug_group(&self, _cx: &mut Cx<'_>, command_encoder: Handle) {
        self.with_encoder(command_encoder, |_, encoder| encoder.pop_debug_group());
    }

    fn command_encoder_push_debug_group(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        group_label: &str,
    ) {
        self.with_encoder(command_encoder, |_, encoder| {
            encoder.push_debug_group(group_label);
        });
    }

    fn command_encoder_resolve_query_set(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        query_set: Handle,
        first_query: u32,
        query_count: u32,
        destination: Handle,
        destination_offset: u64,
    ) {
        let (Some(set), Some(destination)) = (self.query_set(query_set), self.buffer(destination))
        else {
            return;
        };
        self.with_encoder(command_encoder, |state, encoder| {
            let Some(end) = first_query.checked_add(query_count) else {
                self.validation(
                    &state.device,
                    "wgpuCommandEncoderResolveQuerySet: query range overflows.",
                );
                return;
            };
            encoder.resolve_query_set(
                &set.set,
                first_query..end,
                &destination.buffer,
                destination_offset,
            );
        });
    }

    fn command_encoder_write_timestamp(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        query_set: Handle,
        query_index: u32,
    ) {
        let Some(set) = self.query_set(query_set) else {
            return;
        };
        self.with_encoder(command_encoder, |_, encoder| {
            encoder.write_timestamp(&set.set, query_index);
        });
    }

    fn command_encoder_set_label(&self, _cx: &mut Cx<'_>, command_encoder: Handle, _label: &str) {
        let _ = self.command_encoder(command_encoder);
    }

    fn command_encoder_add_ref(&self, _cx: &mut Cx<'_>, command_encoder: Handle) {
        self.add_ref(command_encoder, ObjectKind::CommandEncoder);
    }

    fn command_encoder_release(&self, _cx: &mut Cx<'_>, command_encoder: Handle) {
        self.release(command_encoder, ObjectKind::CommandEncoder);
    }
}

impl CommandBufferApi for Host {
    fn command_buffer_set_label(&self, _cx: &mut Cx<'_>, command_buffer: Handle, _label: &str) {
        let _ = self.command_buffer(command_buffer);
    }

    fn command_buffer_add_ref(&self, _cx: &mut Cx<'_>, command_buffer: Handle) {
        self.add_ref(command_buffer, ObjectKind::CommandBuffer);
    }

    fn command_buffer_release(&self, _cx: &mut Cx<'_>, command_buffer: Handle) {
        self.release(command_buffer, ObjectKind::CommandBuffer);
    }
}
