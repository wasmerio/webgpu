//! Compute passes, render passes and render bundle encoders.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::bind::BindGroupState;
use super::buffer::label;
use super::encoder::{EncoderState, whole_size};
use super::pipeline::RenderPipelineState;
use super::{BufferState, DeviceState, Host, Object, ObjectKind, convert, lock};
use crate::abi::api::{
    ComputePassEncoderApi, RenderBundleApi, RenderBundleEncoderApi, RenderPassEncoderApi,
};
use crate::abi::enums::IndexFormat;
use crate::abi::structs::{Color, RenderBundleDescriptor, RenderBundleEncoderDescriptor};
use crate::abi::{Cx, Handle};

pub(crate) struct ComputePassState {
    pub(crate) device: Arc<DeviceState>,
    /// `None` once ended.
    pub(crate) pass: Mutex<Option<wgpu::ComputePass<'static>>>,
}

pub(crate) struct RenderPassState {
    pub(crate) device: Arc<DeviceState>,
    /// `None` once ended.
    pub(crate) pass: Mutex<Option<wgpu::RenderPass<'static>>>,
    /// The encoder the pass records into.
    pub(crate) encoder: Arc<EncoderState>,
    /// `WGPURenderPassMaxDrawCount`: how many draws the pass said it makes
    /// at most. wgpu has no use for the number, so it is enforced here.
    pub(crate) max_draw_count: u64,
    /// Draws recorded so far, those of executed bundles included.
    pub(crate) draw_count: AtomicU64,
}

/// A finished render bundle.
pub(crate) struct RenderBundleState {
    pub(crate) bundle: wgpu::RenderBundle,
    /// Draws the bundle makes each time it is executed.
    draw_count: u64,
}

/// The byte range `[offset, offset + size)` of a buffer, where a size of
/// `WGPU_WHOLE_SIZE` reaches the end. `None` when it is out of bounds: wgpu
/// panics on such a slice rather than reporting it.
fn buffer_range(buffer: &BufferState, offset: u64, size: u64) -> Option<std::ops::Range<u64>> {
    let end = match whole_size(size) {
        Some(size) => offset.checked_add(size)?,
        None => buffer.size,
    };
    (offset <= end && end <= buffer.size).then_some(offset..end)
}

/// `[first, first + count)`, saturating so that a nonsensical count is left
/// for wgpu to reject instead of overflowing here.
fn span(first: u32, count: u32) -> std::ops::Range<u32> {
    first..first.saturating_add(count)
}

impl Host {
    fn with_compute_pass(&self, pass: Handle, f: impl FnOnce(&mut wgpu::ComputePass<'static>)) {
        let Some(state) = self.compute_pass(pass) else {
            return;
        };
        let mut guard = lock(&state.pass);
        match guard.as_mut() {
            Some(pass) => f(pass),
            None => {
                drop(guard);
                self.validation(&state.device, "The compute pass has already ended.");
            }
        }
    }

    fn with_render_pass(
        &self,
        pass: Handle,
        f: impl FnOnce(&RenderPassState, &mut wgpu::RenderPass<'static>),
    ) {
        let Some(state) = self.render_pass(pass) else {
            return;
        };
        let mut guard = lock(&state.pass);
        match guard.as_mut() {
            Some(pass) => f(&state, pass),
            None => {
                drop(guard);
                self.validation(&state.device, "The render pass has already ended.");
            }
        }
    }

    /// A bind group handle for `SetBindGroup`: null unbinds the slot.
    fn optional_bind_group(&self, group: Handle) -> Result<Option<Arc<BindGroupState>>, ()> {
        match group {
            0 => Ok(None),
            handle => self.bind_group(handle).map(Some).ok_or(()),
        }
    }
}

impl ComputePassEncoderApi for Host {
    fn compute_pass_encoder_insert_debug_marker(
        &self,
        _cx: &mut Cx<'_>,
        compute_pass_encoder: Handle,
        marker_label: &str,
    ) {
        self.with_compute_pass(compute_pass_encoder, |pass| {
            pass.insert_debug_marker(marker_label)
        });
    }

    fn compute_pass_encoder_pop_debug_group(&self, _cx: &mut Cx<'_>, compute_pass_encoder: Handle) {
        self.with_compute_pass(compute_pass_encoder, |pass| pass.pop_debug_group());
    }

    fn compute_pass_encoder_push_debug_group(
        &self,
        _cx: &mut Cx<'_>,
        compute_pass_encoder: Handle,
        group_label: &str,
    ) {
        self.with_compute_pass(compute_pass_encoder, |pass| {
            pass.push_debug_group(group_label)
        });
    }

    fn compute_pass_encoder_set_pipeline(
        &self,
        _cx: &mut Cx<'_>,
        compute_pass_encoder: Handle,
        pipeline: Handle,
    ) {
        let Some(pipeline) = self.compute_pipeline(pipeline) else {
            return;
        };
        self.with_compute_pass(compute_pass_encoder, |pass| {
            pass.set_pipeline(&pipeline.pipeline)
        });
    }

    fn compute_pass_encoder_set_bind_group(
        &self,
        _cx: &mut Cx<'_>,
        compute_pass_encoder: Handle,
        group_index: u32,
        group: Handle,
        dynamic_offsets: &[u32],
    ) {
        let Ok(group) = self.optional_bind_group(group) else {
            return;
        };
        self.with_compute_pass(compute_pass_encoder, |pass| {
            pass.set_bind_group(
                group_index,
                group.as_ref().map(|g| &g.group),
                dynamic_offsets,
            );
        });
    }

    fn compute_pass_encoder_set_immediates(
        &self,
        cx: &mut Cx<'_>,
        compute_pass_encoder: Handle,
        offset: u32,
        data: u32,
        size: u32,
    ) {
        let Ok(bytes) = cx.mem.read_vec(data, u64::from(size)) else {
            return;
        };
        self.with_compute_pass(compute_pass_encoder, |pass| {
            pass.set_immediates(offset, &bytes)
        });
    }

    fn compute_pass_encoder_dispatch_workgroups(
        &self,
        _cx: &mut Cx<'_>,
        compute_pass_encoder: Handle,
        workgroupcountx: u32,
        workgroupcounty: u32,
        workgroupcountz: u32,
    ) {
        self.with_compute_pass(compute_pass_encoder, |pass| {
            pass.dispatch_workgroups(workgroupcountx, workgroupcounty, workgroupcountz);
        });
    }

    fn compute_pass_encoder_dispatch_workgroups_indirect(
        &self,
        _cx: &mut Cx<'_>,
        compute_pass_encoder: Handle,
        indirect_buffer: Handle,
        indirect_offset: u64,
    ) {
        let Some(buffer) = self.buffer(indirect_buffer) else {
            return;
        };
        self.with_compute_pass(compute_pass_encoder, |pass| {
            pass.dispatch_workgroups_indirect(&buffer.buffer, indirect_offset);
        });
    }

    fn compute_pass_encoder_end(&self, _cx: &mut Cx<'_>, compute_pass_encoder: Handle) {
        let Some(state) = self.compute_pass(compute_pass_encoder) else {
            return;
        };
        // Dropping the wgpu pass is what ends it.
        if lock(&state.pass).take().is_none() {
            self.validation(&state.device, "The compute pass has already ended.");
        }
    }

    fn compute_pass_encoder_set_label(
        &self,
        _cx: &mut Cx<'_>,
        compute_pass_encoder: Handle,
        _label: &str,
    ) {
        let _ = self.compute_pass(compute_pass_encoder);
    }

    fn compute_pass_encoder_add_ref(&self, _cx: &mut Cx<'_>, compute_pass_encoder: Handle) {
        self.add_ref(compute_pass_encoder, ObjectKind::ComputePassEncoder);
    }

    fn compute_pass_encoder_release(&self, _cx: &mut Cx<'_>, compute_pass_encoder: Handle) {
        self.release(compute_pass_encoder, ObjectKind::ComputePassEncoder);
    }
}

impl RenderPassEncoderApi for Host {
    fn render_pass_encoder_set_pipeline(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        pipeline: Handle,
    ) {
        let Some(pipeline) = self.render_pipeline(pipeline) else {
            return;
        };
        self.with_render_pass(render_pass_encoder, |_, pass| {
            pass.set_pipeline(&pipeline.pipeline)
        });
    }

    fn render_pass_encoder_set_bind_group(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        group_index: u32,
        group: Handle,
        dynamic_offsets: &[u32],
    ) {
        let Ok(group) = self.optional_bind_group(group) else {
            return;
        };
        self.with_render_pass(render_pass_encoder, |_, pass| {
            pass.set_bind_group(
                group_index,
                group.as_ref().map(|g| &g.group),
                dynamic_offsets,
            );
        });
    }

    fn render_pass_encoder_set_immediates(
        &self,
        cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        offset: u32,
        data: u32,
        size: u32,
    ) {
        let Ok(bytes) = cx.mem.read_vec(data, u64::from(size)) else {
            return;
        };
        self.with_render_pass(render_pass_encoder, |_, pass| {
            pass.set_immediates(offset, &bytes)
        });
    }

    fn render_pass_encoder_draw(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.with_render_pass(render_pass_encoder, |state, pass| {
            state.draw_count.fetch_add(1, Ordering::Relaxed);
            pass.draw(
                span(first_vertex, vertex_count),
                span(first_instance, instance_count),
            );
        });
    }

    fn render_pass_encoder_draw_indexed(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        self.with_render_pass(render_pass_encoder, |state, pass| {
            state.draw_count.fetch_add(1, Ordering::Relaxed);
            pass.draw_indexed(
                span(first_index, index_count),
                base_vertex,
                span(first_instance, instance_count),
            );
        });
    }

    fn render_pass_encoder_draw_indirect(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        indirect_buffer: Handle,
        indirect_offset: u64,
    ) {
        let Some(buffer) = self.buffer(indirect_buffer) else {
            return;
        };
        self.with_render_pass(render_pass_encoder, |state, pass| {
            state.draw_count.fetch_add(1, Ordering::Relaxed);
            pass.draw_indirect(&buffer.buffer, indirect_offset);
        });
    }

    fn render_pass_encoder_draw_indexed_indirect(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        indirect_buffer: Handle,
        indirect_offset: u64,
    ) {
        let Some(buffer) = self.buffer(indirect_buffer) else {
            return;
        };
        self.with_render_pass(render_pass_encoder, |state, pass| {
            state.draw_count.fetch_add(1, Ordering::Relaxed);
            pass.draw_indexed_indirect(&buffer.buffer, indirect_offset);
        });
    }

    fn render_pass_encoder_execute_bundles(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        bundles: &[u32],
    ) {
        let Some(bundles) = bundles
            .iter()
            .map(|&handle| self.render_bundle(handle))
            .collect::<Option<Vec<_>>>()
        else {
            return;
        };
        self.with_render_pass(render_pass_encoder, |state, pass| {
            let draws: u64 = bundles.iter().map(|bundle| bundle.draw_count).sum();
            state.draw_count.fetch_add(draws, Ordering::Relaxed);
            pass.execute_bundles(bundles.iter().map(|bundle| &bundle.bundle));
        });
    }

    fn render_pass_encoder_insert_debug_marker(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        marker_label: &str,
    ) {
        self.with_render_pass(render_pass_encoder, |_, pass| {
            pass.insert_debug_marker(marker_label)
        });
    }

    fn render_pass_encoder_pop_debug_group(&self, _cx: &mut Cx<'_>, render_pass_encoder: Handle) {
        self.with_render_pass(render_pass_encoder, |_, pass| pass.pop_debug_group());
    }

    fn render_pass_encoder_push_debug_group(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        group_label: &str,
    ) {
        self.with_render_pass(render_pass_encoder, |_, pass| {
            pass.push_debug_group(group_label)
        });
    }

    fn render_pass_encoder_set_stencil_reference(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        reference: u32,
    ) {
        self.with_render_pass(render_pass_encoder, |_, pass| {
            pass.set_stencil_reference(reference)
        });
    }

    fn render_pass_encoder_set_blend_constant(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        color: &Color,
    ) {
        self.with_render_pass(render_pass_encoder, |_, pass| {
            pass.set_blend_constant(convert::color(color));
        });
    }

    fn render_pass_encoder_set_viewport(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        self.with_render_pass(render_pass_encoder, |_, pass| {
            pass.set_viewport(x, y, width, height, min_depth, max_depth);
        });
    }

    fn render_pass_encoder_set_scissor_rect(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) {
        self.with_render_pass(render_pass_encoder, |_, pass| {
            pass.set_scissor_rect(x, y, width, height);
        });
    }

    fn render_pass_encoder_set_vertex_buffer(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        slot: u32,
        buffer: Handle,
        offset: u64,
        size: u64,
    ) {
        // A null buffer unbinds the slot. wgpu cannot unbind one, and leaving
        // the previous buffer bound draws the same thing as long as the next
        // pipeline does not read the slot, which is the only valid use.
        if buffer == 0 {
            return;
        }
        let Some(buffer) = self.buffer(buffer) else {
            return;
        };
        self.with_render_pass(render_pass_encoder, |state, pass| {
            match buffer_range(&buffer, offset, size) {
                Some(range) => pass.set_vertex_buffer(slot, buffer.buffer.slice(range)),
                None => self.validation(
                    &state.device,
                    "wgpuRenderPassEncoderSetVertexBuffer: range is outside the buffer.",
                ),
            }
        });
    }

    fn render_pass_encoder_set_index_buffer(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        buffer: Handle,
        format: IndexFormat,
        offset: u64,
        size: u64,
    ) {
        let Some(buffer) = self.buffer(buffer) else {
            return;
        };
        self.with_render_pass(render_pass_encoder, |state, pass| {
            let (Some(range), Some(format)) = (
                buffer_range(&buffer, offset, size),
                convert::index_format(format),
            ) else {
                self.validation(
                    &state.device,
                    "wgpuRenderPassEncoderSetIndexBuffer: invalid range or index format.",
                );
                return;
            };
            pass.set_index_buffer(buffer.buffer.slice(range), format);
        });
    }

    fn render_pass_encoder_begin_occlusion_query(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        query_index: u32,
    ) {
        self.with_render_pass(render_pass_encoder, |_, pass| {
            pass.begin_occlusion_query(query_index)
        });
    }

    fn render_pass_encoder_end_occlusion_query(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
    ) {
        self.with_render_pass(render_pass_encoder, |_, pass| pass.end_occlusion_query());
    }

    fn render_pass_encoder_end(&self, _cx: &mut Cx<'_>, render_pass_encoder: Handle) {
        let Some(state) = self.render_pass(render_pass_encoder) else {
            return;
        };
        if lock(&state.pass).take().is_none() {
            self.validation(&state.device, "The render pass has already ended.");
            return;
        }
        // Too many draws make the whole encoder invalid, which is found out
        // when it is finished.
        let draws = state.draw_count.load(Ordering::Relaxed);
        if draws > state.max_draw_count {
            state.encoder.invalidate(format!(
                "The draw count ({draws}) of the render pass is greater than its \
                 maxDrawCount ({}).",
                state.max_draw_count
            ));
        }
    }

    fn render_pass_encoder_set_label(
        &self,
        _cx: &mut Cx<'_>,
        render_pass_encoder: Handle,
        _label: &str,
    ) {
        let _ = self.render_pass(render_pass_encoder);
    }

    fn render_pass_encoder_add_ref(&self, _cx: &mut Cx<'_>, render_pass_encoder: Handle) {
        self.add_ref(render_pass_encoder, ObjectKind::RenderPassEncoder);
    }

    fn render_pass_encoder_release(&self, _cx: &mut Cx<'_>, render_pass_encoder: Handle) {
        self.release(render_pass_encoder, ObjectKind::RenderPassEncoder);
    }
}

// -- render bundles -----------------------------------------------------------

/// One recorded bundle command.
///
/// wgpu's bundle encoder borrows every resource for its whole lifetime, which
/// a handle table cannot express. So the commands are recorded with owning
/// references and replayed into a wgpu encoder inside `finish`, where
/// everything they borrow is on the stack.
enum BundleCommand {
    SetPipeline(Arc<RenderPipelineState>),
    SetBindGroup(u32, Option<Arc<BindGroupState>>, Vec<u32>),
    SetImmediates(u32, Vec<u8>),
    SetVertexBuffer(u32, Arc<BufferState>, std::ops::Range<u64>),
    SetIndexBuffer(Arc<BufferState>, std::ops::Range<u64>, wgpu::IndexFormat),
    Draw(std::ops::Range<u32>, std::ops::Range<u32>),
    DrawIndexed(std::ops::Range<u32>, i32, std::ops::Range<u32>),
    DrawIndirect(Arc<BufferState>, u64),
    DrawIndexedIndirect(Arc<BufferState>, u64),
}

struct BundleRecording {
    label: String,
    color_formats: Vec<Option<wgpu::TextureFormat>>,
    depth_stencil: Option<wgpu::RenderBundleDepthStencil>,
    sample_count: u32,
    commands: Vec<BundleCommand>,
}

pub(crate) struct BundleEncoderState {
    device: Arc<DeviceState>,
    /// `None` once finished.
    recording: Mutex<Option<BundleRecording>>,
}

impl Host {
    pub(crate) fn create_render_bundle_encoder(
        &self,
        device: Handle,
        descriptor: &RenderBundleEncoderDescriptor,
    ) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        // An undefined color format is a hole in the attachment list.
        let color_formats = descriptor
            .color_formats
            .iter()
            .map(|&format| convert::texture_format(format))
            .collect();
        let depth_stencil =
            convert::texture_format(descriptor.depth_stencil_format).map(|format| {
                wgpu::RenderBundleDepthStencil {
                    format,
                    depth_read_only: descriptor.depth_read_only,
                    stencil_read_only: descriptor.stencil_read_only,
                }
            });
        let state = BundleEncoderState {
            device: device_state,
            recording: Mutex::new(Some(BundleRecording {
                label: descriptor.label.clone(),
                color_formats,
                depth_stencil,
                sample_count: descriptor.sample_count.max(1),
                commands: Vec::new(),
            })),
        };
        self.insert(
            Object::RenderBundleEncoder(Arc::new(state)),
            self.instance_of(device),
        )
    }

    fn record(
        &self,
        encoder: Handle,
        command: impl FnOnce(&BundleEncoderState) -> Option<BundleCommand>,
    ) {
        let Some(state) = self.render_bundle_encoder(encoder) else {
            return;
        };
        let Some(command) = command(&state) else {
            return;
        };
        match lock(&state.recording).as_mut() {
            Some(recording) => recording.commands.push(command),
            None => self.validation(
                &state.device,
                "The render bundle encoder is already finished.",
            ),
        }
    }
}

impl RenderBundleEncoderApi for Host {
    fn render_bundle_encoder_set_pipeline(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        pipeline: Handle,
    ) {
        let Some(pipeline) = self.render_pipeline(pipeline) else {
            return;
        };
        self.record(render_bundle_encoder, |_| {
            Some(BundleCommand::SetPipeline(pipeline))
        });
    }

    fn render_bundle_encoder_set_bind_group(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        group_index: u32,
        group: Handle,
        dynamic_offsets: &[u32],
    ) {
        let Ok(group) = self.optional_bind_group(group) else {
            return;
        };
        self.record(render_bundle_encoder, |_| {
            Some(BundleCommand::SetBindGroup(
                group_index,
                group,
                dynamic_offsets.to_vec(),
            ))
        });
    }

    fn render_bundle_encoder_set_immediates(
        &self,
        cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        offset: u32,
        data: u32,
        size: u32,
    ) {
        let Ok(bytes) = cx.mem.read_vec(data, u64::from(size)) else {
            return;
        };
        self.record(render_bundle_encoder, |_| {
            Some(BundleCommand::SetImmediates(offset, bytes))
        });
    }

    fn render_bundle_encoder_draw(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.record(render_bundle_encoder, |_| {
            Some(BundleCommand::Draw(
                span(first_vertex, vertex_count),
                span(first_instance, instance_count),
            ))
        });
    }

    fn render_bundle_encoder_draw_indexed(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) {
        self.record(render_bundle_encoder, |_| {
            Some(BundleCommand::DrawIndexed(
                span(first_index, index_count),
                base_vertex,
                span(first_instance, instance_count),
            ))
        });
    }

    fn render_bundle_encoder_draw_indirect(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        indirect_buffer: Handle,
        indirect_offset: u64,
    ) {
        let Some(buffer) = self.buffer(indirect_buffer) else {
            return;
        };
        self.record(render_bundle_encoder, |_| {
            Some(BundleCommand::DrawIndirect(buffer, indirect_offset))
        });
    }

    fn render_bundle_encoder_draw_indexed_indirect(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        indirect_buffer: Handle,
        indirect_offset: u64,
    ) {
        let Some(buffer) = self.buffer(indirect_buffer) else {
            return;
        };
        self.record(render_bundle_encoder, |_| {
            Some(BundleCommand::DrawIndexedIndirect(buffer, indirect_offset))
        });
    }

    fn render_bundle_encoder_insert_debug_marker(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        _marker_label: &str,
    ) {
        // wgpu 30 has no debug markers on bundle encoders; they carry no
        // rendering meaning, so dropping them is invisible to the guest.
        let _ = self.render_bundle_encoder(render_bundle_encoder);
    }

    fn render_bundle_encoder_pop_debug_group(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
    ) {
        let _ = self.render_bundle_encoder(render_bundle_encoder);
    }

    fn render_bundle_encoder_push_debug_group(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        _group_label: &str,
    ) {
        let _ = self.render_bundle_encoder(render_bundle_encoder);
    }

    fn render_bundle_encoder_set_vertex_buffer(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        slot: u32,
        buffer: Handle,
        offset: u64,
        size: u64,
    ) {
        // See wgpuRenderPassEncoderSetVertexBuffer about the null buffer.
        if buffer == 0 {
            return;
        }
        let Some(buffer) = self.buffer(buffer) else {
            return;
        };
        self.record(render_bundle_encoder, |state| {
            match buffer_range(&buffer, offset, size) {
                Some(range) => Some(BundleCommand::SetVertexBuffer(slot, buffer, range)),
                None => {
                    self.validation(
                        &state.device,
                        "wgpuRenderBundleEncoderSetVertexBuffer: range is outside the buffer.",
                    );
                    None
                }
            }
        });
    }

    fn render_bundle_encoder_set_index_buffer(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        buffer: Handle,
        format: IndexFormat,
        offset: u64,
        size: u64,
    ) {
        let Some(buffer) = self.buffer(buffer) else {
            return;
        };
        self.record(render_bundle_encoder, |state| {
            match (
                buffer_range(&buffer, offset, size),
                convert::index_format(format),
            ) {
                (Some(range), Some(format)) => {
                    Some(BundleCommand::SetIndexBuffer(buffer, range, format))
                }
                _ => {
                    self.validation(
                        &state.device,
                        "wgpuRenderBundleEncoderSetIndexBuffer: invalid range or index format.",
                    );
                    None
                }
            }
        });
    }

    fn render_bundle_encoder_finish(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        descriptor: Option<&RenderBundleDescriptor>,
    ) -> Handle {
        let Some(state) = self.render_bundle_encoder(render_bundle_encoder) else {
            return 0;
        };
        let Some(recording) = lock(&state.recording).take() else {
            self.validation(
                &state.device,
                "The render bundle encoder is already finished.",
            );
            return 0;
        };
        let mut encoder = state.device.device.create_render_bundle_encoder(
            &wgpu::RenderBundleEncoderDescriptor {
                label: label(&recording.label),
                color_formats: &recording.color_formats,
                depth_stencil: recording.depth_stencil,
                sample_count: recording.sample_count,
                multiview: None,
            },
        );
        for command in &recording.commands {
            match command {
                BundleCommand::SetPipeline(pipeline) => encoder.set_pipeline(&pipeline.pipeline),
                BundleCommand::SetBindGroup(index, group, offsets) => {
                    encoder.set_bind_group(*index, group.as_ref().map(|g| &g.group), offsets);
                }
                BundleCommand::SetImmediates(offset, bytes) => {
                    encoder.set_immediates(*offset, bytes)
                }
                BundleCommand::SetVertexBuffer(slot, buffer, range) => {
                    encoder.set_vertex_buffer(*slot, buffer.buffer.slice(range.clone()));
                }
                BundleCommand::SetIndexBuffer(buffer, range, format) => {
                    encoder.set_index_buffer(buffer.buffer.slice(range.clone()), *format);
                }
                BundleCommand::Draw(vertices, instances) => {
                    encoder.draw(vertices.clone(), instances.clone());
                }
                BundleCommand::DrawIndexed(indices, base_vertex, instances) => {
                    encoder.draw_indexed(indices.clone(), *base_vertex, instances.clone());
                }
                BundleCommand::DrawIndirect(buffer, offset) => {
                    encoder.draw_indirect(&buffer.buffer, *offset);
                }
                BundleCommand::DrawIndexedIndirect(buffer, offset) => {
                    encoder.draw_indexed_indirect(&buffer.buffer, *offset);
                }
            }
        }
        // wgpu treats a bundle that fails validation as fatal. Turn that back
        // into the validation error WebGPU specifies.
        let finished = super::catch_expected_panic(|| {
            encoder.finish(&wgpu::RenderBundleDescriptor {
                label: descriptor.and_then(|d| label(&d.label)),
            })
        });
        match finished {
            Ok(bundle) => self.insert(
                Object::RenderBundle(Arc::new(RenderBundleState {
                    bundle,
                    draw_count: recording
                        .commands
                        .iter()
                        .filter(|command| {
                            matches!(
                                command,
                                BundleCommand::Draw(..)
                                    | BundleCommand::DrawIndexed(..)
                                    | BundleCommand::DrawIndirect(..)
                                    | BundleCommand::DrawIndexedIndirect(..)
                            )
                        })
                        .count() as u64,
                })),
                self.instance_of(render_bundle_encoder),
            ),
            Err(message) => {
                self.validation(&state.device, message);
                0
            }
        }
    }

    fn render_bundle_encoder_set_label(
        &self,
        _cx: &mut Cx<'_>,
        render_bundle_encoder: Handle,
        label: &str,
    ) {
        // wgpu names its objects once, when they are made. This one is only
        // made when the recording is finished, so its name can still change.
        if let Some(state) = self.render_bundle_encoder(render_bundle_encoder)
            && let Some(recording) = lock(&state.recording).as_mut()
        {
            recording.label = label.to_owned();
        }
    }

    fn render_bundle_encoder_add_ref(&self, _cx: &mut Cx<'_>, render_bundle_encoder: Handle) {
        self.add_ref(render_bundle_encoder, ObjectKind::RenderBundleEncoder);
    }

    fn render_bundle_encoder_release(&self, _cx: &mut Cx<'_>, render_bundle_encoder: Handle) {
        self.release(render_bundle_encoder, ObjectKind::RenderBundleEncoder);
    }
}

impl RenderBundleApi for Host {
    fn render_bundle_set_label(&self, _cx: &mut Cx<'_>, render_bundle: Handle, _label: &str) {
        let _ = self.render_bundle(render_bundle);
    }

    fn render_bundle_add_ref(&self, _cx: &mut Cx<'_>, render_bundle: Handle) {
        self.add_ref(render_bundle, ObjectKind::RenderBundle);
    }

    fn render_bundle_release(&self, _cx: &mut Cx<'_>, render_bundle: Handle) {
        self.release(render_bundle, ObjectKind::RenderBundle);
    }
}
