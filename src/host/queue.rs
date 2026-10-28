//! `WGPUQueue`.

use std::sync::Arc;

use super::{Host, ObjectKind, Record, convert, lock};
use crate::abi::api::QueueApi;
use crate::abi::enums::QueueWorkDoneStatus;
use crate::abi::structs::{Extent3D, TexelCopyBufferLayout, TexelCopyTextureInfo};
use crate::abi::{AbiEnum, Cx, EVENT_QUEUE_WORK_DONE, Handle};

impl Host {
    /// Starts `wgpuQueueOnSubmittedWorkDone`.
    pub(crate) fn queue_on_submitted_work_done(&self, queue: Handle, future: u64) {
        let events = Arc::clone(&self.events);
        let Some(device) = self.queue(queue) else {
            events.push(Record {
                future,
                kind: EVENT_QUEUE_WORK_DONE,
                status: QueueWorkDoneStatus::Error.as_u32(),
                message: "Invalid queue.".to_owned(),
                ..Record::default()
            });
            return;
        };
        device.queue.on_submitted_work_done(move || {
            events.push(Record {
                future,
                kind: EVENT_QUEUE_WORK_DONE,
                status: QueueWorkDoneStatus::Success.as_u32(),
                ..Record::default()
            });
        });
    }
}

impl QueueApi for Host {
    fn queue_submit(&self, _cx: &mut Cx<'_>, queue: Handle, commands: &[u32]) {
        let Some(device) = self.queue(queue) else {
            return;
        };
        let mut buffers = Vec::with_capacity(commands.len());
        for &handle in commands {
            let Some(state) = self.command_buffer(handle) else {
                self.validation(&device, "wgpuQueueSubmit: invalid command buffer.");
                return;
            };
            // A command buffer can be submitted once.
            let Some(buffer) = lock(&state.buffer).take() else {
                self.validation(
                    &device,
                    "wgpuQueueSubmit: command buffer was already submitted.",
                );
                return;
            };
            buffers.push(buffer);
        }
        device.queue.submit(buffers);
    }

    fn queue_write_buffer(
        &self,
        cx: &mut Cx<'_>,
        queue: Handle,
        buffer: Handle,
        buffer_offset: u64,
        data: u32,
        size: u32,
    ) {
        let (Some(device), Some(buffer)) = (self.queue(queue), self.buffer(buffer)) else {
            return;
        };
        // Guest memory is shared with the guest's other threads, so the bytes
        // are copied out before wgpu looks at them.
        let Ok(bytes) = cx.mem.read_vec(data, u64::from(size)) else {
            self.validation(
                &device,
                "wgpuQueueWriteBuffer: data is outside guest memory.",
            );
            return;
        };
        device
            .queue
            .write_buffer(&buffer.buffer, buffer_offset, &bytes);
    }

    fn queue_write_texture(
        &self,
        cx: &mut Cx<'_>,
        queue: Handle,
        destination: &TexelCopyTextureInfo,
        data: u32,
        data_size: u32,
        data_layout: &TexelCopyBufferLayout,
        write_size: &Extent3D,
    ) {
        let (Some(device), Some(texture)) = (self.queue(queue), self.texture(destination.texture))
        else {
            return;
        };
        let destination = wgpu::TexelCopyTextureInfo {
            texture: &texture.texture,
            mip_level: destination.mip_level,
            origin: convert::origin3d(&destination.origin),
            aspect: convert::texture_aspect(destination.aspect),
        };
        let layout = convert::texel_copy_buffer_layout(data_layout);
        let size = convert::extent3d(write_size);
        if Arc::ptr_eq(&texture.device, &device)
            && device.uploads.write_texture(
                &device,
                &cx.mem,
                destination,
                data,
                data_size,
                layout,
                size,
            )
        {
            return;
        }
        let Ok(bytes) = cx.mem.read_vec(data, u64::from(data_size)) else {
            self.validation(
                &device,
                "wgpuQueueWriteTexture: data is outside guest memory.",
            );
            return;
        };
        device
            .queue
            .write_texture(destination, &bytes, layout, size);
    }

    fn queue_set_label(&self, _cx: &mut Cx<'_>, queue: Handle, _label: &str) {
        let _ = self.queue(queue);
    }

    fn queue_add_ref(&self, _cx: &mut Cx<'_>, queue: Handle) {
        self.add_ref(queue, ObjectKind::Queue);
    }

    fn queue_release(&self, _cx: &mut Cx<'_>, queue: Handle) {
        self.release(queue, ObjectKind::Queue);
    }
}
