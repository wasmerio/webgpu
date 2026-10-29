//! `WGPUQueue` uploads, and immediate data of passes and bundles: calls whose
//! data is a pointer into guest memory. Dawn gets a host copy of the bytes
//! (see `GuestMemory::with_copy`), which it copies again before the call
//! returns.

use std::sync::Arc;

use super::callbacks::{self, Pending, Simple};
use super::ffi::{self, Kind};
use super::lower::{Lowering, usize_from_guest};
use super::{Host, Object, Record};
use crate::abi::enums::{CallbackMode, QueueWorkDoneStatus};
use crate::abi::structs::{Extent3D, TexelCopyBufferLayout, TexelCopyTextureInfo};
use crate::abi::{AbiEnum, Cx, EVENT_QUEUE_WORK_DONE, Handle};

impl Host {
    pub(crate) fn manual_queue_write_buffer(
        &self,
        cx: &mut Cx<'_>,
        queue: Handle,
        buffer: Handle,
        buffer_offset: u64,
        data: u32,
        size: u32,
    ) {
        let Some(this) = self.object(queue, Kind::Queue) else {
            return;
        };
        let mut l = Lowering::new(self, Some(&this));
        let target = match l.object(Kind::Buffer, buffer, "wgpuQueueWriteBuffer.buffer", false) {
            Ok(target) => target,
            Err(error) => return l.fail("wgpuQueueWriteBuffer", error),
        };
        let size = usize_from_guest(size);
        // A wild guest pointer traps the guest; a bad call is Dawn's to report.
        let _ = cx.mem.with_copy(data, size as u64, |bytes| {
            // SAFETY: live objects; `size` host bytes Dawn copies now.
            unsafe {
                ffi::wgpuQueueWriteBuffer(
                    this.raw.cast(),
                    target.cast(),
                    buffer_offset,
                    bytes.as_ptr().cast(),
                    size,
                )
            }
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn manual_queue_write_texture(
        &self,
        cx: &mut Cx<'_>,
        queue: Handle,
        destination: &TexelCopyTextureInfo,
        data: u32,
        data_size: u32,
        data_layout: &TexelCopyBufferLayout,
        write_size: &Extent3D,
    ) {
        let Some(this) = self.object(queue, Kind::Queue) else {
            return;
        };
        let mut l = Lowering::new(self, Some(&this));
        let lowered = (|| {
            Ok((
                l.lower_ptr(destination)?,
                l.lower_ptr(data_layout)?,
                l.lower_ptr(write_size)?,
            ))
        })();
        let (destination, layout, extent) = match lowered {
            Ok(lowered) => lowered,
            Err(error) => return l.fail("wgpuQueueWriteTexture", error),
        };
        let size = usize_from_guest(data_size);
        let _ = cx.mem.with_copy(data, size as u64, |bytes| {
            // SAFETY: live objects, native structs owned by `l`, and `size`
            // host bytes Dawn copies now.
            unsafe {
                ffi::wgpuQueueWriteTexture(
                    this.raw.cast(),
                    destination,
                    bytes.as_ptr().cast(),
                    size,
                    layout,
                    extent,
                )
            }
        });
    }

    /// Starts `wgpuQueueOnSubmittedWorkDone`. The outcome arrives as an event.
    pub(crate) fn queue_on_submitted_work_done(&self, queue: Handle, future: u64) {
        let Some(this) = self.object(queue, Kind::Queue) else {
            self.events.push(Record {
                future,
                kind: EVENT_QUEUE_WORK_DONE,
                status: QueueWorkDoneStatus::Error.as_u32(),
                message: "Invalid queue.".to_owned(),
                ..Record::default()
            });
            return;
        };
        let context = Simple {
            pending: Pending::new(&self.events, &self.futures, future),
            kind: EVENT_QUEUE_WORK_DONE,
        };
        let info = ffi::WGPUQueueWorkDoneCallbackInfo {
            mode: CallbackMode::AllowSpontaneous.as_u32(),
            callback: Some(callbacks::status_message),
            userdata1: callbacks::userdata(context),
            ..Default::default()
        };
        // SAFETY: a live queue; the callback owns its context.
        let dawn_future = unsafe { ffi::wgpuQueueOnSubmittedWorkDone(this.raw.cast(), info) };
        self.futures.track(future, dawn_future);
    }

    /// The three `Set*Immediates` take their bytes from guest memory.
    fn set_immediates(
        &self,
        cx: &mut Cx<'_>,
        this: Option<Object>,
        offset: u32,
        data: u32,
        size: u32,
        call: impl FnOnce(&Arc<super::Native>, u32, *const std::ffi::c_void, usize),
    ) {
        let Some(this) = this else {
            return;
        };
        let size = usize_from_guest(size);
        let _ = cx.mem.with_copy(data, size as u64, |bytes| {
            call(&this, offset, bytes.as_ptr().cast(), size)
        });
    }

    pub(crate) fn manual_render_pass_encoder_set_immediates(
        &self,
        cx: &mut Cx<'_>,
        encoder: Handle,
        offset: u32,
        data: u32,
        size: u32,
    ) {
        let this = self.object(encoder, Kind::RenderPassEncoder);
        self.set_immediates(cx, this, offset, data, size, |this, offset, data, size| {
            // SAFETY: a live encoder and `size` bytes Dawn copies now.
            unsafe { ffi::wgpuRenderPassEncoderSetImmediates(this.raw.cast(), offset, data, size) }
        });
    }

    pub(crate) fn manual_compute_pass_encoder_set_immediates(
        &self,
        cx: &mut Cx<'_>,
        encoder: Handle,
        offset: u32,
        data: u32,
        size: u32,
    ) {
        let this = self.object(encoder, Kind::ComputePassEncoder);
        self.set_immediates(cx, this, offset, data, size, |this, offset, data, size| {
            // SAFETY: a live encoder and `size` bytes Dawn copies now.
            unsafe { ffi::wgpuComputePassEncoderSetImmediates(this.raw.cast(), offset, data, size) }
        });
    }

    pub(crate) fn manual_render_bundle_encoder_set_immediates(
        &self,
        cx: &mut Cx<'_>,
        encoder: Handle,
        offset: u32,
        data: u32,
        size: u32,
    ) {
        let this = self.object(encoder, Kind::RenderBundleEncoder);
        self.set_immediates(cx, this, offset, data, size, |this, offset, data, size| {
            // SAFETY: a live encoder and `size` bytes Dawn copies now.
            unsafe {
                ffi::wgpuRenderBundleEncoderSetImmediates(this.raw.cast(), offset, data, size)
            }
        });
    }
}
