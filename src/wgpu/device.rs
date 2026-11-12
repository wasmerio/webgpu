//! `WGPUDevice`.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use super::adapter::{has_feature, write_limits};
use super::{Events, Host, Object, ObjectKind, Record, lock, scopes};
use crate::abi::api::DeviceApi;
use crate::abi::enums::{DeviceLostReason, ErrorFilter, ErrorType, Status};
use crate::abi::structs::*;
use crate::abi::{AbiEnum, Cx, EVENT_DEVICE_LOST, EVENT_UNCAPTURED_ERROR, Handle, OutPtr};

/// What a device's callbacks need, shared with the closures wgpu holds.
pub(crate) struct DeviceLink {
    /// Identifies the device in the per-thread error scope stacks.
    pub(crate) id: u64,
    /// The guest's handle to the device, or 0 once the guest dropped it.
    pub(crate) handle: AtomicU32,
    pub(crate) lost: AtomicBool,
    pub(crate) lost_future: u64,
    events: Arc<Events>,
}

impl DeviceLink {
    /// Routes an error to the calling thread's innermost matching scope, or
    /// failing that to the device's uncaptured-error callback.
    pub(crate) fn report(&self, ty: ErrorType, message: String) {
        if scopes::capture(self.id, ty, &message) {
            return;
        }
        // "The uncaptured error callback never fires after the device is lost."
        if self.lost.load(Ordering::Acquire) {
            return;
        }
        self.events.push(Record {
            kind: EVENT_UNCAPTURED_ERROR,
            handle: self.handle.load(Ordering::Acquire),
            aux: ty.as_u32(),
            message,
            ..Record::default()
        });
    }

    fn on_lost(&self, reason: DeviceLostReason, message: String) {
        if self.lost.swap(true, Ordering::AcqRel) {
            return;
        }
        self.events.push(Record {
            future: self.lost_future,
            kind: EVENT_DEVICE_LOST,
            status: reason.as_u32(),
            handle: self.handle.load(Ordering::Acquire),
            message,
            ..Record::default()
        });
    }
}

pub(crate) struct DeviceState {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) link: Arc<DeviceLink>,
    pub(crate) adapter_info: wgpu::AdapterInfo,
    /// The guest's queue handle while it holds one, so `wgpuDeviceGetQueue`
    /// keeps returning the same value.
    queue_handle: Mutex<Handle>,
    /// Staging buffers for large texture uploads.
    pub(crate) uploads: super::upload::Uploads,
}

impl DeviceState {
    /// Wraps a new wgpu device and installs its error and loss handlers.
    pub(crate) fn new(
        host: &Host,
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_info: wgpu::AdapterInfo,
        lost_future: u64,
    ) -> Arc<Self> {
        let link = Arc::new(DeviceLink {
            id: host.next_device_id(),
            handle: AtomicU32::new(0),
            lost: AtomicBool::new(false),
            lost_future,
            events: Arc::clone(&host.events),
        });

        // Every wgpu error funnels through here, on the thread whose call
        // caused it, which is what makes per-thread scopes work.
        let errors = Arc::clone(&link);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            let ty = match &error {
                wgpu::Error::OutOfMemory { .. } => ErrorType::OutOfMemory,
                wgpu::Error::Validation { .. } => ErrorType::Validation,
                wgpu::Error::Internal { .. } => ErrorType::Internal,
            };
            errors.report(ty, error.to_string());
        }));

        let lost = Arc::clone(&link);
        device.set_device_lost_callback(move |reason, message| {
            let reason = match reason {
                wgpu::DeviceLostReason::Destroyed => DeviceLostReason::Destroyed,
                wgpu::DeviceLostReason::Unknown => DeviceLostReason::Unknown,
            };
            lost.on_lost(reason, message);
        });

        Arc::new(Self {
            device,
            queue,
            link,
            adapter_info,
            queue_handle: Mutex::new(0),
            uploads: super::upload::Uploads::new(Arc::clone(&host.staging)),
        })
    }

    pub(crate) fn id(&self) -> u64 {
        self.link.id
    }

    pub(crate) fn report(&self, ty: ErrorType, message: String) {
        self.link.report(ty, message);
    }

    /// Destroys the device. The lost callback fires through wgpu.
    pub(crate) fn destroy(&self) {
        self.uploads.clear();
        self.device.destroy();
        // wgpu reports the loss on its next poll; make it immediate so a
        // guest waiting on the lost future does not depend on anything else
        // driving the device.
        self.link.on_lost(
            DeviceLostReason::Destroyed,
            "Device was destroyed.".to_owned(),
        );
    }

    /// Runs `f` under error scopes of its own, out-of-memory included, so no
    /// error it raises reaches the guest: how an async pipeline creation gets
    /// the outcome for its callback. The scopes are popped even after a
    /// panic, so one call can't leave them to the next on the same thread.
    pub(crate) fn catching<R>(&self, f: impl FnOnce() -> R) -> Result<R, (ErrorType, String)> {
        capture_errors(self.id(), f)
    }
}

fn capture_errors<R>(id: u64, f: impl FnOnce() -> R) -> Result<R, (ErrorType, String)> {
    scopes::push(id, ErrorFilter::OutOfMemory);
    scopes::push(id, ErrorFilter::Internal);
    scopes::push(id, ErrorFilter::Validation);
    let value = super::catch_expected_panic(f);
    let validation = scopes::pop(id).flatten();
    let internal = scopes::pop(id).flatten();
    let oom = scopes::pop(id).flatten();
    if let Some(error) = oom.or(internal).or(validation) {
        return Err(error);
    }
    value.map_err(|message| (ErrorType::Internal, message))
}

impl Drop for DeviceState {
    fn drop(&mut self) {
        scopes::forget(self.id());
    }
}

impl Host {
    /// Drops one guest reference to a device. Returns whether it was the last.
    pub(crate) fn release_device(&self, device: Handle) -> bool {
        let Some(Object::Device(state)) = self.release(device, ObjectKind::Device) else {
            return false;
        };
        // Dropping the last reference implies destroying the device, and the
        // lost callback then receives a null device.
        state.link.handle.store(0, Ordering::Release);
        state.destroy();
        true
    }
}

impl DeviceApi for Host {
    fn device_create_bind_group(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &BindGroupDescriptor,
    ) -> Handle {
        self.create_bind_group(device, descriptor)
    }

    fn device_create_bind_group_layout(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &BindGroupLayoutDescriptor,
    ) -> Handle {
        self.create_bind_group_layout(device, descriptor)
    }

    fn device_create_buffer(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &BufferDescriptor,
    ) -> Handle {
        self.create_buffer(device, descriptor)
    }

    fn device_create_command_encoder(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: Option<&CommandEncoderDescriptor>,
    ) -> Handle {
        self.create_command_encoder(device, descriptor)
    }

    fn device_create_compute_pipeline(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &ComputePipelineDescriptor,
    ) -> Handle {
        self.create_compute_pipeline(device, descriptor)
    }

    fn device_create_pipeline_layout(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &PipelineLayoutDescriptor,
    ) -> Handle {
        self.create_pipeline_layout(device, descriptor)
    }

    fn device_create_query_set(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &QuerySetDescriptor,
    ) -> Handle {
        self.create_query_set(device, descriptor)
    }

    fn device_create_render_bundle_encoder(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &RenderBundleEncoderDescriptor,
    ) -> Handle {
        self.create_render_bundle_encoder(device, descriptor)
    }

    fn device_create_render_pipeline(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &RenderPipelineDescriptor,
    ) -> Handle {
        self.create_render_pipeline(device, descriptor)
    }

    fn device_create_sampler(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: Option<&SamplerDescriptor>,
    ) -> Handle {
        self.create_sampler(device, descriptor)
    }

    fn device_create_shader_module(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &ShaderModuleDescriptor,
    ) -> Handle {
        self.create_shader_module(device, descriptor)
    }

    fn device_create_texture(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &TextureDescriptor,
    ) -> Handle {
        self.create_texture(device, descriptor)
    }

    fn device_destroy(&self, _cx: &mut Cx<'_>, device: Handle) {
        if let Some(device) = self.device(device) {
            device.destroy();
        }
    }

    fn device_get_limits(&self, cx: &mut Cx<'_>, device: Handle, limits: OutPtr<Limits>) -> Status {
        match self.device(device) {
            Some(device) => write_limits(&cx.mem, limits, &device.device.limits()),
            None => Status::Error,
        }
    }

    fn device_has_feature(&self, _cx: &mut Cx<'_>, device: Handle, feature: u32) -> bool {
        self.device(device)
            .is_some_and(|device| has_feature(device.device.features(), feature))
    }

    fn device_get_queue(&self, _cx: &mut Cx<'_>, device: Handle) -> Handle {
        let Some(state) = self.device(device) else {
            return 0;
        };
        let mut queue_handle = lock(&state.queue_handle);
        // Hand back the handle the guest already has, if it still has it.
        if *queue_handle != 0
            && let Some(Object::Queue(existing)) = self.object(*queue_handle)
            && Arc::ptr_eq(&existing, &state)
        {
            self.add_ref(*queue_handle, ObjectKind::Queue);
            return *queue_handle;
        }
        let instance = self.instance_of(device);
        *queue_handle = self.insert(Object::Queue(Arc::clone(&state)), instance);
        *queue_handle
    }

    fn device_push_error_scope(&self, _cx: &mut Cx<'_>, device: Handle, filter: ErrorFilter) {
        if let Some(device) = self.device(device) {
            scopes::push(device.id(), filter);
        }
    }

    fn device_set_label(&self, _cx: &mut Cx<'_>, device: Handle, _label: &str) {
        // wgpu takes labels at creation only; a later label is debug metadata
        // this backend cannot attach.
        let _ = self.device(device);
    }

    fn device_add_ref(&self, _cx: &mut Cx<'_>, device: Handle) {
        self.add_ref(device, ObjectKind::Device);
    }
}

#[cfg(test)]
mod catching_tests {
    use super::*;

    #[test]
    fn pipeline_job_captures_all_error_classes_without_touching_caller_scope() {
        scopes::push(111, ErrorFilter::Validation);
        for ty in [
            ErrorType::Validation,
            ErrorType::Internal,
            ErrorType::OutOfMemory,
        ] {
            let result = capture_errors(111, || {
                assert!(scopes::capture(111, ty, "job-local error"));
                7
            });
            assert_eq!(result, Err((ty, "job-local error".into())));
        }
        assert_eq!(scopes::pop(111), Some(None));
        assert_eq!(scopes::pop(111), None);
    }

    #[test]
    fn panic_does_not_leave_scopes_for_the_next_job() {
        assert!(capture_errors(112, || panic!("injected driver panic")).is_err());
        assert_eq!(scopes::pop(112), None);
        assert_eq!(capture_errors(112, || 42), Ok(42));
    }
}
