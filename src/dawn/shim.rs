//! The imports that exist only for the guest shim: starting asynchronous
//! work, collecting completed events, and the getters whose results the shim
//! has to own.
//!
//! Collecting is where Dawn makes progress: a poll processes Dawn's events,
//! and a wait blocks inside Dawn on the futures it is waiting for (or on the
//! event queue, for events Dawn does not complete, such as another guest
//! thread's).

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::callbacks::{self, DonePipelines, Pending, PipelineRequest, Simple};
use super::ffi::{self, Kind};
use super::instance::{TIMED_WAIT_ANY_MAX_COUNT, wgsl_language_features};
use super::lower::Lowering;
use super::{Extra, Host, Native, Record, lock};
use crate::abi::api::ShimApi;
use crate::abi::enums::{
    CallbackMode, CreatePipelineAsyncStatus, DeviceLostReason, ErrorType, PopErrorScopeStatus,
    RequestAdapterStatus, RequestDeviceStatus, WaitStatus,
};
use crate::abi::structs::{
    ComputePipelineDescriptor, DeviceDescriptor, RenderPipelineDescriptor, RequestAdapterOptions,
};
use crate::abi::{
    AbiEnum, AbiError, AbiResult, Cx, EVENT_COMPILATION_INFO, EVENT_CREATE_COMPUTE_PIPELINE,
    EVENT_CREATE_RENDER_PIPELINE, EVENT_DEVICE_LOST, EVENT_POP_ERROR_SCOPE, EVENT_REQUEST_ADAPTER,
    EVENT_REQUEST_DEVICE, GuestMemory, Handle,
};

/// Longest a waiting thread blocks before it looks again for termination and
/// for events completed by another thread.
const WAIT_SLICE: Duration = Duration::from_millis(20);

/// Copies `bytes` to the guest if they fit. Returns the size either way, so
/// the shim can retry with a buffer that is large enough.
fn write_blob(mem: &GuestMemory<'_>, buf: u32, cap: u32, bytes: &[u8]) -> AbiResult<i32> {
    if bytes.len() <= cap as usize && !bytes.is_empty() {
        mem.write_bytes(buf, bytes)?;
    }
    Ok(bytes.len().min(i32::MAX as usize) as i32)
}

fn u32_blob(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_le_bytes).collect()
}

impl Host {
    /// Sorts out a descriptor that could not be decoded for a call that
    /// completes a future. A pointer outside the guest's memory traps, as
    /// everywhere. Anything else becomes the message of the failed future:
    /// the guest was handed that future, and a callback that never runs
    /// would leave it waiting for ever.
    fn decoded<T>(
        &self,
        function: &'static str,
        result: AbiResult<T>,
    ) -> AbiResult<Result<T, String>> {
        match result {
            Ok(value) => Ok(Ok(value)),
            Err(error @ AbiError::OutOfBounds { .. }) => Err(error),
            Err(error) => {
                super::log(|| format!("{function}: {error}"));
                Ok(Err(format!("{function}: {error}")))
            }
        }
    }

    /// Lets Dawn deliver whatever completed, then gives finished pipelines
    /// their handles.
    fn progress(&self) {
        if let Some(dawn) = self.dawn.get().and_then(Option::as_ref) {
            // SAFETY: the host's live instance.
            unsafe { ffi::wgpuInstanceProcessEvents(dawn.0) };
        }
        self.publish_pipelines();
    }

    /// Registers pipelines Dawn finished compiling and queues their events.
    /// A pipeline whose instance the guest released meanwhile goes with it.
    fn publish_pipelines(&self) {
        let done = std::mem::take(&mut *lock(&self.pipelines.list));
        for mut pipeline in done {
            // A pipeline whose instance is gone is dropped, which releases it.
            if self.object(pipeline.instance, Kind::Instance).is_none() {
                continue;
            }
            let raw = std::mem::replace(&mut pipeline.raw, std::ptr::null_mut());
            let object = (!raw.is_null()).then(|| {
                Arc::new(Native {
                    kind: pipeline.kind,
                    raw,
                    device: Some(Arc::clone(&pipeline.device)),
                    extra: Extra::None,
                })
            });
            let handle = object.map_or(0, |object| self.insert(object, pipeline.instance));
            let status =
                if pipeline.status == CreatePipelineAsyncStatus::Success.as_u32() && handle == 0 {
                    CreatePipelineAsyncStatus::InternalError.as_u32()
                } else {
                    pipeline.status
                };
            self.events.push(Record {
                future: pipeline.future,
                kind: pipeline.event,
                status,
                handle,
                message: std::mem::take(&mut pipeline.message),
                pipeline_instance: pipeline.instance,
                ..Record::default()
            });
        }
    }

    fn fail_pipeline_async(&self, kind: u32, future: u64, message: String) {
        self.events.push(Record {
            future,
            kind,
            status: CreatePipelineAsyncStatus::ValidationError.as_u32(),
            message,
            ..Record::default()
        });
    }

    /// Starts an asynchronous pipeline creation in Dawn.
    fn create_pipeline_async<T: super::lower::Lower>(
        &self,
        device: Handle,
        descriptor: &T,
        future: u64,
        event: u32,
        kind: Kind,
        start: impl FnOnce(ffi::WGPUDevice, *const T::Native, *mut std::ffi::c_void) -> ffi::WGPUFuture,
    ) where
        T::Native: 'static,
    {
        let Some(parent) = self.object(device, Kind::Device) else {
            self.fail_pipeline_async(event, future, "Invalid device.".to_owned());
            return;
        };
        let Some(state) = parent.device.clone() else {
            self.fail_pipeline_async(event, future, "Invalid device.".to_owned());
            return;
        };
        let mut l = Lowering::new(self, Some(&parent));
        // An invalid descriptor fails the future, as a validation error.
        let native = match l.lower_ptr(descriptor) {
            Ok(native) => native,
            Err(error) => {
                self.fail_pipeline_async(event, future, error.to_string());
                return;
            }
        };
        let request = PipelineRequest {
            pending: Pending::new(&self.events, &self.futures, future),
            event,
            kind,
            done: Arc::clone(&self.pipelines),
            device: state.clone(),
            instance: self.instance_of(device),
        };
        let dawn_future = start(state.raw, native, callbacks::userdata(request));
        self.futures.track(future, dawn_future);
    }
}

impl ShimApi for Host {
    fn wasmer_wgpu_abi_version(&self, _cx: &mut Cx<'_>) -> AbiResult<i32> {
        Ok(crate::WEBGPU_ABI_VERSION as i32)
    }

    fn wasmer_wgpu_object_instance(&self, _cx: &mut Cx<'_>, handle: u32) -> AbiResult<i32> {
        Ok(self.instance_of(handle) as i32)
    }

    fn wasmer_wgpu_instance_request_adapter(
        &self,
        cx: &mut Cx<'_>,
        instance: u32,
        options: u32,
        future: u64,
    ) -> AbiResult<()> {
        let options = match self.decoded(
            "wgpuInstanceRequestAdapter",
            cx.mem.read_opt::<RequestAdapterOptions>(options),
        )? {
            Ok(options) => options,
            Err(message) => {
                self.events.push(Record {
                    future,
                    kind: EVENT_REQUEST_ADAPTER,
                    status: RequestAdapterStatus::Error.as_u32(),
                    message,
                    ..Record::default()
                });
                return Ok(());
            }
        };
        let record = match self.request_adapter(instance, options.as_deref()) {
            Ok(adapter) => {
                let handle = self.insert_adapter(adapter, instance);
                Record {
                    future,
                    kind: EVENT_REQUEST_ADAPTER,
                    status: if handle != 0 {
                        RequestAdapterStatus::Success
                    } else {
                        RequestAdapterStatus::Error
                    }
                    .as_u32(),
                    handle,
                    ..Record::default()
                }
            }
            Err(message) => Record {
                future,
                kind: EVENT_REQUEST_ADAPTER,
                status: RequestAdapterStatus::Unavailable.as_u32(),
                message,
                ..Record::default()
            },
        };
        self.events.push(record);
        Ok(())
    }

    fn wasmer_wgpu_adapter_request_device(
        &self,
        cx: &mut Cx<'_>,
        adapter: u32,
        descriptor: u32,
        future: u64,
        lost_future: u64,
    ) -> AbiResult<()> {
        let request = match self.decoded(
            "wgpuAdapterRequestDevice",
            cx.mem.read_opt::<DeviceDescriptor>(descriptor),
        )? {
            Ok(descriptor) => self.request_device(adapter, descriptor.as_deref(), lost_future),
            Err(message) => Err(super::device::DeviceFailure {
                message,
                lost_reported: false,
            }),
        };
        match request {
            Ok(device) => {
                let handle = self.insert_device(&device, self.instance_of(adapter));
                if handle == 0 {
                    device.destroy();
                }
                self.events.push(Record {
                    future,
                    kind: EVENT_REQUEST_DEVICE,
                    status: if handle != 0 {
                        RequestDeviceStatus::Success
                    } else {
                        RequestDeviceStatus::Error
                    }
                    .as_u32(),
                    handle,
                    ..Record::default()
                });
            }
            Err(failure) => {
                self.events.push(Record {
                    future,
                    kind: EVENT_REQUEST_DEVICE,
                    status: RequestDeviceStatus::Error.as_u32(),
                    message: failure.message.clone(),
                    ..Record::default()
                });
                if !failure.lost_reported {
                    // The device-lost future of a device that never existed.
                    self.events.push(Record {
                        future: lost_future,
                        kind: EVENT_DEVICE_LOST,
                        status: DeviceLostReason::FailedCreation.as_u32(),
                        message: failure.message,
                        ..Record::default()
                    });
                }
            }
        }
        Ok(())
    }

    fn wasmer_wgpu_buffer_map_async(
        &self,
        _cx: &mut Cx<'_>,
        buffer: u32,
        mode: u64,
        offset: u32,
        size: u32,
        future: u64,
    ) -> AbiResult<()> {
        self.buffer_map_async(buffer, mode, u64::from(offset), u64::from(size), future);
        Ok(())
    }

    fn wasmer_wgpu_device_pop_error_scope(
        &self,
        _cx: &mut Cx<'_>,
        device: u32,
        future: u64,
    ) -> AbiResult<()> {
        let Some(state) = self.device(device) else {
            self.events.push(Record {
                future,
                kind: EVENT_POP_ERROR_SCOPE,
                status: PopErrorScopeStatus::Error.as_u32(),
                aux: ErrorType::NoError.as_u32(),
                message: "Invalid device.".to_owned(),
                ..Record::default()
            });
            return Ok(());
        };
        let context = Simple {
            pending: Pending::new(&self.events, &self.futures, future),
            kind: EVENT_POP_ERROR_SCOPE,
        };
        let info = ffi::WGPUPopErrorScopeCallbackInfo {
            mode: CallbackMode::AllowSpontaneous.as_u32(),
            callback: Some(callbacks::pop_error_scope),
            userdata1: callbacks::userdata(context),
            ..Default::default()
        };
        // SAFETY: a live device; the callback owns its context.
        let dawn_future = unsafe { ffi::wgpuDevicePopErrorScope(state.raw, info) };
        self.futures.track(future, dawn_future);
        Ok(())
    }

    fn wasmer_wgpu_device_create_compute_pipeline_async(
        &self,
        cx: &mut Cx<'_>,
        device: u32,
        descriptor: u32,
        future: u64,
    ) -> AbiResult<()> {
        const FUNCTION: &str = "wgpuDeviceCreateComputePipelineAsync";
        let descriptor = self.decoded(
            FUNCTION,
            cx.mem
                .read_required::<ComputePipelineDescriptor>(descriptor, "descriptor"),
        )?;
        match descriptor {
            Ok(descriptor) => self.create_pipeline_async(
                device,
                &descriptor,
                future,
                EVENT_CREATE_COMPUTE_PIPELINE,
                Kind::ComputePipeline,
                |device, native, userdata| {
                    let info = ffi::WGPUCreateComputePipelineAsyncCallbackInfo {
                        mode: CallbackMode::AllowSpontaneous.as_u32(),
                        callback: Some(callbacks::compute_pipeline),
                        userdata1: userdata,
                        ..Default::default()
                    };
                    // SAFETY: a live device, a descriptor that outlives the
                    // call; the callback owns its context.
                    unsafe { ffi::wgpuDeviceCreateComputePipelineAsync(device, native, info) }
                },
            ),
            Err(message) => {
                self.fail_pipeline_async(EVENT_CREATE_COMPUTE_PIPELINE, future, message)
            }
        }
        Ok(())
    }

    fn wasmer_wgpu_device_create_render_pipeline_async(
        &self,
        cx: &mut Cx<'_>,
        device: u32,
        descriptor: u32,
        future: u64,
    ) -> AbiResult<()> {
        const FUNCTION: &str = "wgpuDeviceCreateRenderPipelineAsync";
        let descriptor = self.decoded(
            FUNCTION,
            cx.mem
                .read_required::<RenderPipelineDescriptor>(descriptor, "descriptor"),
        )?;
        match descriptor {
            Ok(descriptor) => self.create_pipeline_async(
                device,
                &descriptor,
                future,
                EVENT_CREATE_RENDER_PIPELINE,
                Kind::RenderPipeline,
                |device, native, userdata| {
                    let info = ffi::WGPUCreateRenderPipelineAsyncCallbackInfo {
                        mode: CallbackMode::AllowSpontaneous.as_u32(),
                        callback: Some(callbacks::render_pipeline),
                        userdata1: userdata,
                        ..Default::default()
                    };
                    // SAFETY: a live device, a descriptor that outlives the
                    // call; the callback owns its context.
                    unsafe { ffi::wgpuDeviceCreateRenderPipelineAsync(device, native, info) }
                },
            ),
            Err(message) => self.fail_pipeline_async(EVENT_CREATE_RENDER_PIPELINE, future, message),
        }
        Ok(())
    }

    fn wasmer_wgpu_queue_on_submitted_work_done(
        &self,
        _cx: &mut Cx<'_>,
        queue: u32,
        future: u64,
    ) -> AbiResult<()> {
        self.queue_on_submitted_work_done(queue, future);
        Ok(())
    }

    fn wasmer_wgpu_shader_module_get_compilation_info(
        &self,
        _cx: &mut Cx<'_>,
        shader_module: u32,
        future: u64,
    ) -> AbiResult<()> {
        let Some(module) = self.object(shader_module, Kind::ShaderModule) else {
            // No module, no messages.
            self.events.push(Record {
                future,
                kind: EVENT_COMPILATION_INFO,
                status: crate::abi::enums::CompilationInfoRequestStatus::Success.as_u32(),
                extra: vec![0; 8],
                ..Record::default()
            });
            return Ok(());
        };
        let context = Simple {
            pending: Pending::new(&self.events, &self.futures, future),
            kind: EVENT_COMPILATION_INFO,
        };
        let info = ffi::WGPUCompilationInfoCallbackInfo {
            mode: CallbackMode::AllowSpontaneous.as_u32(),
            callback: Some(callbacks::compilation_info),
            userdata1: callbacks::userdata(context),
            ..Default::default()
        };
        // SAFETY: a live module; the callback owns its context.
        let dawn_future =
            unsafe { ffi::wgpuShaderModuleGetCompilationInfo(module.raw.cast(), info) };
        self.futures.track(future, dawn_future);
        Ok(())
    }

    fn wasmer_wgpu_poll(&self, cx: &mut Cx<'_>, buf: u32, cap: u32) -> AbiResult<i32> {
        self.progress();
        self.events.drain_into(&cx.mem, buf, cap)
    }

    fn wasmer_wgpu_epoch(&self, _cx: &mut Cx<'_>) -> AbiResult<u64> {
        Ok(self.events.epoch())
    }

    fn wasmer_wgpu_wait(
        &self,
        cx: &mut Cx<'_>,
        futures: u32,
        count: u32,
        timeout_ns: u64,
        epoch: u64,
        buf: u32,
        cap: u32,
    ) -> AbiResult<i32> {
        let futures = cx.mem.read_u64_array(futures, count)?;
        // A timeout too large to add to the clock is "forever".
        let deadline = Instant::now().checked_add(Duration::from_nanos(timeout_ns));
        loop {
            self.progress();
            // An event queued since the caller last looked may be the one it
            // waits for, already collected by another guest thread: let the
            // caller look again rather than sleep on a queue that will not
            // show it.
            if self.events.is_ready(&futures)
                || self.events.epoch() != epoch
                || self.is_terminated()
            {
                break;
            }
            // A killed guest would otherwise wait here for as long as it
            // asked to, and for ever if that is what it asked.
            if cx.stopped() {
                return Err(AbiError::Stopped);
            }
            let remaining = match deadline {
                Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                None => WAIT_SLICE,
            };
            if remaining.is_zero() {
                break;
            }
            let slice = remaining.min(WAIT_SLICE);
            // Block where progress will come from: inside Dawn for the
            // futures it completes, otherwise until another thread queues an
            // event.
            let mut ids = self.futures.dawn_ids(&futures);
            ids.truncate(TIMED_WAIT_ANY_MAX_COUNT as usize);
            let waited = match self.dawn.get().and_then(Option::as_ref) {
                Some(dawn) if !ids.is_empty() => {
                    let mut waits: Vec<_> = ids
                        .into_iter()
                        .map(|id| ffi::WGPUFutureWaitInfo {
                            future: ffi::WGPUFuture { id },
                            completed: 0,
                        })
                        .collect();
                    // SAFETY: the host's live instance and futures it started.
                    let status = unsafe {
                        ffi::wgpuInstanceWaitAny(
                            dawn.0,
                            waits.len(),
                            waits.as_mut_ptr(),
                            slice.as_nanos() as u64,
                        )
                    };
                    // Dawn may be unable to wait on this mix of futures at
                    // once (several queues, or a queue's with another kind,
                    // on backends without spontaneous queue events): it says
                    // so at once, and the queue below waits instead.
                    matches!(
                        WaitStatus::from_u32(status),
                        Some(WaitStatus::Success | WaitStatus::TimedOut)
                    )
                }
                _ => false,
            };
            if !waited {
                self.events.wait_for_push(slice, epoch);
            }
        }
        self.events.drain_into(&cx.mem, buf, cap)
    }

    fn wasmer_wgpu_yield(&self, _cx: &mut Cx<'_>) -> AbiResult<()> {
        // Natively nothing needs the guest to yield: polling drives Dawn.
        // Give other guest threads a turn all the same.
        std::thread::yield_now();
        Ok(())
    }

    fn wasmer_wgpu_instance_release(&self, _cx: &mut Cx<'_>, instance: u32) -> AbiResult<i32> {
        let removed = self.release_handle(instance, Kind::Instance);
        if let Some(object) = &removed {
            if let Some(state) = super::instance::InstanceState::of(object) {
                state
                    .active
                    .store(false, std::sync::atomic::Ordering::Release);
            }
            // Pipelines made for the instance and not collected go with it.
            for record in self.events.retire_pipeline_instance(instance) {
                let kind = if record.kind == EVENT_CREATE_COMPUTE_PIPELINE {
                    Kind::ComputePipeline
                } else {
                    Kind::RenderPipeline
                };
                if record.handle != 0 {
                    self.release_handle(record.handle, kind);
                }
            }
        }
        Ok(i32::from(removed.is_some()))
    }

    fn wasmer_wgpu_buffer_release(&self, _cx: &mut Cx<'_>, buffer: u32) -> AbiResult<i32> {
        Ok(i32::from(
            self.release_handle(buffer, Kind::Buffer).is_some(),
        ))
    }

    fn wasmer_wgpu_device_release(&self, _cx: &mut Cx<'_>, device: u32) -> AbiResult<i32> {
        Ok(i32::from(self.release_device(device)))
    }

    fn wasmer_wgpu_buffer_map_info(
        &self,
        cx: &mut Cx<'_>,
        buffer: u32,
        out: u32,
    ) -> AbiResult<i32> {
        match self.buffer_map_info(buffer) {
            Some(info) => {
                cx.mem.write_bytes(out, &info)?;
                Ok(1)
            }
            None => Ok(0),
        }
    }

    fn wasmer_wgpu_adapter_get_info(
        &self,
        cx: &mut Cx<'_>,
        adapter: u32,
        buf: u32,
        cap: u32,
    ) -> AbiResult<i32> {
        match self.adapter_info_blob(adapter) {
            Some(info) => write_blob(&cx.mem, buf, cap, &info),
            None => Ok(-1),
        }
    }

    fn wasmer_wgpu_device_get_adapter_info(
        &self,
        cx: &mut Cx<'_>,
        device: u32,
        buf: u32,
        cap: u32,
    ) -> AbiResult<i32> {
        match self.device(device) {
            Some(device) => write_blob(&cx.mem, buf, cap, &device.adapter_info),
            None => Ok(-1),
        }
    }

    fn wasmer_wgpu_adapter_get_features(
        &self,
        cx: &mut Cx<'_>,
        adapter: u32,
        buf: u32,
        cap: u32,
    ) -> AbiResult<i32> {
        match self.adapter_features(adapter) {
            Some(features) => write_blob(&cx.mem, buf, cap, &u32_blob(features)),
            None => Ok(-1),
        }
    }

    fn wasmer_wgpu_device_get_features(
        &self,
        cx: &mut Cx<'_>,
        device: u32,
        buf: u32,
        cap: u32,
    ) -> AbiResult<i32> {
        match self.device_features(device) {
            Some(features) => write_blob(&cx.mem, buf, cap, &u32_blob(features)),
            None => Ok(-1),
        }
    }

    fn wasmer_wgpu_instance_get_wgsl_language_features(
        &self,
        cx: &mut Cx<'_>,
        instance: u32,
        buf: u32,
        cap: u32,
    ) -> AbiResult<i32> {
        match self.object(instance, Kind::Instance) {
            Some(object) => {
                let features = wgsl_language_features(object.raw.cast());
                write_blob(
                    &cx.mem,
                    buf,
                    cap,
                    &u32_blob(features.into_iter().map(AbiEnum::as_u32)),
                )
            }
            None => Ok(-1),
        }
    }

    fn wasmer_wgpu_surface_get_capabilities(
        &self,
        cx: &mut Cx<'_>,
        surface: u32,
        adapter: u32,
        buf: u32,
        cap: u32,
    ) -> AbiResult<i32> {
        match self.surface_capabilities(surface, adapter) {
            Some(bytes) => write_blob(&cx.mem, buf, cap, &bytes),
            None => Ok(-1),
        }
    }
}

// The pipeline list is the host's; callbacks push to it.
pub(crate) type Pipelines = Arc<DonePipelines>;
