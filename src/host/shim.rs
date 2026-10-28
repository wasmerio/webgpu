//! The imports that exist only for the guest shim: starting asynchronous
//! work, collecting completed events, and the getters whose results the shim
//! has to own.

use std::sync::Arc;
use std::time::{Duration, Instant};

use super::adapter::encode_info;
use super::instance::wgsl_language_features;
use super::{AdapterState, DeviceState, Host, Object, ObjectKind, Record, convert, scopes};
use crate::abi::api::ShimApi;
use crate::abi::enums::{
    DeviceLostReason, ErrorType, FeatureName, PopErrorScopeStatus, RequestAdapterStatus,
    RequestDeviceStatus,
};
use crate::abi::structs::{
    ComputePipelineDescriptor, DeviceDescriptor, RenderPipelineDescriptor, RequestAdapterOptions,
};
use crate::abi::{
    AbiEnum, AbiError, AbiResult, Cx, EVENT_CREATE_COMPUTE_PIPELINE, EVENT_CREATE_RENDER_PIPELINE,
    EVENT_DEVICE_LOST, EVENT_POP_ERROR_SCOPE, EVENT_REQUEST_ADAPTER, EVENT_REQUEST_DEVICE,
    GuestMemory,
};

/// Longest a waiting thread sleeps before it looks again for termination and
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
    /// everywhere. Anything else (an enum value webgpu.h does not have, a
    /// struct chained where it does not belong) becomes the message of the
    /// failed future: the guest was handed that future, and a callback that
    /// never runs would leave it waiting for ever.
    fn decoded<T>(
        &self,
        function: &'static str,
        result: AbiResult<T>,
    ) -> AbiResult<Result<T, String>> {
        match result {
            Ok(value) => Ok(Ok(value)),
            Err(error @ AbiError::OutOfBounds { .. }) => Err(error),
            Err(error) => {
                self.abi_error(function, &error);
                Ok(Err(format!("{function}: {error}")))
            }
        }
    }

    /// Lets every device make progress: completed maps and submissions turn
    /// into events here. Returns whether any device still has work in flight.
    fn poll_devices(&self, wait: Option<Duration>) -> bool {
        let mut busy = false;
        for device in self.live_devices() {
            let poll = match wait {
                Some(timeout) => wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(timeout),
                },
                None => wgpu::PollType::Poll,
            };
            match device.device.poll(poll) {
                Ok(status) => busy |= !status.is_queue_empty(),
                Err(wgpu::PollError::Timeout) => busy = true,
                Err(error) => super::log(|| format!("device poll failed: {error}")),
            }
        }
        busy
    }

    fn request_adapter(
        &self,
        instance: u32,
        options: Option<&RequestAdapterOptions>,
    ) -> Result<AdapterState, String> {
        if self.is_terminated() {
            return Err("The WebGPU context was shut down.".to_owned());
        }
        let instance = self
            .instance(instance)
            .ok_or_else(|| "Invalid instance.".to_owned())?;
        let force_fallback =
            self.ctx.force_fallback_adapter || options.is_some_and(|o| o.force_fallback_adapter);
        let power_preference = options.map_or(wgpu::PowerPreference::None, |o| {
            convert::power_preference(o.power_preference)
        });
        let surface = match options.map_or(0, |o| o.compatible_surface) {
            0 => None,
            handle => Some(
                self.surface(handle)
                    .ok_or_else(|| "Invalid compatible surface.".to_owned())?,
            ),
        };

        // A specific backend can only be honoured by looking at every adapter
        // of that backend ourselves.
        if let Some(backends) = options.and_then(|o| convert::backends_of(o.backend_type)) {
            let adapters = pollster::block_on(instance.instance.enumerate_adapters(backends));
            return adapters
                .into_iter()
                .filter(|adapter| {
                    !force_fallback || adapter.get_info().device_type == wgpu::DeviceType::Cpu
                })
                .find(|adapter| {
                    surface
                        .as_ref()
                        .and_then(|surface| surface.window_surface())
                        .is_none_or(|surface| adapter.is_surface_supported(surface))
                })
                .map(AdapterState::new)
                .ok_or_else(|| {
                    "No adapter of the requested backend type is available.".to_owned()
                });
        }

        let adapter = pollster::block_on(
            instance
                .instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference,
                    force_fallback_adapter: force_fallback,
                    compatible_surface: surface
                        .as_ref()
                        .and_then(|surface| surface.window_surface()),
                }),
        );
        adapter
            .map(AdapterState::new)
            .map_err(|error| error.to_string())
    }

    fn request_device(
        &self,
        adapter: u32,
        descriptor: Option<&DeviceDescriptor>,
        lost_future: u64,
    ) -> Result<Arc<DeviceState>, String> {
        if self.is_terminated() {
            return Err("The WebGPU context was shut down.".to_owned());
        }
        let adapter = self
            .adapter(adapter)
            .ok_or_else(|| "Invalid adapter.".to_owned())?;
        if let Some(max) = self.ctx.limits.max_devices
            && self.live_devices().len() >= max
        {
            return Err(format!("This guest may have at most {max} devices."));
        }

        let available = adapter.adapter.features();
        let mut features = available & convert::implicit_features();
        for &feature in descriptor.map_or(&[][..], |d| &d.required_features) {
            let bits = FeatureName::from_u32(feature)
                .and_then(convert::feature_to_wgpu)
                .filter(|&bits| available.contains(bits))
                .ok_or_else(|| format!("Required feature {feature:#x} is not supported."))?;
            features |= bits;
        }
        // `wgpuCommandEncoderWriteTimestamp` is part of webgpu.h and has no
        // feature of its own there: a device with timestamp queries can write
        // them from an encoder wherever the adapter is able to.
        if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= available & wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }
        let required_limits = match descriptor.and_then(|d| d.required_limits.as_deref()) {
            Some(limits) => convert::limits_from_abi(limits, wgpu::Limits::defaults()),
            None => wgpu::Limits::defaults(),
        };
        let (device, queue) =
            pollster::block_on(adapter.adapter.request_device(&wgpu::DeviceDescriptor {
                label: descriptor.and_then(|d| super::buffer::label(&d.label)),
                required_features: features,
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))
            .map_err(|error| error.to_string())?;
        let state = DeviceState::new(self, device, queue, adapter.info.clone(), lost_future);
        self.register_device(&state);
        Ok(state)
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
                let handle = self.insert(Object::Adapter(Arc::new(adapter)), instance);
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
        let request = self
            .decoded(
                "wgpuAdapterRequestDevice",
                cx.mem.read_opt::<DeviceDescriptor>(descriptor),
            )?
            .and_then(|descriptor| {
                self.request_device(adapter, descriptor.as_deref(), lost_future)
            });
        match request {
            Ok(device) => {
                let handle = self.insert(
                    Object::Device(Arc::clone(&device)),
                    self.instance_of(adapter),
                );
                device
                    .link
                    .handle
                    .store(handle, std::sync::atomic::Ordering::Release);
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
            Err(message) => {
                self.events.push(Record {
                    future,
                    kind: EVENT_REQUEST_DEVICE,
                    status: RequestDeviceStatus::Error.as_u32(),
                    message: message.clone(),
                    ..Record::default()
                });
                // The device-lost future of a device that never existed.
                self.events.push(Record {
                    future: lost_future,
                    kind: EVENT_DEVICE_LOST,
                    status: DeviceLostReason::FailedCreation.as_u32(),
                    message,
                    ..Record::default()
                });
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
        let popped = self
            .device(device)
            .and_then(|device| scopes::pop(device.id()));
        let record = match popped {
            Some(error) => {
                let (ty, message) = error.unwrap_or((ErrorType::NoError, String::new()));
                Record {
                    future,
                    kind: EVENT_POP_ERROR_SCOPE,
                    status: PopErrorScopeStatus::Success.as_u32(),
                    aux: ty.as_u32(),
                    message,
                    ..Record::default()
                }
            }
            // Nothing to pop: the stack was empty (or the device is invalid).
            None => Record {
                future,
                kind: EVENT_POP_ERROR_SCOPE,
                status: PopErrorScopeStatus::Error.as_u32(),
                aux: ErrorType::NoError.as_u32(),
                message: "There is no error scope to pop.".to_owned(),
                ..Record::default()
            },
        };
        self.events.push(record);
        Ok(())
    }

    fn wasmer_wgpu_device_create_compute_pipeline_async(
        &self,
        cx: &mut Cx<'_>,
        device: u32,
        descriptor: u32,
        future: u64,
    ) -> AbiResult<()> {
        let descriptor = self.decoded(
            "wgpuDeviceCreateComputePipelineAsync",
            cx.mem
                .read_required::<ComputePipelineDescriptor>(descriptor, "descriptor"),
        )?;
        match descriptor {
            Ok(descriptor) => self.finish_pipeline_async(
                EVENT_CREATE_COMPUTE_PIPELINE,
                ObjectKind::ComputePipeline,
                future,
                self.device(device).as_deref(),
                || self.create_compute_pipeline(device, &descriptor),
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
        let descriptor = self.decoded(
            "wgpuDeviceCreateRenderPipelineAsync",
            cx.mem
                .read_required::<RenderPipelineDescriptor>(descriptor, "descriptor"),
        )?;
        match descriptor {
            Ok(descriptor) => self.finish_pipeline_async(
                EVENT_CREATE_RENDER_PIPELINE,
                ObjectKind::RenderPipeline,
                future,
                self.device(device).as_deref(),
                || self.create_render_pipeline(device, &descriptor),
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
        self.shader_module_get_compilation_info(shader_module, future);
        Ok(())
    }

    fn wasmer_wgpu_poll(&self, cx: &mut Cx<'_>, buf: u32, cap: u32) -> AbiResult<i32> {
        self.poll_devices(None);
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
            self.poll_devices(None);
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
            // Block where progress will come from: in the GPU while it has
            // work in flight, otherwise until another thread queues an event.
            if !self.poll_devices(Some(slice)) {
                self.events.wait_for_push(slice);
            }
        }
        self.events.drain_into(&cx.mem, buf, cap)
    }

    fn wasmer_wgpu_yield(&self, _cx: &mut Cx<'_>) -> AbiResult<()> {
        // Natively nothing needs the guest to yield: polling drives the
        // backend. Give other guest threads a turn all the same.
        std::thread::yield_now();
        Ok(())
    }

    fn wasmer_wgpu_instance_release(&self, _cx: &mut Cx<'_>, instance: u32) -> AbiResult<i32> {
        Ok(i32::from(
            self.release(instance, ObjectKind::Instance).is_some(),
        ))
    }

    fn wasmer_wgpu_buffer_release(&self, _cx: &mut Cx<'_>, buffer: u32) -> AbiResult<i32> {
        Ok(i32::from(
            self.release(buffer, ObjectKind::Buffer).is_some(),
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
        match self.adapter(adapter) {
            Some(adapter) => write_blob(&cx.mem, buf, cap, &encode_info(&adapter.info)),
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
            Some(device) => write_blob(&cx.mem, buf, cap, &encode_info(&device.adapter_info)),
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
        match self.adapter(adapter) {
            Some(adapter) => {
                let features = convert::features_to_abi(adapter.adapter.features());
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

    fn wasmer_wgpu_device_get_features(
        &self,
        cx: &mut Cx<'_>,
        device: u32,
        buf: u32,
        cap: u32,
    ) -> AbiResult<i32> {
        match self.device(device) {
            Some(device) => {
                let features = convert::features_to_abi(device.device.features());
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

    fn wasmer_wgpu_instance_get_wgsl_language_features(
        &self,
        cx: &mut Cx<'_>,
        instance: u32,
        buf: u32,
        cap: u32,
    ) -> AbiResult<i32> {
        match self.instance(instance) {
            Some(instance) => {
                let features = wgsl_language_features(&instance.instance);
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
