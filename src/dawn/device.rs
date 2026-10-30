//! `WGPUDevice` and its queue.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use super::adapter::{AdapterState, standard_features, write_limits};
use super::callbacks::InFlight;
use super::dawn_ext::{
    FEATURE_BUFFER_MAP_EXTENDED_USAGES, FEATURE_IMPLICIT_DEVICE_SYNCHRONIZATION,
    STYPE_DAWN_TOGGLES_DESCRIPTOR, WGPUDawnTogglesDescriptor, wgpuDeviceInjectError,
};
use super::ffi::{self, Kind};
use super::lower::Lowering;
use super::{
    Events, Extra, Host, Native, PIPELINE_CALLBACKS_WAIT, Record, exclusive, lock, text, view,
};
use crate::abi::enums::{CallbackMode, DeviceLostReason, ErrorType, FeatureName, Status};
use crate::abi::structs::{DeviceDescriptor, Limits};
use crate::abi::{AbiEnum, Cx, EVENT_DEVICE_LOST, EVENT_UNCAPTURED_ERROR, Handle, OutPtr};

/// A process's grant of mappable draw buffers, given back on drop unless the
/// device that took it was made.
struct Grant<'a>(Option<&'a AtomicBool>);

impl Drop for Grant<'_> {
    fn drop(&mut self) {
        if let Some(granted) = self.0 {
            granted.store(false, Ordering::Release);
        }
    }
}

/// What a device's callbacks need. Dawn calls them on any thread.
pub(crate) struct DeviceLink {
    /// The guest's handle to the device, or 0 once the guest dropped it.
    pub(crate) handle: AtomicU32,
    pub(crate) lost: AtomicBool,
    lost_future: u64,
    events: Arc<Events>,
}

impl DeviceLink {
    fn on_uncaptured(&self, ty: u32, message: String) {
        // "The uncaptured error callback never fires after the device is lost."
        if self.lost.load(Ordering::Acquire) {
            return;
        }
        self.events.push(Record {
            kind: EVENT_UNCAPTURED_ERROR,
            handle: self.handle.load(Ordering::Acquire),
            aux: ty,
            message,
            ..Record::default()
        });
    }

    fn on_lost(&self, reason: u32, message: String) {
        if self.lost.swap(true, Ordering::AcqRel) {
            return;
        }
        self.events.push(Record {
            future: self.lost_future,
            kind: EVENT_DEVICE_LOST,
            status: reason,
            handle: self.handle.load(Ordering::Acquire),
            message,
            ..Record::default()
        });
    }
}

/// The device lost callback owns a reference to the link, which it frees:
/// Dawn calls it exactly once, and calls no uncaptured-error callback after.
unsafe extern "C" fn on_lost(
    _device: *const ffi::WGPUDevice,
    reason: u32,
    message: ffi::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    // SAFETY: boxed by `request_device` for this callback.
    let link = unsafe { Box::from_raw(userdata1.cast::<Arc<DeviceLink>>()) };
    // SAFETY: Dawn's view, valid during the callback.
    link.on_lost(reason, unsafe { text(message) });
}

unsafe extern "C" fn on_uncaptured(
    _device: *const ffi::WGPUDevice,
    ty: u32,
    message: ffi::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    // SAFETY: the link the lost callback keeps alive until the device is lost.
    let link = unsafe { &*userdata1.cast::<DeviceLink>() };
    // SAFETY: Dawn's view, valid during the callback.
    link.on_uncaptured(ty, unsafe { text(message) });
}

pub(crate) struct DeviceState {
    /// The host's own references to the device and its queue.
    pub(crate) raw: ffi::WGPUDevice,
    pub(crate) queue: ffi::WGPUQueue,
    pub(crate) link: Arc<DeviceLink>,
    /// The adapter's info, encoded for the shim.
    pub(crate) adapter_info: Vec<u8>,
    /// The guest's queue handle while it holds one, so `wgpuDeviceGetQueue`
    /// keeps returning the same value.
    queue_handle: Mutex<Handle>,
    /// Held shared while the host copies to or from one of the device's
    /// mappings, exclusively while Dawn destroys the device (which frees
    /// them all).
    pub(crate) access: RwLock<()>,
    /// Pipelines Dawn is compiling for this device, and whether it is being
    /// destroyed.
    pub(crate) pipelines: Arc<InFlight>,
}

// SAFETY: devices are created with implicit device synchronization.
unsafe impl Send for DeviceState {}
unsafe impl Sync for DeviceState {}

impl Drop for DeviceState {
    fn drop(&mut self) {
        // SAFETY: the host's own references. The last one destroys the device,
        // whose final submission Metal autoreleases (`autoreleased`).
        crate::native::autoreleased(|| unsafe {
            ffi::wgpuQueueRelease(self.queue);
            ffi::wgpuDeviceRelease(self.raw);
        });
    }
}

impl DeviceState {
    /// Reports an error the host found as the device would: to the calling
    /// thread's innermost matching scope, or the uncaptured-error callback.
    pub(crate) fn inject(&self, ty: ErrorType, message: &str) {
        // SAFETY: a live device; the view outlives the call.
        unsafe { wgpuDeviceInjectError(self.raw, ty.as_u32(), view(message)) };
    }

    pub(crate) fn is_lost(&self) -> bool {
        self.link.lost.load(Ordering::Acquire)
    }

    /// Destroys the device. Dawn reports the loss through the callback; the
    /// host reports it at once too (once is all the guest hears), so a guest
    /// waiting on the lost future does not depend on Dawn's timing.
    ///
    /// Dawn destroys a device holding its lock, and waits for the device's
    /// compilations, whose callbacks take that lock: they finish first. The
    /// first call stops new compilations, reports the loss, and waits for
    /// those in flight for as long as `wait` (for a guest's destroy, as Dawn
    /// would have), holding no lock. Otherwise (a kill waits for nothing) a
    /// thread of its own destroys the device once they are done. Later calls
    /// only report the loss.
    pub(crate) fn destroy(self: &Arc<Self>, wait: Duration) {
        let first = self.pipelines.close();
        self.link.on_lost(
            DeviceLostReason::Destroyed.as_u32(),
            "Device was destroyed.".to_owned(),
        );
        if first {
            if !wait.is_zero() && self.pipelines.wait(wait) {
                self.destroy_now();
            } else {
                let device = Arc::clone(self);
                let spawned = std::thread::Builder::new()
                    .name("webgpu-device-destroy".to_owned())
                    .spawn(move || {
                        device.pipelines.wait_all();
                        device.destroy_now();
                    });
                if spawned.is_err() {
                    // Released with its last reference instead.
                    super::log(|| "device destroy: no thread to finish it".to_owned());
                }
            }
        }
    }

    /// Destroys the device in Dawn, which frees its mappings: no copy to or
    /// from one runs meanwhile.
    fn destroy_now(&self) {
        let _copies = exclusive(&self.access);
        // SAFETY: a live device, with no pipeline compiling.
        crate::native::autoreleased(|| unsafe { ffi::wgpuDeviceDestroy(self.raw) });
    }
}

/// Result slot of a synchronous device request.
struct DeviceRequest {
    device: ffi::WGPUDevice,
    message: String,
}

unsafe extern "C" fn on_device(
    _status: u32,
    device: ffi::WGPUDevice,
    message: ffi::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    // SAFETY: the request the caller waits on, alive until WaitAny returns.
    let request = unsafe { &mut *userdata1.cast::<DeviceRequest>() };
    request.device = device;
    // SAFETY: Dawn's view, valid during the callback.
    request.message = unsafe { text(message) };
}

/// Why a device request failed, and whether Dawn already reported the loss
/// of the device it never made (through the lost callback).
pub(crate) struct DeviceFailure {
    pub(crate) message: String,
    pub(crate) lost_reported: bool,
}

impl Host {
    pub(crate) fn request_device(
        &self,
        adapter: Handle,
        descriptor: Option<&DeviceDescriptor>,
        lost_future: u64,
    ) -> Result<Arc<DeviceState>, DeviceFailure> {
        let early = |message: &str| DeviceFailure {
            message: message.to_owned(),
            lost_reported: false,
        };
        if self.is_terminated() {
            return Err(early("The WebGPU context was shut down."));
        }
        let adapter_object = self
            .object(adapter, Kind::Adapter)
            .ok_or_else(|| early("Invalid adapter."))?;
        let Some(adapter_state) = AdapterState::of(&adapter_object) else {
            return Err(early("Invalid adapter."));
        };
        let dawn = self.dawn().ok_or_else(|| early("No WebGPU instance."))?;
        // A destroyed device may live on in queued work: it doesn't count.
        if let Some(max) = self.ctx.limits.max_devices
            && self
                .live_devices()
                .iter()
                .filter(|device| !device.is_lost())
                .count()
                >= max
        {
            return Err(early(&format!(
                "This guest may have at most {max} devices."
            )));
        }

        // Mappable draw buffers come with Dawn's unsafe APIs: only where the
        // embedder allows them (its guest is its own renderer), and for one
        // device of the process. A request that fails gives the grant back.
        let mapped_draw_buffers = descriptor.is_some_and(|descriptor| {
            descriptor
                .required_features
                .contains(&FEATURE_BUFFER_MAP_EXTENDED_USAGES)
        }) && self.ctx.renderer_mapped_draw_buffers
            && self
                .mapped_draw_buffers
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok();
        let mut grant = Grant(mapped_draw_buffers.then_some(&self.mapped_draw_buffers));
        // Only webgpu.h's features, and those mappable draw buffers: Dawn has
        // more, which a guest never hears of and may not ask for.
        if let Some(feature) = descriptor.and_then(|descriptor| {
            descriptor.required_features.iter().find(|&&feature| {
                FeatureName::from_u32(feature).is_none()
                    && !(feature == FEATURE_BUFFER_MAP_EXTENDED_USAGES && mapped_draw_buffers)
            })
        }) {
            return Err(early(&format!(
                "Required feature {feature:#x} is not supported."
            )));
        }
        let mut l = Lowering::new(self, Some(&adapter_object));
        let mut native = match descriptor {
            // SAFETY: just lowered, owned by `l`.
            Some(descriptor) => unsafe {
                *l.lower_ptr(descriptor)
                    .map_err(|error| early(&format!("wgpuAdapterRequestDevice: {error}")))?
            },
            None => ffi::WGPUDeviceDescriptor::default(),
        };
        // Guest threads share devices: ask Dawn to make them thread-safe.
        let mut features = Vec::with_capacity(native.requiredFeatureCount + 1);
        if !native.requiredFeatures.is_null() {
            // SAFETY: the lowered array.
            features.extend_from_slice(unsafe {
                std::slice::from_raw_parts(native.requiredFeatures, native.requiredFeatureCount)
            });
        }
        // SAFETY: a live adapter.
        if unsafe {
            ffi::wgpuAdapterHasFeature(adapter_state.raw, FEATURE_IMPLICIT_DEVICE_SYNCHRONIZATION)
        } != 0
        {
            features.push(FEATURE_IMPLICIT_DEVICE_SYNCHRONIZATION);
        }
        native.requiredFeatureCount = features.len();
        native.requiredFeatures = features.as_ptr();
        // Dawn grants an experimental feature only with its unsafe APIs: the
        // device that asks for mappable draw buffers gets them, no other one.
        // Dawn fails the request where the adapter cannot map such buffers.
        let unsafe_apis = [c"allow_unsafe_apis".as_ptr()];
        let mut toggles = WGPUDawnTogglesDescriptor {
            chain: ffi::WGPUChainedStruct {
                next: native.nextInChain,
                sType: STYPE_DAWN_TOGGLES_DESCRIPTOR,
            },
            enabledToggleCount: unsafe_apis.len(),
            enabledToggles: unsafe_apis.as_ptr(),
            disabledToggleCount: 0,
            disabledToggles: std::ptr::null(),
        };
        if mapped_draw_buffers {
            native.nextInChain = &mut toggles.chain;
        }

        let link = Arc::new(DeviceLink {
            handle: AtomicU32::new(0),
            lost: AtomicBool::new(false),
            lost_future,
            events: std::sync::Arc::clone(&self.events),
        });
        native.deviceLostCallbackInfo = ffi::WGPUDeviceLostCallbackInfo {
            mode: CallbackMode::AllowSpontaneous.as_u32(),
            callback: Some(on_lost),
            userdata1: Box::into_raw(Box::new(Arc::clone(&link))).cast(),
            ..Default::default()
        };
        native.uncapturedErrorCallbackInfo = ffi::WGPUUncapturedErrorCallbackInfo {
            callback: Some(on_uncaptured),
            userdata1: Arc::as_ptr(&link).cast_mut().cast(),
            ..Default::default()
        };

        let mut request = DeviceRequest {
            device: std::ptr::null_mut(),
            message: String::new(),
        };
        let info = ffi::WGPURequestDeviceCallbackInfo {
            mode: CallbackMode::WaitAnyOnly.as_u32(),
            callback: Some(on_device),
            userdata1: (&mut request as *mut DeviceRequest).cast(),
            ..Default::default()
        };
        // SAFETY: a live adapter and instance; `request` outlives the wait,
        // which runs the callback.
        unsafe {
            let future = ffi::wgpuAdapterRequestDevice(adapter_state.raw, &native, info);
            let mut wait = ffi::WGPUFutureWaitInfo {
                future,
                completed: 0,
            };
            ffi::wgpuInstanceWaitAny(dawn, 1, &mut wait, u64::MAX);
        }
        drop(l);
        if request.device.is_null() {
            // Dawn reports a device it could not make as lost (FailedCreation).
            return Err(DeviceFailure {
                message: if request.message.is_empty() {
                    "The device could not be created.".to_owned()
                } else {
                    request.message
                },
                lost_reported: true,
            });
        }
        // The device has them: keep the grant.
        grant.0 = None;
        // SAFETY: the new device.
        let queue = unsafe { ffi::wgpuDeviceGetQueue(request.device) };
        let state = Arc::new(DeviceState {
            raw: request.device,
            queue,
            link,
            adapter_info: adapter_state.info.clone(),
            queue_handle: Mutex::new(0),
            access: RwLock::new(()),
            pipelines: Arc::default(),
        });
        self.register_device(&state);
        Ok(state)
    }

    /// Registers a device under `instance`.
    pub(crate) fn insert_device(&self, state: &Arc<DeviceState>, instance: Handle) -> Handle {
        // SAFETY: the handle's own reference to a live device.
        unsafe { ffi::wgpuDeviceAddRef(state.raw) };
        let handle = self.insert(
            Arc::new(Native {
                kind: Kind::Device,
                raw: state.raw.cast(),
                device: Some(Arc::clone(state)),
                extra: Extra::None,
            }),
            instance,
        );
        state.link.handle.store(handle, Ordering::Release);
        handle
    }

    /// Drops one guest reference to a device. Returns whether it was the last.
    pub(crate) fn release_device(&self, device: Handle) -> bool {
        let Some(object) = self.release_handle(device, Kind::Device) else {
            return false;
        };
        // Dropping the last reference implies destroying the device, and the
        // lost callback then receives a null device.
        if let Some(state) = &object.device {
            state.link.handle.store(0, Ordering::Release);
            state.destroy(PIPELINE_CALLBACKS_WAIT);
        }
        true
    }

    pub(crate) fn device_features(&self, device: Handle) -> Option<Vec<u32>> {
        let state = self.device(device)?;
        let mut supported = ffi::WGPUSupportedFeatures::default();
        // SAFETY: a live device and a struct for Dawn to fill.
        unsafe { ffi::wgpuDeviceGetFeatures(state.raw, &mut supported) };
        Some(standard_features(supported))
    }

    pub(crate) fn manual_device_destroy(&self, _cx: &mut Cx<'_>, device: Handle) {
        if let Some(state) = self.device(device) {
            state.destroy(PIPELINE_CALLBACKS_WAIT);
        }
    }

    pub(crate) fn manual_device_get_queue(&self, _cx: &mut Cx<'_>, device: Handle) -> Handle {
        let Some(state) = self.device(device) else {
            return 0;
        };
        let mut queue_handle = lock(&state.queue_handle);
        // One table operation: a release on another thread cannot slip in
        // between finding the handle live and adding the reference.
        if *queue_handle != 0 && self.add_ref_handle(*queue_handle, Kind::Queue) {
            return *queue_handle;
        }
        // SAFETY: the handle's own reference to the live queue.
        unsafe { ffi::wgpuQueueAddRef(state.queue) };
        let handle = self.insert(
            Arc::new(Native {
                kind: Kind::Queue,
                raw: state.queue.cast(),
                device: Some(Arc::clone(&state)),
                extra: Extra::None,
            }),
            self.instance_of(device),
        );
        *queue_handle = handle;
        handle
    }

    pub(crate) fn manual_queue_release(&self, _cx: &mut Cx<'_>, queue: Handle) {
        let Some(object) = self.release_handle(queue, Kind::Queue) else {
            return;
        };
        if let Some(state) = &object.device {
            let mut queue_handle = lock(&state.queue_handle);
            if *queue_handle == queue {
                *queue_handle = 0;
            }
        }
    }

    pub(crate) fn manual_device_get_limits(
        &self,
        cx: &mut Cx<'_>,
        device: Handle,
        limits: OutPtr<Limits>,
    ) -> Status {
        let Some(state) = self.device(device) else {
            return Status::Error;
        };
        write_limits(&cx.mem, limits, |native| {
            // SAFETY: a live device and a struct for Dawn to fill.
            unsafe { ffi::wgpuDeviceGetLimits(state.raw, native) }
        })
    }
}
