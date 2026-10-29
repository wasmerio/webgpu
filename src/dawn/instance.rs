//! `WGPUInstance`, the global functions, and adapter requests.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::adapter::AdapterState;
use super::ffi::{self, Kind};
use super::lower::Lowering;
use super::{DawnInstance, Extra, Host, Native, log, text};
use crate::abi::enums::{InstanceFeatureName, Status, WGSLLanguageFeatureName};
use crate::abi::structs::{InstanceDescriptor, InstanceLimits, RequestAdapterOptions};
use crate::abi::{AbiEnum, Cx, Handle, OutPtr};

/// Most futures one timed `wgpuInstanceWaitAny` may wait on.
pub(crate) const TIMED_WAIT_ANY_MAX_COUNT: u32 = 64;

pub(crate) struct InstanceState {
    /// Cleared when the guest releases its last reference, so work started
    /// for the instance is not reported to it any more.
    #[allow(dead_code)]
    pub(crate) active: AtomicBool,
}

/// The Dawn instance a process's guest instances share: timed waits (a guest
/// thread may block in one) and several devices per adapter.
pub(crate) fn create_dawn_instance() -> Option<DawnInstance> {
    let features = [
        InstanceFeatureName::TimedWaitAny.as_u32(),
        InstanceFeatureName::MultipleDevicesPerAdapter.as_u32(),
    ];
    let limits = ffi::WGPUInstanceLimits {
        timedWaitAnyMaxCount: TIMED_WAIT_ANY_MAX_COUNT as usize,
        ..Default::default()
    };
    let descriptor = ffi::WGPUInstanceDescriptor {
        requiredFeatureCount: features.len(),
        requiredFeatures: features.as_ptr(),
        requiredLimits: &limits,
        ..Default::default()
    };
    // SAFETY: a valid descriptor that outlives the call.
    let raw = unsafe { ffi::wgpuCreateInstance(&descriptor) };
    if raw.is_null() {
        log(|| "Dawn could not create an instance".to_owned());
        return None;
    }
    Some(DawnInstance(raw))
}

/// Whether this host has an instance feature of webgpu.h.
///
/// Timed waits (a guest thread can block here) and several devices per
/// adapter, yes. SPIR-V shader sources, no: a browser takes only WGSL, and a
/// program that runs on this host is meant to run there too.
fn has_instance_feature(feature: u32) -> bool {
    matches!(
        InstanceFeatureName::from_u32(feature),
        Some(InstanceFeatureName::TimedWaitAny | InstanceFeatureName::MultipleDevicesPerAdapter)
    )
}

/// The WGSL language features Dawn has, as webgpu.h knows them.
pub(crate) fn wgsl_language_features(instance: ffi::WGPUInstance) -> Vec<WGSLLanguageFeatureName> {
    let mut supported = ffi::WGPUSupportedWGSLLanguageFeatures::default();
    // SAFETY: a live instance and a struct for Dawn to fill.
    unsafe { ffi::wgpuInstanceGetWGSLLanguageFeatures(instance, &mut supported) };
    let mut out = Vec::new();
    if !supported.features.is_null() {
        // SAFETY: `featureCount` values at `features`.
        let features =
            unsafe { std::slice::from_raw_parts(supported.features, supported.featureCount) };
        out.extend(
            features
                .iter()
                .filter_map(|&feature| WGSLLanguageFeatureName::from_u32(feature)),
        );
    }
    // SAFETY: what Dawn filled in.
    unsafe { ffi::wgpuSupportedWGSLLanguageFeaturesFreeMembers(supported) };
    out
}

/// Result slot of a synchronous adapter request.
struct AdapterRequest {
    status: u32,
    adapter: ffi::WGPUAdapter,
    message: String,
}

unsafe extern "C" fn on_adapter(
    status: u32,
    adapter: ffi::WGPUAdapter,
    message: ffi::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    // SAFETY: the request the caller waits on, alive until WaitAny returns.
    let request = unsafe { &mut *userdata1.cast::<AdapterRequest>() };
    request.status = status;
    request.adapter = adapter;
    // SAFETY: Dawn's view, valid during the callback.
    request.message = unsafe { text(message) };
}

impl Host {
    /// Requests an adapter of the process's Dawn instance, synchronously.
    pub(crate) fn request_adapter(
        &self,
        instance: Handle,
        options: Option<&RequestAdapterOptions>,
    ) -> Result<AdapterState, String> {
        if self.is_terminated() {
            return Err("The WebGPU context was shut down.".to_owned());
        }
        let this = self
            .object(instance, Kind::Instance)
            .ok_or_else(|| "Invalid instance.".to_owned())?;
        let dawn = self
            .dawn()
            .ok_or_else(|| "No WebGPU instance.".to_owned())?;
        let mut l = Lowering::new(self, Some(&this));
        let mut native = match options {
            Some(options) => l
                .lower_ptr(options)
                .map(|ptr| {
                    // SAFETY: just lowered, owned by `l`.
                    unsafe { *ptr }
                })
                .map_err(|error| format!("wgpuInstanceRequestAdapter: {error}"))?,
            None => ffi::WGPURequestAdapterOptions::default(),
        };
        if self.ctx.force_fallback_adapter {
            native.forceFallbackAdapter = 1;
        }
        let mut request = AdapterRequest {
            status: 0,
            adapter: std::ptr::null_mut(),
            message: String::new(),
        };
        let info = ffi::WGPURequestAdapterCallbackInfo {
            mode: crate::abi::enums::CallbackMode::WaitAnyOnly.as_u32(),
            callback: Some(on_adapter),
            userdata1: (&mut request as *mut AdapterRequest).cast(),
            ..Default::default()
        };
        // SAFETY: live instance; `request` outlives the wait below, which
        // runs the callback.
        unsafe {
            let future = ffi::wgpuInstanceRequestAdapter(dawn, &native, info);
            let mut wait = ffi::WGPUFutureWaitInfo {
                future,
                completed: 0,
            };
            ffi::wgpuInstanceWaitAny(dawn, 1, &mut wait, u64::MAX);
        }
        drop(l);
        if request.adapter.is_null() {
            return Err(if request.message.is_empty() {
                "No adapter is available.".to_owned()
            } else {
                request.message
            });
        }
        Ok(AdapterState::new(request.adapter))
    }

    /// Registers an adapter under `instance`.
    pub(crate) fn insert_adapter(&self, adapter: AdapterState, instance: Handle) -> Handle {
        let raw = adapter.raw;
        self.insert(
            Arc::new(Native {
                kind: Kind::Adapter,
                raw: raw.cast(),
                device: None,
                extra: Extra::Adapter(adapter),
            }),
            instance,
        )
    }

    pub(crate) fn manual_create_instance(
        &self,
        _cx: &mut Cx<'_>,
        descriptor: Option<&InstanceDescriptor>,
    ) -> Handle {
        if let Some(descriptor) = descriptor {
            if let Some(missing) = descriptor
                .required_features
                .iter()
                .find(|&&feature| !has_instance_feature(feature))
            {
                log(|| {
                    format!("wgpuCreateInstance: instance feature {missing:#x} is not available")
                });
                return 0;
            }
            if let Some(limits) = &descriptor.required_limits
                && limits.timed_wait_any_max_count > TIMED_WAIT_ANY_MAX_COUNT
            {
                return 0;
            }
        }
        let Some(dawn) = self.dawn() else {
            return 0;
        };
        // Each guest instance holds a reference to the shared one.
        // SAFETY: a live instance.
        unsafe { ffi::wgpuInstanceAddRef(dawn) };
        self.insert(
            Arc::new(Native {
                kind: Kind::Instance,
                raw: dawn.cast(),
                device: None,
                extra: Extra::Instance(InstanceState {
                    active: AtomicBool::new(true),
                }),
            }),
            0,
        )
    }

    pub(crate) fn manual_get_instance_limits(
        &self,
        cx: &mut Cx<'_>,
        limits: OutPtr<InstanceLimits>,
    ) -> Status {
        if limits.is_null() {
            return Status::Error;
        }
        let value = InstanceLimits {
            chain: Default::default(),
            timed_wait_any_max_count: TIMED_WAIT_ANY_MAX_COUNT,
        };
        match limits.write(&cx.mem, &value) {
            Ok(()) => Status::Success,
            Err(_) => Status::Error,
        }
    }

    pub(crate) fn manual_has_instance_feature(&self, _cx: &mut Cx<'_>, feature: u32) -> bool {
        has_instance_feature(feature)
    }
}

impl InstanceState {
    pub(crate) fn of(object: &Native) -> Option<&InstanceState> {
        match &object.extra {
            Extra::Instance(state) => Some(state),
            _ => None,
        }
    }
}
