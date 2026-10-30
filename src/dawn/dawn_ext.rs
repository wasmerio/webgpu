//! The few parts of Dawn's API beyond webgpu.h that the host uses. Their
//! values are checked against the linked headers by `dawn_check.c`.

#![allow(non_snake_case)]

use std::os::raw::c_char;

use super::ffi::{
    WGPUChainedStruct, WGPUComputePipeline, WGPUDevice, WGPUErrorType, WGPURenderPipeline,
    WGPUStringView,
};

/// `WGPUFeatureName_ImplicitDeviceSynchronization`: makes a device safe to
/// use from several threads at once, as a guest's threads do.
pub(crate) const FEATURE_IMPLICIT_DEVICE_SYNCHRONIZATION: u32 = 0x0005_0004;

/// `WGPUFeatureName_BufferMapExtendedUsages`: buffers that are mappable and
/// also uniform, vertex, index or storage buffers (on unified memory). A
/// renderer writes its draw data straight into them instead of staging it
/// and copying it on the GPU. Dawn calls it experimental: a device gets it
/// only with the `allow_unsafe_apis` toggle.
pub(crate) const FEATURE_BUFFER_MAP_EXTENDED_USAGES: u32 = 0x0005_0013;

/// `WGPUSType_DawnTogglesDescriptor`.
pub(crate) const STYPE_DAWN_TOGGLES_DESCRIPTOR: u32 = 0x0005_000A;

/// `WGPUDawnTogglesDescriptor`: toggles for the device being created.
#[repr(C)]
pub(crate) struct WGPUDawnTogglesDescriptor {
    pub chain: WGPUChainedStruct,
    pub enabledToggleCount: usize,
    pub enabledToggles: *const *const c_char,
    pub disabledToggleCount: usize,
    pub disabledToggles: *const *const c_char,
}

unsafe extern "C" {
    /// Reports an error as if the device had found it: to the calling
    /// thread's innermost matching error scope, or the uncaptured-error
    /// callback. How the host's own validation reaches the guest.
    pub(crate) fn wgpuDeviceInjectError(
        device: WGPUDevice,
        ty: WGPUErrorType,
        message: WGPUStringView,
    );

    /// An invalid pipeline, made without compiling anything: what Dawn's
    /// asynchronous creation gives on a lost device.
    pub(crate) fn wgpuDeviceCreateErrorComputePipeline(
        device: WGPUDevice,
        label: WGPUStringView,
    ) -> WGPUComputePipeline;
    pub(crate) fn wgpuDeviceCreateErrorRenderPipeline(
        device: WGPUDevice,
        label: WGPUStringView,
    ) -> WGPURenderPipeline;
}
