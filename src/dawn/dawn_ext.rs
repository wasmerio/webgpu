//! The few parts of Dawn's API beyond webgpu.h that the host uses. Their
//! values are checked against the linked headers by `dawn_check.c`.

#![allow(non_snake_case)]

use super::ffi::{
    WGPUComputePipeline, WGPUDevice, WGPUErrorType, WGPURenderPipeline, WGPUStringView,
};

/// `WGPUFeatureName_ImplicitDeviceSynchronization`: makes a device safe to
/// use from several threads at once, as a guest's threads do.
pub(crate) const FEATURE_IMPLICIT_DEVICE_SYNCHRONIZATION: u32 = 0x0005_0004;

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
