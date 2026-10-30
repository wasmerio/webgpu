/* The values src/dawn/dawn_ext.rs spells out, checked against the headers of
   the Dawn being linked. Compiled by build.rs. */
#include <webgpu/webgpu.h>

_Static_assert(WGPUFeatureName_ImplicitDeviceSynchronization == 0x00050004,
               "dawn_ext.rs: FEATURE_IMPLICIT_DEVICE_SYNCHRONIZATION");

/* The signatures dawn_ext.rs declares. */
static void (*const inject_error)(WGPUDevice, WGPUErrorType, WGPUStringView) = wgpuDeviceInjectError;
static WGPUComputePipeline (*const error_compute_pipeline)(WGPUDevice, WGPUStringView) =
    wgpuDeviceCreateErrorComputePipeline;
static WGPURenderPipeline (*const error_render_pipeline)(WGPUDevice, WGPUStringView) =
    wgpuDeviceCreateErrorRenderPipeline;

int wasmer_webgpu_dawn_extensions_checked(void) {
    return inject_error != 0 && error_compute_pipeline != 0 && error_render_pipeline != 0;
}
