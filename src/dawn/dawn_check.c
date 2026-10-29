/* The values src/dawn/dawn_ext.rs spells out, checked against the headers of
   the Dawn being linked. Compiled by build.rs. */
#include <webgpu/webgpu.h>

_Static_assert(WGPUFeatureName_ImplicitDeviceSynchronization == 0x00050004,
               "dawn_ext.rs: FEATURE_IMPLICIT_DEVICE_SYNCHRONIZATION");

/* The signature dawn_ext.rs declares. */
static void (*const inject_error)(WGPUDevice, WGPUErrorType, WGPUStringView) = wgpuDeviceInjectError;

int wasmer_webgpu_dawn_extensions_checked(void) { return inject_error != 0; }
