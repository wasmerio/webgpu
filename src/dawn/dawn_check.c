/* The values src/dawn/dawn_ext.rs spells out, checked against the headers of
   the Dawn being linked. Compiled by build.rs. */
#include <webgpu/webgpu.h>

_Static_assert(WGPUFeatureName_ImplicitDeviceSynchronization == 0x00050004,
               "dawn_ext.rs: FEATURE_IMPLICIT_DEVICE_SYNCHRONIZATION");
_Static_assert(WGPUFeatureName_BufferMapExtendedUsages == 0x00050013,
               "dawn_ext.rs: FEATURE_BUFFER_MAP_EXTENDED_USAGES");
_Static_assert(WGPUSType_DawnTogglesDescriptor == 0x0005000A,
               "dawn_ext.rs: STYPE_DAWN_TOGGLES_DESCRIPTOR");

/* The layout dawn_ext.rs gives WGPUDawnTogglesDescriptor. */
#include <stddef.h>
_Static_assert(offsetof(WGPUDawnTogglesDescriptor, chain) == 0, "dawn_ext.rs: WGPUDawnTogglesDescriptor.chain");
_Static_assert(offsetof(WGPUDawnTogglesDescriptor, enabledToggleCount) == sizeof(WGPUChainedStruct),
               "dawn_ext.rs: WGPUDawnTogglesDescriptor.enabledToggleCount");
_Static_assert(offsetof(WGPUDawnTogglesDescriptor, enabledToggles) == sizeof(WGPUChainedStruct) + sizeof(size_t),
               "dawn_ext.rs: WGPUDawnTogglesDescriptor.enabledToggles");
_Static_assert(offsetof(WGPUDawnTogglesDescriptor, disabledToggleCount) == sizeof(WGPUChainedStruct) + 2 * sizeof(size_t),
               "dawn_ext.rs: WGPUDawnTogglesDescriptor.disabledToggleCount");
_Static_assert(offsetof(WGPUDawnTogglesDescriptor, disabledToggles) == sizeof(WGPUChainedStruct) + 3 * sizeof(size_t),
               "dawn_ext.rs: WGPUDawnTogglesDescriptor.disabledToggles");
_Static_assert(sizeof(WGPUDawnTogglesDescriptor) == sizeof(WGPUChainedStruct) + 4 * sizeof(size_t),
               "dawn_ext.rs: WGPUDawnTogglesDescriptor size");

/* The signatures dawn_ext.rs declares. */
static void (*const inject_error)(WGPUDevice, WGPUErrorType, WGPUStringView) = wgpuDeviceInjectError;
static WGPUComputePipeline (*const error_compute_pipeline)(WGPUDevice, WGPUStringView) =
    wgpuDeviceCreateErrorComputePipeline;
static WGPURenderPipeline (*const error_render_pipeline)(WGPUDevice, WGPUStringView) =
    wgpuDeviceCreateErrorRenderPipeline;

int wasmer_webgpu_dawn_extensions_checked(void) {
    return inject_error != 0 && error_compute_pipeline != 0 && error_render_pipeline != 0;
}
