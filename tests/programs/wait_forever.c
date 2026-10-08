// Blocks in wgpuInstanceWaitAny on a future only the host can complete: the
// device-lost future of a device nobody destroys. tests/embedder_tests.rs
// uses it to prove that an embedder can always get a guest out of a wait.
// (Not in the manifest: on its own it never finishes.)

#include "wgpu_test.h"

static void on_lost(WGPUDevice const* device, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1,
                    void* userdata2) {
    (void)device; (void)message; (void)userdata2;
    *(WGPUDeviceLostReason*)userdata1 = reason;
}

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDeviceLostReason reason = (WGPUDeviceLostReason)0;
    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    descriptor.deviceLostCallbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
    descriptor.deviceLostCallbackInfo.callback = on_lost;
    descriptor.deviceLostCallbackInfo.userdata1 = &reason;
    WGPUDevice device = wt_request_device_with(instance, adapter, &descriptor);

    printf("waiting\n");
    fflush(stdout);
    wt_wait(instance, wgpuDeviceGetLostFuture(device));
    printf("device lost: reason %d\n", (int)reason);
    return 0;
}
