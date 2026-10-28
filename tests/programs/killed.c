// Blocks in wgpuInstanceWaitAny for ever, and is killed while it does.
//
// A thread blocked in a WebGPU wait is inside the host, where WASIX cannot
// reach it to deliver a signal. The host has to notice the kill itself and
// let go of the thread; otherwise this program never ends.
// (Not in the manifest: tests/embedder_tests.rs runs it.)

#include <pthread.h>
#include <signal.h>
#include <unistd.h>

#include "wgpu_test.h"

static void on_lost(WGPUDevice const* device, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1,
                    void* userdata2) {
    (void)device; (void)reason; (void)message; (void)userdata1; (void)userdata2;
}

static void* killer(void* unused) {
    (void)unused;
    usleep(200 * 1000);
    printf("killing\n");
    fflush(stdout);
    kill(getpid(), SIGKILL);
    return NULL;
}

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    descriptor.deviceLostCallbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
    descriptor.deviceLostCallbackInfo.callback = on_lost;
    WGPUDevice device = wt_request_device_with(instance, adapter, &descriptor);

    pthread_t thread;
    CHECK(pthread_create(&thread, NULL, killer, NULL) == 0);
    printf("waiting\n");
    fflush(stdout);
    // Nobody destroys the device, so only the host can end this wait.
    wt_wait(instance, wgpuDeviceGetLostFuture(device));
    printf("the wait returned\n");
    return 0;
}
