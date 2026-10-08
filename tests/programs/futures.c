// The asynchronous model of webgpu.h: futures, the three callback modes,
// wgpuInstanceProcessEvents, wgpuInstanceWaitAny, and what happens to
// callbacks still pending when their instance goes away.

#include "wgpu_test.h"

typedef struct Request {
    int calls;
    WGPURequestAdapterStatus status;
    WGPUAdapter adapter;
} Request;

static void on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1,
                       void* userdata2) {
    (void)message; (void)userdata2;
    Request* request = userdata1;
    request->calls++;
    request->status = status;
    request->adapter = adapter;
}

static WGPUFuture request(WGPUInstance instance, WGPUCallbackMode mode, Request* out) {
    WGPURequestAdapterCallbackInfo callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    callback.mode = mode;
    callback.callback = on_adapter;
    callback.userdata1 = out;
    return wgpuInstanceRequestAdapter(instance, NULL, callback);
}

// A callback that starts more asynchronous work, as real programs do.
typedef struct Chain {
    WGPUInstance instance;
    WGPUDevice device;
    int adapter_calls;
    int device_calls;
} Chain;

static void chain_on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1,
                            void* userdata2) {
    (void)message; (void)userdata2;
    Chain* chain = userdata1;
    chain->device_calls++;
    chain->device = status == WGPURequestDeviceStatus_Success ? device : NULL;
}

static void chain_on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message,
                             void* userdata1, void* userdata2) {
    (void)message; (void)userdata2;
    Chain* chain = userdata1;
    chain->adapter_calls++;
    if (status != WGPURequestAdapterStatus_Success) return;
    WGPURequestDeviceCallbackInfo callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_AllowProcessEvents;
    callback.callback = chain_on_device;
    callback.userdata1 = chain;
    wgpuAdapterRequestDevice(adapter, NULL, callback);
    wgpuAdapterRelease(adapter);
}

typedef struct DeviceRequest {
    int calls;
    WGPURequestDeviceStatus status;
    WGPUDevice device;
} DeviceRequest;

static void on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1,
                      void* userdata2) {
    (void)message; (void)userdata2;
    DeviceRequest* request = userdata1;
    request->calls++;
    request->status = status;
    request->device = device;
}

typedef struct Pipeline {
    int calls;
    WGPUCreatePipelineAsyncStatus status;
    WGPUComputePipeline pipeline;
} Pipeline;

static void on_pipeline(WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline pipeline, WGPUStringView message,
                        void* userdata1, void* userdata2) {
    (void)message; (void)userdata2;
    Pipeline* created = userdata1;
    created->calls++;
    created->status = status;
    created->pipeline = pipeline;
}

typedef struct Lost {
    int calls;
    WGPUDeviceLostReason reason;
    bool device_was_null;
} Lost;

static void on_lost(WGPUDevice const* device, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1,
                    void* userdata2) {
    (void)message; (void)userdata2;
    Lost* lost = userdata1;
    lost->calls++;
    lost->reason = reason;
    lost->device_was_null = *device == NULL;
}

int main(void) {
    WGPUInstance instance = wt_create_instance();

    // AllowProcessEvents: delivered by ProcessEvents, never before.
    Request a = { 0 };
    WGPUFuture future_a = request(instance, WGPUCallbackMode_AllowProcessEvents, &a);
    CHECK(future_a.id != 0);
    CHECK(a.calls == 0);
    while (a.calls == 0) wgpuInstanceProcessEvents(instance);
    CHECK(a.calls == 1 && a.status == WGPURequestAdapterStatus_Success && a.adapter != NULL);
    wgpuInstanceProcessEvents(instance);
    CHECK(a.calls == 1);
    printf("AllowProcessEvents: delivered once by ProcessEvents\n");

    // WaitAnyOnly: ProcessEvents must not deliver it, WaitAny on it does.
    Request b = { 0 };
    WGPUFuture future_b = request(instance, WGPUCallbackMode_WaitAnyOnly, &b);
    for (int i = 0; i < 20; i++) wgpuInstanceProcessEvents(instance);
    CHECK(b.calls == 0);
    WGPUFutureWaitInfo wait_b = { future_b, false };
    CHECK(wgpuInstanceWaitAny(instance, 1, &wait_b, UINT64_MAX) == WGPUWaitStatus_Success);
    CHECK(wait_b.completed && b.calls == 1 && b.adapter != NULL);
    // Waiting again on a completed future succeeds at once and calls nothing.
    wait_b.completed = false;
    CHECK(wgpuInstanceWaitAny(instance, 1, &wait_b, 0) == WGPUWaitStatus_Success);
    CHECK(wait_b.completed && b.calls == 1);
    printf("WaitAnyOnly: delivered once by WaitAny\n");

    // AllowSpontaneous: may arrive at any time; pumping events gets it here.
    Request c = { 0 };
    WGPUFuture future_c = request(instance, WGPUCallbackMode_AllowSpontaneous, &c);
    CHECK(future_c.id != 0);
    while (c.calls == 0) wgpuInstanceProcessEvents(instance);
    CHECK(c.calls == 1 && c.adapter != NULL);
    printf("AllowSpontaneous: delivered once\n");

    // WaitAny over several futures completes each exactly once.
    Request d = { 0 }, e = { 0 };
    WGPUFutureWaitInfo waits[2] = {
        { request(instance, WGPUCallbackMode_WaitAnyOnly, &d), false },
        { request(instance, WGPUCallbackMode_WaitAnyOnly, &e), false },
    };
    CHECK(waits[0].future.id != waits[1].future.id);
    while (!(waits[0].completed && waits[1].completed))
        CHECK(wgpuInstanceWaitAny(instance, 2, waits, UINT64_MAX) == WGPUWaitStatus_Success);
    CHECK(d.calls == 1 && e.calls == 1);
    printf("WaitAny: two futures, one callback each\n");

    // Edge cases of WaitAny.
    CHECK(wgpuInstanceWaitAny(instance, 0, NULL, 0) == WGPUWaitStatus_Success);
    WGPUFutureWaitInfo never_issued = { { 0x7FFFFFFFFFull }, false };
    CHECK(wgpuInstanceWaitAny(instance, 1, &never_issued, 0) == WGPUWaitStatus_Success);
    Request polled = { 0 };
    WGPUFutureWaitInfo wait_polled = { request(instance, WGPUCallbackMode_WaitAnyOnly, &polled), false };
    WGPUWaitStatus status = wgpuInstanceWaitAny(instance, 1, &wait_polled, 0);
    CHECK(status == WGPUWaitStatus_Success || status == WGPUWaitStatus_TimedOut);
    CHECK((status == WGPUWaitStatus_Success) == (polled.calls == 1));
    while (!wait_polled.completed) wgpuInstanceWaitAny(instance, 1, &wait_polled, UINT64_MAX);
    CHECK(polled.calls == 1);
    printf("WaitAny: zero futures, unknown future and zero timeout behave\n");

    // A callback mode webgpu.h does not define yields the null future.
    Request invalid = { 0 };
    CHECK(request(instance, (WGPUCallbackMode)77, &invalid).id == 0);
    wgpuInstanceProcessEvents(instance);
    CHECK(invalid.calls == 0);
    printf("invalid callback mode: null future\n");

    // Callbacks may start more work: adapter -> device, both via ProcessEvents.
    Chain chain = { instance, NULL, 0, 0 };
    WGPURequestAdapterCallbackInfo chained = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    chained.mode = WGPUCallbackMode_AllowProcessEvents;
    chained.callback = chain_on_adapter;
    chained.userdata1 = &chain;
    wgpuInstanceRequestAdapter(instance, NULL, chained);
    while (chain.device_calls == 0) wgpuInstanceProcessEvents(instance);
    CHECK(chain.adapter_calls == 1 && chain.device_calls == 1 && chain.device != NULL);
    wgpuDeviceRelease(chain.device);
    printf("nested requests: ok\n");

    // A request the host cannot make sense of still completes its future:
    // a callback that never ran would leave the program waiting for ever.
    Request unreadable = { 0 };
    WGPURequestAdapterOptions bad_options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    bad_options.featureLevel = (WGPUFeatureLevel)99;
    WGPURequestAdapterCallbackInfo unreadable_callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    unreadable_callback.mode = WGPUCallbackMode_WaitAnyOnly;
    unreadable_callback.callback = on_adapter;
    unreadable_callback.userdata1 = &unreadable;
    wt_wait(instance, wgpuInstanceRequestAdapter(instance, &bad_options, unreadable_callback));
    CHECK(unreadable.calls == 1 && unreadable.status == WGPURequestAdapterStatus_Error);
    CHECK(unreadable.adapter == NULL);

    // For a device that also means its lost callback: the device it would
    // have reported on was never made. The adapter is not used up by this.
    DeviceRequest refused = { 0 };
    Lost unborn = { 0 };
    WGPUShaderSourceWGSL stray = WGPU_SHADER_SOURCE_WGSL_INIT;
    WGPUDeviceDescriptor bad_descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    bad_descriptor.nextInChain = &stray.chain;
    bad_descriptor.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    bad_descriptor.deviceLostCallbackInfo.callback = on_lost;
    bad_descriptor.deviceLostCallbackInfo.userdata1 = &unborn;
    WGPURequestDeviceCallbackInfo refused_callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    refused_callback.mode = WGPUCallbackMode_WaitAnyOnly;
    refused_callback.callback = on_device;
    refused_callback.userdata1 = &refused;
    wt_wait(instance, wgpuAdapterRequestDevice(b.adapter, &bad_descriptor, refused_callback));
    CHECK(refused.calls == 1 && refused.status == WGPURequestDeviceStatus_Error && refused.device == NULL);
    CHECK(unborn.calls == 1 && unborn.reason == WGPUDeviceLostReason_FailedCreation && unborn.device_was_null);
    printf("unreadable requests: completed with errors\n");

    // The device-lost future: pending for the device's whole life, completed
    // by destroying the device.
    Lost lost = { 0 };
    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    descriptor.deviceLostCallbackInfo.mode = WGPUCallbackMode_WaitAnyOnly;
    descriptor.deviceLostCallbackInfo.callback = on_lost;
    descriptor.deviceLostCallbackInfo.userdata1 = &lost;
    WGPUDevice device = wt_request_device_with(instance, b.adapter, &descriptor);

    // The same holds for a pipeline asked for with nothing to make it from.
    // The failure is the callback's to report: it raises no device error.
    Pipeline nothing = { 0 };
    WGPUCreateComputePipelineAsyncCallbackInfo nothing_callback = WGPU_CREATE_COMPUTE_PIPELINE_ASYNC_CALLBACK_INFO_INIT;
    nothing_callback.mode = WGPUCallbackMode_WaitAnyOnly;
    nothing_callback.callback = on_pipeline;
    nothing_callback.userdata1 = &nothing;
    wt_wait(instance, wgpuDeviceCreateComputePipelineAsync(device, NULL, nothing_callback));
    CHECK(nothing.calls == 1 && nothing.status == WGPUCreatePipelineAsyncStatus_ValidationError);
    CHECK(nothing.pipeline == NULL && wt_uncaptured_errors == 0);
    printf("unreadable pipeline request: completed with an error\n");

    WGPUFuture lost_future = wgpuDeviceGetLostFuture(device);
    CHECK(lost_future.id != 0);
    WGPUFutureWaitInfo wait_lost = { lost_future, false };
    CHECK(wgpuInstanceWaitAny(instance, 1, &wait_lost, 0) == WGPUWaitStatus_TimedOut);
    CHECK(lost.calls == 0);
    wgpuDeviceDestroy(device);
    CHECK(wgpuInstanceWaitAny(instance, 1, &wait_lost, UINT64_MAX) == WGPUWaitStatus_Success);
    CHECK(lost.calls == 1 && lost.reason == WGPUDeviceLostReason_Destroyed && !lost.device_was_null);
    wgpuDeviceRelease(device);
    printf("device lost: destroyed\n");

    // Dropping the instance cancels what it could still have delivered: every
    // callback is called exactly once, even if only to say so.
    Lost never = { 0 };
    WGPUDeviceDescriptor other = WGPU_DEVICE_DESCRIPTOR_INIT;
    other.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowProcessEvents;
    other.deviceLostCallbackInfo.callback = on_lost;
    other.deviceLostCallbackInfo.userdata1 = &never;
    WGPUAdapter adapter = wgpuHasInstanceFeature(WGPUInstanceFeatureName_MultipleDevicesPerAdapter) ? b.adapter
                                                                                                    : a.adapter;
    WGPUDevice survivor = wt_request_device_with(instance, adapter, &other);
    wgpuAdapterRelease(a.adapter);
    wgpuAdapterRelease(b.adapter);
    wgpuAdapterRelease(c.adapter);
    wgpuAdapterRelease(d.adapter);
    wgpuAdapterRelease(e.adapter);
    wgpuAdapterRelease(polled.adapter);
    wgpuInstanceRelease(instance);
    CHECK(never.calls == 1 && never.reason == WGPUDeviceLostReason_CallbackCancelled);
    wgpuDeviceRelease(survivor);
    CHECK(never.calls == 1);
    printf("instance dropped: pending callback cancelled\n");

    printf("ok\n");
    return 0;
}
