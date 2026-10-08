// Error reporting: scopes (per device, innermost matching filter wins),
// the uncaptured-error callback, and the status of popping an empty stack.

#include "wgpu_test.h"

typedef struct Scope {
    int calls;
    WGPUPopErrorScopeStatus status;
    WGPUErrorType type;
    size_t message_length;
} Scope;

static void on_scope(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1,
                     void* userdata2) {
    (void)userdata2;
    Scope* scope = userdata1;
    scope->calls++;
    scope->status = status;
    scope->type = type;
    scope->message_length = message.data ? message.length : 0;
}

static Scope pop(WGPUInstance instance, WGPUDevice device) {
    Scope scope = { 0 };
    WGPUPopErrorScopeCallbackInfo callback = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = on_scope;
    callback.userdata1 = &scope;
    wt_wait(instance, wgpuDevicePopErrorScope(device, callback));
    CHECK(scope.calls == 1);
    return scope;
}

// A buffer nobody may create: mappable both ways at once.
static void make_invalid_buffer(WGPUDevice device) {
    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_MapWrite;
    descriptor.size = 16;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &descriptor);
    if (buffer) wgpuBufferRelease(buffer);
}

typedef struct Uncaptured {
    int calls;
    WGPUErrorType type;
    bool device_matches;
    WGPUDevice expected;
} Uncaptured;

static void on_uncaptured(WGPUDevice const* device, WGPUErrorType type, WGPUStringView message, void* userdata1,
                          void* userdata2) {
    (void)message; (void)userdata2;
    Uncaptured* seen = userdata1;
    seen->calls++;
    seen->type = type;
    seen->device_matches = *device == seen->expected;
}

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);

    Uncaptured uncaptured = { 0 };
    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    descriptor.uncapturedErrorCallbackInfo.callback = on_uncaptured;
    descriptor.uncapturedErrorCallbackInfo.userdata1 = &uncaptured;
    WGPUDevice device = wt_request_device_with(instance, adapter, &descriptor);
    uncaptured.expected = device;

    // A scope with nothing in it pops clean.
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    Scope clean = pop(instance, device);
    CHECK(clean.status == WGPUPopErrorScopeStatus_Success && clean.type == WGPUErrorType_NoError);
    CHECK(clean.message_length == 0);
    printf("empty scope: no error\n");

    // A validation error is captured, with a message.
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    make_invalid_buffer(device);
    Scope captured = pop(instance, device);
    CHECK(captured.status == WGPUPopErrorScopeStatus_Success && captured.type == WGPUErrorType_Validation);
    CHECK(captured.message_length > 0);
    CHECK(uncaptured.calls == 0);
    printf("validation scope: captured\n");

    // Scopes nest, and an error goes to the innermost one whose filter
    // matches: here past the out-of-memory scope to the validation scope.
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_OutOfMemory);
    make_invalid_buffer(device);
    make_invalid_buffer(device);
    Scope inner = pop(instance, device);
    CHECK(inner.type == WGPUErrorType_NoError);
    Scope outer = pop(instance, device);
    CHECK(outer.type == WGPUErrorType_Validation);
    printf("nested scopes: error reached the matching scope\n");

    // Popping with no scope on the stack is itself reported.
    Scope empty = pop(instance, device);
    CHECK(empty.status == WGPUPopErrorScopeStatus_Error);
    printf("pop without push: error status\n");

    // With no scope, the error goes to the device's uncaptured callback.
    make_invalid_buffer(device);
    for (int i = 0; i < 1000 && uncaptured.calls == 0; i++) wgpuInstanceProcessEvents(instance);
    CHECK(uncaptured.calls == 1 && uncaptured.type == WGPUErrorType_Validation && uncaptured.device_matches);
    printf("uncaptured error: delivered to the device callback\n");

    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    printf("ok\n");
    return 0;
}
