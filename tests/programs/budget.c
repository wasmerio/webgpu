// What a GPU memory budget of exactly 1 MiB means to a program: storage past
// it is refused as an out-of-memory error, and destroying or releasing
// storage gives it back. The embedder tests of both hosts run this under that
// budget. (Not in the manifest: without the budget nothing here is refused.)

#include "wgpu_test.h"

#define KIB 1024u

static void on_scope(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1,
                     void* userdata2) {
    (void)status; (void)message; (void)userdata2;
    *(WGPUErrorType*)userdata1 = type;
}

static WGPUErrorType pop_scope(WGPUInstance instance, WGPUDevice device) {
    WGPUErrorType type = (WGPUErrorType)0;
    WGPUPopErrorScopeCallbackInfo callback = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = on_scope;
    callback.userdata1 = &type;
    wt_wait(instance, wgpuDevicePopErrorScope(device, callback));
    return type;
}

// Creates a buffer and reports whether the budget had room for it.
static bool buffer_fits(WGPUInstance instance, WGPUDevice device, uint64_t size, WGPUBuffer* out) {
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_OutOfMemory);
    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.size = size;
    descriptor.usage = WGPUBufferUsage_CopyDst;
    *out = wgpuDeviceCreateBuffer(device, &descriptor);
    WGPUErrorType error = pop_scope(instance, device);
    CHECK(error == WGPUErrorType_NoError || error == WGPUErrorType_OutOfMemory);
    return error == WGPUErrorType_NoError;
}

// The same for a 256x256 RGBA8 texture: 256 KiB.
static bool texture_fits(WGPUInstance instance, WGPUDevice device, WGPUTexture* out) {
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_OutOfMemory);
    WGPUTextureDescriptor descriptor = WGPU_TEXTURE_DESCRIPTOR_INIT;
    descriptor.usage = WGPUTextureUsage_CopyDst;
    descriptor.size = (WGPUExtent3D){ 256, 256, 1 };
    descriptor.format = WGPUTextureFormat_RGBA8Unorm;
    *out = wgpuDeviceCreateTexture(device, &descriptor);
    WGPUErrorType error = pop_scope(instance, device);
    CHECK(error == WGPUErrorType_NoError || error == WGPUErrorType_OutOfMemory);
    return error == WGPUErrorType_NoError;
}

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);

    WGPUBuffer first, second, third, fourth;
    CHECK(buffer_fits(instance, device, 768 * KIB, &first));
    printf("768 KiB: fits\n");
    CHECK(!buffer_fits(instance, device, 768 * KIB, &second));
    if (second) wgpuBufferRelease(second);
    printf("768 KiB more: out of memory\n");

    // Destroying storage gives it back, even while the handle is held.
    wgpuBufferDestroy(first);
    CHECK(buffer_fits(instance, device, 768 * KIB, &third));
    printf("after destroying the first: fits\n");
    wgpuBufferRelease(first);

    // So does letting go of the last reference.
    wgpuBufferRelease(third);
    CHECK(buffer_fits(instance, device, 768 * KIB, &fourth));
    printf("after releasing it: fits\n");

    // Textures count too, and the budget is exact: 768 + 256 KiB is all of it.
    WGPUTexture texture, another;
    CHECK(texture_fits(instance, device, &texture));
    CHECK(!texture_fits(instance, device, &another));
    if (another) wgpuTextureRelease(another);
    wgpuTextureDestroy(texture);
    CHECK(texture_fits(instance, device, &another));
    printf("textures: counted, and given back\n");

    wgpuTextureRelease(another);
    wgpuTextureRelease(texture);
    wgpuBufferRelease(fourth);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
