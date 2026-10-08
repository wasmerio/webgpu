// Shared helpers for the conformance programs in this directory.
//
// Each program prints a deterministic transcript on stdout; manifest.json
// records what it must print. The same binaries run under the native host
// (tests/manifest_tests.rs) and in a browser (tests/browser).

#ifndef WGPU_TEST_H_
#define WGPU_TEST_H_

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <webgpu/webgpu.h>
#include <webgpu/webgpu_wasix.h>

#define SV(text) ((WGPUStringView){ (text), WGPU_STRLEN })

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #condition);       \
            fflush(stdout);                                                     \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Errors a test does not expect make it fail loudly.
static int wt_uncaptured_errors;

static void wt_on_uncaptured_error(WGPUDevice const* device, WGPUErrorType type, WGPUStringView message,
                                   void* userdata1, void* userdata2) {
    (void)device; (void)userdata1; (void)userdata2;
    wt_uncaptured_errors++;
    printf("uncaptured error (type %d): %.*s\n", (int)type, (int)message.length, message.data);
}

static WGPUInstance wt_create_instance(void) {
    static const WGPUInstanceFeatureName kFeatures[] = { WGPUInstanceFeatureName_TimedWaitAny };
    WGPUInstanceDescriptor descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    descriptor.requiredFeatureCount = 1;
    descriptor.requiredFeatures = kFeatures;
    WGPUInstance instance = wgpuCreateInstance(&descriptor);
    CHECK(instance != NULL);
    return instance;
}

// Blocks until `future` completes.
static void wt_wait(WGPUInstance instance, WGPUFuture future) {
    WGPUFutureWaitInfo info = { future, false };
    CHECK(wgpuInstanceWaitAny(instance, 1, &info, UINT64_MAX) == WGPUWaitStatus_Success);
    CHECK(info.completed);
}

static void wt_on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message,
                          void* userdata1, void* userdata2) {
    (void)userdata2;
    if (status != WGPURequestAdapterStatus_Success)
        printf("request adapter failed (%d): %.*s\n", (int)status, (int)message.length, message.data);
    *(WGPUAdapter*)userdata1 = adapter;
}

static WGPUAdapter wt_request_adapter(WGPUInstance instance) {
    WGPUAdapter adapter = NULL;
    WGPURequestAdapterCallbackInfo callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = wt_on_adapter;
    callback.userdata1 = &adapter;
    wt_wait(instance, wgpuInstanceRequestAdapter(instance, NULL, callback));
    CHECK(adapter != NULL);
    return adapter;
}

static void wt_on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1,
                         void* userdata2) {
    (void)userdata2;
    if (status != WGPURequestDeviceStatus_Success)
        printf("request device failed (%d): %.*s\n", (int)status, (int)message.length, message.data);
    *(WGPUDevice*)userdata1 = device;
}

static WGPUDevice wt_request_device_with(WGPUInstance instance, WGPUAdapter adapter, WGPUDeviceDescriptor* descriptor) {
    WGPUDevice device = NULL;
    if (!descriptor->uncapturedErrorCallbackInfo.callback)
        descriptor->uncapturedErrorCallbackInfo.callback = wt_on_uncaptured_error;
    WGPURequestDeviceCallbackInfo callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = wt_on_device;
    callback.userdata1 = &device;
    wt_wait(instance, wgpuAdapterRequestDevice(adapter, descriptor, callback));
    CHECK(device != NULL);
    return device;
}

static WGPUDevice wt_request_device(WGPUInstance instance, WGPUAdapter adapter) {
    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    return wt_request_device_with(instance, adapter, &descriptor);
}

static WGPUShaderModule wt_shader(WGPUDevice device, const char* wgsl) {
    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code = SV(wgsl);
    WGPUShaderModuleDescriptor descriptor = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    descriptor.nextInChain = &source.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &descriptor);
    CHECK(module != NULL);
    return module;
}

static void wt_on_map(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
    (void)message; (void)userdata2;
    *(WGPUMapAsyncStatus*)userdata1 = status;
}

// Maps `buffer` and blocks until the map completes. Returns its status.
static WGPUMapAsyncStatus wt_map(WGPUInstance instance, WGPUBuffer buffer, WGPUMapMode mode, size_t offset,
                                 size_t size) {
    WGPUMapAsyncStatus status = (WGPUMapAsyncStatus)0;
    WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = wt_on_map;
    callback.userdata1 = &status;
    wt_wait(instance, wgpuBufferMapAsync(buffer, mode, offset, size, callback));
    return status;
}

// Submits everything `encoder` recorded and releases it.
static void wt_submit(WGPUQueue queue, WGPUCommandEncoder encoder) {
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    CHECK(commands != NULL);
    wgpuQueueSubmit(queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);
}

// Reads a four-bytes-per-pixel 2D texture back into a malloc'd block, rows
// tightly packed. The texture needs the CopySrc usage.
static uint8_t* wt_read_texture(WGPUInstance instance, WGPUDevice device, WGPUQueue queue, WGPUTexture texture,
                                uint32_t width, uint32_t height) {
    uint32_t stride = (width * 4 + 255) & ~255u;  // rows are aligned to 256 bytes in a copy
    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    descriptor.size = (uint64_t)stride * height;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &descriptor);
    CHECK(buffer != NULL);

    WGPUTexelCopyTextureInfo source = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    source.texture = texture;
    WGPUTexelCopyBufferInfo destination = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    destination.buffer = buffer;
    destination.layout.bytesPerRow = stride;
    destination.layout.rowsPerImage = height;
    WGPUExtent3D size = { width, height, 1 };
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &size);
    wt_submit(queue, encoder);

    CHECK(wt_map(instance, buffer, WGPUMapMode_Read, 0, WGPU_WHOLE_MAP_SIZE) == WGPUMapAsyncStatus_Success);
    const uint8_t* mapped = wgpuBufferGetConstMappedRange(buffer, 0, WGPU_WHOLE_MAP_SIZE);
    CHECK(mapped != NULL);
    uint8_t* pixels = malloc((size_t)width * height * 4);
    for (uint32_t y = 0; y < height; y++) memcpy(pixels + (size_t)y * width * 4, mapped + (size_t)y * stride, width * 4);
    wgpuBufferUnmap(buffer);
    wgpuBufferRelease(buffer);
    return pixels;
}

static void wt_print_pixel(const char* what, const uint8_t* pixels, uint32_t width, uint32_t x, uint32_t y) {
    const uint8_t* p = pixels + ((size_t)y * width + x) * 4;
    printf("%s: %u %u %u %u\n", what, p[0], p[1], p[2], p[3]);
}

// A buffer holding `data`, uploaded through a mapping at creation.
static WGPUBuffer wt_buffer_with(WGPUDevice device, WGPUBufferUsage usage, const void* data, size_t size) {
    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.usage = usage;
    descriptor.size = (size + 3) & ~(size_t)3;
    descriptor.mappedAtCreation = true;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &descriptor);
    CHECK(buffer != NULL);
    CHECK(wgpuBufferWriteMappedRange(buffer, 0, data, size) == WGPUStatus_Success);
    wgpuBufferUnmap(buffer);
    return buffer;
}

// A 2D render target that can also be read back and sampled.
static WGPUTexture wt_target(WGPUDevice device, WGPUTextureFormat format, uint32_t width, uint32_t height,
                             uint32_t samples) {
    WGPUTextureDescriptor descriptor = WGPU_TEXTURE_DESCRIPTOR_INIT;
    descriptor.usage = WGPUTextureUsage_RenderAttachment |
                       (samples > 1 ? 0 : WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding);
    descriptor.size = (WGPUExtent3D){ width, height, 1 };
    descriptor.format = format;
    descriptor.sampleCount = samples;
    WGPUTexture texture = wgpuDeviceCreateTexture(device, &descriptor);
    CHECK(texture != NULL);
    return texture;
}

// FNV-1a, for printing a stable fingerprint of rendered pixels.
static uint32_t wt_hash(const void* data, size_t size) {
    const uint8_t* bytes = data;
    uint32_t hash = 2166136261u;
    for (size_t i = 0; i < size; i++) hash = (hash ^ bytes[i]) * 16777619u;
    return hash;
}

// Ends the test: fails if any unexpected error was reported.
static int wt_done(void) {
    if (wt_uncaptured_errors) {
        printf("FAILED: %d uncaptured error(s)\n", wt_uncaptured_errors);
        return 1;
    }
    printf("ok\n");
    return 0;
}

#endif  // WGPU_TEST_H_
