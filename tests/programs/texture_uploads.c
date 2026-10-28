// Large queue writes to textures, the size of a video frame: padded strides,
// offsets, a mip level of an array layer, a stream of frames each copied out
// before the next, a small write from a large buffer, and the invalid writes
// that must still be reported.

#include "wgpu_test.h"

#define WIDTH 512
#define HEIGHT 256

// The texel at (x, y) of the image numbered `seed`.
static void texel(uint8_t out[4], uint32_t x, uint32_t y, uint32_t seed) {
    out[0] = (uint8_t)x;
    out[1] = (uint8_t)y;
    out[2] = (uint8_t)(((x >> 8) | ((y >> 8) << 4)) ^ (seed * 37));
    out[3] = (uint8_t)seed;
}

// `rows` rows of `width` texels at `stride` bytes after `offset` bytes, for
// the image `seed` from (x0, y0) on. Bytes between rows are 0xEE.
static uint8_t* image(uint32_t offset, uint32_t stride, uint32_t width, uint32_t rows, uint32_t x0, uint32_t y0,
                      uint32_t seed, uint32_t* size) {
    *size = offset + stride * (rows - 1) + width * 4;
    uint8_t* data = malloc(*size);
    memset(data, 0xEE, *size);
    for (uint32_t y = 0; y < rows; y++)
        for (uint32_t x = 0; x < width; x++) texel(data + offset + y * stride + x * 4, x0 + x, y0 + y, seed);
    return data;
}

static WGPUTexture texture(WGPUDevice device, WGPUTextureUsage usage, uint32_t width, uint32_t height,
                           uint32_t layers, uint32_t levels) {
    WGPUTextureDescriptor descriptor = WGPU_TEXTURE_DESCRIPTOR_INIT;
    descriptor.usage = usage;
    descriptor.size = (WGPUExtent3D){ width, height, layers };
    descriptor.format = WGPUTextureFormat_RGBA8Unorm;
    descriptor.mipLevelCount = levels;
    WGPUTexture texture = wgpuDeviceCreateTexture(device, &descriptor);
    CHECK(texture != NULL);
    return texture;
}

static void write(WGPUQueue queue, WGPUTexture texture, uint32_t level, WGPUOrigin3D origin, const uint8_t* data,
                  uint32_t size, uint32_t offset, uint32_t stride, uint32_t rows_per_image, WGPUExtent3D extent) {
    WGPUTexelCopyTextureInfo destination = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    destination.texture = texture;
    destination.mipLevel = level;
    destination.origin = origin;
    WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    layout.offset = offset;
    layout.bytesPerRow = stride;
    layout.rowsPerImage = rows_per_image;
    wgpuQueueWriteTexture(queue, &destination, data, size, &layout, &extent);
}

// Records a copy of one level of one layer into a new buffer, rows 256-byte
// aligned, and returns the buffer.
static WGPUBuffer copy_out(WGPUDevice device, WGPUQueue queue, WGPUTexture texture, uint32_t level, uint32_t layer,
                           uint32_t width, uint32_t height) {
    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    descriptor.size = (uint64_t)((width * 4 + 255) & ~255u) * height;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &descriptor);
    CHECK(buffer != NULL);
    WGPUTexelCopyTextureInfo source = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    source.texture = texture;
    source.mipLevel = level;
    source.origin = (WGPUOrigin3D){ 0, 0, layer };
    WGPUTexelCopyBufferInfo destination = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    destination.buffer = buffer;
    destination.layout.bytesPerRow = (width * 4 + 255) & ~255u;
    destination.layout.rowsPerImage = height;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &(WGPUExtent3D){ width, height, 1 });
    wt_submit(queue, encoder);
    return buffer;
}

// Whether the copied-out buffer holds image `seed` in the rectangle at
// (x0, y0) of `rect_width` x `rect_height`, and zeros around it.
static bool holds(WGPUInstance instance, WGPUBuffer buffer, uint32_t width, uint32_t height, uint32_t x0, uint32_t y0,
                  uint32_t rect_width, uint32_t rect_height, uint32_t seed) {
    CHECK(wt_map(instance, buffer, WGPUMapMode_Read, 0, WGPU_WHOLE_MAP_SIZE) == WGPUMapAsyncStatus_Success);
    const uint8_t* mapped = wgpuBufferGetConstMappedRange(buffer, 0, WGPU_WHOLE_MAP_SIZE);
    CHECK(mapped != NULL);
    uint32_t stride = (width * 4 + 255) & ~255u;
    bool same = true;
    for (uint32_t y = 0; y < height && same; y++)
        for (uint32_t x = 0; x < width && same; x++) {
            uint8_t expected[4] = { 0, 0, 0, 0 };
            if (x >= x0 && x < x0 + rect_width && y >= y0 && y < y0 + rect_height) texel(expected, x, y, seed);
            same = memcmp(mapped + y * stride + x * 4, expected, 4) == 0;
            if (!same) {
                const uint8_t* got = mapped + y * stride + x * 4;
                printf("texel (%u, %u): %u %u %u %u, expected %u %u %u %u\n", x, y, got[0], got[1], got[2], got[3],
                       expected[0], expected[1], expected[2], expected[3]);
            }
        }
    wgpuBufferUnmap(buffer);
    wgpuBufferRelease(buffer);
    return same;
}

static void on_scope(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1,
                     void* userdata2) {
    (void)message; (void)userdata2;
    *(WGPUErrorType*)userdata1 = status == WGPUPopErrorScopeStatus_Success ? type : WGPUErrorType_Unknown;
}

static const char* error_name(WGPUInstance instance, WGPUDevice device) {
    WGPUErrorType type = WGPUErrorType_Unknown;
    WGPUPopErrorScopeCallbackInfo callback = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = on_scope;
    callback.userdata1 = &type;
    wt_wait(instance, wgpuDevicePopErrorScope(device, callback));
    switch (type) {
        case WGPUErrorType_NoError: return "no error";
        case WGPUErrorType_Validation: return "validation error";
        default: return "unexpected error";
    }
}

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    const WGPUTextureUsage copies = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
    uint32_t size;

    // 400 x 200 texels at (60, 30), rows 100 bytes apart, 36 bytes in.
    WGPUTexture target = texture(device, copies, WIDTH, HEIGHT, 1, 1);
    uint8_t* data = image(36, 400 * 4 + 100, 400, 200, 60, 30, 1, &size);
    write(queue, target, 0, (WGPUOrigin3D){ 60, 30, 0 }, data, size, 36, 400 * 4 + 100, 200,
          (WGPUExtent3D){ 400, 200, 1 });
    free(data);
    CHECK(holds(instance, copy_out(device, queue, target, 0, 0, WIDTH, HEIGHT), WIDTH, HEIGHT, 60, 30, 400, 200, 1));
    printf("sub-rectangle with a padded stride and an offset: ok\n");

    // One guest buffer rewritten for each frame, every frame copied out right
    // after its write and checked once all are done.
    enum { FRAMES = 12 };
    WGPUBuffer frames[FRAMES];
    data = image(0, WIDTH * 4, WIDTH, HEIGHT, 0, 0, 0, &size);
    for (uint32_t frame = 0; frame < FRAMES; frame++) {
        for (uint32_t y = 0; y < HEIGHT; y++)
            for (uint32_t x = 0; x < WIDTH; x++) texel(data + y * WIDTH * 4 + x * 4, x, y, 100 + frame);
        write(queue, target, 0, (WGPUOrigin3D){ 0, 0, 0 }, data, size, 0, WIDTH * 4, HEIGHT,
              (WGPUExtent3D){ WIDTH, HEIGHT, 1 });
        frames[frame] = copy_out(device, queue, target, 0, 0, WIDTH, HEIGHT);
    }
    free(data);
    bool all = true;
    for (uint32_t frame = 0; frame < FRAMES; frame++)
        all &= holds(instance, frames[frame], WIDTH, HEIGHT, 0, 0, WIDTH, HEIGHT, 100 + frame);
    printf("%d frames in a row, each copied out before the next: %s\n", FRAMES, all ? "ok" : "wrong");

    // Mip level 1 of layer 1 of a 1024 x 512 texture with two layers.
    WGPUTexture layered = texture(device, copies, 2 * WIDTH, 2 * HEIGHT, 2, 2);
    data = image(0, WIDTH * 4, WIDTH, HEIGHT, 0, 0, 7, &size);
    write(queue, layered, 1, (WGPUOrigin3D){ 0, 0, 1 }, data, size, 0, WIDTH * 4, HEIGHT,
          (WGPUExtent3D){ WIDTH, HEIGHT, 1 });
    free(data);
    bool layer1 = holds(instance, copy_out(device, queue, layered, 1, 1, WIDTH, HEIGHT), WIDTH, HEIGHT, 0, 0, WIDTH,
                        HEIGHT, 7);
    bool layer0 = holds(instance, copy_out(device, queue, layered, 1, 0, WIDTH, HEIGHT), WIDTH, HEIGHT, 0, 0, 0, 0, 0);
    printf("mip level 1 of layer 1: %s\n", layer1 && layer0 ? "ok" : "wrong");

    // A few texels, 300000 bytes into a large buffer.
    WGPUTexture small = texture(device, copies, 64, 32, 1, 1);
    data = image(300000, 256, 16, 16, 40, 8, 9, &size);
    write(queue, small, 0, (WGPUOrigin3D){ 40, 8, 0 }, data, size, 300000, 256, 16, (WGPUExtent3D){ 16, 16, 1 });
    free(data);
    CHECK(holds(instance, copy_out(device, queue, small, 0, 0, 64, 32), 64, 32, 40, 8, 16, 16, 9));
    printf("a few texels deep inside a large buffer: ok\n");

    // Invalid writes: each reports one validation error and changes nothing.
    data = image(0, WIDTH * 4, WIDTH, HEIGHT, 0, 0, 3, &size);
    WGPUTexture source_only = texture(device, WGPUTextureUsage_CopySrc, WIDTH, HEIGHT, 1, 1);
    WGPUTexture blank = texture(device, copies, WIDTH, HEIGHT, 1, 1);
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    write(queue, blank, 0, (WGPUOrigin3D){ 0, 0, 0 }, data, size - 4, 0, WIDTH * 4, HEIGHT,
          (WGPUExtent3D){ WIDTH, HEIGHT, 1 });
    printf("one texel too little data: %s\n", error_name(instance, device));
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    write(queue, source_only, 0, (WGPUOrigin3D){ 0, 0, 0 }, data, size, 0, WIDTH * 4, HEIGHT,
          (WGPUExtent3D){ WIDTH, HEIGHT, 1 });
    printf("a texture without CopyDst: %s\n", error_name(instance, device));
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    write(queue, blank, 0, (WGPUOrigin3D){ 1, 0, 0 }, data, size, 0, WIDTH * 4, HEIGHT,
          (WGPUExtent3D){ WIDTH, HEIGHT, 1 });
    printf("one texel past the edge: %s\n", error_name(instance, device));
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    write(queue, blank, 0, (WGPUOrigin3D){ 0, 0, 0 }, data, size, 0, WIDTH * 4, HEIGHT - 1,
          (WGPUExtent3D){ WIDTH, HEIGHT, 1 });
    printf("fewer rows per image than rows: %s\n", error_name(instance, device));
    WGPUTexture destroyed = texture(device, copies, WIDTH, HEIGHT, 1, 1);
    wgpuTextureDestroy(destroyed);
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    write(queue, destroyed, 0, (WGPUOrigin3D){ 0, 0, 0 }, data, size, 0, WIDTH * 4, HEIGHT,
          (WGPUExtent3D){ WIDTH, HEIGHT, 1 });
    printf("a destroyed texture: %s\n", error_name(instance, device));
    free(data);
    CHECK(holds(instance, copy_out(device, queue, blank, 0, 0, WIDTH, HEIGHT), WIDTH, HEIGHT, 0, 0, 0, 0, 0));
    printf("invalid writes changed nothing: ok\n");

    return wt_done();
}
