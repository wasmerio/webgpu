// Moving texels and bytes around: queue writes, every copy command between
// buffers and textures, mip levels and array layers, buffer clears, and the
// "submitted work is done" notification.

#include "wgpu_test.h"

static void on_work_done(WGPUQueueWorkDoneStatus status, WGPUStringView message, void* userdata1, void* userdata2) {
    (void)message; (void)userdata2;
    *(WGPUQueueWorkDoneStatus*)userdata1 = status;
}

// Reads `width` x `height` texels of one mip level and array layer.
static uint8_t* read_level(WGPUInstance instance, WGPUDevice device, WGPUQueue queue, WGPUTexture texture,
                           uint32_t level, uint32_t layer, uint32_t width, uint32_t height) {
    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    descriptor.size = 256 * height;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &descriptor);
    WGPUTexelCopyTextureInfo source = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    source.texture = texture;
    source.mipLevel = level;
    source.origin = (WGPUOrigin3D){ 0, 0, layer };
    source.aspect = WGPUTextureAspect_All;
    WGPUTexelCopyBufferInfo destination = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    destination.buffer = buffer;
    destination.layout.offset = 0;
    destination.layout.bytesPerRow = 256;
    destination.layout.rowsPerImage = height;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &(WGPUExtent3D){ width, height, 1 });
    wt_submit(queue, encoder);
    CHECK(wt_map(instance, buffer, WGPUMapMode_Read, 0, WGPU_WHOLE_MAP_SIZE) == WGPUMapAsyncStatus_Success);
    uint8_t* texels = malloc((size_t)width * height * 4);
    for (uint32_t y = 0; y < height; y++)
        CHECK(wgpuBufferReadMappedRange(buffer, 256 * y, texels + (size_t)y * width * 4, width * 4) ==
              WGPUStatus_Success);
    wgpuBufferUnmap(buffer);
    wgpuBufferRelease(buffer);
    return texels;
}

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    // A 4x4 texture with two mip levels and three array layers.
    WGPUTextureDescriptor descriptor = WGPU_TEXTURE_DESCRIPTOR_INIT;
    descriptor.usage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst | WGPUTextureUsage_TextureBinding;
    descriptor.dimension = WGPUTextureDimension_2D;
    descriptor.size = (WGPUExtent3D){ 4, 4, 3 };
    descriptor.format = WGPUTextureFormat_RGBA8Unorm;
    descriptor.mipLevelCount = 2;
    descriptor.sampleCount = 1;
    WGPUTexture source = wgpuDeviceCreateTexture(device, &descriptor);
    CHECK(source != NULL);
    CHECK(wgpuTextureGetDepthOrArrayLayers(source) == 3 && wgpuTextureGetMipLevelCount(source) == 2);
    (void)wgpuTextureGetTextureBindingViewDimension(source);

    // A view of one layer of the second mip level.
    WGPUTextureViewDescriptor view_descriptor = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    view_descriptor.format = WGPUTextureFormat_RGBA8Unorm;
    view_descriptor.dimension = WGPUTextureViewDimension_2D;
    view_descriptor.baseMipLevel = 1;
    view_descriptor.mipLevelCount = 1;
    view_descriptor.baseArrayLayer = 2;
    view_descriptor.arrayLayerCount = 1;
    WGPUTextureView view = wgpuTextureCreateView(source, &view_descriptor);
    CHECK(view != NULL);
    // And the whole thing, as an array.
    WGPUTextureViewDescriptor array_descriptor = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    array_descriptor.dimension = WGPUTextureViewDimension_2DArray;
    WGPUTextureView array_view = wgpuTextureCreateView(source, &array_descriptor);
    CHECK(array_view != NULL);

    // Level 0 of layer 0: texel (x, y) = (x, y, x + y, 255). Level 1 of
    // layer 2: all (9, 9, 9, 9). Written straight from guest memory.
    uint8_t level0[4 * 4 * 4];
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            uint8_t* texel = &level0[(y * 4 + x) * 4];
            texel[0] = (uint8_t)x; texel[1] = (uint8_t)y; texel[2] = (uint8_t)(x + y); texel[3] = 255;
        }
    WGPUTexelCopyTextureInfo write_to = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    write_to.texture = source;
    WGPUTexelCopyBufferLayout write_layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    write_layout.bytesPerRow = 16;
    write_layout.rowsPerImage = 4;
    wgpuQueueWriteTexture(queue, &write_to, level0, sizeof level0, &write_layout, &(WGPUExtent3D){ 4, 4, 1 });
    uint8_t level1[2 * 2 * 4];
    memset(level1, 9, sizeof level1);
    write_to.mipLevel = 1;
    write_to.origin = (WGPUOrigin3D){ 0, 0, 2 };
    write_layout.bytesPerRow = 8;
    write_layout.rowsPerImage = 2;
    wgpuQueueWriteTexture(queue, &write_to, level1, sizeof level1, &write_layout, &(WGPUExtent3D){ 2, 2, 1 });

    // Texture to texture, then a 2x2 patch from a buffer on top.
    descriptor.size = (WGPUExtent3D){ 4, 4, 1 };
    descriptor.mipLevelCount = 1;
    WGPUTexture destination = wgpuDeviceCreateTexture(device, &descriptor);
    static const uint8_t kPatch[2 * 2 * 4] = {
        200, 201, 202, 203,   210, 211, 212, 213,
        220, 221, 222, 223,   230, 231, 232, 233,
    };
    // Rows of a buffer-to-texture copy are bytesPerRow apart: 256 here.
    uint8_t staged[256 + 8] = { 0 };
    memcpy(staged, kPatch, 8);
    memcpy(staged + 256, kPatch + 8, 8);
    WGPUBuffer staging = wt_buffer_with(device, WGPUBufferUsage_CopySrc, staged, sizeof staged);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPUTexelCopyTextureInfo from = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    from.texture = source;
    WGPUTexelCopyTextureInfo to = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    to.texture = destination;
    wgpuCommandEncoderCopyTextureToTexture(encoder, &from, &to, &(WGPUExtent3D){ 4, 4, 1 });
    WGPUTexelCopyBufferInfo patch = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    patch.buffer = staging;
    patch.layout.bytesPerRow = 256;
    patch.layout.rowsPerImage = 2;
    to.origin = (WGPUOrigin3D){ 1, 1, 0 };
    wgpuCommandEncoderCopyBufferToTexture(encoder, &patch, &to, &(WGPUExtent3D){ 2, 2, 1 });
    wt_submit(queue, encoder);

    uint8_t* texels = read_level(instance, device, queue, destination, 0, 0, 4, 4);
    // Outside the patch the copy of level 0 shows; inside, the patch.
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++) {
            const uint8_t* texel = &texels[(y * 4 + x) * 4];
            bool patched = x >= 1 && x <= 2 && y >= 1 && y <= 2;
            if (patched) {
                CHECK(memcmp(texel, &kPatch[((y - 1) * 2 + (x - 1)) * 4], 4) == 0);
            } else {
                CHECK(texel[0] == x && texel[1] == y && texel[2] == x + y && texel[3] == 255);
            }
        }
    free(texels);
    printf("texture copies: ok\n");

    texels = read_level(instance, device, queue, source, 1, 2, 2, 2);
    for (int i = 0; i < 16; i++) CHECK(texels[i] == 9);
    free(texels);
    printf("mip level 1 of layer 2: ok\n");

    // Buffers: a write at an offset, a clear in the middle, read back.
    WGPUBufferDescriptor buffer_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    buffer_descriptor.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    buffer_descriptor.size = 32;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &buffer_descriptor);
    uint8_t bytes[32];
    for (int i = 0; i < 32; i++) bytes[i] = (uint8_t)(i + 1);
    wgpuQueueWriteBuffer(queue, buffer, 0, bytes, 32);
    static const uint8_t kMark[4] = { 0xAA, 0xBB, 0xCC, 0xDD };
    wgpuQueueWriteBuffer(queue, buffer, 4, kMark, 4);
    WGPUBufferDescriptor readback_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    readback_descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    readback_descriptor.size = 32;
    WGPUBuffer readback = wgpuDeviceCreateBuffer(device, &readback_descriptor);
    encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    wgpuCommandEncoderClearBuffer(encoder, buffer, 16, 8);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, buffer, 0, readback, 0, WGPU_WHOLE_SIZE);
    wt_submit(queue, encoder);

    // The queue says when everything submitted so far has run.
    WGPUQueueWorkDoneStatus done = (WGPUQueueWorkDoneStatus)0;
    WGPUQueueWorkDoneCallbackInfo done_callback = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    done_callback.mode = WGPUCallbackMode_WaitAnyOnly;
    done_callback.callback = on_work_done;
    done_callback.userdata1 = &done;
    wt_wait(instance, wgpuQueueOnSubmittedWorkDone(queue, done_callback));
    CHECK(done == WGPUQueueWorkDoneStatus_Success);
    printf("submitted work done: ok\n");

    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 0, 32) == WGPUMapAsyncStatus_Success);
    const uint8_t* result = wgpuBufferGetConstMappedRange(readback, 0, 32);
    CHECK(result != NULL);
    for (int i = 0; i < 32; i++) {
        uint8_t expected = (uint8_t)(i + 1);
        if (i >= 4 && i < 8) expected = kMark[i - 4];
        if (i >= 16 && i < 24) expected = 0;
        CHECK(result[i] == expected);
    }
    wgpuBufferUnmap(readback);
    printf("buffer write, clear and copy: ok\n");

    wgpuTextureDestroy(destination);
    wgpuTextureViewRelease(array_view);
    wgpuTextureViewRelease(view);
    wgpuBufferRelease(readback);
    wgpuBufferRelease(buffer);
    wgpuBufferRelease(staging);
    wgpuTextureRelease(destination);
    wgpuTextureRelease(source);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
