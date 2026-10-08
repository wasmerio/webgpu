// Presenting: a surface from the host-provided target, its capabilities, the
// configure / get-current-texture / present cycle with its error cases, and
// a native-style blocking frame loop.

#include "wgpu_test.h"

static WGPUSurfaceGetCurrentTextureStatus acquire(WGPUSurface surface, WGPUTexture* texture) {
    WGPUSurfaceTexture frame = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(surface, &frame);
    *texture = frame.texture;
    return frame.status;
}

int main(void) {
    WGPUInstance instance = wt_create_instance();

    WGPUWasixSurfaceSourceCanvas source = WGPU_WASIX_SURFACE_SOURCE_CANVAS_INIT;
    source.selector = SV("#canvas");
    WGPUSurfaceDescriptor surface_descriptor = WGPU_SURFACE_DESCRIPTOR_INIT;
    surface_descriptor.label = SV("main surface");
    surface_descriptor.nextInChain = &source.chain;
    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surface_descriptor);
    CHECK(surface != NULL);
    wgpuSurfaceSetLabel(surface, SV("renamed"));
    wgpuSurfaceAddRef(surface);
    wgpuSurfaceRelease(surface);

    // Ask for an adapter that can present to it.
    WGPUAdapter adapter = NULL;
    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    options.compatibleSurface = surface;
    options.powerPreference = WGPUPowerPreference_HighPerformance;
    WGPURequestAdapterCallbackInfo callback = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = wt_on_adapter;
    callback.userdata1 = &adapter;
    wt_wait(instance, wgpuInstanceRequestAdapter(instance, &options, callback));
    CHECK(adapter != NULL);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    WGPUSurfaceCapabilities capabilities = WGPU_SURFACE_CAPABILITIES_INIT;
    CHECK(wgpuSurfaceGetCapabilities(surface, adapter, &capabilities) == WGPUStatus_Success);
    CHECK(capabilities.formatCount >= 1 && capabilities.formats != NULL);
    CHECK(capabilities.alphaModeCount >= 1);
    CHECK(capabilities.usages & WGPUTextureUsage_RenderAttachment);
    bool fifo = false;
    for (size_t i = 0; i < capabilities.presentModeCount; i++)
        fifo |= capabilities.presentModes[i] == WGPUPresentMode_Fifo;
    CHECK(fifo);
    WGPUTextureFormat format = capabilities.formats[0];
    WGPUCompositeAlphaMode alpha = capabilities.alphaModes[0];
    wgpuSurfaceCapabilitiesFreeMembers(capabilities);
    printf("capabilities: ok\n");

    uint32_t width = 0, height = 0;
    CHECK(wgpuWasixSurfaceGetSize(surface, &width, &height) == WGPUStatus_Success);
    CHECK(width > 0 && height > 0);

    // Unconfigured: no frame, nothing to present.
    WGPUTexture texture = NULL;
    CHECK(acquire(surface, &texture) == WGPUSurfaceGetCurrentTextureStatus_Error && texture == NULL);
    CHECK(wgpuSurfacePresent(surface) == WGPUStatus_Error);
    printf("unconfigured: no current texture\n");

    WGPUSurfaceConfiguration configuration = WGPU_SURFACE_CONFIGURATION_INIT;
    configuration.device = device;
    configuration.format = format;
    configuration.usage = WGPUTextureUsage_RenderAttachment;
    configuration.width = width;
    configuration.height = height;
    configuration.alphaMode = alpha;
    configuration.presentMode = WGPUPresentMode_Fifo;
    // Colour management, spelled out with the values every target takes.
    WGPUSurfaceColorManagement colors = WGPU_SURFACE_COLOR_MANAGEMENT_INIT;
    colors.colorSpace = WGPUPredefinedColorSpace_SRGB;
    colors.toneMappingMode = WGPUToneMappingMode_Standard;
    configuration.nextInChain = &colors.chain;
    wgpuSurfaceConfigure(surface, &configuration);

    // The frame loop a native program writes: acquire, draw, present, repeat.
    for (int frame = 0; frame < 3; frame++) {
        WGPUSurfaceGetCurrentTextureStatus status = acquire(surface, &texture);
        CHECK(status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
              status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal);
        CHECK(texture != NULL);
        CHECK(wgpuTextureGetWidth(texture) == width && wgpuTextureGetHeight(texture) == height);
        CHECK(wgpuTextureGetFormat(texture) == format);
        if (frame == 0) {
            // One frame at a time: asking again before presenting is an error.
            WGPUTexture again = (WGPUTexture)1;
            CHECK(acquire(surface, &again) == WGPUSurfaceGetCurrentTextureStatus_Error && again == NULL);
        }

        WGPUTextureView view = wgpuTextureCreateView(texture, NULL);
        WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        attachment.view = view;
        attachment.loadOp = WGPULoadOp_Clear;
        attachment.storeOp = WGPUStoreOp_Store;
        // Red, then green, then blue.
        attachment.clearValue = (WGPUColor){ frame == 0, frame == 1, frame == 2, 1 };
        WGPURenderPassDescriptor pass_descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        pass_descriptor.colorAttachmentCount = 1;
        pass_descriptor.colorAttachments = &attachment;
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
        wt_submit(queue, encoder);
        wgpuTextureViewRelease(view);

        CHECK(wgpuSurfacePresent(surface) == WGPUStatus_Success);
        wgpuTextureRelease(texture);
        wgpuInstanceProcessEvents(instance);
    }
    // Presenting consumed the frame.
    CHECK(wgpuSurfacePresent(surface) == WGPUStatus_Error);
    printf("presented 3 frames\n");

    wgpuSurfaceUnconfigure(surface);
    CHECK(acquire(surface, &texture) == WGPUSurfaceGetCurrentTextureStatus_Error && texture == NULL);
    printf("unconfigured again: no current texture\n");

    wgpuSurfaceRelease(surface);

    // A released surface gives its target back: a program can make another
    // one for it. Programs written for Emscripten name the source struct
    // differently; it is the same struct.
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector html = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
    html.selector = SV("#canvas");
    WGPUSurfaceDescriptor again_descriptor = WGPU_SURFACE_DESCRIPTOR_INIT;
    again_descriptor.nextInChain = &html.chain;
    WGPUSurface again = wgpuInstanceCreateSurface(instance, &again_descriptor);
    CHECK(again != NULL);
    CHECK(wgpuWasixSurfaceGetSize(again, &width, &height) == WGPUStatus_Success);
    wgpuSurfaceRelease(again);
    printf("surface made again for the same target\n");

    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
