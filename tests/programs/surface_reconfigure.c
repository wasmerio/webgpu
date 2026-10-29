// Ending a surface's configuration: Unconfigure, and Configure replacing it
// (even with a configuration that is refused), destroy the frame a program
// acquired and did not present, though it still holds the texture; and a
// surface configured for another device presents what that device drew. The
// embedder tests run this against a capturing target and check that the last
// frame is the second device's green.

#include "wgpu_test.h"

static WGPUTexture acquire(WGPUSurface surface) {
    WGPUSurfaceTexture frame = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(surface, &frame);
    CHECK(frame.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
          frame.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal);
    CHECK(frame.texture != NULL);
    return frame.texture;
}

// Whether the surface hands out a frame now (one it hands out is released).
static bool has_frame(WGPUSurface surface) {
    WGPUSurfaceTexture frame = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(surface, &frame);
    if (frame.texture == NULL) return false;
    wgpuTextureRelease(frame.texture);
    return true;
}

static void on_scope(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1,
                     void* userdata2) {
    (void)message; (void)userdata2;
    *(WGPUErrorType*)userdata1 = status == WGPUPopErrorScopeStatus_Success ? type : WGPUErrorType_Unknown;
}

// The error type the scope pushed last caught.
static WGPUErrorType pop_error(WGPUInstance instance, WGPUDevice device) {
    WGPUErrorType type = WGPUErrorType_Unknown;
    WGPUPopErrorScopeCallbackInfo callback = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = on_scope;
    callback.userdata1 = &type;
    wt_wait(instance, wgpuDevicePopErrorScope(device, callback));
    return type;
}

// Clears `texture` to `color` from `device`: "drawn", or "destroyed" when the
// submit is refused for using a destroyed texture.
static const char* clear(WGPUInstance instance, WGPUDevice device, WGPUTexture texture, WGPUColor color) {
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPUTextureView view = wgpuTextureCreateView(texture, NULL);
    WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    attachment.view = view;
    attachment.loadOp = WGPULoadOp_Clear;
    attachment.storeOp = WGPUStoreOp_Store;
    attachment.clearValue = color;
    WGPURenderPassDescriptor descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = &attachment;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    wt_submit(queue, encoder);
    wgpuQueueRelease(queue);
    wgpuTextureViewRelease(view);
    switch (pop_error(instance, device)) {
        case WGPUErrorType_NoError: return "drawn";
        case WGPUErrorType_Validation: return "destroyed";
        default: return "unexpected error";
    }
}

static const char* error_name(WGPUErrorType type) {
    switch (type) {
        case WGPUErrorType_NoError: return "no error";
        case WGPUErrorType_Validation: return "validation error";
        default: return "unexpected error";
    }
}

// Configures the surface and says what error that was for `device`.
static const char* configure(WGPUInstance instance, WGPUDevice device, WGPUSurface surface,
                             const WGPUSurfaceConfiguration* configuration) {
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    wgpuSurfaceConfigure(surface, configuration);
    return error_name(pop_error(instance, device));
}

// Presents the current frame: "presented", or what went wrong for the device
// it was configured with.
static const char* present(WGPUInstance instance, WGPUDevice device, WGPUSurface surface) {
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPUStatus status = wgpuSurfacePresent(surface);
    WGPUErrorType error = pop_error(instance, device);
    if (status != WGPUStatus_Success) return "failed";
    return error == WGPUErrorType_NoError ? "presented" : "presented with a validation error";
}

int main(void) {
    WGPUInstance instance = wt_create_instance();

    WGPUWasixSurfaceSourceCanvas source = WGPU_WASIX_SURFACE_SOURCE_CANVAS_INIT;
    source.selector = SV("#canvas");
    WGPUSurfaceDescriptor surface_descriptor = WGPU_SURFACE_DESCRIPTOR_INIT;
    surface_descriptor.nextInChain = &source.chain;
    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surface_descriptor);
    CHECK(surface != NULL);

    // Two devices, each from an adapter of its own (a browser gives one
    // device per adapter).
    WGPUAdapter first_adapter = wt_request_adapter(instance);
    WGPUAdapter second_adapter = wt_request_adapter(instance);
    WGPUDevice first = wt_request_device(instance, first_adapter);
    WGPUDevice second = wt_request_device(instance, second_adapter);

    WGPUSurfaceCapabilities capabilities = WGPU_SURFACE_CAPABILITIES_INIT;
    CHECK(wgpuSurfaceGetCapabilities(surface, first_adapter, &capabilities) == WGPUStatus_Success);
    CHECK(capabilities.formatCount >= 1 && capabilities.alphaModeCount >= 1);
    WGPUSurfaceConfiguration configuration = WGPU_SURFACE_CONFIGURATION_INIT;
    configuration.device = first;
    configuration.format = capabilities.formats[0];
    configuration.usage = WGPUTextureUsage_RenderAttachment;
    configuration.alphaMode = capabilities.alphaModes[0];
    configuration.presentMode = WGPUPresentMode_Fifo;
    wgpuSurfaceCapabilitiesFreeMembers(capabilities);
    CHECK(wgpuWasixSurfaceGetSize(surface, &configuration.width, &configuration.height) == WGPUStatus_Success);

    // Unconfigure with a frame out.
    wgpuSurfaceConfigure(surface, &configuration);
    WGPUTexture held = acquire(surface);
    wgpuSurfaceUnconfigure(surface);
    printf("a frame held past Unconfigure: %s\n", clear(instance, first, held, (WGPUColor){ 1, 0, 0, 1 }));
    wgpuTextureRelease(held);

    // Configure again with a frame out: the old frame goes, the new
    // configuration's frames work.
    wgpuSurfaceConfigure(surface, &configuration);
    held = acquire(surface);
    wgpuSurfaceConfigure(surface, &configuration);
    printf("a frame held past Configure: %s\n", clear(instance, first, held, (WGPUColor){ 1, 0, 0, 1 }));
    wgpuTextureRelease(held);

    // A Configure that is refused still unconfigures first: the frame out is
    // destroyed, and the surface has no configuration to hand frames out for.
    held = acquire(surface);
    WGPUSurfaceConfiguration empty = configuration;
    empty.width = 0;
    printf("a frame held past a Configure of size 0: %s, ", configure(instance, first, surface, &empty));
    printf("%s\n", clear(instance, first, held, (WGPUColor){ 1, 0, 0, 1 }));
    wgpuTextureRelease(held);
    wgpuSurfaceConfigure(surface, &configuration);
    WGPUSurfaceConfiguration formatless = configuration;
    formatless.format = WGPUTextureFormat_Undefined;
    printf("a Configure without a format: %s, ", configure(instance, first, surface, &formatless));
    printf("%s\n", has_frame(surface) ? "still configured" : "unconfigured");

    wgpuSurfaceConfigure(surface, &configuration);
    WGPUTexture frame = acquire(surface);
    printf("red from the first device: %s, ", clear(instance, first, frame, (WGPUColor){ 1, 0, 0, 1 }));
    printf("%s\n", present(instance, first, surface));
    wgpuTextureRelease(frame);

    // The same surface, at the same size, for the second device.
    configuration.device = second;
    wgpuSurfaceConfigure(surface, &configuration);
    frame = acquire(surface);
    printf("green from the second device: %s, ", clear(instance, second, frame, (WGPUColor){ 0, 1, 0, 1 }));
    printf("%s\n", present(instance, second, surface));
    wgpuTextureRelease(frame);

    wgpuSurfaceUnconfigure(surface);
    wgpuSurfaceRelease(surface);
    wgpuDeviceRelease(second);
    wgpuDeviceRelease(first);
    wgpuAdapterRelease(second_adapter);
    wgpuAdapterRelease(first_adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
