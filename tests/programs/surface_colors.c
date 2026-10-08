// A surface configured for a wide colour space and extended tone mapping
// (WGPUSurfaceColorManagement), presenting one frame. The embedder tests run
// it against targets that can show such frames, and check what they were
// told. (Not in the manifest: a target that cannot, a plain native window for
// one, refuses the configuration, which is the right answer there.)

#include "wgpu_test.h"

int main(void) {
    WGPUInstance instance = wt_create_instance();

    WGPUWasixSurfaceSourceCanvas source = WGPU_WASIX_SURFACE_SOURCE_CANVAS_INIT;
    source.selector = SV("#canvas");
    WGPUSurfaceDescriptor surface_descriptor = WGPU_SURFACE_DESCRIPTOR_INIT;
    surface_descriptor.nextInChain = &source.chain;
    WGPUSurface surface = wgpuInstanceCreateSurface(instance, &surface_descriptor);
    CHECK(surface != NULL);

    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    WGPUSurfaceCapabilities capabilities = WGPU_SURFACE_CAPABILITIES_INIT;
    CHECK(wgpuSurfaceGetCapabilities(surface, adapter, &capabilities) == WGPUStatus_Success);
    CHECK(capabilities.formatCount >= 1);
    WGPUTextureFormat format = capabilities.formats[0];
    wgpuSurfaceCapabilitiesFreeMembers(capabilities);
    uint32_t width = 0, height = 0;
    CHECK(wgpuWasixSurfaceGetSize(surface, &width, &height) == WGPUStatus_Success);

    WGPUSurfaceColorManagement colors = WGPU_SURFACE_COLOR_MANAGEMENT_INIT;
    colors.colorSpace = WGPUPredefinedColorSpace_DisplayP3;
    colors.toneMappingMode = WGPUToneMappingMode_Extended;
    WGPUSurfaceConfiguration configuration = WGPU_SURFACE_CONFIGURATION_INIT;
    configuration.nextInChain = &colors.chain;
    configuration.device = device;
    configuration.format = format;
    configuration.width = width;
    configuration.height = height;
    wgpuSurfaceConfigure(surface, &configuration);

    WGPUSurfaceTexture frame = WGPU_SURFACE_TEXTURE_INIT;
    wgpuSurfaceGetCurrentTexture(surface, &frame);
    CHECK(frame.texture != NULL);
    WGPUTextureView view = wgpuTextureCreateView(frame.texture, NULL);
    WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    attachment.view = view;
    attachment.loadOp = WGPULoadOp_Clear;
    attachment.storeOp = WGPUStoreOp_Store;
    attachment.clearValue = (WGPUColor){ 1, 0, 0, 1 };
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
    wgpuTextureRelease(frame.texture);
    printf("presented in display-p3\n");

    wgpuSurfaceRelease(surface);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
