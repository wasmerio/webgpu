// Rendering to an offscreen texture: a render pipeline with vertex and index
// buffers, two passes (clear, then load), viewport and scissor state, and the
// pixels read back to prove what was drawn.

#include "wgpu_test.h"

static const char* kShader =
    "struct VertexOut {\n"
    "    @builtin(position) position: vec4f,\n"
    "    @location(0) color: vec4f,\n"
    "}\n"
    "@vertex fn vs(@location(0) position: vec2f, @location(1) color: vec3f) -> VertexOut {\n"
    "    var out: VertexOut;\n"
    "    out.position = vec4f(position, 0.0, 1.0);\n"
    "    out.color = vec4f(color, 1.0);\n"
    "    return out;\n"
    "}\n"
    "@fragment fn fs(in: VertexOut) -> @location(0) vec4f {\n"
    "    return in.color;\n"
    "}\n";

enum { kSize = 64 };

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    WGPUTexture target = wt_target(device, WGPUTextureFormat_RGBA8Unorm, kSize, kSize, 1);
    CHECK(wgpuTextureGetWidth(target) == kSize && wgpuTextureGetHeight(target) == kSize);
    CHECK(wgpuTextureGetDepthOrArrayLayers(target) == 1 && wgpuTextureGetMipLevelCount(target) == 1);
    CHECK(wgpuTextureGetSampleCount(target) == 1);
    CHECK(wgpuTextureGetDimension(target) == WGPUTextureDimension_2D);
    CHECK(wgpuTextureGetFormat(target) == WGPUTextureFormat_RGBA8Unorm);
    CHECK(wgpuTextureGetUsage(target) ==
          (WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding));
    WGPUTextureView view = wgpuTextureCreateView(target, NULL);
    CHECK(view != NULL);

    // x, y, r, g, b: a red triangle around the centre, then a green quad
    // covering everything (the scissor will cut it down).
    static const float kVertices[] = {
        -0.8f, -0.8f, 1, 0, 0,   0.8f, -0.8f, 1, 0, 0,   0.0f, 0.8f, 1, 0, 0,
        -1, -1, 0, 1, 0,   1, -1, 0, 1, 0,   1, 1, 0, 1, 0,   -1, 1, 0, 1, 0,
    };
    static const uint16_t kIndices[] = { 0, 1, 2, 0, 2, 3 };
    WGPUBuffer vertices = wt_buffer_with(device, WGPUBufferUsage_Vertex, kVertices, sizeof kVertices);
    WGPUBuffer indices = wt_buffer_with(device, WGPUBufferUsage_Index, kIndices, sizeof kIndices);

    WGPUShaderModule module = wt_shader(device, kShader);
    WGPUVertexAttribute attributes[2] = { WGPU_VERTEX_ATTRIBUTE_INIT, WGPU_VERTEX_ATTRIBUTE_INIT };
    attributes[0].format = WGPUVertexFormat_Float32x2;
    attributes[0].offset = 0;
    attributes[0].shaderLocation = 0;
    attributes[1].format = WGPUVertexFormat_Float32x3;
    attributes[1].offset = 2 * sizeof(float);
    attributes[1].shaderLocation = 1;
    WGPUVertexBufferLayout layout = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
    layout.stepMode = WGPUVertexStepMode_Vertex;
    layout.arrayStride = 5 * sizeof(float);
    layout.attributeCount = 2;
    layout.attributes = attributes;

    WGPUColorTargetState color_target = WGPU_COLOR_TARGET_STATE_INIT;
    color_target.format = WGPUTextureFormat_RGBA8Unorm;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = SV("fs");
    fragment.targetCount = 1;
    fragment.targets = &color_target;

    WGPURenderPipelineDescriptor pipeline_descriptor = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pipeline_descriptor.label = SV("triangle");
    pipeline_descriptor.vertex.module = module;
    pipeline_descriptor.vertex.entryPoint = SV("vs");
    pipeline_descriptor.vertex.bufferCount = 1;
    pipeline_descriptor.vertex.buffers = &layout;
    pipeline_descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipeline_descriptor.fragment = &fragment;
    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device, &pipeline_descriptor);
    CHECK(pipeline != NULL);

    // Pass 1: clear to blue, draw the triangle.
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    attachment.view = view;
    attachment.loadOp = WGPULoadOp_Clear;
    attachment.storeOp = WGPUStoreOp_Store;
    attachment.clearValue = (WGPUColor){ 0, 0, 1, 1 };
    WGPURenderPassDescriptor pass_descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    pass_descriptor.colorAttachmentCount = 1;
    pass_descriptor.colorAttachments = &attachment;
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
    CHECK(pass != NULL);
    wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    wgpuRenderPassEncoderSetViewport(pass, 0, 0, kSize, kSize, 0, 1);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertices, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    // Pass 2: keep what is there, draw the quad through a small scissor.
    attachment.loadOp = WGPULoadOp_Load;
    pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    wgpuRenderPassEncoderSetScissorRect(pass, 48, 48, 8, 8);
    wgpuRenderPassEncoderSetBlendConstant(pass, &(WGPUColor){ 1, 1, 1, 1 });
    wgpuRenderPassEncoderSetStencilReference(pass, 0);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertices, 3 * 5 * sizeof(float), 4 * 5 * sizeof(float));
    wgpuRenderPassEncoderSetIndexBuffer(pass, indices, WGPUIndexFormat_Uint16, 0, sizeof kIndices);
    wgpuRenderPassEncoderDrawIndexed(pass, 6, 1, 0, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    wt_submit(queue, encoder);

    uint8_t* pixels = wt_read_texture(instance, device, queue, target, kSize, kSize);
    wt_print_pixel("inside the triangle", pixels, kSize, 32, 32);
    wt_print_pixel("outside the triangle", pixels, kSize, 2, 2);
    wt_print_pixel("inside the scissor", pixels, kSize, 50, 50);
    wt_print_pixel("outside the scissor", pixels, kSize, 60, 60);
    free(pixels);

    wgpuTextureDestroy(target);
    wgpuRenderPipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
    wgpuBufferRelease(indices);
    wgpuBufferRelease(vertices);
    wgpuTextureViewRelease(view);
    wgpuTextureRelease(target);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
