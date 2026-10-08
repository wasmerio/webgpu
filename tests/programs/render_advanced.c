// The rest of the render path: depth testing, instancing, explicit bind group
// and pipeline layouts with dynamic uniform offsets, indirect draws, render
// bundles, occlusion queries, multisampled rendering with a resolve target,
// texture sampling, and debug groups.

#include "wgpu_test.h"

enum { kSize = 64 };

static const char* kQuadShader =
    "struct Uniforms { color: vec4f, offset: vec4f }\n"
    "@group(0) @binding(0) var<uniform> u: Uniforms;\n"
    "struct VertexOut { @builtin(position) position: vec4f, @location(0) color: vec4f }\n"
    "@vertex fn vs(@location(0) corner: vec2f, @location(1) centre: vec2f) -> VertexOut {\n"
    "    var out: VertexOut;\n"
    "    out.position = vec4f(corner * 0.25 + centre + u.offset.xy, u.offset.z, 1.0);\n"
    "    out.color = u.color;\n"
    "    return out;\n"
    "}\n"
    "@fragment fn fs(in: VertexOut) -> @location(0) vec4f { return in.color; }\n";

static const char* kTexturedShader =
    "@group(0) @binding(0) var s: sampler;\n"
    "@group(0) @binding(1) var t: texture_2d<f32>;\n"
    "struct VertexOut { @builtin(position) position: vec4f, @location(0) uv: vec2f }\n"
    "@vertex fn vs(@builtin(vertex_index) index: u32) -> VertexOut {\n"
    "    var corners = array<vec2f, 6>(vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1),\n"
    "                                  vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1));\n"
    "    let p = corners[index];\n"
    "    var out: VertexOut;\n"
    "    out.position = vec4f(p, 0.0, 1.0);\n"
    "    out.uv = vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);\n"
    "    return out;\n"
    "}\n"
    "@fragment fn fs(in: VertexOut) -> @location(0) vec4f { return textureSample(t, s, in.uv); }\n";

// One 256-byte-aligned uniform slot per draw: a colour and an offset (xy
// moves the quads, z is their depth).
typedef struct Slot {
    float color[4];
    float offset[4];
    float padding[56];
} Slot;

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    // ---- depth, instancing, layouts, indirect draws, bundles, occlusion ----

    WGPUTexture target = wt_target(device, WGPUTextureFormat_RGBA8Unorm, kSize, kSize, 1);
    WGPUTextureView target_view = wgpuTextureCreateView(target, NULL);
    WGPUTextureDescriptor depth_descriptor = WGPU_TEXTURE_DESCRIPTOR_INIT;
    depth_descriptor.usage = WGPUTextureUsage_RenderAttachment;
    depth_descriptor.size = (WGPUExtent3D){ kSize, kSize, 1 };
    depth_descriptor.format = WGPUTextureFormat_Depth24Plus;
    WGPUTexture depth = wgpuDeviceCreateTexture(device, &depth_descriptor);
    WGPUTextureViewDescriptor depth_view_descriptor = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    depth_view_descriptor.label = SV("depth view");
    depth_view_descriptor.format = WGPUTextureFormat_Depth24Plus;
    depth_view_descriptor.dimension = WGPUTextureViewDimension_2D;
    depth_view_descriptor.aspect = WGPUTextureAspect_DepthOnly;
    depth_view_descriptor.baseMipLevel = 0;
    depth_view_descriptor.mipLevelCount = 1;
    depth_view_descriptor.baseArrayLayer = 0;
    depth_view_descriptor.arrayLayerCount = 1;
    WGPUTextureView depth_view = wgpuTextureCreateView(depth, &depth_view_descriptor);
    CHECK(depth_view != NULL);

    static const float kCorners[] = { -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1 };
    static const float kCentres[] = { -0.5f, 0.5f, 0.5f, 0.5f };  // two instances, top row
    static const uint32_t kIndices[] = { 0, 1, 2, 3, 4, 5 };
    WGPUBuffer corners = wt_buffer_with(device, WGPUBufferUsage_Vertex, kCorners, sizeof kCorners);
    WGPUBuffer centres = wt_buffer_with(device, WGPUBufferUsage_Vertex, kCentres, sizeof kCentres);
    WGPUBuffer indices = wt_buffer_with(device, WGPUBufferUsage_Index, kIndices, sizeof kIndices);

    static const Slot kSlots[4] = {
        { { 1, 0, 0, 1 }, { 0, 0, 0.5f, 0 }, { 0 } },      // red, top row
        { { 0, 1, 0, 1 }, { 0, -1, 0.25f, 0 }, { 0 } },    // green, bottom row
        { { 0, 0, 1, 1 }, { 0, 0, 0.75f, 0 }, { 0 } },     // blue, top row, behind red
        { { 1, 1, 1, 1 }, { 0, 0, 0.25f, 0 }, { 0 } },     // white, top row, in front of red
    };
    WGPUBufferDescriptor uniform_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    uniform_descriptor.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    uniform_descriptor.size = sizeof kSlots;
    WGPUBuffer uniforms = wgpuDeviceCreateBuffer(device, &uniform_descriptor);
    wgpuQueueWriteBuffer(queue, uniforms, 0, kSlots, sizeof kSlots);

    static const uint32_t kDrawTwo[] = { 6, 2, 0, 0 };            // vertices, instances, first vertex, first instance
    static const uint32_t kDrawOneIndexed[] = { 6, 1, 0, 0, 0 };  // indices, instances, first index, base vertex, first instance
    WGPUBuffer indirect = wt_buffer_with(device, WGPUBufferUsage_Indirect, kDrawTwo, sizeof kDrawTwo);
    WGPUBuffer indirect_indexed =
        wt_buffer_with(device, WGPUBufferUsage_Indirect, kDrawOneIndexed, sizeof kDrawOneIndexed);

    WGPUBindGroupLayoutEntry layout_entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    layout_entry.binding = 0;
    layout_entry.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    layout_entry.buffer.type = WGPUBufferBindingType_Uniform;
    layout_entry.buffer.hasDynamicOffset = true;
    layout_entry.buffer.minBindingSize = 32;
    WGPUBindGroupLayoutDescriptor layout_descriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layout_descriptor.label = SV("uniform layout");
    layout_descriptor.entryCount = 1;
    layout_descriptor.entries = &layout_entry;
    WGPUBindGroupLayout bind_layout = wgpuDeviceCreateBindGroupLayout(device, &layout_descriptor);
    CHECK(bind_layout != NULL);
    WGPUPipelineLayoutDescriptor pipeline_layout_descriptor = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_descriptor.bindGroupLayoutCount = 1;
    pipeline_layout_descriptor.bindGroupLayouts = &bind_layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_descriptor);
    CHECK(pipeline_layout != NULL);

    WGPUBindGroupEntry bind_entry = WGPU_BIND_GROUP_ENTRY_INIT;
    bind_entry.binding = 0;
    bind_entry.buffer = uniforms;
    bind_entry.offset = 0;
    bind_entry.size = 32;
    WGPUBindGroupDescriptor bind_descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bind_descriptor.layout = bind_layout;
    bind_descriptor.entryCount = 1;
    bind_descriptor.entries = &bind_entry;
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bind_descriptor);
    CHECK(bind_group != NULL);

    WGPUShaderModule quad_module = wt_shader(device, kQuadShader);
    WGPUVertexAttribute corner_attribute = WGPU_VERTEX_ATTRIBUTE_INIT;
    corner_attribute.format = WGPUVertexFormat_Float32x2;
    corner_attribute.shaderLocation = 0;
    WGPUVertexAttribute centre_attribute = WGPU_VERTEX_ATTRIBUTE_INIT;
    centre_attribute.format = WGPUVertexFormat_Float32x2;
    centre_attribute.shaderLocation = 1;
    WGPUVertexBufferLayout vertex_layouts[2] = { WGPU_VERTEX_BUFFER_LAYOUT_INIT, WGPU_VERTEX_BUFFER_LAYOUT_INIT };
    vertex_layouts[0].stepMode = WGPUVertexStepMode_Vertex;
    vertex_layouts[0].arrayStride = 2 * sizeof(float);
    vertex_layouts[0].attributeCount = 1;
    vertex_layouts[0].attributes = &corner_attribute;
    vertex_layouts[1].stepMode = WGPUVertexStepMode_Instance;
    vertex_layouts[1].arrayStride = 2 * sizeof(float);
    vertex_layouts[1].attributeCount = 1;
    vertex_layouts[1].attributes = &centre_attribute;

    WGPUColorTargetState color_target = WGPU_COLOR_TARGET_STATE_INIT;
    color_target.format = WGPUTextureFormat_RGBA8Unorm;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = quad_module;
    fragment.entryPoint = SV("fs");
    fragment.targetCount = 1;
    fragment.targets = &color_target;
    WGPUDepthStencilState depth_state = WGPU_DEPTH_STENCIL_STATE_INIT;
    depth_state.format = WGPUTextureFormat_Depth24Plus;
    depth_state.depthWriteEnabled = WGPUOptionalBool_True;
    depth_state.depthCompare = WGPUCompareFunction_Less;
    WGPURenderPipelineDescriptor pipeline_descriptor = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pipeline_descriptor.layout = pipeline_layout;
    pipeline_descriptor.vertex.module = quad_module;
    pipeline_descriptor.vertex.entryPoint = SV("vs");
    pipeline_descriptor.vertex.bufferCount = 2;
    pipeline_descriptor.vertex.buffers = vertex_layouts;
    pipeline_descriptor.primitive.cullMode = WGPUCullMode_None;
    pipeline_descriptor.primitive.frontFace = WGPUFrontFace_CCW;
    pipeline_descriptor.depthStencil = &depth_state;
    pipeline_descriptor.fragment = &fragment;
    WGPURenderPipeline quad_pipeline = wgpuDeviceCreateRenderPipeline(device, &pipeline_descriptor);
    CHECK(quad_pipeline != NULL);

    // The green row is recorded once into a bundle. The extra draws repeat it
    // through every other kind of draw call; equal depth, so they change nothing.
    WGPUTextureFormat bundle_format = WGPUTextureFormat_RGBA8Unorm;
    WGPURenderBundleEncoderDescriptor bundle_descriptor = WGPU_RENDER_BUNDLE_ENCODER_DESCRIPTOR_INIT;
    bundle_descriptor.colorFormatCount = 1;
    bundle_descriptor.colorFormats = &bundle_format;
    bundle_descriptor.depthStencilFormat = WGPUTextureFormat_Depth24Plus;
    bundle_descriptor.sampleCount = 1;
    WGPURenderBundleEncoder bundle_encoder = wgpuDeviceCreateRenderBundleEncoder(device, &bundle_descriptor);
    CHECK(bundle_encoder != NULL);
    uint32_t green_slot = 1 * sizeof(Slot);
    wgpuRenderBundleEncoderPushDebugGroup(bundle_encoder, SV("green row"));
    wgpuRenderBundleEncoderSetPipeline(bundle_encoder, quad_pipeline);
    wgpuRenderBundleEncoderSetBindGroup(bundle_encoder, 0, bind_group, 1, &green_slot);
    wgpuRenderBundleEncoderSetVertexBuffer(bundle_encoder, 0, corners, 0, WGPU_WHOLE_SIZE);
    wgpuRenderBundleEncoderSetVertexBuffer(bundle_encoder, 1, centres, 0, sizeof kCentres);
    wgpuRenderBundleEncoderSetIndexBuffer(bundle_encoder, indices, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);
    wgpuRenderBundleEncoderDraw(bundle_encoder, 6, 2, 0, 0);
    wgpuRenderBundleEncoderInsertDebugMarker(bundle_encoder, SV("repeats"));
    wgpuRenderBundleEncoderDrawIndexed(bundle_encoder, 6, 2, 0, 0, 0);
    wgpuRenderBundleEncoderDrawIndirect(bundle_encoder, indirect, 0);
    wgpuRenderBundleEncoderDrawIndexedIndirect(bundle_encoder, indirect_indexed, 0);
    wgpuRenderBundleEncoderPopDebugGroup(bundle_encoder);
    WGPURenderBundleDescriptor finished_descriptor = WGPU_RENDER_BUNDLE_DESCRIPTOR_INIT;
    finished_descriptor.label = SV("green row");
    WGPURenderBundle bundle = wgpuRenderBundleEncoderFinish(bundle_encoder, &finished_descriptor);
    CHECK(bundle != NULL);
    wgpuRenderBundleEncoderRelease(bundle_encoder);

    WGPUQuerySetDescriptor query_descriptor = WGPU_QUERY_SET_DESCRIPTOR_INIT;
    query_descriptor.type = WGPUQueryType_Occlusion;
    query_descriptor.count = 2;
    WGPUQuerySet queries = wgpuDeviceCreateQuerySet(device, &query_descriptor);
    CHECK(queries != NULL);
    CHECK(wgpuQuerySetGetType(queries) == WGPUQueryType_Occlusion && wgpuQuerySetGetCount(queries) == 2);
    WGPUBufferDescriptor resolve_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    resolve_descriptor.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
    resolve_descriptor.size = 16;
    WGPUBuffer resolved = wgpuDeviceCreateBuffer(device, &resolve_descriptor);
    WGPUBufferDescriptor readback_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    readback_descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    readback_descriptor.size = 16;
    WGPUBuffer readback = wgpuDeviceCreateBuffer(device, &readback_descriptor);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    wgpuCommandEncoderPushDebugGroup(encoder, SV("depth pass"));
    wgpuCommandEncoderInsertDebugMarker(encoder, SV("begin"));
    WGPURenderPassColorAttachment color_attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    color_attachment.view = target_view;
    color_attachment.loadOp = WGPULoadOp_Clear;
    color_attachment.storeOp = WGPUStoreOp_Store;
    color_attachment.clearValue = (WGPUColor){ 0, 0, 0, 1 };
    WGPURenderPassDepthStencilAttachment depth_attachment = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
    depth_attachment.view = depth_view;
    depth_attachment.depthLoadOp = WGPULoadOp_Clear;
    depth_attachment.depthStoreOp = WGPUStoreOp_Store;
    depth_attachment.depthClearValue = 1.0f;
    WGPURenderPassDescriptor pass_descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    pass_descriptor.label = SV("depth pass");
    pass_descriptor.colorAttachmentCount = 1;
    pass_descriptor.colorAttachments = &color_attachment;
    pass_descriptor.depthStencilAttachment = &depth_attachment;
    pass_descriptor.occlusionQuerySet = queries;
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
    CHECK(pass != NULL);
    wgpuRenderPassEncoderPushDebugGroup(pass, SV("quads"));

    // Red: both instances, through an indirect draw.
    uint32_t slot = 0;
    wgpuRenderPassEncoderSetPipeline(pass, quad_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 1, &slot);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, corners, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 1, centres, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDrawIndirect(pass, indirect, 0);

    // Green: the bundle. Executing bundles resets the pass state.
    wgpuRenderPassEncoderExecuteBundles(pass, 1, &bundle);
    wgpuRenderPassEncoderInsertDebugMarker(pass, SV("after bundle"));
    wgpuRenderPassEncoderSetPipeline(pass, quad_pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, corners, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 1, centres, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderSetIndexBuffer(pass, indices, WGPUIndexFormat_Uint32, 0, WGPU_WHOLE_SIZE);

    // Blue, behind red: fails the depth test everywhere.
    slot = 2 * sizeof(Slot);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 1, &slot);
    wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
    wgpuRenderPassEncoderDraw(pass, 6, 2, 0, 0);
    wgpuRenderPassEncoderEndOcclusionQuery(pass);

    // White, in front of red, first instance only, indexed and indirect.
    slot = 3 * sizeof(Slot);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 1, &slot);
    wgpuRenderPassEncoderBeginOcclusionQuery(pass, 1);
    wgpuRenderPassEncoderDrawIndexedIndirect(pass, indirect_indexed, 0);
    wgpuRenderPassEncoderEndOcclusionQuery(pass);

    wgpuRenderPassEncoderPopDebugGroup(pass);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    wgpuCommandEncoderResolveQuerySet(encoder, queries, 0, 2, resolved, 0);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, resolved, 0, readback, 0, 16);
    wgpuCommandEncoderPopDebugGroup(encoder);
    wt_submit(queue, encoder);

    uint8_t* pixels = wt_read_texture(instance, device, queue, target, kSize, kSize);
    wt_print_pixel("top left (white over red)", pixels, kSize, 16, 16);
    wt_print_pixel("top right (red, blue hidden)", pixels, kSize, 48, 16);
    wt_print_pixel("bottom left (green, bundle)", pixels, kSize, 16, 48);
    wt_print_pixel("bottom right (green, bundle)", pixels, kSize, 48, 48);
    wt_print_pixel("centre (clear)", pixels, kSize, 32, 32);
    free(pixels);

    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 0, 16) == WGPUMapAsyncStatus_Success);
    uint64_t samples[2];
    CHECK(wgpuBufferReadMappedRange(readback, 0, samples, sizeof samples) == WGPUStatus_Success);
    wgpuBufferUnmap(readback);
    printf("occlusion: hidden draw %s, visible draw %s\n", samples[0] == 0 ? "passed no samples" : "PASSED SAMPLES",
           samples[1] != 0 ? "passed samples" : "PASSED NONE");

    // ---- multisampling with a resolve target, sampling a texture ----

    static const uint8_t kTexels[16] = {
        255, 0, 0, 255,   0, 255, 0, 255,      // red, green
        0, 0, 255, 255,   255, 255, 255, 255,  // blue, white
    };
    WGPUTextureDescriptor image_descriptor = WGPU_TEXTURE_DESCRIPTOR_INIT;
    image_descriptor.label = SV("2x2 image");
    image_descriptor.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    image_descriptor.dimension = WGPUTextureDimension_2D;
    image_descriptor.size = (WGPUExtent3D){ 2, 2, 1 };
    image_descriptor.format = WGPUTextureFormat_RGBA8Unorm;
    WGPUTexture image = wgpuDeviceCreateTexture(device, &image_descriptor);
    WGPUTexelCopyTextureInfo image_destination = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    image_destination.texture = image;
    WGPUTexelCopyBufferLayout image_layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    image_layout.bytesPerRow = 8;
    image_layout.rowsPerImage = 2;
    wgpuQueueWriteTexture(queue, &image_destination, kTexels, sizeof kTexels, &image_layout,
                          &(WGPUExtent3D){ 2, 2, 1 });
    WGPUTextureView image_view = wgpuTextureCreateView(image, NULL);
    WGPUSampler sampler = wgpuDeviceCreateSampler(device, NULL);
    CHECK(sampler != NULL);

    WGPUTexture multisampled = wt_target(device, WGPUTextureFormat_RGBA8Unorm, kSize, kSize, 4);
    CHECK(wgpuTextureGetSampleCount(multisampled) == 4);
    WGPUTextureView multisampled_view = wgpuTextureCreateView(multisampled, NULL);
    WGPUTexture resolve_target = wt_target(device, WGPUTextureFormat_RGBA8Unorm, kSize, kSize, 1);
    WGPUTextureView resolve_view = wgpuTextureCreateView(resolve_target, NULL);

    WGPUShaderModule textured_module = wt_shader(device, kTexturedShader);
    WGPUFragmentState textured_fragment = WGPU_FRAGMENT_STATE_INIT;
    textured_fragment.module = textured_module;
    textured_fragment.entryPoint = SV("fs");
    textured_fragment.targetCount = 1;
    textured_fragment.targets = &color_target;
    WGPURenderPipelineDescriptor textured_descriptor = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    textured_descriptor.vertex.module = textured_module;
    textured_descriptor.vertex.entryPoint = SV("vs");
    textured_descriptor.multisample.count = 4;
    textured_descriptor.fragment = &textured_fragment;
    WGPURenderPipeline textured_pipeline = wgpuDeviceCreateRenderPipeline(device, &textured_descriptor);
    CHECK(textured_pipeline != NULL);

    // The layout was derived from the shader; ask the pipeline for it.
    WGPUBindGroupLayout derived_layout = wgpuRenderPipelineGetBindGroupLayout(textured_pipeline, 0);
    WGPUBindGroupEntry textured_entries[2] = { WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT };
    textured_entries[0].binding = 0;
    textured_entries[0].sampler = sampler;
    textured_entries[1].binding = 1;
    textured_entries[1].textureView = image_view;
    WGPUBindGroupDescriptor textured_bind_descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    textured_bind_descriptor.layout = derived_layout;
    textured_bind_descriptor.entryCount = 2;
    textured_bind_descriptor.entries = textured_entries;
    WGPUBindGroup textured_group = wgpuDeviceCreateBindGroup(device, &textured_bind_descriptor);
    CHECK(textured_group != NULL);

    encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPURenderPassColorAttachment resolving = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    resolving.view = multisampled_view;
    resolving.resolveTarget = resolve_view;
    resolving.loadOp = WGPULoadOp_Clear;
    resolving.storeOp = WGPUStoreOp_Discard;
    resolving.clearValue = (WGPUColor){ 0, 0, 0, 1 };
    WGPURenderPassDescriptor resolving_pass = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    resolving_pass.colorAttachmentCount = 1;
    resolving_pass.colorAttachments = &resolving;
    pass = wgpuCommandEncoderBeginRenderPass(encoder, &resolving_pass);
    wgpuRenderPassEncoderSetPipeline(pass, textured_pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, textured_group, 0, NULL);
    wgpuRenderPassEncoderDraw(pass, 6, 1, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);
    wt_submit(queue, encoder);

    pixels = wt_read_texture(instance, device, queue, resolve_target, kSize, kSize);
    wt_print_pixel("sampled top left", pixels, kSize, 16, 16);
    wt_print_pixel("sampled top right", pixels, kSize, 48, 16);
    wt_print_pixel("sampled bottom left", pixels, kSize, 16, 48);
    wt_print_pixel("sampled bottom right", pixels, kSize, 48, 48);
    free(pixels);

    wgpuQuerySetDestroy(queries);
    wgpuQuerySetRelease(queries);
    wgpuRenderBundleRelease(bundle);
    wgpuBindGroupRelease(textured_group);
    wgpuBindGroupLayoutRelease(derived_layout);
    wgpuRenderPipelineRelease(textured_pipeline);
    wgpuShaderModuleRelease(textured_module);
    wgpuTextureViewRelease(resolve_view);
    wgpuTextureRelease(resolve_target);
    wgpuTextureViewRelease(multisampled_view);
    wgpuTextureRelease(multisampled);
    wgpuSamplerRelease(sampler);
    wgpuTextureViewRelease(image_view);
    wgpuTextureRelease(image);
    wgpuBufferRelease(readback);
    wgpuBufferRelease(resolved);
    wgpuRenderPipelineRelease(quad_pipeline);
    wgpuShaderModuleRelease(quad_module);
    wgpuBindGroupRelease(bind_group);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuBindGroupLayoutRelease(bind_layout);
    wgpuBufferRelease(indirect_indexed);
    wgpuBufferRelease(indirect);
    wgpuBufferRelease(uniforms);
    wgpuBufferRelease(indices);
    wgpuBufferRelease(centres);
    wgpuBufferRelease(corners);
    wgpuTextureViewRelease(depth_view);
    wgpuTextureRelease(depth);
    wgpuTextureViewRelease(target_view);
    wgpuTextureRelease(target);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
