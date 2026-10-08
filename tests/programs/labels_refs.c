// Every object type: debug labels, and reference counting (an object stays
// usable until its last reference is released).

#include "wgpu_test.h"

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    WGPUBufferDescriptor buffer_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    buffer_descriptor.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    buffer_descriptor.size = 64;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &buffer_descriptor);

    WGPUTextureDescriptor texture_descriptor = WGPU_TEXTURE_DESCRIPTOR_INIT;
    texture_descriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding;
    texture_descriptor.size = (WGPUExtent3D){ 4, 4, 1 };
    texture_descriptor.format = WGPUTextureFormat_RGBA8Unorm;
    WGPUTexture texture = wgpuDeviceCreateTexture(device, &texture_descriptor);
    WGPUTextureView view = wgpuTextureCreateView(texture, NULL);

    WGPUSamplerDescriptor sampler_descriptor = WGPU_SAMPLER_DESCRIPTOR_INIT;
    sampler_descriptor.addressModeU = WGPUAddressMode_Repeat;
    sampler_descriptor.addressModeV = WGPUAddressMode_MirrorRepeat;
    sampler_descriptor.addressModeW = WGPUAddressMode_ClampToEdge;
    sampler_descriptor.magFilter = WGPUFilterMode_Linear;
    sampler_descriptor.minFilter = WGPUFilterMode_Linear;
    sampler_descriptor.mipmapFilter = WGPUMipmapFilterMode_Linear;
    sampler_descriptor.lodMinClamp = 0;
    sampler_descriptor.lodMaxClamp = 8;
    sampler_descriptor.maxAnisotropy = 4;
    WGPUSampler sampler = wgpuDeviceCreateSampler(device, &sampler_descriptor);
    WGPUSamplerDescriptor comparison_descriptor = WGPU_SAMPLER_DESCRIPTOR_INIT;
    comparison_descriptor.compare = WGPUCompareFunction_LessEqual;
    WGPUSampler comparison = wgpuDeviceCreateSampler(device, &comparison_descriptor);
    CHECK(sampler != NULL && comparison != NULL);

    // One layout entry of each kind.
    WGPUBindGroupLayoutEntry entries[4] = {
        WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
        WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT, WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
    };
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Fragment;
    entries[0].buffer.type = WGPUBufferBindingType_Uniform;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Fragment;
    entries[1].sampler.type = WGPUSamplerBindingType_Filtering;
    entries[2].binding = 2;
    entries[2].visibility = WGPUShaderStage_Fragment;
    entries[2].texture.sampleType = WGPUTextureSampleType_Float;
    entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;
    entries[3].binding = 3;
    entries[3].visibility = WGPUShaderStage_Compute;
    entries[3].storageTexture.access = WGPUStorageTextureAccess_WriteOnly;
    entries[3].storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
    entries[3].storageTexture.viewDimension = WGPUTextureViewDimension_2D;
    WGPUBindGroupLayoutDescriptor layout_descriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layout_descriptor.entryCount = 3;
    layout_descriptor.entries = entries;
    WGPUBindGroupLayout layout = wgpuDeviceCreateBindGroupLayout(device, &layout_descriptor);
    CHECK(layout != NULL);
    WGPUBindGroupLayoutDescriptor storage_layout_descriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    storage_layout_descriptor.entryCount = 1;
    storage_layout_descriptor.entries = &entries[3];
    WGPUBindGroupLayout storage_layout = wgpuDeviceCreateBindGroupLayout(device, &storage_layout_descriptor);
    CHECK(storage_layout != NULL);

    WGPUBindGroupEntry bindings[3] = { WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
                                       WGPU_BIND_GROUP_ENTRY_INIT };
    bindings[0].binding = 0;
    bindings[0].buffer = buffer;
    bindings[0].size = WGPU_WHOLE_SIZE;
    bindings[1].binding = 1;
    bindings[1].sampler = sampler;
    bindings[2].binding = 2;
    bindings[2].textureView = view;
    WGPUBindGroupDescriptor group_descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    group_descriptor.layout = layout;
    group_descriptor.entryCount = 3;
    group_descriptor.entries = bindings;
    WGPUBindGroup group = wgpuDeviceCreateBindGroup(device, &group_descriptor);
    CHECK(group != NULL);

    // A layout may leave a group slot empty.
    WGPUBindGroupLayout slots[2] = { layout, NULL };
    WGPUPipelineLayoutDescriptor pipeline_layout_descriptor = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_descriptor.bindGroupLayoutCount = 2;
    pipeline_layout_descriptor.bindGroupLayouts = slots;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_descriptor);
    CHECK(pipeline_layout != NULL);

    WGPUShaderModule module = wt_shader(
        device,
        "@vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0, 0.0, 0.0, 1.0); }\n"
        "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }\n"
        "@compute @workgroup_size(1) fn cs() {}\n");
    WGPUComputePipelineDescriptor compute_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    compute_descriptor.compute.module = module;
    compute_descriptor.compute.entryPoint = SV("cs");
    WGPUComputePipeline compute_pipeline = wgpuDeviceCreateComputePipeline(device, &compute_descriptor);
    WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
    target.format = WGPUTextureFormat_RGBA8Unorm;
    WGPUBlendState blend = WGPU_BLEND_STATE_INIT;
    blend.color.operation = WGPUBlendOperation_Add;
    blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend.alpha.operation = WGPUBlendOperation_Max;
    blend.alpha.srcFactor = WGPUBlendFactor_One;
    blend.alpha.dstFactor = WGPUBlendFactor_One;
    target.blend = &blend;
    target.writeMask = WGPUColorWriteMask_Red | WGPUColorWriteMask_Green | WGPUColorWriteMask_Blue;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = SV("fs");
    fragment.targetCount = 1;
    fragment.targets = &target;
    WGPURenderPipelineDescriptor render_descriptor = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    render_descriptor.vertex.module = module;
    render_descriptor.vertex.entryPoint = SV("vs");
    render_descriptor.primitive.topology = WGPUPrimitiveTopology_TriangleStrip;
    render_descriptor.primitive.stripIndexFormat = WGPUIndexFormat_Uint16;
    render_descriptor.primitive.frontFace = WGPUFrontFace_CW;
    render_descriptor.primitive.cullMode = WGPUCullMode_Back;
    render_descriptor.fragment = &fragment;
    WGPURenderPipeline render_pipeline = wgpuDeviceCreateRenderPipeline(device, &render_descriptor);
    CHECK(compute_pipeline != NULL && render_pipeline != NULL);

    WGPUQuerySetDescriptor query_descriptor = WGPU_QUERY_SET_DESCRIPTOR_INIT;
    query_descriptor.type = WGPUQueryType_Occlusion;
    query_descriptor.count = 1;
    WGPUQuerySet queries = wgpuDeviceCreateQuerySet(device, &query_descriptor);

    WGPUTextureFormat bundle_format = WGPUTextureFormat_RGBA8Unorm;
    WGPURenderBundleEncoderDescriptor bundle_descriptor = WGPU_RENDER_BUNDLE_ENCODER_DESCRIPTOR_INIT;
    bundle_descriptor.colorFormatCount = 1;
    bundle_descriptor.colorFormats = &bundle_format;
    bundle_descriptor.sampleCount = 1;
    WGPURenderBundleEncoder bundle_encoder = wgpuDeviceCreateRenderBundleEncoder(device, &bundle_descriptor);
    wgpuRenderBundleEncoderSetLabel(bundle_encoder, SV("bundle encoder"));
    wgpuRenderBundleEncoderAddRef(bundle_encoder);
    wgpuRenderBundleEncoderRelease(bundle_encoder);
    WGPURenderBundle bundle = wgpuRenderBundleEncoderFinish(bundle_encoder, NULL);
    CHECK(bundle != NULL);

    WGPUCommandEncoderDescriptor encoder_descriptor = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    encoder_descriptor.label = SV("encoder");
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encoder_descriptor);
    wgpuCommandEncoderSetLabel(encoder, SV("renamed encoder"));
    wgpuCommandEncoderAddRef(encoder);
    wgpuCommandEncoderRelease(encoder);

    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    wgpuComputePassEncoderSetLabel(compute_pass, SV("compute pass"));
    wgpuComputePassEncoderAddRef(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuComputePassEncoderEnd(compute_pass);

    WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    attachment.view = view;
    attachment.loadOp = WGPULoadOp_Clear;
    attachment.storeOp = WGPUStoreOp_Store;
    WGPURenderPassDescriptor pass_descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    pass_descriptor.colorAttachmentCount = 1;
    pass_descriptor.colorAttachments = &attachment;
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
    wgpuRenderPassEncoderSetLabel(render_pass, SV("render pass"));
    wgpuRenderPassEncoderAddRef(render_pass);
    wgpuRenderPassEncoderRelease(render_pass);
    wgpuRenderPassEncoderEnd(render_pass);

    WGPUCommandBufferDescriptor commands_descriptor = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    commands_descriptor.label = SV("commands");
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &commands_descriptor);
    wgpuCommandBufferSetLabel(commands, SV("renamed commands"));
    wgpuCommandBufferAddRef(commands);
    wgpuCommandBufferRelease(commands);

    // Labels.
    wgpuBindGroupSetLabel(group, SV("group"));
    wgpuBindGroupLayoutSetLabel(layout, SV("layout"));
    wgpuBufferSetLabel(buffer, SV("buffer"));
    wgpuComputePipelineSetLabel(compute_pipeline, SV("compute pipeline"));
    wgpuDeviceSetLabel(device, SV("device"));
    wgpuPipelineLayoutSetLabel(pipeline_layout, SV("pipeline layout"));
    wgpuQuerySetSetLabel(queries, SV("queries"));
    wgpuQueueSetLabel(queue, SV("queue"));
    wgpuRenderBundleSetLabel(bundle, SV("bundle"));
    wgpuRenderPipelineSetLabel(render_pipeline, SV("render pipeline"));
    wgpuSamplerSetLabel(sampler, SV("sampler"));
    wgpuShaderModuleSetLabel(module, SV("module"));
    wgpuTextureSetLabel(texture, SV("texture"));
    wgpuTextureViewSetLabel(view, (WGPUStringView){ "view with a length", 4 });
    printf("labels: ok\n");

    // An extra reference keeps an object alive past a release.
    wgpuBindGroupAddRef(group);
    wgpuBindGroupRelease(group);
    wgpuBindGroupLayoutAddRef(layout);
    wgpuBindGroupLayoutRelease(layout);
    wgpuComputePipelineAddRef(compute_pipeline);
    wgpuComputePipelineRelease(compute_pipeline);
    wgpuPipelineLayoutAddRef(pipeline_layout);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuQuerySetAddRef(queries);
    wgpuQuerySetRelease(queries);
    wgpuRenderBundleAddRef(bundle);
    wgpuRenderBundleRelease(bundle);
    wgpuRenderPipelineAddRef(render_pipeline);
    wgpuRenderPipelineRelease(render_pipeline);
    wgpuSamplerAddRef(sampler);
    wgpuSamplerRelease(sampler);
    wgpuShaderModuleAddRef(module);
    wgpuShaderModuleRelease(module);
    wgpuTextureAddRef(texture);
    wgpuTextureRelease(texture);
    wgpuTextureViewAddRef(view);
    wgpuTextureViewRelease(view);
    wgpuBufferAddRef(buffer);
    wgpuBufferRelease(buffer);
    CHECK(wgpuBufferGetSize(buffer) == 64);
    CHECK(wgpuTextureGetWidth(texture) == 4);
    CHECK(wgpuQuerySetGetCount(queries) == 1);
    wgpuQueueSubmit(queue, 1, &commands);
    printf("references: ok\n");

    // webgpu.h declares WGPUExternalTexture but offers no way to make one;
    // its functions must at least tolerate the null handle.
    wgpuExternalTextureSetLabel(NULL, SV("none"));
    wgpuExternalTextureAddRef(NULL);
    wgpuExternalTextureRelease(NULL);
    printf("external texture: nothing to create\n");

    wgpuCommandBufferRelease(commands);
    wgpuRenderPassEncoderRelease(render_pass);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuRenderBundleRelease(bundle);
    wgpuRenderBundleEncoderRelease(bundle_encoder);
    wgpuQuerySetRelease(queries);
    wgpuRenderPipelineRelease(render_pipeline);
    wgpuComputePipelineRelease(compute_pipeline);
    wgpuShaderModuleRelease(module);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuBindGroupRelease(group);
    wgpuBindGroupLayoutRelease(storage_layout);
    wgpuBindGroupLayoutRelease(layout);
    wgpuSamplerRelease(comparison);
    wgpuSamplerRelease(sampler);
    wgpuTextureViewRelease(view);
    wgpuTextureRelease(texture);
    wgpuBufferRelease(buffer);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
