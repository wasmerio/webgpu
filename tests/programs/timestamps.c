// Timestamp queries (an optional feature) and immediate data. Hosts differ in
// what they offer here, so this program checks that each path either works or
// is refused through WebGPU's own error reporting, and prints the same
// transcript either way.

#include "wgpu_test.h"

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

static const char* kShader =
    "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
    "    return vec4f(f32(i) - 1.0, 0.0, 0.0, 1.0);\n"
    "}\n"
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }\n"
    "@compute @workgroup_size(1) fn cs() {}\n";

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);

    bool timestamps = wgpuAdapterHasFeature(adapter, WGPUFeatureName_TimestampQuery);
    WGPUFeatureName feature = WGPUFeatureName_TimestampQuery;
    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    if (timestamps) {
        descriptor.requiredFeatureCount = 1;
        descriptor.requiredFeatures = &feature;
    }
    WGPUDevice device = wt_request_device_with(instance, adapter, &descriptor);
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    WGPUShaderModule module = wt_shader(device, kShader);

    WGPUTexture target = wt_target(device, WGPUTextureFormat_RGBA8Unorm, 4, 4, 1);
    WGPUTextureView view = wgpuTextureCreateView(target, NULL);
    WGPUColorTargetState color_target = WGPU_COLOR_TARGET_STATE_INIT;
    color_target.format = WGPUTextureFormat_RGBA8Unorm;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = SV("fs");
    fragment.targetCount = 1;
    fragment.targets = &color_target;
    WGPURenderPipelineDescriptor render_descriptor = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    render_descriptor.vertex.module = module;
    render_descriptor.vertex.entryPoint = SV("vs");
    render_descriptor.fragment = &fragment;
    WGPURenderPipeline render_pipeline = wgpuDeviceCreateRenderPipeline(device, &render_descriptor);
    WGPUComputePipelineDescriptor compute_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    compute_descriptor.compute.module = module;
    compute_descriptor.compute.entryPoint = SV("cs");
    WGPUComputePipeline compute_pipeline = wgpuDeviceCreateComputePipeline(device, &compute_descriptor);

    // Timestamps written at the start and end of passes.
    if (timestamps) {
        WGPUQuerySetDescriptor query_descriptor = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        query_descriptor.type = WGPUQueryType_Timestamp;
        query_descriptor.count = 4;
        WGPUQuerySet queries = wgpuDeviceCreateQuerySet(device, &query_descriptor);
        CHECK(queries != NULL && wgpuQuerySetGetType(queries) == WGPUQueryType_Timestamp);
        WGPUBufferDescriptor resolve_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        resolve_descriptor.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
        resolve_descriptor.size = 32;
        WGPUBuffer resolved = wgpuDeviceCreateBuffer(device, &resolve_descriptor);

        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
        WGPUPassTimestampWrites compute_writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
        compute_writes.querySet = queries;
        compute_writes.beginningOfPassWriteIndex = 0;
        compute_writes.endOfPassWriteIndex = 1;
        WGPUComputePassDescriptor compute_pass_descriptor = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        compute_pass_descriptor.timestampWrites = &compute_writes;
        WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, &compute_pass_descriptor);
        wgpuComputePassEncoderSetPipeline(compute_pass, compute_pipeline);
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);
        wgpuComputePassEncoderEnd(compute_pass);
        wgpuComputePassEncoderRelease(compute_pass);

        WGPUPassTimestampWrites render_writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
        render_writes.querySet = queries;
        render_writes.beginningOfPassWriteIndex = 2;
        render_writes.endOfPassWriteIndex = 3;
        WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        attachment.view = view;
        attachment.loadOp = WGPULoadOp_Clear;
        attachment.storeOp = WGPUStoreOp_Store;
        WGPURenderPassDescriptor render_pass_descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        render_pass_descriptor.colorAttachmentCount = 1;
        render_pass_descriptor.colorAttachments = &attachment;
        render_pass_descriptor.timestampWrites = &render_writes;
        WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_descriptor);
        wgpuRenderPassEncoderSetPipeline(render_pass, render_pipeline);
        wgpuRenderPassEncoderDraw(render_pass, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(render_pass);
        wgpuRenderPassEncoderRelease(render_pass);

        wgpuCommandEncoderResolveQuerySet(encoder, queries, 0, 4, resolved, 0);
        wt_submit(queue, encoder);
        wgpuBufferRelease(resolved);
        wgpuQuerySetRelease(queries);
    }
    printf("pass timestamps: handled\n");

    // Passes told to write timestamps into a query set that is gone: both
    // kinds refuse, and both say so.
    {
        WGPUQuerySetDescriptor gone_descriptor = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        gone_descriptor.type = WGPUQueryType_Occlusion;
        gone_descriptor.count = 2;
        WGPUQuerySet gone = wgpuDeviceCreateQuerySet(device, &gone_descriptor);
        CHECK(gone != NULL);
        wgpuQuerySetRelease(gone);
        WGPUPassTimestampWrites writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
        writes.querySet = gone;
        writes.beginningOfPassWriteIndex = 0;
        writes.endOfPassWriteIndex = 1;

        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
        WGPUComputePassDescriptor compute_pass_descriptor = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        compute_pass_descriptor.timestampWrites = &writes;
        WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, &compute_pass_descriptor);
        if (compute_pass) {
            wgpuComputePassEncoderEnd(compute_pass);
            wgpuComputePassEncoderRelease(compute_pass);
        }
        CHECK(pop_scope(instance, device) == WGPUErrorType_Validation);

        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
        WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        attachment.view = view;
        attachment.loadOp = WGPULoadOp_Clear;
        attachment.storeOp = WGPUStoreOp_Store;
        WGPURenderPassDescriptor render_pass_descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        render_pass_descriptor.colorAttachmentCount = 1;
        render_pass_descriptor.colorAttachments = &attachment;
        render_pass_descriptor.timestampWrites = &writes;
        WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_descriptor);
        if (render_pass) {
            wgpuRenderPassEncoderEnd(render_pass);
            wgpuRenderPassEncoderRelease(render_pass);
        }
        CHECK(pop_scope(instance, device) == WGPUErrorType_Validation);
        wgpuCommandEncoderRelease(encoder);
    }
    printf("pass timestamps into a released query set: refused\n");

    // Writing a timestamp from the encoder itself is in webgpu.h but no longer
    // in the WebGPU specification. A host whose backend can still do it lets
    // it through, the others refuse it as a validation error, and into a
    // query set that does not hold timestamps it is refused everywhere.
    {
        WGPUQuerySetDescriptor query_descriptor = WGPU_QUERY_SET_DESCRIPTOR_INIT;
        query_descriptor.type = timestamps ? WGPUQueryType_Timestamp : WGPUQueryType_Occlusion;
        query_descriptor.count = 1;
        WGPUQuerySet queries = wgpuDeviceCreateQuerySet(device, &query_descriptor);
        CHECK(queries != NULL);

        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
        wgpuCommandEncoderWriteTimestamp(encoder, queries, 0);
        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
        wgpuCommandEncoderRelease(encoder);
        WGPUErrorType error = pop_scope(instance, device);
        CHECK(error == WGPUErrorType_NoError || error == WGPUErrorType_Validation);
        if (!timestamps) CHECK(error == WGPUErrorType_Validation);
        if (error == WGPUErrorType_NoError) {
            // What was accepted has to be something the queue can run.
            CHECK(commands != NULL);
            wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
            wgpuQueueSubmit(queue, 1, &commands);
            CHECK(pop_scope(instance, device) == WGPUErrorType_NoError);
        }
        if (commands) wgpuCommandBufferRelease(commands);
        wgpuQuerySetRelease(queries);
        // Which of the two it was depends on the host: said aside, for the
        // tests that know what theirs should do.
        fprintf(stderr, "encoder timestamp: %s\n", error == WGPUErrorType_NoError ? "written" : "refused");
    }
    printf("encoder timestamp: handled\n");

    // Immediate data. Setting it past the device's maxImmediateSize (which is
    // 0 where the device has none) is a validation error on every host.
    {
        WGPULimits limits = WGPU_LIMITS_INIT;
        CHECK(wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success);
        static const float kData[4] = { 1, 2, 3, 4 };
        wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);

        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
        WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
        wgpuComputePassEncoderSetPipeline(compute_pass, compute_pipeline);
        wgpuComputePassEncoderSetImmediates(compute_pass, limits.maxImmediateSize, kData, sizeof kData);
        wgpuComputePassEncoderEnd(compute_pass);
        wgpuComputePassEncoderRelease(compute_pass);

        WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        attachment.view = view;
        attachment.loadOp = WGPULoadOp_Clear;
        attachment.storeOp = WGPUStoreOp_Store;
        WGPURenderPassDescriptor render_pass_descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        render_pass_descriptor.colorAttachmentCount = 1;
        render_pass_descriptor.colorAttachments = &attachment;
        WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(encoder, &render_pass_descriptor);
        wgpuRenderPassEncoderSetPipeline(render_pass, render_pipeline);
        wgpuRenderPassEncoderSetImmediates(render_pass, limits.maxImmediateSize, kData, sizeof kData);
        wgpuRenderPassEncoderEnd(render_pass);
        wgpuRenderPassEncoderRelease(render_pass);
        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
        if (commands) wgpuCommandBufferRelease(commands);
        wgpuCommandEncoderRelease(encoder);

        WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm;
        WGPURenderBundleEncoderDescriptor bundle_descriptor = WGPU_RENDER_BUNDLE_ENCODER_DESCRIPTOR_INIT;
        bundle_descriptor.colorFormatCount = 1;
        bundle_descriptor.colorFormats = &format;
        bundle_descriptor.sampleCount = 1;
        WGPURenderBundleEncoder bundle_encoder = wgpuDeviceCreateRenderBundleEncoder(device, &bundle_descriptor);
        wgpuRenderBundleEncoderSetPipeline(bundle_encoder, render_pipeline);
        wgpuRenderBundleEncoderSetImmediates(bundle_encoder, limits.maxImmediateSize, kData, sizeof kData);
        WGPURenderBundle bundle = wgpuRenderBundleEncoderFinish(bundle_encoder, NULL);
        if (bundle) wgpuRenderBundleRelease(bundle);
        wgpuRenderBundleEncoderRelease(bundle_encoder);

        // Past the end of what the device allows is out of range everywhere:
        // the WebGPU specification checks the range against maxImmediateSize,
        // and an implementation that checks it against the pipeline's layout
        // (which declares no immediate data here) refuses it all the more.
        WGPUErrorType error = pop_scope(instance, device);
        CHECK(error == WGPUErrorType_Validation);
    }
    printf("immediates without a slot: refused\n");

    wgpuComputePipelineRelease(compute_pipeline);
    wgpuRenderPipelineRelease(render_pipeline);
    wgpuTextureViewRelease(view);
    wgpuTextureRelease(target);
    wgpuShaderModuleRelease(module);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
