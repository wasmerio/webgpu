// Rules a render pass is held to that not every backend enforces by itself:
// what a transient attachment may be used for, and how many draws a pass that
// announced a number may make. A program sees them enforced the same way on
// every host.

#include "wgpu_test.h"

static const char* kShader =
    "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
    "    return vec4f(f32(i) - 1.0, 0.0, 0.0, 1.0);\n"
    "}\n"
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }\n";

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

typedef struct Context {
    WGPUInstance instance;
    WGPUDevice device;
    WGPURenderPipeline pipeline;
} Context;

typedef struct Pass {
    WGPUTextureView view;
    WGPULoadOp load;
    WGPUStoreOp store;
    uint64_t max_draws;  // 0: the pass does not say
    int draws;
    WGPURenderBundle bundle;  // executed once, when set
} Pass;

// What recording the pass and what finishing its encoder each report.
typedef struct Outcome {
    WGPUErrorType recording;
    WGPUErrorType finishing;
} Outcome;

static Outcome encode(const Context* context, const Pass* what) {
    WGPUDevice device = context->device;
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    attachment.view = what->view;
    attachment.loadOp = what->load;
    attachment.storeOp = what->store;
    WGPURenderPassMaxDrawCount limit = WGPU_RENDER_PASS_MAX_DRAW_COUNT_INIT;
    limit.maxDrawCount = what->max_draws;
    WGPURenderPassDescriptor descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    descriptor.colorAttachmentCount = 1;
    descriptor.colorAttachments = &attachment;
    if (what->max_draws) descriptor.nextInChain = &limit.chain;

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &descriptor);
    if (pass) {
        wgpuRenderPassEncoderSetPipeline(pass, context->pipeline);
        for (int i = 0; i < what->draws; i++) wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        if (what->bundle) wgpuRenderPassEncoderExecuteBundles(pass, 1, &what->bundle);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    }
    Outcome outcome;
    outcome.recording = pop_scope(context->instance, device);

    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    if (commands) wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);
    outcome.finishing = pop_scope(context->instance, device);
    return outcome;
}

static bool refused(Outcome outcome) {
    return outcome.recording == WGPUErrorType_Validation || outcome.finishing == WGPUErrorType_Validation;
}

static bool accepted(Outcome outcome) {
    return outcome.recording == WGPUErrorType_NoError && outcome.finishing == WGPUErrorType_NoError;
}

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);

    WGPUShaderModule module = wt_shader(device, kShader);
    WGPUColorTargetState color_target = WGPU_COLOR_TARGET_STATE_INIT;
    color_target.format = WGPUTextureFormat_RGBA8Unorm;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = module;
    fragment.entryPoint = SV("fs");
    fragment.targetCount = 1;
    fragment.targets = &color_target;
    WGPURenderPipelineDescriptor pipeline_descriptor = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pipeline_descriptor.vertex.module = module;
    pipeline_descriptor.vertex.entryPoint = SV("vs");
    pipeline_descriptor.fragment = &fragment;
    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device, &pipeline_descriptor);
    CHECK(pipeline != NULL);
    Context context = { instance, device, pipeline };

    // ---- transient attachments ----
    // A texture that only ever holds what one pass draws: on GPUs that render
    // in tiles it needs no memory of its own. Nothing may go in or come out.
    WGPUTextureDescriptor transient_descriptor = WGPU_TEXTURE_DESCRIPTOR_INIT;
    transient_descriptor.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TransientAttachment;
    transient_descriptor.size = (WGPUExtent3D){ 4, 4, 1 };
    transient_descriptor.format = WGPUTextureFormat_RGBA8Unorm;
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPUTexture transient = wgpuDeviceCreateTexture(device, &transient_descriptor);
    CHECK(pop_scope(instance, device) == WGPUErrorType_NoError && transient != NULL);
    CHECK(wgpuTextureGetUsage(transient) == transient_descriptor.usage);
    WGPUTextureView transient_view = wgpuTextureCreateView(transient, NULL);

    Pass pass = { transient_view, WGPULoadOp_Clear, WGPUStoreOp_Discard, 0, 1, NULL };
    CHECK(accepted(encode(&context, &pass)));
    printf("transient attachment: cleared and discarded, accepted\n");
    pass.store = WGPUStoreOp_Store;
    CHECK(refused(encode(&context, &pass)));
    printf("transient attachment: stored, refused\n");
    pass.store = WGPUStoreOp_Discard;
    pass.load = WGPULoadOp_Load;
    CHECK(refused(encode(&context, &pass)));
    printf("transient attachment: loaded, refused\n");

    transient_descriptor.usage |= WGPUTextureUsage_CopySrc;
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPUTexture readable = wgpuDeviceCreateTexture(device, &transient_descriptor);
    CHECK(pop_scope(instance, device) == WGPUErrorType_Validation);
    if (readable) wgpuTextureRelease(readable);
    printf("transient attachment: with another usage, refused\n");

    // ---- the draw count a pass announces ----
    WGPUTexture target = wt_target(device, WGPUTextureFormat_RGBA8Unorm, 4, 4, 1);
    WGPUTextureView target_view = wgpuTextureCreateView(target, NULL);
    Pass counted = { target_view, WGPULoadOp_Clear, WGPUStoreOp_Store, 2, 2, NULL };
    CHECK(accepted(encode(&context, &counted)));
    printf("max draw count: within it, accepted\n");

    // One draw too many makes the encoder invalid. That is found out when the
    // encoder is finished, not at the draw or at the end of the pass.
    counted.draws = 3;
    Outcome over = encode(&context, &counted);
    CHECK(over.recording == WGPUErrorType_NoError && over.finishing == WGPUErrorType_Validation);
    printf("max draw count: exceeded, refused when the encoder is finished\n");

    // The draws of a bundle count each time it is executed.
    WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm;
    WGPURenderBundleEncoderDescriptor bundle_descriptor = WGPU_RENDER_BUNDLE_ENCODER_DESCRIPTOR_INIT;
    bundle_descriptor.colorFormatCount = 1;
    bundle_descriptor.colorFormats = &format;
    bundle_descriptor.sampleCount = 1;
    WGPURenderBundleEncoder bundle_encoder = wgpuDeviceCreateRenderBundleEncoder(device, &bundle_descriptor);
    wgpuRenderBundleEncoderSetPipeline(bundle_encoder, pipeline);
    wgpuRenderBundleEncoderDraw(bundle_encoder, 3, 1, 0, 0);
    wgpuRenderBundleEncoderDraw(bundle_encoder, 3, 1, 0, 0);
    WGPURenderBundle bundle = wgpuRenderBundleEncoderFinish(bundle_encoder, NULL);
    CHECK(bundle != NULL);
    wgpuRenderBundleEncoderRelease(bundle_encoder);
    Pass bundled = { target_view, WGPULoadOp_Clear, WGPUStoreOp_Store, 3, 1, bundle };
    CHECK(accepted(encode(&context, &bundled)));
    bundled.max_draws = 2;
    CHECK(refused(encode(&context, &bundled)));
    printf("max draw count: a bundle's draws count\n");

    wgpuRenderBundleRelease(bundle);
    wgpuTextureViewRelease(target_view);
    wgpuTextureRelease(target);
    wgpuTextureViewRelease(transient_view);
    wgpuTextureRelease(transient);
    wgpuRenderPipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
