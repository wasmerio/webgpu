// Shader modules and pipelines: asynchronous pipeline creation (which
// reports failure through its callback, not as a device error), compilation
// messages, pipeline constants, and SPIR-V where the host offers it.

#include "wgpu_test.h"

typedef struct Created {
    int calls;
    WGPUCreatePipelineAsyncStatus status;
    void* pipeline;
    size_t message_length;
} Created;

static void on_compute_pipeline(WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline pipeline,
                                WGPUStringView message, void* userdata1, void* userdata2) {
    (void)userdata2;
    Created* created = userdata1;
    created->calls++;
    created->status = status;
    created->pipeline = pipeline;
    created->message_length = message.data ? message.length : 0;
}

static void on_render_pipeline(WGPUCreatePipelineAsyncStatus status, WGPURenderPipeline pipeline,
                               WGPUStringView message, void* userdata1, void* userdata2) {
    (void)userdata2;
    Created* created = userdata1;
    created->calls++;
    created->status = status;
    created->pipeline = pipeline;
    created->message_length = message.data ? message.length : 0;
}

typedef struct Info {
    int calls;
    WGPUCompilationInfoRequestStatus status;
    size_t messages;
    size_t errors;
    bool located;  // every error names a line
} Info;

static void on_compilation_info(WGPUCompilationInfoRequestStatus status, WGPUCompilationInfo const* info,
                                void* userdata1, void* userdata2) {
    (void)userdata2;
    Info* out = userdata1;
    out->calls++;
    out->status = status;
    out->messages = info->messageCount;
    out->located = true;
    for (size_t i = 0; i < info->messageCount; i++) {
        const WGPUCompilationMessage* message = &info->messages[i];
        if (message->type != WGPUCompilationMessageType_Error) continue;
        out->errors++;
        out->located &= message->lineNum >= 1 && message->message.length > 0;
    }
}

static Info compilation_info(WGPUInstance instance, WGPUShaderModule module) {
    Info info = { 0 };
    WGPUCompilationInfoCallbackInfo callback = WGPU_COMPILATION_INFO_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = on_compilation_info;
    callback.userdata1 = &info;
    wt_wait(instance, wgpuShaderModuleGetCompilationInfo(module, callback));
    CHECK(info.calls == 1 && info.status == WGPUCompilationInfoRequestStatus_Success);
    return info;
}

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

static const char* kCompute =
    "override scale: u32 = 1u;\n"
    "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n"
    "@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "    data[id.x] = data[id.x] * scale;\n"
    "}\n";

static const char* kRender =
    "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
    "    return vec4f(f32(i), 0.0, 0.0, 1.0);\n"
    "}\n"
    "@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }\n";

// The smallest SPIR-V compute shader: an empty `main` with local size 1.
static const uint32_t kSpirv[] = {
    0x07230203, 0x00010000, 0, 6, 0,
    (2u << 16) | 17, 1,                                  // OpCapability Shader
    (3u << 16) | 14, 0, 1,                               // OpMemoryModel Logical GLSL450
    (5u << 16) | 15, 5, 4, 0x6E69616D, 0,                // OpEntryPoint GLCompute %4 "main"
    (6u << 16) | 16, 4, 17, 1, 1, 1,                     // OpExecutionMode %4 LocalSize 1 1 1
    (2u << 16) | 19, 2,                                  // %2 = OpTypeVoid
    (3u << 16) | 33, 3, 2,                               // %3 = OpTypeFunction %2
    (5u << 16) | 54, 2, 4, 0, 3,                         // %4 = OpFunction %2 None %3
    (2u << 16) | 248, 5,                                 // %5 = OpLabel
    (1u << 16) | 253,                                    // OpReturn
    (1u << 16) | 56,                                     // OpFunctionEnd
};

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    // A valid module compiles without errors.
    WGPUShaderModule compute_module = wt_shader(device, kCompute);
    Info clean = compilation_info(instance, compute_module);
    CHECK(clean.errors == 0);
    printf("valid shader: no errors\n");

    // An invalid one raises a validation error and explains itself.
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPUShaderModule broken = wt_shader(device, "@compute @workgroup_size(1) fn main() {\n    let x = ;\n}\n");
    CHECK(pop_scope(instance, device) == WGPUErrorType_Validation);
    Info explained = compilation_info(instance, broken);
    CHECK(explained.errors >= 1 && explained.located);
    printf("invalid shader: validation error with located messages\n");

    // Compute pipeline, asynchronously, with a pipeline constant.
    WGPUConstantEntry constant = WGPU_CONSTANT_ENTRY_INIT;
    constant.key = SV("scale");
    constant.value = 3;
    WGPUComputePipelineDescriptor compute_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    compute_descriptor.compute.module = compute_module;
    compute_descriptor.compute.entryPoint = SV("main");
    compute_descriptor.compute.constantCount = 1;
    compute_descriptor.compute.constants = &constant;
    Created compute = { 0 };
    WGPUCreateComputePipelineAsyncCallbackInfo compute_callback = WGPU_CREATE_COMPUTE_PIPELINE_ASYNC_CALLBACK_INFO_INIT;
    compute_callback.mode = WGPUCallbackMode_WaitAnyOnly;
    compute_callback.callback = on_compute_pipeline;
    compute_callback.userdata1 = &compute;
    wt_wait(instance, wgpuDeviceCreateComputePipelineAsync(device, &compute_descriptor, compute_callback));
    CHECK(compute.calls == 1 && compute.status == WGPUCreatePipelineAsyncStatus_Success && compute.pipeline != NULL);
    WGPUComputePipeline compute_pipeline = compute.pipeline;

    // Prove the pipeline works and the constant took: 7 * 3.
    uint32_t value = 7;
    WGPUBuffer storage =
        wt_buffer_with(device, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, &value, sizeof value);
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(compute_pipeline, 0);
    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.buffer = storage;
    WGPUBindGroupDescriptor group_descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    group_descriptor.layout = layout;
    group_descriptor.entryCount = 1;
    group_descriptor.entries = &entry;
    WGPUBindGroup group = wgpuDeviceCreateBindGroup(device, &group_descriptor);
    WGPUBufferDescriptor readback_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    readback_descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    readback_descriptor.size = 4;
    WGPUBuffer readback = wgpuDeviceCreateBuffer(device, &readback_descriptor);
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPUComputePassDescriptor pass_descriptor = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    pass_descriptor.label = SV("scale");
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_descriptor);
    wgpuComputePassEncoderPushDebugGroup(pass, SV("dispatch"));
    wgpuComputePassEncoderSetPipeline(pass, compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
    wgpuComputePassEncoderInsertDebugMarker(pass, SV("one workgroup"));
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    wgpuComputePassEncoderPopDebugGroup(pass);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, readback, 0, 4);
    wt_submit(queue, encoder);
    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 0, 4) == WGPUMapAsyncStatus_Success);
    uint32_t scaled = 0;
    CHECK(wgpuBufferReadMappedRange(readback, 0, &scaled, 4) == WGPUStatus_Success);
    wgpuBufferUnmap(readback);
    printf("async compute pipeline: 7 * scale = %u\n", scaled);

    // The same pass, driven by an indirect buffer.
    static const uint32_t kWorkgroups[3] = { 1, 1, 1 };
    WGPUBuffer indirect = wt_buffer_with(device, WGPUBufferUsage_Indirect, kWorkgroups, sizeof kWorkgroups);
    encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    wgpuComputePassEncoderSetPipeline(pass, compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroupsIndirect(pass, indirect, 0);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, readback, 0, 4);
    wt_submit(queue, encoder);
    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 0, 4) == WGPUMapAsyncStatus_Success);
    CHECK(wgpuBufferReadMappedRange(readback, 0, &scaled, 4) == WGPUStatus_Success);
    wgpuBufferUnmap(readback);
    printf("indirect dispatch: 21 * scale = %u\n", scaled);

    // Render pipeline, asynchronously.
    WGPUShaderModule render_module = wt_shader(device, kRender);
    WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
    target.format = WGPUTextureFormat_RGBA8Unorm;
    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = render_module;
    fragment.entryPoint = SV("fs");
    fragment.targetCount = 1;
    fragment.targets = &target;
    WGPURenderPipelineDescriptor render_descriptor = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    render_descriptor.vertex.module = render_module;
    render_descriptor.vertex.entryPoint = SV("vs");
    render_descriptor.fragment = &fragment;
    Created render = { 0 };
    WGPUCreateRenderPipelineAsyncCallbackInfo render_callback = WGPU_CREATE_RENDER_PIPELINE_ASYNC_CALLBACK_INFO_INIT;
    render_callback.mode = WGPUCallbackMode_AllowProcessEvents;
    render_callback.callback = on_render_pipeline;
    render_callback.userdata1 = &render;
    wgpuDeviceCreateRenderPipelineAsync(device, &render_descriptor, render_callback);
    while (render.calls == 0) wgpuInstanceProcessEvents(instance);
    CHECK(render.status == WGPUCreatePipelineAsyncStatus_Success && render.pipeline != NULL);
    wgpuRenderPipelineRelease(render.pipeline);
    printf("async render pipeline: created\n");

    // A pipeline that cannot be created says so in its callback. That is the
    // whole report: no device error is raised for it.
    compute_descriptor.compute.entryPoint = SV("no_such_entry_point");
    Created failed = { 0 };
    compute_callback.userdata1 = &failed;
    wt_wait(instance, wgpuDeviceCreateComputePipelineAsync(device, &compute_descriptor, compute_callback));
    CHECK(failed.calls == 1 && failed.status == WGPUCreatePipelineAsyncStatus_ValidationError);
    CHECK(failed.pipeline == NULL && failed.message_length > 0);
    for (int i = 0; i < 20; i++) wgpuInstanceProcessEvents(instance);
    CHECK(wt_uncaptured_errors == 0);
    printf("async pipeline failure: reported through the callback\n");

    // SPIR-V is an instance feature. Where it is missing, offering SPIR-V is
    // a validation error rather than a crash.
    WGPUShaderSourceSPIRV spirv = WGPU_SHADER_SOURCE_SPIRV_INIT;
    spirv.codeSize = sizeof kSpirv / sizeof kSpirv[0];
    spirv.code = kSpirv;
    WGPUShaderModuleDescriptor spirv_descriptor = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    spirv_descriptor.nextInChain = &spirv.chain;
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    WGPUShaderModule spirv_module = wgpuDeviceCreateShaderModule(device, &spirv_descriptor);
    WGPUErrorType spirv_error = pop_scope(instance, device);
    if (wgpuHasInstanceFeature(WGPUInstanceFeatureName_ShaderSourceSPIRV)) {
        CHECK(spirv_module != NULL && spirv_error == WGPUErrorType_NoError);
        WGPUComputePipelineDescriptor spirv_pipeline_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        spirv_pipeline_descriptor.compute.module = spirv_module;
        spirv_pipeline_descriptor.compute.entryPoint = SV("main");
        WGPUComputePipeline spirv_pipeline = wgpuDeviceCreateComputePipeline(device, &spirv_pipeline_descriptor);
        CHECK(spirv_pipeline != NULL);
        wgpuComputePipelineRelease(spirv_pipeline);
    } else {
        CHECK(spirv_error == WGPUErrorType_Validation);
    }
    if (spirv_module) wgpuShaderModuleRelease(spirv_module);
    printf("spir-v: handled as the instance features say\n");

    wgpuBufferRelease(indirect);
    wgpuBufferRelease(readback);
    wgpuBindGroupRelease(group);
    wgpuBindGroupLayoutRelease(layout);
    wgpuBufferRelease(storage);
    wgpuComputePipelineRelease(compute_pipeline);
    wgpuShaderModuleRelease(render_module);
    wgpuShaderModuleRelease(broken);
    wgpuShaderModuleRelease(compute_module);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
