// Native integration: the import must copy every descriptor byte and retain
// resolved native shader/layout objects before it returns. Heap contents are
// poisoned/freed and guest resource handles released before any owner polling.
// The same-worker held-lane Rust tests supply deterministic worker delay; this
// program tests the real Wasm ABI, C callback mode/thread and GPU output.

#include "wgpu_test.h"
#include <pthread.h>
#include <time.h>

typedef struct Created {
    int calls;
    WGPUCreatePipelineAsyncStatus status;
    void* pipeline;
    pthread_t owner;
} Created;

static void on_compute(WGPUCreatePipelineAsyncStatus status, WGPUComputePipeline pipeline,
                       WGPUStringView message, void* userdata1, void* userdata2) {
    (void)message; (void)userdata2;
    Created* result = userdata1;
    CHECK(pthread_equal(result->owner, pthread_self()));
    result->calls++;
    result->status = status;
    result->pipeline = pipeline;
}

static void on_render(WGPUCreatePipelineAsyncStatus status, WGPURenderPipeline pipeline,
                      WGPUStringView message, void* userdata1, void* userdata2) {
    (void)message; (void)userdata2;
    Created* result = userdata1;
    CHECK(pthread_equal(result->owner, pthread_self()));
    result->calls++;
    result->status = status;
    result->pipeline = pipeline;
}

static char* owned_string(const char* value) {
    char* out = malloc(strlen(value) + 1);
    CHECK(out != NULL);
    strcpy(out, value);
    return out;
}

static void retire(void* memory, size_t size) {
    // Keep actual stores even at -O2; plain memset-before-free can be removed.
    volatile unsigned char* bytes = memory;
    for (size_t i = 0; i < size; i++) bytes[i] = 0xa5;
    free(memory);
}

static uint64_t monotonic_ns(void) {
    struct timespec time;
    CHECK(clock_gettime(CLOCK_MONOTONIC, &time) == 0);
    return (uint64_t)time.tv_sec * 1000000000ull + (uint64_t)time.tv_nsec;
}

static void wait_owned(WGPUInstance instance, WGPUFuture future, Created* result) {
    CHECK(future.id != 0);
    for (int i = 0; i < 20; i++) wgpuInstanceProcessEvents(instance);
    CHECK(result->calls == 0); // WaitAnyOnly must not deliver via ProcessEvents.
    uint64_t deadline = monotonic_ns() + 15000000000ull;
    while (!result->calls) {
        WGPUFutureWaitInfo info = { future, false };
        WGPUWaitStatus status = wgpuInstanceWaitAny(instance, 1, &info, 50000000ull);
        CHECK(status == WGPUWaitStatus_Success || status == WGPUWaitStatus_TimedOut);
        CHECK(monotonic_ns() < deadline);
        if (info.completed) CHECK(status == WGPUWaitStatus_Success && result->calls == 1);
    }
    CHECK(result->calls == 1 && result->status == WGPUCreatePipelineAsyncStatus_Success && result->pipeline != NULL);
    for (int i = 0; i < 20; i++) wgpuInstanceProcessEvents(instance);
    CHECK(result->calls == 1);
}

static const char* kCompute =
    "override scale: u32 = 1u;\n"
    "@group(0) @binding(0) var<storage,read_write> data: array<u32>;\n"
    "@compute @workgroup_size(1) fn main() { data[0] *= scale; }\n";
static const char* kRender =
    "override tint: u32 = 0u;\n"
    "@vertex fn vs(@location(0) pos:vec2f)->@builtin(position) vec4f { return vec4f(pos,0.0,1.0); }\n"
    "@fragment fn fs()->@location(0) vec4f { return vec4f(f32(tint),0.0,0.0,1.0); }\n";

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    WGPUShaderModule module = wt_shader(device, kCompute);

    WGPUBindGroupLayoutEntry binding = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    binding.visibility = WGPUShaderStage_Compute;
    binding.buffer.type = WGPUBufferBindingType_Storage;
    binding.buffer.minBindingSize = 4;
    WGPUBindGroupLayoutDescriptor bgl_descriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bgl_descriptor.entryCount = 1; bgl_descriptor.entries = &binding;
    WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(device, &bgl_descriptor);
    CHECK(bgl != NULL);
    WGPUPipelineLayoutDescriptor layout_descriptor = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    layout_descriptor.bindGroupLayoutCount = 1; layout_descriptor.bindGroupLayouts = &bgl;
    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device, &layout_descriptor);
    CHECK(layout != NULL);

    char* compute_entry = owned_string("main");
    char* compute_key = owned_string("scale");
    char* compute_label = owned_string("owned-compute");
    WGPUConstantEntry* constant = malloc(sizeof *constant);
    WGPUComputePipelineDescriptor* descriptor = malloc(sizeof *descriptor);
    CHECK(constant && descriptor);
    *constant = (WGPUConstantEntry)WGPU_CONSTANT_ENTRY_INIT;
    constant->key = SV(compute_key); constant->value = 3;
    *descriptor = (WGPUComputePipelineDescriptor)WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    descriptor->layout = layout; descriptor->label = SV(compute_label);
    descriptor->compute.module = module; descriptor->compute.entryPoint = SV(compute_entry);
    descriptor->compute.constantCount = 1; descriptor->compute.constants = constant;
    Created compute = { .owner = pthread_self() };
    WGPUCreateComputePipelineAsyncCallbackInfo callback = WGPU_CREATE_COMPUTE_PIPELINE_ASYNC_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly; callback.callback = on_compute; callback.userdata1 = &compute;
    WGPUFuture future = wgpuDeviceCreateComputePipelineAsync(device, descriptor, callback);
    retire(descriptor, sizeof *descriptor); retire(constant, sizeof *constant);
    retire(compute_entry, 5); retire(compute_key, 6); retire(compute_label, 14);
    wgpuShaderModuleRelease(module); wgpuPipelineLayoutRelease(layout); wgpuBindGroupLayoutRelease(bgl);
    wait_owned(instance, future, &compute);

    uint32_t value = 7;
    WGPUBuffer storage = wt_buffer_with(device, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, &value, sizeof value);
    WGPUBindGroupLayout actual_layout = wgpuComputePipelineGetBindGroupLayout(compute.pipeline, 0);
    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.buffer = storage; entry.size = 4;
    WGPUBindGroupDescriptor group_descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    group_descriptor.layout = actual_layout; group_descriptor.entryCount = 1; group_descriptor.entries = &entry;
    WGPUBindGroup group = wgpuDeviceCreateBindGroup(device, &group_descriptor);
    CHECK(group != NULL);
    WGPUBufferDescriptor readback_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    readback_descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst; readback_descriptor.size = 4;
    WGPUBuffer readback = wgpuDeviceCreateBuffer(device, &readback_descriptor);
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    wgpuComputePassEncoderSetPipeline(pass, compute.pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    wgpuComputePassEncoderEnd(pass); wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, readback, 0, 4); wt_submit(queue, encoder);
    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 0, 4) == WGPUMapAsyncStatus_Success);
    uint32_t scaled = 0;
    CHECK(wgpuBufferReadMappedRange(readback, 0, &scaled, 4) == WGPUStatus_Success);
    CHECK(scaled == 21); wgpuBufferUnmap(readback);
    printf("owned compute: WaitAnyOnly owner, released inputs, 7 * 3 = 21\n");

    module = wt_shader(device, kRender);
    layout_descriptor.bindGroupLayoutCount = 0; layout_descriptor.bindGroupLayouts = NULL;
    layout = wgpuDeviceCreatePipelineLayout(device, &layout_descriptor); CHECK(layout != NULL);
    char* vertex_entry = owned_string("vs"); char* fragment_entry = owned_string("fs");
    char* fragment_key = owned_string("tint"); char* render_label = owned_string("owned-render");
    WGPUVertexAttribute* attribute = malloc(sizeof *attribute);
    WGPUVertexBufferLayout* vertex_layout = malloc(sizeof *vertex_layout);
    WGPUColorTargetState* color = malloc(sizeof *color);
    WGPUFragmentState* fragment = malloc(sizeof *fragment);
    WGPURenderPipelineDescriptor* render_descriptor = malloc(sizeof *render_descriptor);
    constant = malloc(sizeof *constant);
    CHECK(attribute && vertex_layout && color && fragment && render_descriptor && constant);
    *attribute = (WGPUVertexAttribute)WGPU_VERTEX_ATTRIBUTE_INIT;
    attribute->format = WGPUVertexFormat_Float32x2;
    *vertex_layout = (WGPUVertexBufferLayout)WGPU_VERTEX_BUFFER_LAYOUT_INIT;
    vertex_layout->arrayStride = 2 * sizeof(float); vertex_layout->stepMode = WGPUVertexStepMode_Vertex;
    vertex_layout->attributeCount = 1; vertex_layout->attributes = attribute;
    *color = (WGPUColorTargetState)WGPU_COLOR_TARGET_STATE_INIT; color->format = WGPUTextureFormat_RGBA8Unorm;
    *constant = (WGPUConstantEntry)WGPU_CONSTANT_ENTRY_INIT; constant->key = SV(fragment_key); constant->value = 1;
    *fragment = (WGPUFragmentState)WGPU_FRAGMENT_STATE_INIT;
    fragment->module = module; fragment->entryPoint = SV(fragment_entry);
    fragment->constantCount = 1; fragment->constants = constant; fragment->targetCount = 1; fragment->targets = color;
    *render_descriptor = (WGPURenderPipelineDescriptor)WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    render_descriptor->label = SV(render_label); render_descriptor->layout = layout;
    render_descriptor->vertex.module = module; render_descriptor->vertex.entryPoint = SV(vertex_entry);
    render_descriptor->vertex.bufferCount = 1; render_descriptor->vertex.buffers = vertex_layout;
    render_descriptor->primitive.topology = WGPUPrimitiveTopology_TriangleList; render_descriptor->fragment = fragment;
    Created render = { .owner = pthread_self() };
    WGPUCreateRenderPipelineAsyncCallbackInfo render_callback = WGPU_CREATE_RENDER_PIPELINE_ASYNC_CALLBACK_INFO_INIT;
    render_callback.mode = WGPUCallbackMode_WaitAnyOnly; render_callback.callback = on_render; render_callback.userdata1 = &render;
    future = wgpuDeviceCreateRenderPipelineAsync(device, render_descriptor, render_callback);
    retire(render_descriptor, sizeof *render_descriptor); retire(fragment, sizeof *fragment); retire(color, sizeof *color);
    retire(vertex_layout, sizeof *vertex_layout); retire(attribute, sizeof *attribute); retire(constant, sizeof *constant);
    retire(vertex_entry, 3); retire(fragment_entry, 3); retire(fragment_key, 5); retire(render_label, 13);
    wgpuShaderModuleRelease(module); wgpuPipelineLayoutRelease(layout);
    wait_owned(instance, future, &render);

    static const float vertices[] = { -1, -1, 3, -1, -1, 3 };
    WGPUBuffer vertex_buffer = wt_buffer_with(device, WGPUBufferUsage_Vertex, vertices, sizeof vertices);
    WGPUTexture target = wt_target(device, WGPUTextureFormat_RGBA8Unorm, 4, 4, 1);
    WGPUTextureView view = wgpuTextureCreateView(target, NULL);
    WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
    attachment.view = view; attachment.loadOp = WGPULoadOp_Clear; attachment.storeOp = WGPUStoreOp_Store;
    attachment.clearValue = (WGPUColor){ 0, 0, 1, 1 };
    WGPURenderPassDescriptor pass_descriptor = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
    pass_descriptor.colorAttachmentCount = 1; pass_descriptor.colorAttachments = &attachment;
    encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPURenderPassEncoder render_pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_descriptor);
    wgpuRenderPassEncoderSetPipeline(render_pass, render.pipeline);
    wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, vertex_buffer, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(render_pass, 3, 1, 0, 0);
    wgpuRenderPassEncoderEnd(render_pass); wgpuRenderPassEncoderRelease(render_pass); wt_submit(queue, encoder);
    uint8_t* pixels = wt_read_texture(instance, device, queue, target, 4, 4);
    for (int i = 0; i < 16; i++) CHECK(pixels[i*4] == 255 && pixels[i*4+1] == 0 && pixels[i*4+2] == 0 && pixels[i*4+3] == 255);
    free(pixels);
    printf("owned render: WaitAnyOnly owner, released inputs, 16 red pixels\n");

    wgpuTextureViewRelease(view); wgpuTextureRelease(target); wgpuBufferRelease(vertex_buffer);
    wgpuRenderPipelineRelease(render.pipeline); wgpuBufferRelease(readback); wgpuBindGroupRelease(group);
    wgpuBindGroupLayoutRelease(actual_layout); wgpuBufferRelease(storage); wgpuComputePipelineRelease(compute.pipeline);
    wgpuQueueRelease(queue); wgpuDeviceRelease(device); wgpuAdapterRelease(adapter); wgpuInstanceRelease(instance);
    return wt_done();
}
