// Submits work the GPU needs a moment for and returns from main without
// waiting for it or releasing anything. tests/browser/run.mjs starts another
// program on the same thread right afterwards, as the Wasmer SDK does: a
// queue that goes away while the browser still owes it an answer must not
// take the thread down with it.
// (Not in the manifest: on its own there is nothing to compare.)

#include "wgpu_test.h"

// Slow on purpose: tens of milliseconds of a desktop GPU.
static const char* kShader =
    "@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n"
    "@compute @workgroup_size(64)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "    var x = f32(id.x);\n"
    "    for (var i = 0u; i < 60000u; i++) { x = sin(x) + 1.0; }\n"
    "    data[id.x] = x;\n"
    "}\n";

enum { kGroups = 4096, kGroupSize = 64 };

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    WGPUBufferDescriptor buffer_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    buffer_descriptor.usage = WGPUBufferUsage_Storage;
    buffer_descriptor.size = sizeof(float) * kGroups * kGroupSize;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &buffer_descriptor);
    CHECK(buffer != NULL);

    WGPUComputePipelineDescriptor pipeline_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipeline_descriptor.compute.module = wt_shader(device, kShader);
    pipeline_descriptor.compute.entryPoint = SV("main");
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &pipeline_descriptor);
    CHECK(pipeline != NULL);

    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.binding = 0;
    entry.buffer = buffer;
    WGPUBindGroupDescriptor group_descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    group_descriptor.layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    group_descriptor.entryCount = 1;
    group_descriptor.entries = &entry;
    WGPUBindGroup group = wgpuDeviceCreateBindGroup(device, &group_descriptor);
    CHECK(group != NULL);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, kGroups, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wt_submit(queue, encoder);

    printf("submitted\n");
    return 0;
}
