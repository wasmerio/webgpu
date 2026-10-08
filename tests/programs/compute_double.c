// A compute pass end to end: upload through a mapped-at-creation buffer, run
// a shader over it, copy the result out and read it back through a map.

#include "wgpu_test.h"

static const char* kShader =
    "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n"
    "@compute @workgroup_size(4)\n"
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
    "    if (id.x < arrayLength(&data)) {\n"
    "        data[id.x] = data[id.x] * 2u + 1u;\n"
    "    }\n"
    "}\n";

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    uint32_t input[16];
    for (uint32_t i = 0; i < 16; i++) input[i] = i * 10;

    WGPUBufferDescriptor storage_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    storage_descriptor.label = SV("storage");
    storage_descriptor.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    storage_descriptor.size = sizeof input;
    storage_descriptor.mappedAtCreation = true;
    WGPUBuffer storage = wgpuDeviceCreateBuffer(device, &storage_descriptor);
    CHECK(storage != NULL);
    CHECK(wgpuBufferGetMapState(storage) == WGPUBufferMapState_Mapped);
    void* mapped = wgpuBufferGetMappedRange(storage, 0, sizeof input);
    CHECK(mapped != NULL);
    memcpy(mapped, input, sizeof input);
    wgpuBufferUnmap(storage);
    CHECK(wgpuBufferGetMapState(storage) == WGPUBufferMapState_Unmapped);

    WGPUBufferDescriptor readback_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    readback_descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    readback_descriptor.size = sizeof input;
    WGPUBuffer readback = wgpuDeviceCreateBuffer(device, &readback_descriptor);
    CHECK(wgpuBufferGetSize(readback) == sizeof input);
    CHECK(wgpuBufferGetUsage(readback) == (WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst));

    WGPUShaderModule module = wt_shader(device, kShader);
    WGPUComputePipelineDescriptor pipeline_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipeline_descriptor.compute.module = module;
    pipeline_descriptor.compute.entryPoint = SV("main");
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &pipeline_descriptor);
    CHECK(pipeline != NULL);

    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
    entry.binding = 0;
    entry.buffer = storage;
    WGPUBindGroupDescriptor group_descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    group_descriptor.layout = layout;
    group_descriptor.entryCount = 1;
    group_descriptor.entries = &entry;
    WGPUBindGroup group = wgpuDeviceCreateBindGroup(device, &group_descriptor);
    CHECK(group != NULL);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 4, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, storage, 0, readback, 0, sizeof input);
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);

    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 0, sizeof input) == WGPUMapAsyncStatus_Success);
    CHECK(wgpuBufferGetMapState(readback) == WGPUBufferMapState_Mapped);
    const uint32_t* output = wgpuBufferGetConstMappedRange(readback, 0, sizeof input);
    CHECK(output != NULL);
    // A read mapping does not hand out writable pointers.
    CHECK(wgpuBufferGetMappedRange(readback, 0, sizeof input) == NULL);
    for (uint32_t i = 0; i < 16; i++) {
        CHECK(output[i] == input[i] * 2 + 1);
        printf("%u%s", output[i], i == 15 ? "\n" : " ");
    }
    wgpuBufferUnmap(readback);

    wgpuBindGroupRelease(group);
    wgpuBindGroupLayoutRelease(layout);
    wgpuComputePipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
    wgpuBufferRelease(readback);
    wgpuBufferRelease(storage);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
