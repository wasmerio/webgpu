// Buffer mapping: asynchronous maps for read and for write, the pointer
// rules of GetMappedRange, the copying Read/WriteMappedRange calls, map
// state, and maps that are cancelled or refused.

#include "wgpu_test.h"

static WGPUBuffer make_buffer(WGPUDevice device, WGPUBufferUsage usage, uint64_t size, bool mapped) {
    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.usage = usage;
    descriptor.size = size;
    descriptor.mappedAtCreation = mapped;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &descriptor);
    CHECK(buffer != NULL);
    return buffer;
}

typedef struct Scope {
    int calls;
    WGPUPopErrorScopeStatus status;
    WGPUErrorType type;
} Scope;

static void on_scope(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1,
                     void* userdata2) {
    (void)message; (void)userdata2;
    Scope* scope = userdata1;
    scope->calls++;
    scope->status = status;
    scope->type = type;
}

static WGPUErrorType pop_scope(WGPUInstance instance, WGPUDevice device) {
    Scope scope = { 0 };
    WGPUPopErrorScopeCallbackInfo callback = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = on_scope;
    callback.userdata1 = &scope;
    wt_wait(instance, wgpuDevicePopErrorScope(device, callback));
    CHECK(scope.calls == 1 && scope.status == WGPUPopErrorScopeStatus_Success);
    return scope.type;
}

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);

    // Map for writing, fill through two disjoint pointers.
    WGPUBuffer source = make_buffer(device, WGPUBufferUsage_MapWrite | WGPUBufferUsage_CopySrc, 64, false);
    CHECK(wgpuBufferGetMapState(source) == WGPUBufferMapState_Unmapped);
    CHECK(wgpuBufferGetMappedRange(source, 0, 64) == NULL);
    CHECK(wt_map(instance, source, WGPUMapMode_Write, 0, 64) == WGPUMapAsyncStatus_Success);
    CHECK(wgpuBufferGetMapState(source) == WGPUBufferMapState_Mapped);
    uint8_t* low = wgpuBufferGetMappedRange(source, 0, 32);
    uint8_t* high = wgpuBufferGetMappedRange(source, 32, 32);
    CHECK(low != NULL && high != NULL && low != high);
    CHECK(((uintptr_t)low & 15) == 0);
    // Ranges of one mapping may not overlap, whichever accessor asks.
    CHECK(wgpuBufferGetMappedRange(source, 16, 32) == NULL);
    CHECK(wgpuBufferGetConstMappedRange(source, 0, 8) == NULL);
    // Offsets are multiples of 8 and sizes multiples of 4.
    for (int i = 0; i < 32; i++) low[i] = (uint8_t)i;
    for (int i = 0; i < 32; i++) high[i] = (uint8_t)(100 + i);
    wgpuBufferUnmap(source);
    CHECK(wgpuBufferGetMapState(source) == WGPUBufferMapState_Unmapped);
    printf("write mapping: two ranges written\n");

    // Copy to a readable buffer and map a window of it.
    WGPUBuffer readback = make_buffer(device, WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst, 64, false);
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, source, 0, readback, 0, 64);
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);

    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 16, 32) == WGPUMapAsyncStatus_Success);
    // Outside the mapped window, or writable, or misaligned: no pointer.
    CHECK(wgpuBufferGetConstMappedRange(readback, 0, 64) == NULL);
    CHECK(wgpuBufferGetConstMappedRange(readback, 8, 16) == NULL);
    CHECK(wgpuBufferGetMappedRange(readback, 16, 32) == NULL);
    CHECK(wgpuBufferGetConstMappedRange(readback, 20, 8) == NULL);
    CHECK(wgpuBufferGetConstMappedRange(readback, 16, 6) == NULL);
    const uint8_t* window = wgpuBufferGetConstMappedRange(readback, 16, WGPU_WHOLE_MAP_SIZE);
    CHECK(window != NULL);
    for (int i = 0; i < 16; i++) CHECK(window[i] == 16 + i);
    for (int i = 0; i < 16; i++) CHECK(window[16 + i] == 100 + i);
    printf("read mapping: window [16, 48) matches\n");

    // The copying calls need no pointer and no alignment.
    uint8_t copy[10];
    CHECK(wgpuBufferReadMappedRange(readback, 27, copy, sizeof copy) == WGPUStatus_Success);
    for (int i = 0; i < 5; i++) CHECK(copy[i] == 27 + i);
    for (int i = 5; i < 10; i++) CHECK(copy[i] == 100 + (i - 5));
    CHECK(wgpuBufferReadMappedRange(readback, 8, copy, sizeof copy) == WGPUStatus_Error);
    CHECK(wgpuBufferWriteMappedRange(readback, 16, copy, sizeof copy) == WGPUStatus_Error);
    wgpuBufferUnmap(readback);
    CHECK(wgpuBufferReadMappedRange(readback, 16, copy, sizeof copy) == WGPUStatus_Error);
    printf("ReadMappedRange: ok\n");

    // mappedAtCreation works for any usage, and WriteMappedRange fills it.
    WGPUBuffer vertex = make_buffer(device, WGPUBufferUsage_Vertex | WGPUBufferUsage_CopySrc, 32, true);
    CHECK(wgpuBufferGetMapState(vertex) == WGPUBufferMapState_Mapped);
    uint8_t pattern[32];
    for (int i = 0; i < 32; i++) pattern[i] = (uint8_t)(200 - i);
    CHECK(wgpuBufferWriteMappedRange(vertex, 0, pattern, 32) == WGPUStatus_Success);
    CHECK(wgpuBufferReadMappedRange(vertex, 4, copy, 4) == WGPUStatus_Success);
    CHECK(copy[0] == 196 && copy[3] == 193);
    // A pointer taken after the write sees it, and what is written through
    // the pointer lands on unmap.
    uint8_t* tail = wgpuBufferGetMappedRange(vertex, 24, 8);
    CHECK(tail != NULL && tail[0] == 176);
    tail[7] = 7;
    wgpuBufferUnmap(vertex);

    encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, vertex, 0, readback, 0, 32);
    commands = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(queue, 1, &commands);
    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);
    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 0, WGPU_WHOLE_MAP_SIZE) == WGPUMapAsyncStatus_Success);
    const uint8_t* all = wgpuBufferGetConstMappedRange(readback, 0, 64);
    CHECK(all != NULL);
    for (int i = 0; i < 31; i++) CHECK(all[i] == 200 - i);
    CHECK(all[31] == 7);
    // The second half still holds the first copy.
    CHECK(all[32] == 100);
    wgpuBufferUnmap(readback);
    printf("mappedAtCreation: ok\n");

    // Mapping a mapped buffer is refused: the callback says so, and the
    // device gets a validation error.
    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 0, 64) == WGPUMapAsyncStatus_Success);
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);
    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 0, 64) == WGPUMapAsyncStatus_Error);
    CHECK(pop_scope(instance, device) == WGPUErrorType_Validation);
    CHECK(wgpuBufferGetMapState(readback) == WGPUBufferMapState_Mapped);
    wgpuBufferUnmap(readback);
    printf("map while mapped: refused\n");

    // Unmapping with a map still pending aborts it.
    WGPUMapAsyncStatus aborted = (WGPUMapAsyncStatus)0;
    WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = wt_on_map;
    callback.userdata1 = &aborted;
    WGPUFuture pending = wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, 64, callback);
    wgpuBufferUnmap(readback);
    wt_wait(instance, pending);
    // The map may have finished before the unmap arrived; either way the
    // buffer ends unmapped and the callback ran exactly once.
    CHECK(aborted == WGPUMapAsyncStatus_Aborted || aborted == WGPUMapAsyncStatus_Success);
    CHECK(wgpuBufferGetMapState(readback) == WGPUBufferMapState_Unmapped);
    printf("unmap with a map pending: handled\n");

    // Destroy unmaps.
    CHECK(wt_map(instance, readback, WGPUMapMode_Read, 0, 64) == WGPUMapAsyncStatus_Success);
    CHECK(wgpuBufferGetConstMappedRange(readback, 0, 64) != NULL);
    wgpuBufferDestroy(readback);
    CHECK(wgpuBufferGetMapState(readback) == WGPUBufferMapState_Unmapped);
    CHECK(wgpuBufferGetConstMappedRange(readback, 0, 64) == NULL);
    printf("destroy: unmapped\n");

    wgpuBufferSetLabel(source, SV("source"));
    wgpuBufferAddRef(source);
    wgpuBufferRelease(source);
    wgpuBufferRelease(source);
    wgpuBufferRelease(vertex);
    wgpuBufferRelease(readback);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
