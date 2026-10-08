// One device, several guest threads: objects created on one thread are used
// on others, each thread waits on its own futures, and error scopes are per
// thread. (Native hosts only: in a browser, WebGPU objects belong to the
// thread that created them.)

#include <pthread.h>

#include "wgpu_test.h"

typedef struct Worker {
    WGPUInstance instance;
    WGPUDevice device;
    WGPUQueue queue;
    WGPUBuffer shared;  // created by the main thread
    uint32_t seed;
    bool provoke_error;
    // Results.
    bool data_ok;
    WGPUErrorType scope;
} Worker;

static void on_scope(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1,
                     void* userdata2) {
    (void)status; (void)message; (void)userdata2;
    *(WGPUErrorType*)userdata1 = type;
}

static void* work(void* argument) {
    Worker* worker = argument;
    wgpuDevicePushErrorScope(worker->device, WGPUErrorFilter_Validation);

    worker->data_ok = true;
    for (int round = 0; round < 8; round++) {
        uint32_t values[64];
        for (uint32_t i = 0; i < 64; i++) values[i] = worker->seed * 1000 + (uint32_t)round * 100 + i;
        WGPUBuffer source =
            wt_buffer_with(worker->device, WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, values, sizeof values);
        WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        descriptor.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        descriptor.size = sizeof values;
        WGPUBuffer readback = wgpuDeviceCreateBuffer(worker->device, &descriptor);

        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(worker->device, NULL);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, source, 0, readback, 0, sizeof values);
        wt_submit(worker->queue, encoder);

        // Each thread blocks on its own future while the other does the same.
        if (wt_map(worker->instance, readback, WGPUMapMode_Read, 0, sizeof values) != WGPUMapAsyncStatus_Success) {
            worker->data_ok = false;
        } else {
            const uint32_t* mapped = wgpuBufferGetConstMappedRange(readback, 0, sizeof values);
            worker->data_ok &= mapped != NULL && memcmp(mapped, values, sizeof values) == 0;
            wgpuBufferUnmap(readback);
        }
        wgpuBufferRelease(readback);
        wgpuBufferRelease(source);
    }
    // A buffer from another thread is just as usable here.
    worker->data_ok &= wgpuBufferGetSize(worker->shared) == 256;

    if (worker->provoke_error) {
        WGPUBufferDescriptor invalid = WGPU_BUFFER_DESCRIPTOR_INIT;
        invalid.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_MapWrite;
        invalid.size = 16;
        WGPUBuffer buffer = wgpuDeviceCreateBuffer(worker->device, &invalid);
        if (buffer) wgpuBufferRelease(buffer);
    }
    worker->scope = (WGPUErrorType)0;
    WGPUPopErrorScopeCallbackInfo callback = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = on_scope;
    callback.userdata1 = &worker->scope;
    wt_wait(worker->instance, wgpuDevicePopErrorScope(worker->device, callback));
    return NULL;
}

int main(void) {
    WGPUInstance instance = wt_create_instance();
    WGPUAdapter adapter = wt_request_adapter(instance);
    WGPUDevice device = wt_request_device(instance, adapter);
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.usage = WGPUBufferUsage_Uniform;
    descriptor.size = 256;
    WGPUBuffer shared = wgpuDeviceCreateBuffer(device, &descriptor);

    enum { kThreads = 4 };
    Worker workers[kThreads];
    pthread_t threads[kThreads];
    for (int i = 0; i < kThreads; i++) {
        workers[i] = (Worker){ instance, device, queue, shared, (uint32_t)i + 1, i % 2 == 1, false, 0 };
        CHECK(pthread_create(&threads[i], NULL, work, &workers[i]) == 0);
    }
    for (int i = 0; i < kThreads; i++) CHECK(pthread_join(threads[i], NULL) == 0);

    for (int i = 0; i < kThreads; i++) {
        CHECK(workers[i].data_ok);
        // Only the threads that made a mistake see one in their scope.
        CHECK(workers[i].scope == (workers[i].provoke_error ? WGPUErrorType_Validation : WGPUErrorType_NoError));
        printf("thread %d: data ok, scope %s\n", i, workers[i].provoke_error ? "caught its error" : "clean");
    }

    wgpuBufferRelease(shared);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    return wt_done();
}
