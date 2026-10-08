// Everything a program can ask before it draws: instance features and
// limits, the proc table, adapter and device info, features and limits, and
// the ownership rules of the results (FreeMembers, AddRef/Release).

#include "wgpu_test.h"

// Like wt_on_device, but a failure is the expected outcome here.
static void on_device_quiet(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1,
                            void* userdata2) {
    (void)message; (void)userdata2;
    *(WGPUDevice*)userdata1 = status == WGPURequestDeviceStatus_Success ? device : NULL;
}

static bool has_feature(const WGPUSupportedFeatures* features, WGPUFeatureName name) {
    for (size_t i = 0; i < features->featureCount; i++) {
        if (features->features[i] == name) return true;
    }
    return false;
}

int main(void) {
    // Instance features, before any instance exists.
    WGPUSupportedInstanceFeatures instance_features = { 0 };
    wgpuGetInstanceFeatures(&instance_features);
    bool timed_wait = false;
    for (size_t i = 0; i < instance_features.featureCount; i++) {
        CHECK(wgpuHasInstanceFeature(instance_features.features[i]));
        timed_wait |= instance_features.features[i] == WGPUInstanceFeatureName_TimedWaitAny;
    }
    CHECK(timed_wait);
    wgpuSupportedInstanceFeaturesFreeMembers(instance_features);
    CHECK(!wgpuHasInstanceFeature((WGPUInstanceFeatureName)0x7000));

    WGPUInstanceLimits instance_limits = WGPU_INSTANCE_LIMITS_INIT;
    CHECK(wgpuGetInstanceLimits(&instance_limits) == WGPUStatus_Success);
    CHECK(instance_limits.timedWaitAnyMaxCount >= 1);
    printf("instance features and limits: ok\n");

    // The proc table resolves names to the very functions the program calls.
    CHECK(wgpuGetProcAddress(SV("wgpuCreateInstance")) == (WGPUProc)wgpuCreateInstance);
    WGPUStringView sized = { "wgpuDeviceCreateBufferAndMore", 22 };
    CHECK(wgpuGetProcAddress(sized) == (WGPUProc)wgpuDeviceCreateBuffer);
    CHECK(wgpuGetProcAddress(SV("wgpuWasixSurfaceGetSize")) == (WGPUProc)wgpuWasixSurfaceGetSize);
    CHECK(wgpuGetProcAddress(SV("wgpuNoSuchFunction")) == NULL);
    printf("proc table: ok\n");

    WGPUInstance instance = wt_create_instance();
    wgpuInstanceAddRef(instance);
    wgpuInstanceRelease(instance);

    WGPUSupportedWGSLLanguageFeatures wgsl = { 0 };
    wgpuInstanceGetWGSLLanguageFeatures(instance, &wgsl);
    for (size_t i = 0; i < wgsl.featureCount; i++)
        CHECK(wgpuInstanceHasWGSLLanguageFeature(instance, wgsl.features[i]));
    CHECK(!wgpuInstanceHasWGSLLanguageFeature(instance, (WGPUWGSLLanguageFeatureName)0x7000));
    wgpuSupportedWGSLLanguageFeaturesFreeMembers(wgsl);
    printf("wgsl language features: ok\n");

    WGPUAdapter adapter = wt_request_adapter(instance);
    wgpuAdapterAddRef(adapter);
    wgpuAdapterRelease(adapter);

    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    CHECK(wgpuAdapterGetInfo(adapter, &info) == WGPUStatus_Success);
    CHECK(info.backendType != WGPUBackendType_Undefined);
    CHECK(info.adapterType >= WGPUAdapterType_DiscreteGPU && info.adapterType <= WGPUAdapterType_Unknown);
    // Strings are views into memory the caller owns until FreeMembers.
    size_t text = info.vendor.length + info.architecture.length + info.device.length + info.description.length;
    char* copy = malloc(text + 1);
    size_t at = 0;
    memcpy(copy + at, info.vendor.data, info.vendor.length); at += info.vendor.length;
    memcpy(copy + at, info.architecture.data, info.architecture.length); at += info.architecture.length;
    memcpy(copy + at, info.device.data, info.device.length); at += info.device.length;
    memcpy(copy + at, info.description.data, info.description.length); at += info.description.length;
    CHECK(at == text);
    free(copy);
    wgpuAdapterInfoFreeMembers(info);

    WGPUSupportedFeatures adapter_features = { 0 };
    wgpuAdapterGetFeatures(adapter, &adapter_features);
    CHECK(has_feature(&adapter_features, WGPUFeatureName_CoreFeaturesAndLimits));
    for (size_t i = 0; i < adapter_features.featureCount; i++)
        CHECK(wgpuAdapterHasFeature(adapter, adapter_features.features[i]));
    CHECK(!wgpuAdapterHasFeature(adapter, (WGPUFeatureName)0x7000));

    WGPULimits adapter_limits = WGPU_LIMITS_INIT;
    CHECK(wgpuAdapterGetLimits(adapter, &adapter_limits) == WGPUStatus_Success);
    CHECK(adapter_limits.maxTextureDimension2D >= 8192);
    CHECK(adapter_limits.maxBindGroups >= 4);
    CHECK(adapter_limits.maxBufferSize >= 268435456ull);
    printf("adapter: ok\n");

    // A device asking for nothing gets the specification's default limits,
    // not the adapter's.
    WGPUDevice device = wt_request_device(instance, adapter);
    wgpuDeviceAddRef(device);
    wgpuDeviceRelease(device);
    WGPULimits limits = WGPU_LIMITS_INIT;
    CHECK(wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success);
    printf("device limits: maxBindGroups=%u maxTextureDimension2D=%u maxColorAttachments=%u\n", limits.maxBindGroups,
           limits.maxTextureDimension2D, limits.maxColorAttachments);

    WGPUSupportedFeatures device_features = { 0 };
    wgpuDeviceGetFeatures(device, &device_features);
    CHECK(has_feature(&device_features, WGPUFeatureName_CoreFeaturesAndLimits));
    CHECK(wgpuDeviceHasFeature(device, WGPUFeatureName_CoreFeaturesAndLimits));
    // Nothing optional was required, so nothing optional is enabled.
    CHECK(!wgpuDeviceHasFeature(device, WGPUFeatureName_TextureCompressionBC));
    for (size_t i = 0; i < device_features.featureCount; i++)
        CHECK(has_feature(&adapter_features, device_features.features[i]));
    wgpuSupportedFeaturesFreeMembers(device_features);

    WGPUAdapterInfo device_info = WGPU_ADAPTER_INFO_INIT;
    CHECK(wgpuDeviceGetAdapterInfo(device, &device_info) == WGPUStatus_Success);
    CHECK(device_info.backendType != WGPUBackendType_Undefined);
    wgpuAdapterInfoFreeMembers(device_info);

    // The queue is one object: asking again returns the same handle.
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    WGPUQueue again = wgpuDeviceGetQueue(device);
    CHECK(queue != NULL && queue == again);
    wgpuQueueAddRef(queue);
    wgpuQueueRelease(queue);
    wgpuQueueRelease(again);
    wgpuQueueSetLabel(queue, SV("the queue"));
    wgpuDeviceSetLabel(device, SV("the device"));
    printf("device: ok\n");

    // Requiring a limit raises it; requiring a feature enables it.
    WGPULimits required = WGPU_LIMITS_INIT;
    required.maxBindGroups = adapter_limits.maxBindGroups;
    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    descriptor.label = SV("second device");
    descriptor.requiredLimits = &required;
    WGPUFeatureName optional = (WGPUFeatureName)0;
    static const WGPUFeatureName kCandidates[] = {
        WGPUFeatureName_Depth32FloatStencil8, WGPUFeatureName_DepthClipControl, WGPUFeatureName_TimestampQuery,
        WGPUFeatureName_IndirectFirstInstance, WGPUFeatureName_ShaderF16,
    };
    for (size_t i = 0; i < sizeof kCandidates / sizeof kCandidates[0] && !optional; i++) {
        if (has_feature(&adapter_features, kCandidates[i])) optional = kCandidates[i];
    }
    if (optional) {
        descriptor.requiredFeatureCount = 1;
        descriptor.requiredFeatures = &optional;
    }
    // Several devices per adapter are an instance feature; hosts without it
    // need a fresh adapter for the second device.
    WGPUAdapter second_adapter = adapter;
    if (!wgpuHasInstanceFeature(WGPUInstanceFeatureName_MultipleDevicesPerAdapter))
        second_adapter = wt_request_adapter(instance);
    WGPUDevice second = wt_request_device_with(instance, second_adapter, &descriptor);
    WGPULimits second_limits = WGPU_LIMITS_INIT;
    CHECK(wgpuDeviceGetLimits(second, &second_limits) == WGPUStatus_Success);
    CHECK(second_limits.maxBindGroups == adapter_limits.maxBindGroups);
    if (optional) CHECK(wgpuDeviceHasFeature(second, optional));
    printf("required limits and features: ok\n");

    // A feature the adapter lacks fails the request instead of being ignored.
    WGPUFeatureName bogus = (WGPUFeatureName)0x7000;
    WGPUDeviceDescriptor bad = WGPU_DEVICE_DESCRIPTOR_INIT;
    bad.requiredFeatureCount = 1;
    bad.requiredFeatures = &bogus;
    WGPUDevice none = (WGPUDevice)1;
    WGPURequestDeviceCallbackInfo callback = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    callback.mode = WGPUCallbackMode_WaitAnyOnly;
    callback.callback = on_device_quiet;
    callback.userdata1 = &none;
    WGPUAdapter third_adapter = second_adapter;
    if (!wgpuHasInstanceFeature(WGPUInstanceFeatureName_MultipleDevicesPerAdapter))
        third_adapter = wt_request_adapter(instance);
    wt_wait(instance, wgpuAdapterRequestDevice(third_adapter, &bad, callback));
    CHECK(none == NULL);
    printf("unsupported feature: request failed\n");

    if (third_adapter != second_adapter) wgpuAdapterRelease(third_adapter);
    if (second_adapter != adapter) wgpuAdapterRelease(second_adapter);
    wgpuSupportedFeaturesFreeMembers(adapter_features);
    wgpuDeviceRelease(second);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(device);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);
    printf("ok\n");
    return 0;
}
