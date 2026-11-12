//! `WGPUInstance` and the global functions.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use super::{Host, Object, ObjectKind};
use crate::abi::api::{GlobalApi, InstanceApi};
use crate::abi::enums::{InstanceFeatureName, Status, WGSLLanguageFeatureName};
use crate::abi::structs::{InstanceDescriptor, InstanceLimits, SurfaceDescriptor};
use crate::abi::{AbiEnum, Cx, Handle, OutPtr};

/// Most futures one timed `wgpuInstanceWaitAny` may wait on.
pub(crate) const TIMED_WAIT_ANY_MAX_COUNT: u32 = 64;

pub(crate) struct InstanceState {
    pub(crate) instance: wgpu::Instance,
    pub(crate) active: AtomicBool,
}

/// The WGSL language features of this backend, as C enum values.
pub(crate) fn wgsl_language_features(instance: &wgpu::Instance) -> Vec<WGSLLanguageFeatureName> {
    use wgpu::WgslLanguageFeatures as W;
    let have = instance.wgsl_language_features();
    [
        (
            W::ReadOnlyAndReadWriteStorageTextures,
            WGSLLanguageFeatureName::ReadonlyAndReadwriteStorageTextures,
        ),
        (
            W::Packed4x8IntegerDotProduct,
            WGSLLanguageFeatureName::Packed4x8IntegerDotProduct,
        ),
        (
            W::UnrestrictedPointerParameters,
            WGSLLanguageFeatureName::UnrestrictedPointerParameters,
        ),
        (
            W::PointerCompositeAccess,
            WGSLLanguageFeatureName::PointerCompositeAccess,
        ),
    ]
    .into_iter()
    .filter_map(|(flag, name)| have.contains(flag).then_some(name))
    .collect()
}

/// Whether this host has an instance feature of webgpu.h.
///
/// Timed waits (a guest thread can block here) and several devices per
/// adapter, yes. SPIR-V shader sources, no: a browser takes only WGSL, and a
/// program that runs on this host is meant to run there too.
fn has_instance_feature(feature: u32) -> bool {
    matches!(
        InstanceFeatureName::from_u32(feature),
        Some(InstanceFeatureName::TimedWaitAny | InstanceFeatureName::MultipleDevicesPerAdapter)
    )
}

impl GlobalApi for Host {
    fn create_instance(&self, _cx: &mut Cx<'_>, descriptor: Option<&InstanceDescriptor>) -> Handle {
        if let Some(descriptor) = descriptor {
            if let Some(missing) = descriptor
                .required_features
                .iter()
                .find(|&&feature| !has_instance_feature(feature))
            {
                super::log(|| {
                    format!("wgpuCreateInstance: instance feature {missing:#x} is not available")
                });
                return 0;
            }
            if let Some(limits) = &descriptor.required_limits
                && limits.timed_wait_any_max_count > TIMED_WAIT_ANY_MAX_COUNT
            {
                return 0;
            }
        }

        let mut options = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        // The backend's own validation always runs. Driver debug layers are
        // an embedder's debugging aid, never something a guest pays for.
        options.flags = wgpu::InstanceFlags::empty().with_env();
        let instance = wgpu::Instance::new(options);
        self.insert(
            Object::Instance(Arc::new(InstanceState {
                instance,
                active: AtomicBool::new(true),
            })),
            0,
        )
    }

    fn get_instance_limits(&self, cx: &mut Cx<'_>, limits: OutPtr<InstanceLimits>) -> Status {
        if limits.is_null() {
            return Status::Error;
        }
        let value = InstanceLimits {
            chain: Default::default(),
            timed_wait_any_max_count: TIMED_WAIT_ANY_MAX_COUNT,
        };
        match limits.write(&cx.mem, &value) {
            Ok(()) => Status::Success,
            Err(_) => Status::Error,
        }
    }

    fn has_instance_feature(&self, _cx: &mut Cx<'_>, feature: u32) -> bool {
        has_instance_feature(feature)
    }
}

impl InstanceApi for Host {
    fn instance_create_surface(
        &self,
        _cx: &mut Cx<'_>,
        instance: Handle,
        descriptor: &SurfaceDescriptor,
    ) -> Handle {
        self.create_surface(instance, descriptor)
    }

    fn instance_has_wgsl_language_feature(
        &self,
        _cx: &mut Cx<'_>,
        instance: Handle,
        feature: u32,
    ) -> bool {
        let Some(instance) = self.instance(instance) else {
            return false;
        };
        WGSLLanguageFeatureName::from_u32(feature)
            .is_some_and(|feature| wgsl_language_features(&instance.instance).contains(&feature))
    }

    fn instance_add_ref(&self, _cx: &mut Cx<'_>, instance: Handle) {
        self.add_ref(instance, ObjectKind::Instance);
    }
}
