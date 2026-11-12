//! `WGPUAdapter`.

use super::{Host, ObjectKind, convert};
use crate::abi::api::AdapterApi;
use crate::abi::enums::{FeatureName, Status};
use crate::abi::structs::{CompatibilityModeLimits, Limits};
use crate::abi::{AbiEnum, Cx, GuestMemory, GuestWrite, Handle, OutPtr};

pub(crate) struct AdapterState {
    pub(crate) adapter: wgpu::Adapter,
    pub(crate) info: wgpu::AdapterInfo,
}

impl AdapterState {
    pub(crate) fn new(adapter: wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        Self { adapter, info }
    }
}

/// Fills a guest `WGPULimits` and the limit structs chained to it.
pub(crate) fn write_limits(
    mem: &GuestMemory<'_>,
    out: OutPtr<Limits>,
    limits: &wgpu::Limits,
) -> Status {
    if out.is_null() {
        return Status::Error;
    }
    // Reading first validates the chain the guest hung off the struct.
    let Ok(guest) = out.read(mem) else {
        return Status::Error;
    };
    if out.write(mem, &convert::limits_to_abi(limits)).is_err() {
        return Status::Error;
    }
    if let Some(ptr) = guest.chain.link(CompatibilityModeLimits::S_TYPE) {
        // Compatibility mode is not something this backend offers, so its
        // extra limits take the values of the core limits they refine.
        let compat = CompatibilityModeLimits {
            max_storage_buffers_in_vertex_stage: limits.max_storage_buffers_per_shader_stage,
            max_storage_textures_in_vertex_stage: limits.max_storage_textures_per_shader_stage,
            max_storage_buffers_in_fragment_stage: limits.max_storage_buffers_per_shader_stage,
            max_storage_textures_in_fragment_stage: limits.max_storage_textures_per_shader_stage,
        };
        if compat.write(mem, ptr).is_err() {
            return Status::Error;
        }
    }
    Status::Success
}

/// Serializes adapter info for the shim, which owns the strings it hands out:
///
/// ```text
/// u32 backendType, adapterType, vendorID, deviceID, subgroupMinSize, subgroupMaxSize;
/// u32 vendorLength, architectureLength, deviceLength, descriptionLength;
/// u8  strings[] (in that order, no terminators)
/// ```
pub(crate) fn encode_info(info: &wgpu::AdapterInfo) -> Vec<u8> {
    let vendor = convert::vendor_name(info.vendor);
    let architecture = "";
    let device = info.name.as_str();
    let description = match (info.driver.is_empty(), info.driver_info.is_empty()) {
        (true, true) => String::new(),
        (false, true) => info.driver.clone(),
        (true, false) => info.driver_info.clone(),
        (false, false) => format!("{} ({})", info.driver, info.driver_info),
    };
    let mut out = Vec::new();
    for value in [
        convert::backend_type(info.backend).as_u32(),
        convert::adapter_type(info.device_type).as_u32(),
        info.vendor,
        info.device,
        info.subgroup_min_size,
        info.subgroup_max_size,
        vendor.len() as u32,
        architecture.len() as u32,
        device.len() as u32,
        description.len() as u32,
    ] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for text in [vendor, architecture, device, description.as_str()] {
        out.extend_from_slice(text.as_bytes());
    }
    out
}

pub(crate) fn has_feature(features: wgpu::Features, feature: u32) -> bool {
    FeatureName::from_u32(feature)
        .and_then(convert::feature_to_wgpu)
        .is_some_and(|bits| features.contains(bits))
}

impl AdapterApi for Host {
    fn adapter_get_limits(
        &self,
        cx: &mut Cx<'_>,
        adapter: Handle,
        limits: OutPtr<Limits>,
    ) -> Status {
        match self.adapter(adapter) {
            Some(adapter) => write_limits(&cx.mem, limits, &adapter.adapter.limits()),
            None => Status::Error,
        }
    }

    fn adapter_has_feature(&self, _cx: &mut Cx<'_>, adapter: Handle, feature: u32) -> bool {
        self.adapter(adapter)
            .is_some_and(|adapter| has_feature(adapter.adapter.features(), feature))
    }

    fn adapter_add_ref(&self, _cx: &mut Cx<'_>, adapter: Handle) {
        self.add_ref(adapter, ObjectKind::Adapter);
    }

    fn adapter_release(&self, _cx: &mut Cx<'_>, adapter: Handle) {
        self.release(adapter, ObjectKind::Adapter);
    }
}
