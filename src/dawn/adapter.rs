//! `WGPUAdapter`: limits, and the info the shim owns as a blob.

use super::ffi::{self, Kind};
use super::lower_gen::{raise_compatibility_mode_limits, raise_limits};
use super::{Extra, Host, Native, text};
use crate::abi::enums::{FeatureName, Status};
use crate::abi::structs::{CompatibilityModeLimits, Limits};
use crate::abi::{AbiEnum, Cx, GuestMemory, GuestWrite, Handle, OutPtr};

pub(crate) struct AdapterState {
    /// Dawn's adapter, borrowed from the owning [`Native`].
    pub(crate) raw: ffi::WGPUAdapter,
    /// The adapter info, encoded for the shim.
    pub(crate) info: Vec<u8>,
}

// SAFETY: Dawn adapters are thread-safe; `raw` is owned by the Native.
unsafe impl Send for AdapterState {}
unsafe impl Sync for AdapterState {}

impl AdapterState {
    /// Takes Dawn's reference to `raw`, which the owning [`Native`] releases.
    pub(crate) fn new(raw: ffi::WGPUAdapter) -> Self {
        let mut info = ffi::WGPUAdapterInfo::default();
        // SAFETY: a live adapter and a struct for Dawn to fill.
        unsafe { ffi::wgpuAdapterGetInfo(raw, &mut info) };
        let encoded = encode_info(&info);
        // SAFETY: what Dawn filled in.
        unsafe { ffi::wgpuAdapterInfoFreeMembers(info) };
        Self { raw, info: encoded }
    }

    pub(crate) fn of(object: &Native) -> Option<&AdapterState> {
        match &object.extra {
            Extra::Adapter(state) => Some(state),
            _ => None,
        }
    }
}

/// Serializes adapter info for the shim, which owns the strings it hands out:
///
/// ```text
/// u32 backendType, adapterType, vendorID, deviceID, subgroupMinSize, subgroupMaxSize;
/// u32 vendorLength, architectureLength, deviceLength, descriptionLength;
/// u8  strings[] (in that order, no terminators)
/// ```
pub(crate) fn encode_info(info: &ffi::WGPUAdapterInfo) -> Vec<u8> {
    // SAFETY: views Dawn filled in, alive until FreeMembers.
    let strings = unsafe {
        [
            text(info.vendor),
            text(info.architecture),
            text(info.device),
            text(info.description),
        ]
    };
    let mut out = Vec::new();
    for value in [
        info.backendType,
        info.adapterType,
        info.vendorID,
        info.deviceID,
        info.subgroupMinSize,
        info.subgroupMaxSize,
    ]
    .into_iter()
    .chain(strings.iter().map(|text| text.len() as u32))
    {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for text in &strings {
        out.extend_from_slice(text.as_bytes());
    }
    out
}

/// The features webgpu.h knows of a feature list Dawn filled in.
pub(crate) fn standard_features(supported: ffi::WGPUSupportedFeatures) -> Vec<u32> {
    let mut out = Vec::new();
    if !supported.features.is_null() {
        // SAFETY: `featureCount` values at `features`.
        let features =
            unsafe { std::slice::from_raw_parts(supported.features, supported.featureCount) };
        out.extend(
            features
                .iter()
                .filter(|&&feature| FeatureName::from_u32(feature).is_some())
                .copied(),
        );
    }
    // SAFETY: what Dawn filled in.
    unsafe { ffi::wgpuSupportedFeaturesFreeMembers(supported) };
    out
}

/// Fills a guest `WGPULimits`, and a `WGPUCompatibilityModeLimits` chained to
/// it, from what `get` writes into native ones.
pub(crate) fn write_limits(
    mem: &GuestMemory<'_>,
    out: OutPtr<Limits>,
    get: impl FnOnce(*mut ffi::WGPULimits) -> u32,
) -> Status {
    if out.is_null() {
        return Status::Error;
    }
    // Reading first validates the chain the guest hung off the struct.
    let Ok(guest) = out.read(mem) else {
        return Status::Error;
    };
    let compat_ptr = guest.chain.link(CompatibilityModeLimits::S_TYPE);
    let mut compat = ffi::WGPUCompatibilityModeLimits::default();
    compat.chain.sType = CompatibilityModeLimits::S_TYPE;
    let mut native = ffi::WGPULimits::default();
    if compat_ptr.is_some() {
        native.nextInChain = (&mut compat as *mut ffi::WGPUCompatibilityModeLimits).cast();
    }
    if get(&mut native) != Status::Success.as_u32() {
        return Status::Error;
    }
    let mut limits = raise_limits(&native);
    limits.chain = guest.chain.clone();
    if out.write(mem, &limits).is_err() {
        return Status::Error;
    }
    if let Some(ptr) = compat_ptr
        && raise_compatibility_mode_limits(&compat)
            .write(mem, ptr)
            .is_err()
    {
        return Status::Error;
    }
    Status::Success
}

impl Host {
    pub(crate) fn manual_adapter_get_limits(
        &self,
        cx: &mut Cx<'_>,
        adapter: Handle,
        limits: OutPtr<Limits>,
    ) -> Status {
        let Some(object) = self.object(adapter, Kind::Adapter) else {
            return Status::Error;
        };
        write_limits(&cx.mem, limits, |native| {
            // SAFETY: a live adapter and a struct for Dawn to fill.
            unsafe { ffi::wgpuAdapterGetLimits(object.raw.cast(), native) }
        })
    }

    pub(crate) fn adapter_info_blob(&self, adapter: Handle) -> Option<Vec<u8>> {
        let object = self.object(adapter, Kind::Adapter)?;
        AdapterState::of(&object).map(|state| state.info.clone())
    }

    pub(crate) fn adapter_features(&self, adapter: Handle) -> Option<Vec<u32>> {
        let object = self.object(adapter, Kind::Adapter)?;
        let mut supported = ffi::WGPUSupportedFeatures::default();
        // SAFETY: a live adapter and a struct for Dawn to fill.
        unsafe { ffi::wgpuAdapterGetFeatures(object.raw.cast(), &mut supported) };
        Some(standard_features(supported))
    }
}
