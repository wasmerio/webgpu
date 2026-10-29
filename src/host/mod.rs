//! The native host: every WebGPU import, implemented on [`wgpu`].
//!
//! One [`Host`] exists per guest process. It owns the handle table that maps
//! the guest's opaque `WGPU*` handles to wgpu objects, and the queue of
//! completed events the guest shim drains. Each object of the C API has its
//! own file implementing the matching generated `*Api` trait.

mod adapter;
mod async_pipeline;
mod bind;
mod buffer;
mod convert;
mod device;
mod encoder;
mod instance;
mod pass;
mod pipeline;
mod pipeline_worker;
mod queue;
mod scopes;
mod shader;
mod shim;
mod surface;
mod table;
mod texture;
mod upload;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use wasmer::StoreMut;

use crate::abi::enums::ErrorType;
use crate::abi::{AbiError, Handle};
use crate::ctx::{CtxInner, WebGpuCtx};
use crate::native::{DEFAULT_MAX_PENDING_EVENTS, HostControl, host_for_memory};

pub(crate) use crate::native::events::{Events, Record};
pub(crate) use crate::native::{catch_expected_panic, lock, log};
pub(crate) use table::{Object, ObjectKind};

pub(crate) use adapter::AdapterState;
pub(crate) use buffer::BufferState;
pub(crate) use device::DeviceState;
pub(crate) use instance::InstanceState;
pub(crate) use surface::SurfaceState;
pub(crate) use texture::TextureState;

/// Async pipeline creations a process's compiler lane holds at once: queued,
/// compiling, or finished but not collected yet. Beyond it, a creation is
/// compiled on the calling thread.
const PIPELINE_LANE_CAPACITY: usize = 64;

/// Everything the WebGPU imports of one guest process share.
pub(crate) struct Host {
    pub(crate) ctx: Arc<CtxInner>,
    table: Mutex<table::Table>,
    pub(crate) events: Arc<Events>,
    /// Every device of the process, for polling.
    devices: Mutex<Vec<Weak<DeviceState>>>,
    next_device_id: AtomicU64,
    /// Bytes of buffer and texture storage currently allocated.
    pub(crate) gpu_memory: Arc<AtomicU64>,
    /// What the staging buffers of large texture uploads may hold.
    pub(crate) staging: Arc<upload::StagingBudget>,
    terminated: AtomicBool,
    pipelines: pipeline_worker::Worker<async_pipeline::Request, async_pipeline::Outcome>,
}

impl Host {
    fn new(ctx: Arc<CtxInner>) -> Self {
        let events = Arc::new(Events::with_waker(
            ctx.limits
                .max_pending_events
                .or(Some(DEFAULT_MAX_PENDING_EVENTS)),
            ctx.event_waker.clone(),
        ));
        let notify = Arc::clone(&events);
        let pipelines = pipeline_worker::Worker::new(
            PIPELINE_LANE_CAPACITY,
            Arc::new(move || notify.notify_work()),
        );
        Self {
            table: Mutex::new(table::Table::new(ctx.limits.max_objects)),
            pipelines,
            events,
            devices: Mutex::new(Vec::new()),
            next_device_id: AtomicU64::new(1),
            gpu_memory: Arc::new(AtomicU64::new(0)),
            staging: Arc::new(upload::StagingBudget::new(ctx.limits.max_gpu_memory_bytes)),
            terminated: AtomicBool::new(ctx.stopped.load(Ordering::Acquire)),
            ctx,
        }
    }

    /// The host of the process `memory` belongs to, created on first use.
    pub(crate) fn for_memory(
        ctx: &WebGpuCtx,
        store: &mut StoreMut<'_>,
        memory: &wasmer::Memory,
    ) -> Arc<Self> {
        host_for_memory(ctx, store, memory, || Self::new(Arc::clone(&ctx.inner)))
    }

    pub(crate) fn is_terminated(&self) -> bool {
        self.terminated.load(Ordering::Acquire) || self.ctx.stopped.load(Ordering::Acquire)
    }

    /// Stops the process's GPU work: wakes waiters and destroys every device.
    pub(crate) fn terminate(&self) {
        self.terminated.store(true, Ordering::Release);
        self.pipelines.terminate();
        for device in self.live_devices() {
            device.destroy();
        }
        self.events.wake_all();
    }

    pub(crate) fn register_device(&self, device: &Arc<DeviceState>) {
        let mut devices = lock(&self.devices);
        devices.retain(|device| device.strong_count() > 0);
        devices.push(Arc::downgrade(device));
    }

    pub(crate) fn live_devices(&self) -> Vec<Arc<DeviceState>> {
        let mut devices = lock(&self.devices);
        devices.retain(|device| device.strong_count() > 0);
        devices.iter().filter_map(Weak::upgrade).collect()
    }

    pub(crate) fn next_device_id(&self) -> u64 {
        self.next_device_id.fetch_add(1, Ordering::Relaxed)
    }

    // -- handle table ---------------------------------------------------------

    /// Stores `object` and returns the guest's (single) reference to it, or
    /// the null handle when the process is at its object limit.
    pub(crate) fn insert(&self, object: Object, instance: Handle) -> Handle {
        let kind = object.kind();
        let handle = lock(&self.table).insert(object, instance);
        if handle == 0 {
            log(|| format!("object limit reached; refusing to create a {kind:?}"));
        }
        handle
    }

    pub(crate) fn object(&self, handle: Handle) -> Option<Object> {
        lock(&self.table).get(handle)
    }

    /// The instance an object descends from (itself, for an instance).
    pub(crate) fn instance_of(&self, handle: Handle) -> Handle {
        lock(&self.table).instance_of(handle)
    }

    pub(crate) fn add_ref(&self, handle: Handle, kind: ObjectKind) {
        if !lock(&self.table).add_ref(handle, kind) {
            log(|| format!("AddRef on an invalid {kind:?} handle {handle:#x}"));
        }
    }

    /// Drops one guest reference. Returns the object when that was the last
    /// one, so the caller decides what destroying it means.
    pub(crate) fn release(&self, handle: Handle, kind: ObjectKind) -> Option<Object> {
        match lock(&self.table).release(handle, kind) {
            Ok(removed) => removed,
            Err(()) => {
                log(|| format!("Release on an invalid {kind:?} handle {handle:#x}"));
                None
            }
        }
    }

    // -- errors ---------------------------------------------------------------

    /// Reports a WebGPU error on `device`: to the calling thread's innermost
    /// matching error scope, or else as an uncaptured-error event.
    pub(crate) fn error(&self, device: &DeviceState, ty: ErrorType, message: impl Into<String>) {
        device.report(ty, message.into());
    }

    /// Reports a validation error on `device`.
    pub(crate) fn validation(&self, device: &DeviceState, message: impl Into<String>) {
        self.error(device, ErrorType::Validation, message);
    }

    /// A decoding failure that is the guest's bug (an out-of-range enum, an
    /// sType that does not belong in a chain). There is no device to blame it
    /// on in general, so it is only logged; the call returns its error value.
    fn abi_error(&self, function: &'static str, error: &AbiError) {
        log(|| format!("{function}: {error}"));
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        // Worker::drop is nonblocking and owns no Host/guest state. Its native
        // inputs survive until any in-flight driver call exits and discards.
        self.terminated.store(true, Ordering::Release);
        self.pipelines.close();
    }
}

impl HostControl for Host {
    fn terminate(&self) {
        Host::terminate(self);
    }

    fn usage(&self) -> (usize, usize, u64) {
        (
            self.live_devices().len(),
            lock(&self.table).live(),
            self.gpu_memory.load(Ordering::Relaxed),
        )
    }
}

/// The browser host (`js/webgpu_host.js`) is a second implementation of this
/// one. What it shares with it by layout comes from the generated tables;
/// these few numbers it writes out by hand, and they must not drift.
#[cfg(test)]
mod parity {
    const BROWSER_HOST: &str = include_str!("../../js/webgpu_host.js");

    /// The value of `const NAME = <expression>;` in the browser host, for
    /// the expressions it uses: integers and one shift.
    fn browser_constant(name: &str) -> u64 {
        let declaration = format!("const {name} = ");
        let start = BROWSER_HOST
            .find(&declaration)
            .unwrap_or_else(|| panic!("webgpu_host.js has no `{name}`"))
            + declaration.len();
        let expression = BROWSER_HOST[start..]
            .split(';')
            .next()
            .expect("the declaration ends")
            .trim();
        let number = |text: &str| -> u64 {
            let text = text.trim().replace('_', "");
            match text.strip_prefix("0x") {
                Some(hex) => u64::from_str_radix(hex, 16),
                None => text.parse(),
            }
            .unwrap_or_else(|_| panic!("`{name} = {expression}` is not a number this test reads"))
        };
        match expression.split_once("<<") {
            Some((value, shift)) => number(value) << number(shift),
            None => number(expression),
        }
    }

    #[test]
    fn the_browser_host_agrees_on_the_numbers_it_spells_out() {
        assert_eq!(
            browser_constant("ABI_VERSION"),
            u64::from(crate::WEBGPU_ABI_VERSION)
        );
        assert_eq!(
            browser_constant("MAX_ARRAY_LENGTH"),
            u64::from(crate::abi::MAX_ARRAY_LENGTH)
        );
        assert_eq!(
            browser_constant("MAX_STRING_BYTES"),
            u64::from(crate::abi::MAX_STRING_BYTES)
        );
        assert_eq!(
            browser_constant("MAX_CHAIN_LENGTH"),
            crate::abi::structs::MAX_CHAIN_LENGTH as u64
        );
        assert_eq!(
            browser_constant("TIMED_WAIT_ANY_MAX_COUNT"),
            u64::from(super::instance::TIMED_WAIT_ANY_MAX_COUNT)
        );
        assert_eq!(
            browser_constant("DEFAULT_MAX_PENDING_EVENTS"),
            crate::native::DEFAULT_MAX_PENDING_EVENTS as u64
        );
        // The event record's header, which the shim parses too.
        let header = format!("size: {} }}", crate::native::events::RECORD_HEADER_SIZE);
        assert!(
            BROWSER_HOST.contains(&header),
            "the browser host's event record header is not {} bytes",
            crate::native::events::RECORD_HEADER_SIZE
        );
    }
}
