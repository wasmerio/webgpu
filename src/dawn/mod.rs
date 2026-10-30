//! The native host on [Dawn](https://dawn.googlesource.com/dawn), Chromium's
//! WebGPU implementation (the `dawn` feature).
//!
//! Dawn implements `webgpu.h` itself, at the revision `spec/` pins, so most
//! imports are forwards: decode the guest's arguments, resolve its handles,
//! lower the structs to their native layout and call the C function of the
//! same name. Those are generated (`forward.gen.rs`, `lower.gen.rs`) from
//! `spec/webgpu.json`, like the guest's side. What is written by hand is what
//! a host decides rather than forwards: futures and callbacks (they become
//! event records), mapped ranges and uploads (they copy guest memory), the
//! GPU memory budget, surfaces (the guest names a target and the embedder
//! decides what it is), and process-wide state.
//!
//! One [`Host`] exists per guest process, as with the wgpu host. Its guest
//! instances share one Dawn instance; every device is created with Dawn's
//! implicit device synchronization, because a guest's threads share devices.

mod adapter;
mod buffer;
mod callbacks;
mod dawn_ext;
mod device;
#[rustfmt::skip]
#[path = "ffi.gen.rs"]
pub(crate) mod ffi;
#[rustfmt::skip]
#[path = "forward.gen.rs"]
mod forward;
mod instance;
mod lower;
#[rustfmt::skip]
#[path = "lower.gen.rs"]
mod lower_gen;
mod queue;
mod shim;
mod surface;
#[cfg(test)]
mod tests;
mod texture;

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard, Weak};

use wasmer::StoreMut;

use crate::abi::Handle;
use crate::ctx::{CtxInner, WebGpuCtx};
use crate::native::table::{Table, TableObject};
use crate::native::{DEFAULT_MAX_PENDING_EVENTS, HostControl, host_for_memory};

pub(crate) use crate::native::events::{Events, Record};
pub(crate) use crate::native::{lock, log};

use adapter::AdapterState;
use buffer::BufferState;
pub(crate) use device::DeviceState;
use ffi::Kind;
use instance::InstanceState;
use surface::SurfaceState;
use texture::TextureState;

/// A GPU object behind a guest handle: Dawn's object, the device it belongs
/// to (for errors the host finds itself), and what the host keeps beside it.
pub(crate) struct Native {
    pub(crate) kind: Kind,
    /// One reference to Dawn's object, released when this is dropped.
    pub(crate) raw: *mut c_void,
    /// The device this object belongs to; a device's or queue's own.
    pub(crate) device: Option<Arc<DeviceState>>,
    pub(crate) extra: Extra,
}

/// What the host keeps beside some objects.
pub(crate) enum Extra {
    None,
    Instance(InstanceState),
    Adapter(AdapterState),
    Buffer(BufferState),
    Texture(TextureState),
    Surface(Arc<SurfaceState>),
}

// SAFETY: Dawn objects may be used from any thread. Every device is created
// with implicit device synchronization, which makes that true of devices and
// what they create; instances and adapters are thread-safe in Dawn.
unsafe impl Send for Native {}
unsafe impl Sync for Native {}

impl Drop for Native {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: `raw` is a live object of this kind, and this its reference.
            crate::native::autoreleased(|| unsafe { self.kind.release(self.raw) });
        }
    }
}

pub(crate) type Object = Arc<Native>;

impl TableObject for Object {
    type Kind = Kind;
    fn kind(&self) -> Kind {
        self.kind
    }
    fn is_instance(&self) -> bool {
        self.kind == Kind::Instance
    }
}

/// The Dawn instance a process's guest instances share.
pub(crate) struct DawnInstance(pub(crate) ffi::WGPUInstance);

// SAFETY: Dawn instances are thread-safe.
unsafe impl Send for DawnInstance {}
unsafe impl Sync for DawnInstance {}

impl Drop for DawnInstance {
    fn drop(&mut self) {
        // SAFETY: the host's own reference.
        crate::native::autoreleased(|| unsafe { ffi::wgpuInstanceRelease(self.0) });
    }
}

/// Everything the WebGPU imports of one guest process share.
pub(crate) struct Host {
    pub(crate) ctx: Arc<CtxInner>,
    table: Mutex<Table<Object>>,
    pub(crate) events: Arc<Events>,
    /// Every device of the process, for termination and accounting.
    devices: Mutex<Vec<Weak<DeviceState>>>,
    /// Bytes of buffer and texture storage currently allocated.
    pub(crate) gpu_memory: Arc<AtomicU64>,
    terminated: AtomicBool,
    dawn: OnceLock<Option<DawnInstance>>,
    /// Dawn futures behind the guest futures a wait may name.
    pub(crate) futures: callbacks::Futures,
    /// Pipelines Dawn compiled, waiting for a guest thread to give them handles.
    pub(crate) pipelines: shim::Pipelines,
}

impl Host {
    fn new(ctx: Arc<CtxInner>) -> Self {
        let events = Arc::new(Events::with_waker(
            ctx.limits
                .max_pending_events
                .or(Some(DEFAULT_MAX_PENDING_EVENTS)),
            ctx.event_waker.clone(),
        ));
        Self {
            table: Mutex::new(Table::new(ctx.limits.max_objects)),
            events,
            devices: Mutex::new(Vec::new()),
            gpu_memory: Arc::new(AtomicU64::new(0)),
            terminated: AtomicBool::new(ctx.stopped.load(Ordering::Acquire)),
            dawn: OnceLock::new(),
            futures: callbacks::Futures::default(),
            pipelines: Default::default(),
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

    /// The process's Dawn instance, made on first use. None if Dawn could not
    /// make one.
    pub(crate) fn dawn(&self) -> Option<ffi::WGPUInstance> {
        self.dawn
            .get_or_init(instance::create_dawn_instance)
            .as_ref()
            .map(|instance| instance.0)
    }

    pub(crate) fn is_terminated(&self) -> bool {
        self.terminated.load(Ordering::Acquire) || self.ctx.stopped.load(Ordering::Acquire)
    }

    /// Stops the process's GPU work: destroys every device and wakes waiters.
    pub(crate) fn terminate(&self) {
        self.terminated.store(true, Ordering::Release);
        // Killing never waits for the GPU: a device still compiling is
        // destroyed once it is done (`DeviceState::destroy`).
        for device in self.live_devices() {
            device.destroy(std::time::Duration::ZERO);
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

    // -- handle table ---------------------------------------------------------

    /// Stores `object` and returns the guest's (single) reference to it, or
    /// the null handle when the process is at its object limit (the object
    /// is then dropped, which releases it).
    pub(crate) fn insert(&self, object: Object, instance: Handle) -> Handle {
        let kind = object.kind;
        let handle = lock(&self.table).insert(object, instance);
        if handle == 0 {
            log(|| format!("object limit reached; refusing to create a {kind:?}"));
        }
        handle
    }

    /// The object behind `handle`, if it is live and of this kind.
    pub(crate) fn object(&self, handle: Handle, kind: Kind) -> Option<Object> {
        match lock(&self.table).get(handle) {
            Some(object) if object.kind == kind => Some(object),
            other => {
                log(|| {
                    format!(
                        "expected a {kind:?} handle, got {handle:#x} ({})",
                        match &other {
                            Some(object) => format!("a {:?}", object.kind),
                            None if handle == 0 => "null".to_owned(),
                            None => "not a live handle".to_owned(),
                        }
                    )
                });
                None
            }
        }
    }

    /// The instance an object descends from (itself, for an instance).
    pub(crate) fn instance_of(&self, handle: Handle) -> Handle {
        lock(&self.table).instance_of(handle)
    }

    /// Registers an object Dawn created from `parent`: it belongs to the
    /// parent's device and descends from the parent's instance. A null
    /// object (Dawn could not make even an error object) is the null handle.
    pub(crate) fn adopt(
        &self,
        kind: Kind,
        raw: *mut c_void,
        parent: &Object,
        parent_handle: Handle,
    ) -> Handle {
        if raw.is_null() {
            return 0;
        }
        let object = Arc::new(Native {
            kind,
            raw,
            device: parent.device.clone(),
            extra: Extra::None,
        });
        self.insert(object, self.instance_of(parent_handle))
    }

    /// Adds a guest reference. Returns whether `handle` was a live object of
    /// this kind.
    pub(crate) fn add_ref_handle(&self, handle: Handle, kind: Kind) -> bool {
        let added = lock(&self.table).add_ref(handle, kind);
        if !added {
            log(|| format!("AddRef on an invalid {kind:?} handle {handle:#x}"));
        }
        added
    }

    /// Drops one guest reference. Returns the object when that was the last
    /// one; dropping it (outside the table's lock) releases Dawn's object.
    pub(crate) fn release_handle(&self, handle: Handle, kind: Kind) -> Option<Object> {
        let released = lock(&self.table).release(handle, kind);
        match released {
            Ok(removed) => removed,
            Err(()) => {
                log(|| format!("Release on an invalid {kind:?} handle {handle:#x}"));
                None
            }
        }
    }

    // -- typed views ----------------------------------------------------------

    pub(crate) fn device(&self, handle: Handle) -> Option<Arc<DeviceState>> {
        self.object(handle, Kind::Device)
            .and_then(|object| object.device.clone())
    }
}

impl HostControl for Host {
    fn terminate(&self) {
        Host::terminate(self);
    }

    fn usage(&self) -> (usize, usize, u64) {
        (
            self.live_devices()
                .iter()
                .filter(|device| !device.is_lost())
                .count(),
            lock(&self.table).live(),
            self.gpu_memory.load(Ordering::Relaxed),
        )
    }
}

/// How long a guest's destroy of a device waits for Dawn to finish the
/// pipelines it is compiling (`DeviceState::destroy`).
pub(crate) const PIPELINE_CALLBACKS_WAIT: std::time::Duration = std::time::Duration::from_secs(10);

impl Drop for Host {
    fn drop(&mut self) {
        self.terminated.store(true, Ordering::Release);
        // A pipeline Dawn is still compiling keeps its device, and the list
        // of compiled pipelines, until its callback runs; the last callback
        // drops them on a thread of its own (`callbacks::pipeline`).
    }
}

/// Takes a lock shared, ignoring poisoning (see [`lock`]).
pub(crate) fn shared(lock: &RwLock<()>) -> RwLockReadGuard<'_, ()> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Takes a lock exclusively, ignoring poisoning (see [`lock`]).
pub(crate) fn exclusive(lock: &RwLock<()>) -> RwLockWriteGuard<'_, ()> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Counts `bytes` against the guest's GPU memory, saturating.
pub(crate) fn charge(counter: &AtomicU64, bytes: u64) {
    let _ = counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
        Some(used.saturating_add(bytes))
    });
}

/// Gives `bytes` back, saturating.
pub(crate) fn refund(counter: &AtomicU64, bytes: u64) {
    let _ = counter.try_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
        Some(used.saturating_sub(bytes))
    });
}

/// A string view of `text` for a call that returns before `text` goes.
pub(crate) fn view(text: &str) -> ffi::WGPUStringView {
    ffi::WGPUStringView {
        data: text.as_ptr().cast(),
        length: text.len(),
    }
}

/// The text of a string view Dawn handed out.
///
/// # Safety
/// `view` must be one Dawn made, valid for the duration of the call.
pub(crate) unsafe fn text(view: ffi::WGPUStringView) -> String {
    if view.data.is_null() {
        return String::new();
    }
    let bytes = if view.length == ffi::WGPU_STRLEN {
        // SAFETY: a NUL-terminated string, as the header says of WGPU_STRLEN.
        unsafe { std::ffi::CStr::from_ptr(view.data) }.to_bytes()
    } else {
        // SAFETY: `length` bytes at `data`.
        unsafe { std::slice::from_raw_parts(view.data.cast::<u8>(), view.length) }
    };
    String::from_utf8_lossy(bytes).into_owned()
}
