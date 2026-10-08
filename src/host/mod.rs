//! The native host: every WebGPU import, implemented on [`wgpu`].
//!
//! One [`Host`] exists per guest process. It owns the handle table that maps
//! the guest's opaque `WGPU*` handles to wgpu objects, and the queue of
//! completed events the guest shim drains. Each object of the C API has its
//! own file implementing the matching generated `*Api` trait.

mod adapter;
mod bind;
mod buffer;
mod convert;
mod device;
mod encoder;
mod events;
mod instance;
mod pass;
mod pipeline;
mod queue;
mod scopes;
mod shader;
mod shim;
mod surface;
mod table;
mod texture;

use std::any::Any;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use wasmer::{FunctionEnvMut, RuntimeError, StoreMut};

use crate::abi::enums::ErrorType;
use crate::abi::{AbiError, AbiResult, Cx, GuestMemory, Handle};
use crate::ctx::{CtxInner, WebGpuCtx};
use crate::env::WebGpuEnv;

pub(crate) use events::{Events, Record};
pub(crate) use table::{Object, ObjectKind};

pub(crate) use adapter::AdapterState;
pub(crate) use buffer::BufferState;
pub(crate) use device::DeviceState;
pub(crate) use instance::InstanceState;
pub(crate) use surface::SurfaceState;
pub(crate) use texture::TextureState;

/// Locks a mutex, ignoring poisoning: a panic inside one import is turned into
/// a guest trap by [`enter`], and must not take the rest of the process's GPU
/// state down with it.
pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn logging() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("WASMER_WEBGPU_LOG").is_some())
}

/// Diagnostics for embedders, enabled with `WASMER_WEBGPU_LOG=1`. Guests learn
/// about their mistakes through WebGPU's own error reporting, not from this.
pub(crate) fn log(message: impl FnOnce() -> String) {
    if logging() {
        eprintln!("[wasmer-webgpu] {}", message());
    }
}

/// Futureless events (uncaptured errors) kept for a guest that does not
/// collect them, unless the embedder says otherwise.
const DEFAULT_MAX_PENDING_EVENTS: usize = 4096;

/// What the embedder-facing context needs from a guest process's host.
pub(crate) trait HostControl: Any + Send + Sync {
    fn terminate(&self);
    /// (devices, live object handles, bytes of buffer and texture storage)
    fn usage(&self) -> (usize, usize, u64);
}

/// Host state of every guest process of one [`WebGpuCtx`].
///
/// Held as trait objects so that the context's own `Send`/`Sync` do not
/// depend on wgpu's object graph (a proof deep enough to need a raised
/// recursion limit, which embedders should not have to copy).
#[derive(Default)]
pub(crate) struct HostRegistry {
    /// Threads of a process share one linear memory, and every instance of the
    /// process must reach the same [`Host`]: keyed by the memory's base.
    by_memory: HashMap<usize, Weak<dyn Any + Send + Sync>>,
    all: Vec<Weak<dyn HostControl>>,
}

impl HostRegistry {
    pub(crate) fn live(&mut self) -> Vec<Arc<dyn HostControl>> {
        self.all.retain(|host| host.strong_count() > 0);
        self.by_memory.retain(|_, host| host.strong_count() > 0);
        self.all.iter().filter_map(Weak::upgrade).collect()
    }
}

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
    terminated: AtomicBool,
}

impl Host {
    fn new(ctx: Arc<CtxInner>) -> Self {
        let events = Arc::new(Events::new(
            ctx.limits
                .max_pending_events
                .or(Some(DEFAULT_MAX_PENDING_EVENTS)),
        ));
        Self {
            table: Mutex::new(table::Table::new(ctx.limits.max_objects)),
            events,
            devices: Mutex::new(Vec::new()),
            next_device_id: AtomicU64::new(1),
            gpu_memory: Arc::new(AtomicU64::new(0)),
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
        let shared = memory.ty(store).shared;
        let base = memory.view(store).data_ptr() as usize;
        let mut registry = lock(&ctx.inner.hosts);
        registry.live();
        if shared
            && base != 0
            && let Some(host) = registry.by_memory.get(&base).and_then(Weak::upgrade)
            && let Ok(host) = host.downcast::<Self>()
        {
            return host;
        }
        let host = Arc::new(Self::new(Arc::clone(&ctx.inner)));
        if shared && base != 0 {
            let any: Arc<dyn Any + Send + Sync> = host.clone();
            registry.by_memory.insert(base, Arc::downgrade(&any));
        }
        let control: Arc<dyn HostControl> = host.clone();
        registry.all.push(Arc::downgrade(&control));
        host
    }

    pub(crate) fn is_terminated(&self) -> bool {
        self.terminated.load(Ordering::Acquire) || self.ctx.stopped.load(Ordering::Acquire)
    }

    /// Stops the process's GPU work: wakes waiters and destroys every device.
    pub(crate) fn terminate(&self) {
        self.terminated.store(true, Ordering::Release);
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

thread_local! {
    /// Set while this thread runs a call whose panic is an expected outcome.
    static EXPECTING_PANIC: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f`, turning a panic into `Err(message)` without the panic hook
/// printing it.
///
/// wgpu reports a few validation failures by panicking. Where the host knows
/// such a call can fail on guest input, the failure is an ordinary WebGPU
/// error, and must not show up on the embedder's stderr as a crash report.
pub(crate) fn catch_expected_panic<R>(f: impl FnOnce() -> R) -> Result<R, String> {
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !EXPECTING_PANIC.get() {
                previous(info);
            }
        }));
    });
    EXPECTING_PANIC.set(true);
    let result = catch_unwind(AssertUnwindSafe(f));
    EXPECTING_PANIC.set(false);
    result.map_err(|panic| panic_message(&*panic))
}

fn panic_message(panic: &(dyn Any + Send)) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_owned())
}

/// Runs one import: resolves the calling instance's memory and host, decodes
/// through `f`, and turns failures into the right kind of guest-visible
/// outcome.
///
/// - A guest pointer outside linear memory traps, like the segfault it is,
///   whatever the implementation did with the failed copy.
/// - Any other decoding error makes the call return its zero value.
/// - A panic below (wgpu panics on a few misuse paths) traps this call
///   instead of unwinding through the wasm frames into the embedder.
pub(crate) fn enter<R: Default>(
    env: &mut FunctionEnvMut<'_, WebGpuEnv>,
    function: &'static str,
    f: impl FnOnce(&Host, &mut Cx<'_>) -> AbiResult<R>,
) -> Result<R, RuntimeError> {
    let (data, store) = env.data_and_store_mut();
    let (Some(host), Some(memory)) = (data.host.clone(), data.memory.as_ref()) else {
        return Err(RuntimeError::new(format!(
            "{function}: WebGPU was called before the instance was set up"
        )));
    };
    let mut cx = Cx {
        mem: GuestMemory::new(memory.view(&store)),
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| f(&host, &mut cx)));
    // Checked before anything else: an implementation may have turned a
    // failed copy into a status of its own, and a wild pointer still traps.
    if let Some(fault) = cx.mem.take_fault() {
        return Err(RuntimeError::new(format!("{function}: {fault}")));
    }
    match outcome {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error @ AbiError::OutOfBounds { .. })) => {
            Err(RuntimeError::new(format!("{function}: {error}")))
        }
        Ok(Err(error)) => {
            host.abi_error(function, &error);
            Ok(R::default())
        }
        Err(panic) => {
            let message = panic_message(&*panic);
            log(|| format!("{function} panicked: {message}"));
            Err(RuntimeError::new(format!(
                "{function}: the WebGPU backend failed: {message}"
            )))
        }
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
            super::DEFAULT_MAX_PENDING_EVENTS as u64
        );
        // The event record's header, which the shim parses too.
        let header = format!("size: {} }}", super::events::RECORD_HEADER_SIZE);
        assert!(
            BROWSER_HOST.contains(&header),
            "the browser host's event record header is not {} bytes",
            super::events::RECORD_HEADER_SIZE
        );
    }
}
