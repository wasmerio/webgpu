//! What the native hosts share, whichever GPU library implements WebGPU for
//! them: running an import ([`enter`]), the registry of guest processes, the
//! queue of completed events, and the handle table.
//!
//! A build has one native host, `crate::host`: the wgpu one by default, the
//! Dawn one with the `dawn` feature. Code here names it only through
//! `crate::host::Host`.

pub(crate) mod events;
pub(crate) mod table;

use std::any::Any;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use wasmer::{FunctionEnvMut, RuntimeError, StoreMut};

use crate::abi::{AbiError, AbiResult, Cx, GuestMemory};
use crate::ctx::WebGpuCtx;
use crate::env::WebGpuEnv;
use crate::host::Host;

/// Futureless events (uncaptured errors) kept for a guest that does not
/// collect them, unless the embedder says otherwise.
pub(crate) const DEFAULT_MAX_PENDING_EVENTS: usize = 4096;

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

/// What the embedder-facing context needs from a guest process's host.
pub(crate) trait HostControl: Any + Send + Sync {
    fn terminate(&self);
    /// (devices, live object handles, bytes of buffer and texture storage)
    fn usage(&self) -> (usize, usize, u64);
}

/// Host state of every guest process of one [`WebGpuCtx`].
///
/// Held as trait objects so that the context's own `Send`/`Sync` do not
/// depend on the GPU library's object graph (a proof deep enough to need a
/// raised recursion limit, which embedders should not have to copy).
#[derive(Default)]
pub(crate) struct HostRegistry {
    /// Threads of a process share one linear memory, and every instance of the
    /// process must reach the same host: keyed by the memory's base.
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

/// The host of the process `memory` belongs to, made by `create` on first use.
pub(crate) fn host_for_memory<H: HostControl>(
    ctx: &WebGpuCtx,
    store: &mut StoreMut<'_>,
    memory: &wasmer::Memory,
    create: impl FnOnce() -> H,
) -> Arc<H> {
    let shared = memory.ty(store).shared;
    let base = memory.view(store).data_ptr() as usize;
    let mut registry = lock(&ctx.inner.hosts);
    registry.live();
    if shared
        && base != 0
        && let Some(host) = registry.by_memory.get(&base).and_then(Weak::upgrade)
        && let Ok(host) = host.downcast::<H>()
    {
        return host;
    }
    let host = Arc::new(create());
    if shared && base != 0 {
        let any: Arc<dyn Any + Send + Sync> = host.clone();
        registry.by_memory.insert(base, Arc::downgrade(&any));
    }
    let control: Arc<dyn HostControl> = host.clone();
    registry.all.push(Arc::downgrade(&control));
    host
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
#[cfg_attr(feature = "dawn", allow(dead_code))]
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

/// Whether the process whose memory this is was killed. WASIX disables the
/// atomics of a killed process's memory to shake its threads out of their
/// waits; a wait of no duration on an address nobody notifies tells whether
/// that has happened, and nothing else.
fn killed(probe: &wasmer::MemoryOps) -> bool {
    matches!(
        probe.wait(
            wasmer::MemoryLocation::new_32(u32::MAX),
            Some(std::time::Duration::ZERO)
        ),
        Err(wasmer::AtomicsError::AtomicsDisabled)
    )
}

/// Runs one import: resolves the calling instance's memory and host, decodes
/// through `f`, and turns failures into the right kind of guest-visible
/// outcome.
///
/// - A guest pointer outside linear memory traps, like the segfault it is,
///   whatever the implementation did with the failed copy.
/// - A guest stopped while the call waited is stopped: the call traps, as
///   soon as the call saw it, whatever the call returned.
/// - Any other decoding error makes the call return its zero value.
/// - A panic below (the GPU library failing on a misuse path) traps this
///   call instead of unwinding through the wasm frames into the embedder.
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
    // Only a kill stops a call. A context shut down by its embedder
    // (`terminate_all`) leaves its guests running, with their devices lost.
    let probe = data.kill_probe.clone();
    let stopped = || probe.as_ref().is_some_and(killed);
    let mut cx = Cx::new(GuestMemory::new(memory.view(&store)), &stopped);
    let outcome = catch_unwind(AssertUnwindSafe(|| f(&host, &mut cx)));
    finish(function, &cx, outcome)
}

/// What a call that ran becomes for the guest; see [`enter`].
fn finish<R: Default>(
    function: &'static str,
    cx: &Cx<'_>,
    outcome: std::thread::Result<AbiResult<R>>,
) -> Result<R, RuntimeError> {
    // Checked before anything else: an implementation may have turned a
    // failed copy into a status of its own, and a wild pointer still traps.
    if let Some(fault) = cx.mem.take_fault() {
        return Err(RuntimeError::new(format!("{function}: {fault}")));
    }
    // Likewise a call that found its guest stopped, even one that went on to
    // return a status or nothing at all.
    if cx.take_stop() {
        return Err(RuntimeError::new(format!(
            "{function}: {}",
            AbiError::Stopped
        )));
    }
    match outcome {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error @ (AbiError::OutOfBounds { .. } | AbiError::Stopped))) => {
            Err(RuntimeError::new(format!("{function}: {error}")))
        }
        Ok(Err(error)) => {
            // A decoding failure that is the guest's bug (an out-of-range
            // enum, an sType that does not belong in a chain). There is no
            // device to blame it on in general, so it is only logged; the
            // call returns its error value.
            log(|| format!("{function}: {error}"));
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

#[cfg(test)]
mod finish_tests {
    use super::*;

    fn with_cx(stopped: bool, f: impl FnOnce(&Cx<'_>)) {
        let mut store = wasmer::Store::default();
        let memory =
            wasmer::Memory::new(&mut store, wasmer::MemoryType::new(1, None, false)).unwrap();
        let stopped = move || stopped;
        f(&Cx::new(GuestMemory::new(memory.view(&store)), &stopped));
    }

    #[test]
    fn a_call_that_saw_its_guest_stopped_traps_whatever_it_returned() {
        with_cx(true, |cx| {
            assert!(cx.stopped());
            let error = finish("wgpuTest", cx, Ok(Ok(7))).expect_err("traps");
            assert!(error.to_string().contains("stopped"), "{error}");
        });
    }

    #[test]
    fn a_call_that_never_asked_returns_what_it_returned() {
        // Only a check that sees the stop makes the call trap.
        with_cx(true, |cx| {
            assert_eq!(finish("wgpuTest", cx, Ok(Ok(7))).unwrap(), 7);
        });
        with_cx(false, |cx| {
            assert!(!cx.stopped());
            assert_eq!(finish("wgpuTest", cx, Ok(Ok(7))).unwrap(), 7);
        });
    }
}
