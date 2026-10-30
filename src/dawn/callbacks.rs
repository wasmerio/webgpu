//! Dawn's callbacks, turned into the event records the guest shim collects.
//!
//! Every future the host starts in Dawn uses `AllowSpontaneous`: its callback
//! may run on any thread, as soon as Dawn notices the work is done, and all it
//! does is queue a record (the host never calls into the guest). Each
//! callback owns a boxed context through `userdata1`, which it frees; Dawn
//! calls every future's callback exactly once, cancelled futures included.
//!
//! Pipelines created asynchronously are the one result that becomes an
//! object: the callback hands it to the host, which gives it a handle the
//! next time a guest thread polls or waits (see `Host::publish_pipelines`).

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use super::buffer::MapInfo;
use super::ffi::{self, WGPUStringView};
use super::{DeviceState, Events, Record, lock, text};
use crate::abi::enums::{BufferMapState, CompilationInfoRequestStatus, MapAsyncStatus};
use crate::abi::{AbiEnum, EVENT_BUFFER_MAP, EVENT_COMPILATION_INFO, Handle};

/// Guest futures whose Dawn future has not completed yet, and Dawn's id for
/// each.
#[derive(Default)]
struct FutureMap {
    pending: HashMap<u64, u64>,
    /// Futures whose callback ran before the call that started them returned
    /// (Dawn completes an event that is already ready inside that call):
    /// `track` then has nothing to record.
    finished_early: HashSet<u64>,
}

/// The Dawn futures behind guest futures that are still pending, so a guest
/// waiting for some of them can wait inside Dawn for exactly those.
#[derive(Default)]
pub(crate) struct Futures {
    map: Arc<Mutex<FutureMap>>,
}

impl Futures {
    /// Records the Dawn future a call started for `guest`, unless its
    /// callback has run already.
    pub(crate) fn track(&self, guest: u64, dawn: ffi::WGPUFuture) {
        let mut map = lock(&self.map);
        if map.finished_early.remove(&guest) {
            return;
        }
        if guest != 0 && dawn.id != 0 {
            map.pending.insert(guest, dawn.id);
        }
    }

    /// Dawn's ids of the listed guest futures that are still pending.
    pub(crate) fn dawn_ids(&self, guest: &[u64]) -> Vec<u64> {
        let map = lock(&self.map);
        guest
            .iter()
            .filter_map(|future| map.pending.get(future).copied())
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn len_for_test(&self) -> usize {
        let map = lock(&self.map);
        map.pending.len() + map.finished_early.len()
    }
}

/// What every callback needs to report its future.
pub(crate) struct Pending {
    events: Arc<Events>,
    futures: Arc<Mutex<FutureMap>>,
    pub(crate) future: u64,
}

impl Pending {
    pub(crate) fn new(events: &Arc<Events>, futures: &Futures, future: u64) -> Self {
        Self {
            events: Arc::clone(events),
            futures: Arc::clone(&futures.map),
            future,
        }
    }

    /// Forgets the future's Dawn id: it no longer needs waiting for. A
    /// callback that runs before `track` leaves a mark for it instead.
    fn settle(&self) {
        let mut map = lock(&self.futures);
        if map.pending.remove(&self.future).is_none() {
            map.finished_early.insert(self.future);
        }
    }

    fn finish(&self, record: Record) {
        self.settle();
        self.events.push(record);
    }
}

/// Boxes a callback context for `userdata1`.
pub(crate) fn userdata<T>(context: T) -> *mut c_void {
    Box::into_raw(Box::new(context)).cast()
}

/// Takes back a context boxed by [`userdata`].
///
/// # Safety
/// `userdata` must come from [`userdata`] with the same `T`, once.
unsafe fn take<T>(userdata: *mut c_void) -> Box<T> {
    // SAFETY: as the caller promises.
    unsafe { Box::from_raw(userdata.cast::<T>()) }
}

// -- simple futures --------------------------------------------------------

/// A future whose record is its kind, status and message.
pub(crate) struct Simple {
    pub(crate) pending: Pending,
    pub(crate) kind: u32,
}

pub(crate) unsafe extern "C" fn status_message(
    status: u32,
    message: WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    // SAFETY: the context boxed for this callback.
    let simple = unsafe { take::<Simple>(userdata1) };
    simple.pending.finish(Record {
        future: simple.pending.future,
        kind: simple.kind,
        status,
        // SAFETY: Dawn's view, valid during the callback.
        message: unsafe { text(message) },
        ..Record::default()
    });
}

pub(crate) unsafe extern "C" fn pop_error_scope(
    status: u32,
    ty: u32,
    message: WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    // SAFETY: the context boxed for this callback.
    let simple = unsafe { take::<Simple>(userdata1) };
    simple.pending.finish(Record {
        future: simple.pending.future,
        kind: simple.kind,
        status,
        aux: ty,
        // SAFETY: Dawn's view, valid during the callback.
        message: unsafe { text(message) },
        ..Record::default()
    });
}

// -- buffer mapping --------------------------------------------------------

/// A pending `wgpuBufferMapAsync`, and the mapping it updates.
pub(crate) struct Map {
    pub(crate) pending: Pending,
    pub(crate) map: Arc<Mutex<MapInfo>>,
    /// Which map request of the buffer this is: an unmap or a later request
    /// makes a late callback's state change moot.
    pub(crate) generation: u64,
}

pub(crate) unsafe extern "C" fn buffer_map(
    status: u32,
    message: WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    // SAFETY: the context boxed for this callback.
    let map = unsafe { take::<Map>(userdata1) };
    settle_map(
        &map.map,
        map.generation,
        status == MapAsyncStatus::Success.as_u32(),
    );
    map.pending.finish(Record {
        future: map.pending.future,
        kind: EVENT_BUFFER_MAP,
        status,
        // SAFETY: Dawn's view, valid during the callback.
        message: unsafe { text(message) },
        ..Record::default()
    });
}

/// Ends a pending map request: mapped if it succeeded, unmapped otherwise.
/// Only the buffer's current request may: an unmap or a later request makes
/// a late callback moot.
pub(crate) fn settle_map(map: &Mutex<MapInfo>, generation: u64, success: bool) {
    let mut info = lock(map);
    if info.generation == generation && info.state == BufferMapState::Pending {
        info.state = if success {
            BufferMapState::Mapped
        } else {
            BufferMapState::Unmapped
        };
    }
}

// -- shader compilation info ------------------------------------------------

/// Serializes compilation messages for the shim:
///
/// ```text
/// u32 count, pad;
/// per message: u32 type, length; u64 line, position, offset, length;
///              u8 text[length], padded to 8
/// ```
pub(crate) unsafe extern "C" fn compilation_info(
    status: u32,
    info: *const ffi::WGPUCompilationInfo,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    // SAFETY: the context boxed for this callback.
    let simple = unsafe { take::<Simple>(userdata1) };
    let mut extra = Vec::new();
    let mut count = 0u32;
    // SAFETY: Dawn's info, valid during the callback, or null.
    if let Some(info) = unsafe { info.as_ref() }
        && !info.messages.is_null()
    {
        // SAFETY: `messageCount` messages at `messages`.
        let messages = unsafe { std::slice::from_raw_parts(info.messages, info.messageCount) };
        for message in messages {
            // SAFETY: Dawn's view, valid during the callback.
            let body = unsafe { text(message.message) };
            extra.extend_from_slice(&message.r#type.to_le_bytes());
            extra.extend_from_slice(&(body.len() as u32).to_le_bytes());
            for value in [
                message.lineNum,
                message.linePos,
                message.offset,
                message.length,
            ] {
                extra.extend_from_slice(&value.to_le_bytes());
            }
            extra.extend_from_slice(body.as_bytes());
            extra.resize((extra.len() + 7) & !7, 0);
            count += 1;
        }
    }
    let mut payload = Vec::with_capacity(extra.len() + 8);
    payload.extend_from_slice(&count.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&extra);
    let status = if status == CompilationInfoRequestStatus::Success.as_u32() {
        status
    } else {
        // Anything else (an instance gone) is a callback with no messages.
        status
    };
    simple.pending.finish(Record {
        future: simple.pending.future,
        kind: EVENT_COMPILATION_INFO,
        status,
        extra: payload,
        ..Record::default()
    });
}

// -- pipelines created asynchronously ----------------------------------------

/// A pipeline Dawn finished compiling, waiting for a guest thread to give it
/// a handle.
pub(crate) struct DonePipeline {
    pub(crate) future: u64,
    pub(crate) event: u32,
    pub(crate) kind: ffi::Kind,
    pub(crate) status: u32,
    pub(crate) raw: *mut c_void,
    pub(crate) message: String,
    pub(crate) device: Arc<DeviceState>,
    pub(crate) instance: Handle,
}

// SAFETY: a Dawn object, which may move between threads.
unsafe impl Send for DonePipeline {}

impl Drop for DonePipeline {
    /// A pipeline nobody gave a handle (its host went first) is released
    /// here; one that was published has had `raw` taken.
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // SAFETY: the reference Dawn handed the callback.
            crate::native::autoreleased(|| unsafe { self.kind.release(self.raw) });
        }
    }
}

#[derive(Default)]
pub(crate) struct DonePipelines {
    pub(crate) list: Mutex<Vec<DonePipeline>>,
}

/// How many of a device's pipeline requests still owe their callback. It
/// holds no GPU object, so a compiler thread may be the last to let go of it.
#[derive(Default)]
pub(crate) struct InFlight {
    count: Mutex<usize>,
    settled: Condvar,
}

impl InFlight {
    /// A request is about to start: its callback is owed. The caller holds
    /// the device's `access` shared, so a destroy waits for it.
    fn start(&self) {
        *lock(&self.count) += 1;
    }

    fn called_back(&self) {
        let mut count = lock(&self.count);
        *count -= 1;
        if *count == 0 {
            self.settled.notify_all();
        }
    }

    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        *lock(&self.count)
    }

    /// Waits until every request's callback has run.
    pub(crate) fn wait_all(&self) {
        let count = lock(&self.count);
        drop(
            self.settled
                .wait_while(count, |count| *count > 0)
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        );
    }

    /// Waits, for at most `timeout`, until every request's callback has run.
    /// Whether they all did.
    pub(crate) fn wait(&self, timeout: Duration) -> bool {
        let count = lock(&self.count);
        let (count, _) = self
            .settled
            .wait_timeout_while(count, timeout, |count| *count > 0)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *count == 0
    }
}

/// A pending `wgpuDeviceCreate*PipelineAsync`.
pub(crate) struct PipelineRequest {
    pending: Pending,
    event: u32,
    kind: ffi::Kind,
    done: Arc<DonePipelines>,
    device: Arc<DeviceState>,
    instance: Handle,
    in_flight: Arc<InFlight>,
}

impl PipelineRequest {
    /// The context of a request Dawn is about to start, counted in flight
    /// until its callback has run.
    pub(crate) fn new(
        pending: Pending,
        event: u32,
        kind: ffi::Kind,
        done: &Arc<DonePipelines>,
        device: Arc<DeviceState>,
        instance: Handle,
    ) -> Self {
        device.pipelines.start();
        Self {
            pending,
            event,
            kind,
            in_flight: Arc::clone(&device.pipelines),
            done: Arc::clone(done),
            device,
            instance,
        }
    }
}

pub(crate) unsafe extern "C" fn pipeline(
    status: u32,
    raw: *mut c_void,
    message: WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    // SAFETY: the context boxed for this callback.
    let request = unsafe { take::<PipelineRequest>(userdata1) };
    let PipelineRequest {
        pending,
        event,
        kind,
        done,
        device,
        instance,
        in_flight,
    } = *request;
    pending.settle();
    lock(&done.list).push(DonePipeline {
        future: pending.future,
        event,
        kind,
        status,
        raw,
        // SAFETY: Dawn's view, valid during the callback.
        message: unsafe { text(message) },
        device,
        instance,
    });
    // Destroying the device, and a host that is going, wait for this
    // (`DeviceState::destroy`, `Host::drop`). Let go of everything that can
    // hold a device first: its last reference must not go on this thread,
    // inside Dawn's task, where releasing the device waits for that task. A
    // host that gave up waiting has left the list to this callback; another
    // thread drops it.
    if let Some(orphan) = Arc::into_inner(done) {
        // Without a thread to drop it on, it leaks: dropping it here would
        // wait for this very task.
        let orphan = std::mem::ManuallyDrop::new(orphan);
        let _ = std::thread::Builder::new()
            .name("webgpu-pipelines-release".to_owned())
            .spawn(move || drop(std::mem::ManuallyDrop::into_inner(orphan)));
    }
    in_flight.called_back();
    // Waiting threads look again; the next poll or wait publishes it.
    pending.events.notify_work();
}

/// The two pipeline callbacks differ only in the pipeline's C type.
pub(crate) unsafe extern "C" fn compute_pipeline(
    status: u32,
    pipeline: ffi::WGPUComputePipeline,
    message: WGPUStringView,
    userdata1: *mut c_void,
    userdata2: *mut c_void,
) {
    // SAFETY: forwarded as received.
    unsafe { self::pipeline(status, pipeline.cast(), message, userdata1, userdata2) }
}

pub(crate) unsafe extern "C" fn render_pipeline(
    status: u32,
    pipeline: ffi::WGPURenderPipeline,
    message: WGPUStringView,
    userdata1: *mut c_void,
    userdata2: *mut c_void,
) {
    // SAFETY: forwarded as received.
    unsafe { self::pipeline(status, pipeline.cast(), message, userdata1, userdata2) }
}
