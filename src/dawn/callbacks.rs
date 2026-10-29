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

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use super::buffer::MapInfo;
use super::ffi::{self, WGPUStringView};
use super::{DeviceState, Events, Record, lock, text};
use crate::abi::enums::{BufferMapState, CompilationInfoRequestStatus, MapAsyncStatus};
use crate::abi::{AbiEnum, EVENT_BUFFER_MAP, EVENT_COMPILATION_INFO, Handle};

/// The Dawn futures behind guest futures that are still pending, so a guest
/// waiting for some of them can wait inside Dawn for exactly those.
#[derive(Default)]
pub(crate) struct Futures {
    pending: Arc<Mutex<HashMap<u64, u64>>>,
}

impl Futures {
    pub(crate) fn track(&self, guest: u64, dawn: ffi::WGPUFuture) {
        if guest != 0 && dawn.id != 0 {
            lock(&self.pending).insert(guest, dawn.id);
        }
    }

    /// Dawn's ids of the listed guest futures that are still pending.
    pub(crate) fn dawn_ids(&self, guest: &[u64]) -> Vec<u64> {
        let pending = lock(&self.pending);
        guest
            .iter()
            .filter_map(|future| pending.get(future).copied())
            .collect()
    }
}

/// What every callback needs to report its future.
pub(crate) struct Pending {
    events: Arc<Events>,
    futures: Arc<Mutex<HashMap<u64, u64>>>,
    pub(crate) future: u64,
}

impl Pending {
    pub(crate) fn new(events: &Arc<Events>, futures: &Futures, future: u64) -> Self {
        Self {
            events: Arc::clone(events),
            futures: Arc::clone(&futures.pending),
            future,
        }
    }

    fn finish(&self, record: Record) {
        lock(&self.futures).remove(&self.future);
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

#[derive(Default)]
pub(crate) struct DonePipelines {
    pub(crate) list: Mutex<Vec<DonePipeline>>,
}

/// A pending `wgpuDeviceCreate*PipelineAsync`.
pub(crate) struct PipelineRequest {
    pub(crate) pending: Pending,
    pub(crate) event: u32,
    pub(crate) kind: ffi::Kind,
    pub(crate) done: Arc<DonePipelines>,
    pub(crate) device: Arc<DeviceState>,
    pub(crate) instance: Handle,
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
    lock(&request.pending.futures).remove(&request.pending.future);
    lock(&request.done.list).push(DonePipeline {
        future: request.pending.future,
        event: request.event,
        kind: request.kind,
        status,
        raw,
        // SAFETY: Dawn's view, valid during the callback.
        message: unsafe { text(message) },
        device: Arc::clone(&request.device),
        instance: request.instance,
    });
    // Waiting threads look again; the next poll or wait publishes it.
    request.pending.events.notify_work();
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
