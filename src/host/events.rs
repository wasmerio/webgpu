//! Completed asynchronous work, waiting for the guest to collect it.
//!
//! The host never calls into the guest. An operation that finishes (an adapter
//! request, a buffer map, a lost device, an uncaptured error) becomes a
//! [`Record`] in this queue; the guest shim drains the queue from
//! `wgpuInstanceProcessEvents` / `wgpuInstanceWaitAny` and invokes the user's
//! callbacks itself.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use crate::abi::{AbiResult, GuestMemory};

/// Bytes of the fixed part of an encoded record.
pub(crate) const RECORD_HEADER_SIZE: usize = 32;

/// One completed event. Encoded for the guest as:
///
/// ```text
/// u64 future;      // 0 for events that belong to no future
/// u32 kind;        // EVENT_* of the import ABI
/// u32 status;      // the callback's status enum, per kind
/// u32 handle;      // result object, or the device an error belongs to
/// u32 aux;         // per kind (an error type, for example)
/// u32 message_len;
/// u32 extra_len;
/// u8  message[message_len], padded to 8 bytes
/// u8  extra[extra_len], padded to 8 bytes
/// ```
#[derive(Clone, Debug, Default)]
pub(crate) struct Record {
    pub future: u64,
    pub kind: u32,
    pub status: u32,
    pub handle: u32,
    pub aux: u32,
    pub message: String,
    pub extra: Vec<u8>,
    // Host-only, absent from the wire format: the compiler lane admission an
    // async pipeline's event holds until the guest collects it (only its Drop
    // matters), and the instance whose release retires the event.
    pub(super) _pipeline_slot: Option<Arc<super::pipeline_worker::Slot>>,
    pub(super) pipeline_instance: u32,
}

fn pad8(len: usize) -> usize {
    (len + 7) & !7
}

impl Record {
    pub(crate) fn encoded_len(&self) -> usize {
        RECORD_HEADER_SIZE + pad8(self.message.len()) + pad8(self.extra.len())
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(&self.future.to_le_bytes());
        out.extend_from_slice(&self.kind.to_le_bytes());
        out.extend_from_slice(&self.status.to_le_bytes());
        out.extend_from_slice(&self.handle.to_le_bytes());
        out.extend_from_slice(&self.aux.to_le_bytes());
        out.extend_from_slice(&(self.message.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.extra.len() as u32).to_le_bytes());
        out.extend_from_slice(self.message.as_bytes());
        out.resize(RECORD_HEADER_SIZE + pad8(self.message.len()), 0);
        out.extend_from_slice(&self.extra);
        out.resize(self.encoded_len(), 0);
        out
    }
}

pub(crate) struct Events {
    ready: Mutex<VecDeque<Record>>,
    pushed: Condvar,
    max_pending: Option<usize>,
    dropped: AtomicU64,
    /// Records queued so far. Guest threads share one queue, so a thread
    /// about to block compares this with what it saw before it looked for
    /// its events: any change means another thread may have collected them.
    epoch: AtomicU64,
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Events {
    #[cfg(test)]
    pub(crate) fn new(max_pending: Option<usize>) -> Self {
        Self::with_waker(max_pending, None)
    }

    pub(crate) fn with_waker(
        max_pending: Option<usize>,
        wake: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Self {
        Self {
            ready: Mutex::new(VecDeque::new()),
            pushed: Condvar::new(),
            max_pending,
            dropped: AtomicU64::new(0),
            epoch: AtomicU64::new(0),
            wake,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<Record>> {
        self.ready
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Queues a completed event. Events of a future are never dropped: the
    /// guest is owed exactly one callback per future. Futureless events
    /// (uncaptured errors) are dropped once the queue is over its limit, so a
    /// guest that never polls cannot grow host memory without bound.
    pub(crate) fn push(&self, record: Record) {
        if self.queue(record) {
            self.wake_owner();
        }
    }

    /// Like [`push`](Self::push), but leaves waking the embedder to the
    /// caller, who calls [`wake_owner`](Self::wake_owner) once it released its
    /// own locks. Guest threads waiting for an event are woken. Returns
    /// whether the record was queued.
    pub(crate) fn queue(&self, record: Record) -> bool {
        let mut ready = self.lock();
        if record.future == 0
            && self
                .max_pending
                .is_some_and(|max_pending| ready.len() >= max_pending)
        {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        ready.push_back(record);
        self.epoch.fetch_add(1, Ordering::Release);
        drop(ready);
        self.pushed.notify_all();
        true
    }

    /// Calls the embedder's event waker, if it has one.
    pub(crate) fn wake_owner(&self) {
        if let Some(wake) = &self.wake {
            // Embedder signaling must not strand GPU futures if it panics.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| wake()));
        }
    }

    /// Owned compiler results are ready; publication happens on the next import.
    pub(crate) fn notify_work(&self) {
        let ready = self.lock();
        self.epoch.fetch_add(1, Ordering::Release);
        drop(ready);
        self.pushed.notify_all();
        self.wake_owner();
    }

    pub(crate) fn retire_pipeline_instance(&self, instance: u32) -> Vec<Record> {
        let mut ready = self.lock();
        let mut retained = VecDeque::new();
        let mut retired = Vec::new();
        for record in ready.drain(..) {
            // Only async pipeline events name an instance (handles start at 1).
            if record.pipeline_instance == instance {
                retired.push(record);
            } else {
                retained.push_back(record);
            }
        }
        *ready = retained;
        retired
    }

    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(super) fn take_for_test(&self) -> Vec<Record> {
        self.lock().drain(..).collect()
    }

    pub(crate) fn is_ready(&self, futures: &[u64]) -> bool {
        self.lock()
            .iter()
            .any(|record| record.future != 0 && futures.contains(&record.future))
    }

    /// Sleeps until an event is queued or `timeout` passes.
    pub(crate) fn wait_for_push(&self, timeout: Duration, epoch: u64) {
        let ready = self.lock();
        if !ready.is_empty() || self.epoch() != epoch {
            return;
        }
        let _ = self
            .pushed
            .wait_timeout(ready, timeout)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }

    pub(crate) fn wake_all(&self) {
        self.notify_work();
    }

    /// Moves as many whole records as fit into the guest buffer.
    ///
    /// Returns the bytes written, or the negated size of the first record when
    /// not even that one fits (the shim grows its buffer and asks again).
    pub(crate) fn drain_into(&self, mem: &GuestMemory<'_>, buf: u32, cap: u32) -> AbiResult<i32> {
        let mut ready = self.lock();
        let mut written = 0usize;
        while let Some(record) = ready.front() {
            let len = record.encoded_len();
            if written + len > cap as usize {
                if written == 0 {
                    return Ok(-(len.min(i32::MAX as usize) as i32));
                }
                break;
            }
            mem.write_bytes(buf + written as u32, &record.encode())?;
            written += len;
            ready.pop_front();
        }
        Ok(written as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_padded_to_eight_bytes() {
        let record = Record {
            future: 7,
            kind: 3,
            status: 1,
            handle: 0x0100_0002,
            aux: 5,
            message: "abc".into(),
            extra: vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            ..Record::default()
        };
        let bytes = record.encode();
        assert_eq!(bytes.len(), 32 + 8 + 16);
        assert_eq!(bytes.len(), record.encoded_len());
        assert_eq!(u64::from_le_bytes(bytes[0..8].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(bytes[24..28].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(bytes[28..32].try_into().unwrap()), 9);
        assert_eq!(&bytes[32..35], b"abc");
        assert_eq!(&bytes[35..40], &[0; 5]);
        assert_eq!(&bytes[40..49], &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn futureless_events_are_dropped_over_the_limit_but_futures_never_are() {
        let events = Events::new(Some(2));
        for _ in 0..5 {
            events.push(Record::default());
        }
        events.push(Record {
            future: 9,
            ..Record::default()
        });
        assert_eq!(events.lock().len(), 3);
        assert_eq!(events.dropped.load(Ordering::Relaxed), 3);
        assert!(events.is_ready(&[9]));
        assert!(!events.is_ready(&[8]));
    }

    #[test]
    fn compiler_notification_before_wait_registration_is_not_lost() {
        let events = Events::new(None);
        let before = events.epoch();
        events.notify_work();
        // An already queued compiler notification must not sleep for the
        // timeout, even though no ordinary event record exists yet.
        let started = std::time::Instant::now();
        events.wait_for_push(Duration::from_secs(2), before);
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "did not sleep for stale epoch"
        );
        assert_ne!(events.epoch(), before);
        assert!(
            events.lock().is_empty(),
            "notification does not forge a future"
        );
    }

    #[test]
    fn waker_runs_after_event_mutex_is_released() {
        let weak = std::sync::OnceLock::<std::sync::Weak<Events>>::new();
        let shared = Arc::new(weak);
        let capture = Arc::clone(&shared);
        let observed = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed_in_callback = Arc::clone(&observed);
        let events = Arc::new(Events::with_waker(
            None,
            Some(Arc::new(move || {
                let events = capture.get().unwrap().upgrade().unwrap();
                observed_in_callback.store(events.ready.try_lock().is_ok(), Ordering::Release);
            })),
        ));
        assert!(shared.set(Arc::downgrade(&events)).is_ok());
        events.push(Record::default());
        assert!(observed.swap(false, Ordering::AcqRel));
        events.notify_work();
        assert!(observed.load(Ordering::Acquire));
    }
}
