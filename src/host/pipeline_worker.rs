//! A single lazy compiler lane. Only owned, Send inputs enter this module.
//!
//! An admission slot covers queued, running AND completed-unconsumed work.
//! Work the lane cannot take is handed back, so the caller can run it itself.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const LIVE: u8 = 0;
const TERMINATED: u8 = 1;
const DISCARDED: u8 = 2;

// One lane per Host, i.e. per guest process. Detached, nonpreemptible calls
// from retired Hosts still count, so repeated process teardown cannot create
// unbounded stuck compiler threads. Past the limit, callers compile on their
// own thread, as they did before compiler lanes existed.
const MAX_COMPILER_THREADS: usize = 64;
static COMPILER_THREADS: AtomicUsize = AtomicUsize::new(0);
struct ThreadSlot(&'static AtomicUsize);
impl ThreadSlot {
    fn acquire() -> Option<Self> {
        Self::from_counter(&COMPILER_THREADS)
    }
    fn from_counter(counter: &'static AtomicUsize) -> Option<Self> {
        counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                (count < MAX_COMPILER_THREADS).then_some(count + 1)
            })
            .ok()
            .map(|_| Self(counter))
    }
}
impl Drop for ThreadSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Kept by the ready queue, then by the event until it is actually consumed.
#[derive(Debug)]
pub(super) struct Slot(Arc<AtomicUsize>);

impl Drop for Slot {
    fn drop(&mut self) {
        let previous = self.0.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
    }
}

struct Job<I, T> {
    key: u32,
    info: I,
    run: Box<dyn FnOnce() -> T + Send>,
    slot: Arc<Slot>,
    cancellation: Arc<AtomicU8>,
}

/// Why the lane did not take a job.
pub(super) enum Rejected<I, T> {
    /// The lane was stopped: its Host is terminated or gone.
    Stopped(I),
    /// The lane is at capacity (its queue is full, or no compiler thread is
    /// available); the caller may run the job itself.
    Busy(I, Box<dyn FnOnce() -> T + Send>),
}

impl<I, T> std::fmt::Debug for Rejected<I, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Stopped(_) => "Stopped",
            Self::Busy(..) => "Busy",
        })
    }
}

pub(super) struct Completed<I, T> {
    pub key: u32,
    pub info: I,
    pub result: Result<T, String>,
    pub slot: Arc<Slot>,
}

struct State<I, T> {
    queued: VecDeque<Job<I, T>>,
    completed: VecDeque<Completed<I, T>>,
    running: Option<(u32, Arc<AtomicU8>)>,
    stopped: bool,
}

struct Shared<I, T> {
    state: Mutex<State<I, T>>,
    changed: Condvar,
    outstanding: Arc<AtomicUsize>,
    limit: usize,
    wake: Arc<dyn Fn() + Send + Sync>,
}

pub(super) struct Worker<I: Send + 'static, T: Send + 'static> {
    shared: Arc<Shared<I, T>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl<I: Send + 'static, T: Send + 'static> Worker<I, T> {
    pub fn new(limit: usize, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        assert!(limit > 0);
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    queued: VecDeque::new(),
                    completed: VecDeque::new(),
                    running: None,
                    stopped: false,
                }),
                changed: Condvar::new(),
                outstanding: Arc::new(AtomicUsize::new(0)),
                limit,
                wake,
            }),
            thread: Mutex::new(None),
        }
    }

    /// Whether a job is queued or compiling: a result may come at any time.
    pub fn is_working(&self) -> bool {
        let state = lock(&self.shared.state);
        !state.queued.is_empty() || state.running.is_some()
    }

    /// Queues `run`, or hands it back: the lane never runs rejected work.
    pub fn enqueue(
        &self,
        key: u32,
        info: I,
        run: impl FnOnce() -> T + Send + 'static,
    ) -> Result<(), Rejected<I, T>> {
        let mut thread = lock(&self.thread);
        let mut state = lock(&self.shared.state);
        if state.stopped {
            return Err(Rejected::Stopped(info));
        }
        if self.shared.outstanding.load(Ordering::Acquire) >= self.shared.limit {
            return Err(Rejected::Busy(info, Box::new(run)));
        }
        if thread.is_none() {
            let Some(thread_slot) = ThreadSlot::acquire() else {
                return Err(Rejected::Busy(info, Box::new(run)));
            };
            let shared = Arc::clone(&self.shared);
            match thread::Builder::new()
                .name("wasmer-webgpu-pipelines".into())
                .spawn(move || {
                    let _thread_slot = thread_slot;
                    work(shared);
                }) {
                Ok(handle) => *thread = Some(handle),
                // A later call tries to start the thread again.
                Err(_) => return Err(Rejected::Busy(info, Box::new(run))),
            }
        }
        self.shared.outstanding.fetch_add(1, Ordering::AcqRel);
        state.queued.push_back(Job {
            key,
            info,
            run: Box::new(run),
            slot: Arc::new(Slot(Arc::clone(&self.shared.outstanding))),
            cancellation: Arc::new(AtomicU8::new(LIVE)),
        });
        drop(state);
        self.shared.changed.notify_one();
        Ok(())
    }

    pub fn take_completed(&self) -> Vec<Completed<I, T>> {
        lock(&self.shared.state).completed.drain(..).collect()
    }

    #[cfg(test)]
    pub fn completed_count_for_test(&self) -> usize {
        lock(&self.shared.state).completed.len()
    }

    #[cfg(test)]
    pub fn any_completed_for_test(
        &self,
        predicate: impl Fn(&I, &Result<T, String>) -> bool,
    ) -> bool {
        lock(&self.shared.state)
            .completed
            .iter()
            .any(|done| predicate(&done.info, &done.result))
    }

    /// Instance release already cancels guest callbacks. Retire unpublished
    /// results and queued ownership; a nonpreemptible running call discards late.
    pub fn discard(&self, key: u32) {
        let mut state = lock(&self.shared.state);
        if let Some((running_key, cancellation)) = &state.running
            && *running_key == key
        {
            cancellation.store(DISCARDED, Ordering::Release);
        }
        let mut retired_jobs = Vec::new();
        let mut retired_results = Vec::new();
        let mut queued = VecDeque::new();
        for job in state.queued.drain(..) {
            if job.key == key {
                retired_jobs.push(job);
            } else {
                queued.push_back(job);
            }
        }
        state.queued = queued;
        let mut completed = VecDeque::new();
        for result in state.completed.drain(..) {
            if result.key == key {
                retired_results.push(result);
            } else {
                completed.push_back(result);
            }
        }
        state.completed = completed;
        drop(state);
        // Native objects/closures are always destroyed outside the queue lock.
        drop((retired_jobs, retired_results));
    }

    /// Stop admission and cancel queued work, without waiting for the driver.
    /// The guest remains alive after context termination and is owed outcomes.
    pub fn terminate(&self) {
        let mut state = lock(&self.shared.state);
        state.stopped = true;
        if let Some((_, cancellation)) = &state.running {
            cancellation.fetch_max(TERMINATED, Ordering::AcqRel);
        }
        let mut retired = Vec::new();
        let mut retired_runs = Vec::new();
        for done in &mut state.completed {
            retired.push(std::mem::replace(
                &mut done.result,
                Err("Pipeline compiler was terminated.".into()),
            ));
        }
        let queued = std::mem::take(&mut state.queued);
        for job in queued {
            let Job {
                key,
                info,
                run,
                slot,
                ..
            } = job;
            retired_runs.push(run);
            state.completed.push_back(Completed {
                key,
                info,
                slot,
                result: Err("Pipeline compiler was terminated.".into()),
            });
        }
        drop(state);
        drop((retired, retired_runs));
        self.shared.changed.notify_all();
        (self.shared.wake)();
    }
}

fn work<I: Send + 'static, T: Send + 'static>(shared: Arc<Shared<I, T>>) {
    loop {
        let mut state = lock(&shared.state);
        while state.queued.is_empty() && !state.stopped {
            state = shared
                .changed
                .wait(state)
                .unwrap_or_else(|p| p.into_inner());
        }
        let Some(job) = state.queued.pop_front() else {
            return;
        };
        state.running = Some((job.key, Arc::clone(&job.cancellation)));
        drop(state);
        let Job {
            key,
            info,
            run,
            slot,
            cancellation,
        } = job;
        let mut result =
            catch_unwind(AssertUnwindSafe(run)).map_err(|_| "Pipeline compiler panicked.".into());
        let mut state = lock(&shared.state);
        state.running = None;
        let status = cancellation.load(Ordering::Acquire);
        let mut retired = None;
        if status == DISCARDED {
            drop(state);
            // Dropping a result can drop a device. Should that panic, this
            // lane must live on for the jobs queued behind it.
            let _ = catch_unwind(AssertUnwindSafe(move || drop((result, info, slot))));
            continue;
        }
        if status == TERMINATED || state.stopped {
            retired = Some(result);
            result = Err("Pipeline compiler was terminated.".into());
        }
        state.completed.push_back(Completed {
            key,
            info,
            result,
            slot,
        });
        drop(state);
        let _ = catch_unwind(AssertUnwindSafe(move || drop(retired)));
        // A waker must not unwind out of this lane and strand later futures.
        let _ = catch_unwind(AssertUnwindSafe(|| (shared.wake)()));
    }
}

impl<I: Send + 'static, T: Send + 'static> Worker<I, T> {
    pub fn close(&self) {
        let mut thread = lock(&self.thread);
        let handle = thread.take();
        let mut state = lock(&self.shared.state);
        state.stopped = true;
        if let Some((_, cancellation)) = &state.running {
            cancellation.store(DISCARDED, Ordering::Release);
        }
        let retired = (
            std::mem::take(&mut state.queued),
            std::mem::take(&mut state.completed),
        );
        drop(state);
        drop(thread);
        drop(retired);
        self.shared.changed.notify_all();
        // Detach: cleanup/navigation must not wait for a nonpreemptible driver.
        // The worker holds only Shared and owned native inputs. Discarded late
        // results never wake/publish. Its global ThreadSlot stays held until exit.
        drop(handle);
    }
}

impl<I: Send + 'static, T: Send + 'static> Drop for Worker<I, T> {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{self, Receiver};
    use std::time::Duration;

    fn receive<T>(receiver: &Receiver<T>) -> T {
        receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("bounded test wait")
    }

    #[test]
    fn detached_thread_admission_is_bounded_and_released() {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let mut slots: Vec<_> = (0..MAX_COMPILER_THREADS)
            .map(|_| ThreadSlot::from_counter(&COUNT).unwrap())
            .collect();
        assert!(ThreadSlot::from_counter(&COUNT).is_none());
        slots.pop();
        let last = ThreadSlot::from_counter(&COUNT).unwrap();
        drop((slots, last));
        assert_eq!(COUNT.load(Ordering::Acquire), 0);
    }

    #[test]
    fn submission_returns_while_work_is_held_and_runs_on_another_thread() {
        let (wake, woke) = mpsc::channel();
        let worker = Worker::new(
            2,
            Arc::new(move || {
                wake.send(()).unwrap();
            }),
        );
        let (entered, entry) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let owner = thread::current().id();
        worker
            .enqueue(1, 7, move || {
                entered.send(thread::current().id()).unwrap();
                receive(&released);
                42
            })
            .unwrap();
        assert_ne!(receive(&entry), owner);
        assert!(worker.take_completed().is_empty());
        release.send(()).unwrap();
        receive(&woke);
        let result = worker.take_completed().pop().unwrap();
        assert_eq!((result.info, result.result), (7, Ok(42)));
        assert!(worker.take_completed().is_empty());
    }

    #[test]
    fn slot_covers_ready_and_consumed_event_ownership() {
        let (wake, woke) = mpsc::channel();
        let worker = Worker::new(
            1,
            Arc::new(move || {
                let _ = wake.send(());
            }),
        );
        worker.enqueue(1, (), || 42).unwrap();
        receive(&woke);
        assert!(
            worker.enqueue(1, (), || 0).is_err(),
            "ready result retains slot"
        );
        let result = worker.take_completed().pop().unwrap();
        assert!(
            worker.enqueue(1, (), || 0).is_err(),
            "event retains slot after queue drain"
        );
        drop(result);
        worker.enqueue(1, (), || 1).unwrap();
        receive(&woke);
    }

    #[test]
    fn queue_pressure_hands_the_work_back_without_running_it() {
        let worker = Worker::new(1, Arc::new(|| {}));
        let (release, released) = mpsc::channel();
        worker
            .enqueue(1, 0, move || {
                receive(&released);
                0
            })
            .unwrap();
        let Err(Rejected::Busy(info, run)) = worker.enqueue(1, 7, || 42) else {
            panic!("a full lane hands the work back");
        };
        assert_eq!(info, 7);
        // The caller runs it, on its own thread.
        assert_eq!(run(), 42);
        release.send(()).unwrap();
    }

    #[test]
    fn panic_is_one_result_and_worker_keeps_progressing() {
        let (wake, woke) = mpsc::channel();
        let worker = Worker::new(
            2,
            Arc::new(move || {
                let _ = wake.send(());
            }),
        );
        worker
            .enqueue(1, 1, || panic!("injected driver failure"))
            .unwrap();
        worker.enqueue(1, 2, || 42).unwrap();
        receive(&woke);
        receive(&woke);
        let done = worker.take_completed();
        assert_eq!(done.len(), 2);
        assert!(done[0].result.is_err());
        assert_eq!(done[1].result, Ok(42));
    }

    #[test]
    fn instance_discard_drops_queued_and_late_results() {
        let (wake, woke) = mpsc::channel();
        let worker = Worker::new(
            3,
            Arc::new(move || {
                let _ = wake.send(());
            }),
        );
        let (entered, entry) = mpsc::channel();
        let (release, released) = mpsc::channel();
        worker
            .enqueue(1, 1, move || {
                entered.send(()).unwrap();
                receive(&released);
                1
            })
            .unwrap();
        receive(&entry);
        worker
            .enqueue(1, 2, || panic!("discarded queued work ran"))
            .unwrap();
        worker.discard(1);
        worker.enqueue(2, 3, || 3).unwrap();
        release.send(()).unwrap();
        receive(&woke);
        let done = worker.take_completed();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].info, 3);
    }

    #[test]
    fn termination_produces_one_error_per_accepted_job_and_stops_admission() {
        let worker = Worker::new(2, Arc::new(|| {}));
        let (entered, entry) = mpsc::channel();
        let (release, released) = mpsc::channel();
        worker
            .enqueue(1, 1, move || {
                entered.send(()).unwrap();
                receive(&released);
                1
            })
            .unwrap();
        receive(&entry);
        worker
            .enqueue(1, 2, || panic!("terminated queued work ran"))
            .unwrap();
        worker.terminate();
        assert!(matches!(
            worker.enqueue(1, 3, || 3),
            Err(Rejected::Stopped(3))
        ));
        let queued = worker.take_completed();
        assert_eq!(queued.len(), 1);
        assert_eq!(queued[0].info, 2);
        assert!(queued[0].result.is_err());
        release.send(()).unwrap();
        // Termination does not join the driver. Drop discards its late result.
        drop(worker);
        drop(queued);
    }

    #[test]
    fn drop_returns_before_running_work_and_destroys_queued_captures() {
        struct OnDrop(mpsc::Sender<()>);
        impl Drop for OnDrop {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        let worker = Worker::new(2, Arc::new(|| {}));
        let (entered, entry) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let (retired, retirement) = mpsc::channel();
        let running_resource = Arc::new(OnDrop(retired));
        let running_weak = Arc::downgrade(&running_resource);
        worker
            .enqueue(1, (), move || {
                entered.send(()).unwrap();
                receive(&released);
                running_resource
            })
            .unwrap();
        receive(&entry);
        let queued_resource = Arc::new(());
        let queued_weak = Arc::downgrade(&queued_resource);
        worker
            .enqueue(1, (), move || {
                drop(queued_resource);
                Arc::new(OnDrop(mpsc::channel().0))
            })
            .unwrap();
        drop(worker);
        assert!(queued_weak.upgrade().is_none());
        assert!(
            running_weak.upgrade().is_some(),
            "nonpreemptible call still owns inputs"
        );
        release.send(()).unwrap();
        receive(&retirement);
        assert!(running_weak.upgrade().is_none());
    }
}
