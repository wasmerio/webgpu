//! The runtime hooks in a browser, as the Wasmer SDK uses them.
//!
//! The other browser tests create the JavaScript host by hand, and the Rust
//! checks only compile the browser backend. Here a guest is instantiated
//! through `WebGpuRuntimeHooks`, and what the backend hands an embedder is
//! checked: the context's `event_waker` and `__wasmerWebGpuHostCreated`.
//!
//! `build-runtime-hooks.sh` builds it with wasm-bindgen, and `run.mjs` runs it
//! in a worker (`runtime_hooks`).

#![cfg(target_arch = "wasm32")]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Result, anyhow, bail, ensure};
use js_sys::{Array, Date, Object, Promise, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use wasmer::{AsStoreMut, Instance, Module, Store, Value};
use wasmer_webgpu::WebGpuCtx;

#[wasm_bindgen(inline_js = "
export function listen(limits, failure) {
  // Each host the embedder is told about, with what became of it.
  const announced = [];
  globalThis.__wasmerWebGpuHostCreated = (host, scope) => {
    const entry = { host, scope, disposed: false };
    // The guest's memory is bound by now: the host writes into it.
    entry.status = host.imports.wgpuGetInstanceLimits(limits);
    const dispose = host.dispose;
    host.dispose = () => {
      entry.disposed = true;
      dispose.call(host);
    };
    announced.push(entry);
    if (failure) throw new Error(failure);
  };
  return announced;
}
export function stop_listening() {
  delete globalThis.__wasmerWebGpuHostCreated;
}
export function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
")]
extern "C" {
    fn listen(limits: u32, failure: Option<String>) -> Array;
    fn stop_listening();
    fn sleep(milliseconds: u32) -> Promise;
}

/// A request for an adapter is answered at once for an instance that does not
/// exist, and by the browser for one that does.
const GUEST: &str = r#"(module
  (import "wasmer_webgpu_v0" "wgpuCreateInstance" (func $create_instance (param i32) (result i32)))
  (import "wasmer_webgpu_v0" "wasmer_wgpu_instance_request_adapter"
    (func $request_adapter (param i32 i32 i64)))
  (import "wasmer_webgpu_v0" "wasmer_wgpu_poll" (func $poll (param i32 i32) (result i32)))
  (memory (export "memory") 1)
  (func (export "create_instance") (result i32) (call $create_instance (i32.const 0)))
  (func (export "request_adapter") (param i32 i64)
    (call $request_adapter (local.get 0) (i32.const 0) (local.get 1)))
  (func (export "poll") (param i32 i32) (result i32) (call $poll (local.get 0) (local.get 1))))"#;

const SCOPE: u32 = 7;
/// Where the announced host writes the instance limits.
const LIMITS: u32 = 64;
/// Where the guest collects event records, and how many bytes it has there.
const RECORDS: u32 = 1024;
const RECORDS_CAPACITY: u32 = 4096;
/// `WGPUStatus_Success`.
const STATUS_SUCCESS: f64 = 1.0;
/// The `timedWaitAnyMaxCount` of a host that may suspend its guest.
const TIMED_WAIT_ANY_MAX_COUNT: u32 = 64;
/// The event kind of an answered adapter request.
const EVENT_REQUEST_ADAPTER: u32 = 1;

/// A guest instance, and what the tests ask of it.
struct Guest(Instance);

impl Guest {
    /// Instantiates `wat` the way a WASIX runtime does with the hooks.
    fn new(store: &mut Store, ctx: &WebGpuCtx, wat: &str) -> Result<Self> {
        let module = Module::new(&*store, wat::parse_str(wat)?)
            .map_err(|error| anyhow!("compiling: {error}"))?;
        let hooks = ctx.runtime_hooks();
        let (imports, state) = hooks.additional_imports(&module, &mut store.as_store_mut())?;
        let instance = Instance::new(store, &module, &imports)
            .map_err(|error| anyhow!("instantiating: {error}"))?;
        hooks.configure_instance(&module, &mut store.as_store_mut(), &instance, None, state)?;
        Ok(Self(instance))
    }

    fn call(&self, store: &mut Store, name: &str, args: &[Value]) -> Result<Box<[Value]>> {
        let function = self.0.exports.get_function(name)?;
        function
            .call(store, args)
            .map_err(|error| anyhow!("{name}: {error}"))
    }

    fn create_instance(&self, store: &mut Store) -> Result<Value> {
        Ok(self.call(store, "create_instance", &[])?[0].clone())
    }

    fn request_adapter(&self, store: &mut Store, instance: Value, future: i64) -> Result<()> {
        self.call(store, "request_adapter", &[instance, Value::I64(future)])?;
        Ok(())
    }

    fn read(&self, store: &Store, offset: u32, length: u32) -> Result<Vec<u8>> {
        let mut bytes = vec![0; length as usize];
        let memory = self.0.exports.get_memory("memory")?;
        memory.view(store).read(offset.into(), &mut bytes)?;
        Ok(bytes)
    }

    /// The futures of the adapter requests one poll collects.
    fn collect(&self, store: &mut Store) -> Result<Vec<i64>> {
        let args = [RECORDS, RECORDS_CAPACITY].map(|value| Value::I32(value as i32));
        let written = match *self.call(store, "poll", &args)? {
            [Value::I32(written)] if written >= 0 => written as u32,
            ref other => bail!("wasmer_wgpu_poll returned {other:?}"),
        };
        let bytes = self.read(store, RECORDS, written)?;
        let mut futures = Vec::new();
        let mut at = 0;
        // A record is 32 bytes, then its message and its extra bytes, each
        // padded to 8.
        while at < bytes.len() {
            let record = bytes
                .get(at..at + 32)
                .ok_or_else(|| anyhow!("a record is cut short at {at}"))?;
            let future = i64::from_le_bytes(record[..8].try_into()?);
            let kind = u32_at(record, 8);
            ensure!(
                kind == EVENT_REQUEST_ADAPTER,
                "future {future} completed as kind {kind}"
            );
            futures.push(future);
            let padded = |length: u32| (length as usize + 7) & !7;
            at += 32 + padded(u32_at(record, 24)) + padded(u32_at(record, 28));
        }
        Ok(futures)
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// A waker that counts how often it is called.
fn counting_waker() -> (Arc<AtomicUsize>, Arc<dyn Fn() + Send + Sync>) {
    let wakes = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&wakes);
    let waker = Arc::new(move || {
        counted.fetch_add(1, Ordering::SeqCst);
    });
    (wakes, waker)
}

fn field(object: &JsValue, key: &str) -> JsValue {
    Reflect::get(object, &key.into()).unwrap_or(JsValue::UNDEFINED)
}

/// The one host the embedder was told about.
fn announced_once(announced: &Array) -> Result<JsValue> {
    let times = announced.length();
    ensure!(times == 1, "the host was announced {times} times");
    Ok(announced.get(0))
}

fn disposed(entry: &JsValue) -> Result<()> {
    ensure!(
        field(entry, "disposed") == true,
        "the host outlived its store"
    );
    Ok(())
}

async fn wait_for(what: &str, done: impl Fn() -> bool) -> Result<()> {
    for _ in 0..1000 {
        if done() {
            return Ok(());
        }
        let _ = JsFuture::from(sleep(10)).await;
    }
    bail!("{what} did not come within 10 s")
}

async fn the_embedder_learns_of_the_host_and_is_woken() -> Result<()> {
    let (wakes, waker) = counting_waker();
    let woken = || wakes.load(Ordering::SeqCst);
    let ctx = WebGpuCtx::builder()
        .canvas_scope(SCOPE)
        .event_waker(Arc::clone(&waker))
        .build();
    let owners = Arc::strong_count(&waker);
    let announced = listen(LIMITS, None);
    let mut store = Store::default();
    let guest = Guest::new(&mut store, &ctx, GUEST)?;

    let entry = announced_once(&announced)?;
    let scope = field(&entry, "scope").as_f64();
    ensure!(scope == Some(SCOPE.into()), "announced in scope {scope:?}");
    let status = field(&entry, "status").as_f64();
    let limit = u32_at(&guest.read(&store, LIMITS + 4, 4)?, 0);
    ensure!(
        status == Some(STATUS_SUCCESS) && limit == TIMED_WAIT_ANY_MAX_COUNT,
        "the announced host wrote the instance limits with status {status:?}, as {limit}"
    );
    ensure!(
        Arc::strong_count(&waker) == owners + 1,
        "the host does not hold the waker"
    );

    // An event that a call makes ready wakes during that call.
    guest.request_adapter(&mut store, Value::I32(0), 1)?;
    ensure!(
        woken() == 1,
        "an answer made at once woke {} times",
        woken()
    );
    // One the browser makes ready wakes when it settles, once the guest has
    // given the worker back.
    let instance = guest.create_instance(&mut store)?;
    guest.request_adapter(&mut store, instance, 2)?;
    ensure!(
        woken() == 1,
        "the browser's answer came during the call (no navigator.gpu?)"
    );
    wait_for("the browser's answer", || woken() >= 2).await?;
    ensure!(woken() == 2, "two answers woke {} times", woken());
    // The guest still collects both.
    let futures = guest.collect(&mut store)?;
    ensure!(futures == [1, 2], "collected the futures {futures:?}");

    drop(guest);
    drop(store);
    ensure!(
        Arc::strong_count(&waker) == owners,
        "the host kept the waker after its store was dropped"
    );
    disposed(&entry)
}

async fn an_embedder_that_throws_fails_the_instance() -> Result<()> {
    let announced = listen(LIMITS, Some("the embedder is not ready".into()));
    let mut store = Store::default();
    let error = Guest::new(&mut store, &WebGpuCtx::default(), GUEST)
        .err()
        .ok_or_else(|| anyhow!("the instance was set up"))?;
    let error = error.to_string();
    ensure!(
        error.contains("__wasmerWebGpuHostCreated") && error.contains("the embedder is not ready"),
        "the instance failed with: {error}"
    );
    let entry = announced_once(&announced)?;
    drop(store);
    disposed(&entry)
}

async fn a_host_is_announced_only_once_its_instance_is_set_up() -> Result<()> {
    let (_, waker) = counting_waker();
    let ctx = WebGpuCtx::builder().event_waker(Arc::clone(&waker)).build();
    let owners = Arc::strong_count(&waker);
    let announced = listen(LIMITS, None);
    let mut store = Store::default();
    // An import nothing provides: the instance is never created.
    let unlinked = r#"(module
      (import "wasmer_webgpu_v0" "wgpuCreateInstance" (func (param i32) (result i32)))
      (import "env" "missing" (func))
      (memory (export "memory") 1))"#;
    ensure!(
        Guest::new(&mut store, &ctx, unlinked).is_err(),
        "a guest missing an import was instantiated"
    );
    // No memory: the instance exists, and WebGPU cannot be set up for it.
    let memoryless = r#"(module
      (import "wasmer_webgpu_v0" "wgpuCreateInstance" (func (param i32) (result i32))))"#;
    let error = Guest::new(&mut store, &ctx, memoryless)
        .err()
        .ok_or_else(|| anyhow!("a guest without memory was set up"))?;
    ensure!(
        error.to_string().contains("linear memory"),
        "the guest without memory failed with: {error}"
    );
    let times = announced.length();
    ensure!(times == 0, "announced {times} times");
    // Both hosts were made, and go with the store.
    ensure!(
        Arc::strong_count(&waker) == owners + 2,
        "the hosts do not hold the waker"
    );
    drop(store);
    ensure!(
        Arc::strong_count(&waker) == owners,
        "a host kept the waker after its store was dropped"
    );
    Ok(())
}

async fn an_embedder_needs_neither_waker_nor_listener() -> Result<()> {
    stop_listening();
    let mut store = Store::default();
    let guest = Guest::new(&mut store, &WebGpuCtx::default(), GUEST)?;
    guest.request_adapter(&mut store, Value::I32(0), 1)?;
    let futures = guest.collect(&mut store)?;
    ensure!(futures == [1], "collected the futures {futures:?}");
    Ok(())
}

type Test = Pin<Box<dyn Future<Output = Result<()>>>>;

/// Runs every test in turn: `[{ name, failure, milliseconds }]`, where
/// `failure` is null for a test that passed.
#[wasm_bindgen]
pub async fn run() -> Array {
    let tests: [(&str, Test); 4] = [
        (
            "the embedder learns of the host once its memory is bound, and is woken by its events",
            Box::pin(the_embedder_learns_of_the_host_and_is_woken()),
        ),
        (
            "an embedder that throws fails the instance, whose host is still disposed of",
            Box::pin(an_embedder_that_throws_fails_the_instance()),
        ),
        (
            "a host is announced only once its instance is set up, and goes with its store",
            Box::pin(a_host_is_announced_only_once_its_instance_is_set_up()),
        ),
        (
            "an embedder needs neither a waker nor a listener",
            Box::pin(an_embedder_needs_neither_waker_nor_listener()),
        ),
    ];
    let results = Array::new();
    for (name, test) in tests {
        let started = Date::now();
        let failure = match test.await {
            Ok(()) => JsValue::NULL,
            Err(error) => format!("{error:#}").into(),
        };
        stop_listening();
        let result = Object::new();
        for (key, value) in [
            ("name", JsValue::from(name)),
            ("failure", failure),
            ("milliseconds", (Date::now() - started).into()),
        ] {
            let _ = Reflect::set(&result, &key.into(), &value);
        }
        results.push(&result);
    }
    results
}
