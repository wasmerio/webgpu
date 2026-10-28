//! The browser backend.
//!
//! In a JavaScript realm the guest's WebGPU imports are plain JavaScript
//! functions (`js/webgpu_host.js`): a call goes from the guest's wasm straight
//! into JavaScript and on to the browser's WebGPU, with no Rust in between.
//! This module only does the wiring: it asks the JavaScript host for its
//! functions, checks them against the import table generated for both
//! backends, and hands them to Wasmer as the instance's imports.
//!
//! The few imports that have to wait for the browser (a timed wait, a frame
//! boundary, a canvas arriving from another thread) return promises. Those go
//! through Wasmer's asynchronous host functions, which suspend the guest with
//! JSPI until the promise settles.
//!
//! WebGPU objects belong to the JavaScript realm that created them, so each
//! guest instance (each web worker) gets its own host. A handle created on one
//! guest thread is not usable on another; `webgpu.h` allows that restriction.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, anyhow, bail};
use js_sys::{Array, BigInt, Function as JsFunction, Object, Promise, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use wasmer::js::AsJs;
use wasmer::{
    Exports, ExternType, Function, FunctionEnv, FunctionType, Imports, Memory, Module,
    RuntimeError, StoreMut, Type, Value,
};

use crate::abi::{self, ImportSig, ValType};
use crate::ctx::WebGpuCtx;
use crate::env::WebGpuEnv;

#[wasm_bindgen(module = "/js/webgpu_host.js")]
extern "C" {
    #[wasm_bindgen(js_name = createWebGpuHost, catch)]
    fn create_webgpu_host(options: &JsValue) -> Result<JsValue, JsValue>;
}

// Importing from the generated tables makes wasm-bindgen ship that file next
// to the host, which imports it; the value doubles as a check that both
// backends were generated from the same ABI description.
#[wasm_bindgen(module = "/js/webgpu_abi.gen.js")]
extern "C" {
    #[wasm_bindgen(thread_local_v2, js_name = IMPORT_MODULE)]
    static JS_IMPORT_MODULE: JsValue;
}

#[wasm_bindgen(inline_js = "
export function wasmer_webgpu_default_get_canvas(selector, scope) {
  // The embedding JavaScript decides what a selector means. The Wasmer SDK
  // installs a resolver in each worker that fetches the canvas from the page;
  // a plain map is enough for embedders with everything in one realm.
  const resolve = globalThis.__wasmerWebGpuGetCanvas;
  if (typeof resolve === 'function') return resolve(selector, scope);
  const canvases = globalThis.__wasmerWebGpuCanvases;
  return canvases && typeof canvases.get === 'function' ? canvases.get(selector) ?? null : null;
}
export function wasmer_webgpu_has_jspi() {
  return typeof WebAssembly.Suspending === 'function' && typeof WebAssembly.promising === 'function';
}
export function wasmer_webgpu_host_created(host, scope) {
  // An embedder with work of its own in this worker learns of each host, for
  // what the host offers beyond the guest's imports: `copyExternalImage`
  // hands the guest an image the embedder holds.
  const created = globalThis.__wasmerWebGpuHostCreated;
  if (typeof created === 'function') created(host, scope);
}
")]
extern "C" {
    #[wasm_bindgen(js_name = wasmer_webgpu_default_get_canvas)]
    fn default_get_canvas(selector: &JsValue, scope: u32) -> JsValue;
    #[wasm_bindgen(js_name = wasmer_webgpu_has_jspi)]
    fn has_jspi() -> bool;
    #[wasm_bindgen(js_name = wasmer_webgpu_host_created, catch)]
    fn host_created(host: &JsValue, scope: u32) -> Result<(), JsValue>;
}

/// The JavaScript host of one guest instance, with the callbacks it was given.
pub(crate) struct JsHost {
    host: JsValue,
    /// The context's canvas scope, which the embedder is told with the host.
    scope: u32,
    _get_canvas: Closure<dyn Fn(JsValue) -> JsValue>,
    _is_terminated: Closure<dyn Fn() -> bool>,
    _event_waker: Option<Closure<dyn Fn()>>,
}

// SAFETY: a JavaScript value never leaves the realm (the worker) it was
// created in: the instance, its imports and this value are created, used and
// dropped on that one thread. The marker only satisfies `FunctionEnv`'s bound.
unsafe impl Send for JsHost {}

impl Drop for JsHost {
    fn drop(&mut self) {
        if let Ok(dispose) = Reflect::get(&self.host, &"dispose".into())
            && let Some(dispose) = dispose.dyn_ref::<JsFunction>()
        {
            let _ = dispose.call0(&self.host);
        }
    }
}

fn js_error(what: &str, error: &JsValue) -> anyhow::Error {
    let detail = error
        .as_string()
        .or_else(|| {
            Reflect::get(error, &"message".into())
                .ok()
                .and_then(|message| message.as_string())
        })
        .unwrap_or_else(|| format!("{error:?}"));
    anyhow!("{what}: {detail}")
}

fn wasm_type(ty: ValType) -> Type {
    match ty {
        ValType::I32 => Type::I32,
        ValType::I64 => Type::I64,
        ValType::F32 => Type::F32,
        ValType::F64 => Type::F64,
    }
}

fn function_type(sig: &ImportSig) -> FunctionType {
    FunctionType::new(
        sig.params
            .iter()
            .copied()
            .map(wasm_type)
            .collect::<Vec<_>>(),
        sig.results
            .iter()
            .copied()
            .map(wasm_type)
            .collect::<Vec<_>>(),
    )
}

/// A wasm value as JavaScript sees it across the import boundary: numbers,
/// and BigInt for i64.
fn to_js(value: &Value) -> JsValue {
    match value {
        Value::I32(v) => JsValue::from(*v),
        Value::I64(v) => BigInt::from(*v).into(),
        Value::F32(v) => JsValue::from(*v),
        Value::F64(v) => JsValue::from(*v),
        _ => JsValue::UNDEFINED,
    }
}

fn from_js(ty: ValType, value: &JsValue) -> Value {
    match ty {
        // `as` saturates, and JavaScript hands handles back as doubles that
        // may exceed i32::MAX: go through i64 to wrap instead.
        ValType::I32 => Value::I32(value.as_f64().unwrap_or(0.0) as i64 as i32),
        ValType::I64 => Value::I64(
            value
                .clone()
                .dyn_into::<BigInt>()
                .ok()
                .and_then(|big| i64::try_from(big).ok())
                .unwrap_or_else(|| value.as_f64().unwrap_or(0.0) as i64),
        ),
        ValType::F32 => Value::F32(value.as_f64().unwrap_or(0.0) as f32),
        ValType::F64 => Value::F64(value.as_f64().unwrap_or(0.0)),
    }
}

/// An import whose JavaScript implementation returns a promise: the guest is
/// suspended until it settles.
fn suspending_import(
    store: &mut StoreMut<'_>,
    env: &FunctionEnv<WebGpuEnv>,
    sig: &'static ImportSig,
    function: JsFunction,
) -> Function {
    Function::new_with_env_async(
        store,
        env,
        function_type(sig),
        move |_env, args: &[Value]| {
            let arguments: Array = args.iter().map(to_js).collect();
            let outcome = function.apply(&JsValue::NULL, &arguments);
            async move {
                let failed = |error: JsValue| {
                    RuntimeError::new(format!(
                        "{}: {}",
                        sig.name,
                        js_error("JavaScript host", &error)
                    ))
                };
                let returned = outcome.map_err(failed)?;
                // A host that could answer at once may not have returned a promise.
                let settled = match returned.dyn_into::<Promise>() {
                    Ok(promise) => JsFuture::from(promise).await.map_err(failed)?,
                    Err(value) => value,
                };
                Ok(sig
                    .results
                    .iter()
                    .map(|ty| from_js(*ty, &settled))
                    .collect::<Vec<_>>())
            }
        },
    )
}

/// Creates the JavaScript host for a new instance and registers its functions
/// as the instance's WebGPU imports.
pub(crate) fn add_imports(
    ctx: &WebGpuCtx,
    module: &Module,
    store: &mut StoreMut<'_>,
    env: &FunctionEnv<WebGpuEnv>,
    imports: &mut Imports,
) -> Result<()> {
    let tables = JS_IMPORT_MODULE.with(JsValue::as_string);
    if tables.as_deref() != Some(abi::IMPORT_MODULE) {
        bail!(
            "the JavaScript WebGPU tables are for {tables:?}, this build speaks `{}`",
            abi::IMPORT_MODULE
        );
    }

    let jspi = has_jspi();
    let options = Object::new();
    let set = |target: &Object, key: &str, value: JsValue| {
        Reflect::set(target, &key.into(), &value)
            .map(|_| ())
            .map_err(|error| js_error("setting a WebGPU host option", &error))
    };
    let scope = ctx.inner.canvas_scope;
    let get_canvas = Closure::<dyn Fn(JsValue) -> JsValue>::new(move |selector: JsValue| {
        default_get_canvas(&selector, scope)
    });
    // The context lives in the embedder's shared memory, so the flag an
    // embedder sets from any thread is the one this instance's worker reads.
    let stopped = Arc::clone(&ctx.inner);
    let is_terminated =
        Closure::<dyn Fn() -> bool>::new(move || stopped.stopped.load(Ordering::Acquire));
    set(&options, "getCanvas", get_canvas.as_ref().clone())?;
    set(&options, "isTerminated", is_terminated.as_ref().clone())?;
    // The host calls it in this worker, when an event becomes ready.
    let event_waker = ctx
        .inner
        .event_waker
        .clone()
        .map(|wake| Closure::<dyn Fn()>::new(move || wake()));
    if let Some(waker) = &event_waker {
        set(&options, "eventWaker", waker.as_ref().clone())?;
    }
    set(&options, "suspending", JsValue::from(jspi))?;
    set(
        &options,
        "forceFallbackAdapter",
        JsValue::from(ctx.inner.force_fallback_adapter),
    )?;
    let limits = Object::new();
    let configured = &ctx.inner.limits;
    for (key, value) in [
        ("maxDevices", configured.max_devices.map(|v| v as f64)),
        ("maxObjects", configured.max_objects.map(|v| v as f64)),
        (
            "maxPendingEvents",
            configured.max_pending_events.map(|v| v as f64),
        ),
        (
            "maxGpuMemoryBytes",
            configured.max_gpu_memory_bytes.map(|v| v as f64),
        ),
    ] {
        if let Some(value) = value {
            set(&limits, key, JsValue::from(value))?;
        }
    }
    set(&options, "limits", limits.into())?;

    let host = create_webgpu_host(&options)
        .map_err(|error| js_error("creating the JavaScript WebGPU host", &error))?;
    let functions = Reflect::get(&host, &"imports".into())
        .map_err(|error| js_error("reading the JavaScript WebGPU imports", &error))?;
    // The host calls these for as long as it lives; it owns them from here.
    let host = JsHost {
        host,
        scope,
        _get_canvas: get_canvas,
        _is_terminated: is_terminated,
        _event_waker: event_waker,
    };

    let mut exports = Exports::new();
    for import in module.imports() {
        if import.module() != abi::IMPORT_MODULE {
            continue;
        }
        let ExternType::Function(wanted) = import.ty() else {
            bail!("WebGPU import `{}` is not a function", import.name());
        };
        let sig = abi::import_sig(import.name())
            .with_context(|| format!("unknown WebGPU import `{}`", import.name()))?;
        let ty = function_type(sig);
        // A module that reached this worker as a bare `WebAssembly.Module`
        // has no type information: Wasmer then reports every function import
        // as `() -> ()`, which says nothing about what the guest declared.
        let untyped = wanted.params().is_empty() && wanted.results().is_empty();
        if *wanted != ty && !untyped {
            bail!(
                "WebGPU import `{}` has signature {wanted}, this build provides {ty}",
                import.name()
            );
        }
        let function = Reflect::get(&functions, &sig.name.into())
            .ok()
            .and_then(|value| value.dyn_into::<JsFunction>().ok())
            .with_context(|| format!("the JavaScript WebGPU host has no `{}`", sig.name))?;
        // Without JSPI nothing can suspend the guest: the host was told so
        // (`suspending`), and these imports then answer without blocking.
        let function = if sig.suspending && jspi {
            suspending_import(store, env, sig, function)
        } else {
            Function::from_jsvalue(store, &ty, &function.into())
                .map_err(|error| anyhow!("wrapping WebGPU import `{}`: {error:?}", sig.name))?
        };
        exports.insert(sig.name, function);
    }
    imports.register_namespace(abi::IMPORT_MODULE, exports);
    env.as_mut(store).js_host = Some(host);
    Ok(())
}

/// Tells the instance's JavaScript host which memory its pointers refer to,
/// then tells the embedder about the host: only a host whose instance exists
/// is announced. It is disposed of with the instance, after which
/// `copyExternalImage` answers false.
pub(crate) fn configure_instance(
    store: &mut StoreMut<'_>,
    env: &FunctionEnv<WebGpuEnv>,
    memory: &Memory,
) -> Result<()> {
    let memory = memory.as_jsvalue(store);
    let data = env.as_ref(store);
    let host = data
        .js_host
        .as_ref()
        .context("the JavaScript WebGPU host is missing for this instance")?;
    let set_memory = Reflect::get(&host.host, &"setMemory".into())
        .ok()
        .and_then(|value| value.dyn_into::<JsFunction>().ok())
        .context("the JavaScript WebGPU host has no `setMemory`")?;
    set_memory
        .call1(&host.host, &memory)
        .map_err(|error| js_error("binding guest memory to the JavaScript WebGPU host", &error))?;
    host_created(&host.host, host.scope)
        .map_err(|error| js_error("the embedder's __wasmerWebGpuHostCreated", &error))?;
    Ok(())
}
