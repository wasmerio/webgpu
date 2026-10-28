//! Embedder-facing configuration and the runtime hooks that hand WebGPU
//! imports to guest instances.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result, bail};
use wasmer::{FunctionEnv, Imports, Instance, Module, StoreMut};

use crate::{WebGpuAbi, env::WebGpuEnv, module_needs_webgpu};

/// Ceilings on what one guest process may hold. `None` means unlimited.
#[derive(Debug, Clone, Default)]
pub struct WebGpuLimits {
    /// Devices alive at once.
    pub max_devices: Option<usize>,
    /// Live object handles (buffers, textures, pipelines, ...).
    pub max_objects: Option<usize>,
    /// Completed events the guest has not collected yet. Beyond it, further
    /// uncaptured errors are dropped (and counted) rather than queued.
    pub max_pending_events: Option<usize>,
    /// Bytes of buffer and texture storage the guest may have allocated. The
    /// staging buffers the host keeps for large texture uploads don't count:
    /// they hold at most 64 MiB per guest, and no more than this.
    pub max_gpu_memory_bytes: Option<u64>,
}

/// A snapshot of what the guests of a [`WebGpuCtx`] hold.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WebGpuUsage {
    /// Guest processes that have used WebGPU and are still alive.
    pub processes: usize,
    pub devices: usize,
    /// Live object handles of every kind.
    pub objects: usize,
    /// Bytes of buffer and texture storage (textures are estimated).
    pub gpu_memory_bytes: u64,
}

/// Builds a [`WebGpuCtx`].
#[derive(Default)]
pub struct WebGpuCtxBuilder {
    limits: WebGpuLimits,
    force_fallback_adapter: bool,
    #[cfg(not(target_arch = "wasm32"))]
    surfaces: Option<Arc<dyn crate::surface::SurfaceProvider>>,
    #[cfg(not(target_arch = "wasm32"))]
    event_waker: Option<Arc<dyn Fn() + Send + Sync>>,
    #[cfg(target_arch = "wasm32")]
    canvas_scope: u32,
}

impl WebGpuCtxBuilder {
    pub fn limits(mut self, limits: WebGpuLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn max_devices(mut self, max: usize) -> Self {
        self.limits.max_devices = Some(max);
        self
    }

    pub fn max_objects(mut self, max: usize) -> Self {
        self.limits.max_objects = Some(max);
        self
    }

    pub fn max_gpu_memory_bytes(mut self, bytes: u64) -> Self {
        self.limits.max_gpu_memory_bytes = Some(bytes);
        self
    }

    /// Only ever hand guests a software adapter, whatever they ask for.
    pub fn force_fallback_adapter(mut self, force: bool) -> Self {
        self.force_fallback_adapter = force;
        self
    }

    /// Where guest surfaces present to. The default is headless: surfaces
    /// work, and presented frames are dropped.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn surface_provider(mut self, provider: Arc<dyn crate::surface::SurfaceProvider>) -> Self {
        self.surfaces = Some(provider);
        self
    }

    /// Called whenever an event becomes ready for a guest of this context, so
    /// an embedder whose own loop waits on something else (window events, for
    /// example) can wake it. The guest still collects the event the usual way,
    /// by polling.
    ///
    /// It runs on whichever thread made the event ready, sometimes while that
    /// thread holds locks of this crate, so it must only signal. It may not
    /// call back into WebGPU, the store or the guest, nor wait for anything a
    /// thread inside WebGPU might hold: a lock such a thread takes, or a
    /// bounded channel it drains. Hold only weak references to what it wakes.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn event_waker(mut self, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        self.event_waker = Some(wake);
        self
    }

    /// Names the set of canvases the guests of this context may draw to.
    ///
    /// In a browser a guest runs in a web worker, and the canvases it presents
    /// to belong to the page: nothing this context could hold reaches across.
    /// Instead, when a guest creates a surface, its worker asks the embedding
    /// JavaScript for the canvas (`globalThis.__wasmerWebGpuGetCanvas(selector,
    /// scope)`), and this number tells that code whose canvases to look in.
    /// The default scope is 0.
    #[cfg(target_arch = "wasm32")]
    pub fn canvas_scope(mut self, scope: u32) -> Self {
        self.canvas_scope = scope;
        self
    }

    pub fn build(self) -> WebGpuCtx {
        WebGpuCtx {
            inner: Arc::new(CtxInner {
                limits: self.limits,
                force_fallback_adapter: self.force_fallback_adapter,
                #[cfg(not(target_arch = "wasm32"))]
                surfaces: self
                    .surfaces
                    .unwrap_or_else(|| Arc::new(crate::surface::HeadlessSurfaces::default())),
                #[cfg(not(target_arch = "wasm32"))]
                hosts: Mutex::new(crate::host::HostRegistry::default()),
                #[cfg(not(target_arch = "wasm32"))]
                event_waker: self.event_waker,
                #[cfg(target_arch = "wasm32")]
                canvas_scope: self.canvas_scope,
                stopped: AtomicBool::new(false),
            }),
        }
    }
}

pub(crate) struct CtxInner {
    pub(crate) limits: WebGpuLimits,
    pub(crate) force_fallback_adapter: bool,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) surfaces: Arc<dyn crate::surface::SurfaceProvider>,
    /// Host state of every live guest process started through this context.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) hosts: Mutex<crate::host::HostRegistry>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) event_waker: Option<Arc<dyn Fn() + Send + Sync>>,
    #[cfg(target_arch = "wasm32")]
    #[cfg_attr(not(feature = "js"), allow(dead_code))]
    pub(crate) canvas_scope: u32,
    pub(crate) stopped: AtomicBool,
}

impl std::fmt::Debug for CtxInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGpuCtx")
            .field("limits", &self.limits)
            .field("force_fallback_adapter", &self.force_fallback_adapter)
            .finish_non_exhaustive()
    }
}

/// Shared WebGPU configuration for the guests of one embedder (one app, one
/// sandbox). Cheap to clone.
#[derive(Clone, Debug)]
pub struct WebGpuCtx {
    pub(crate) inner: Arc<CtxInner>,
}

impl Default for WebGpuCtx {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl WebGpuCtx {
    pub fn builder() -> WebGpuCtxBuilder {
        WebGpuCtxBuilder::default()
    }

    pub fn limits(&self) -> &WebGpuLimits {
        &self.inner.limits
    }

    /// Hooks to register on the WASIX runtime (they implement
    /// `wasmer_wasix::runtime::InstantiationHook` with the `wasix` feature).
    pub fn runtime_hooks(&self) -> WebGpuRuntimeHooks {
        WebGpuRuntimeHooks { ctx: self.clone() }
    }

    /// What the guests of this context hold right now, summed over every
    /// live guest process.
    pub fn usage(&self) -> WebGpuUsage {
        // In a browser every guest thread's objects live in that thread's
        // JavaScript realm, out of reach from here: nothing is reported.
        #[cfg(target_arch = "wasm32")]
        let usage = WebGpuUsage::default();
        #[cfg(not(target_arch = "wasm32"))]
        let usage = {
            let hosts = self
                .inner
                .hosts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .live();
            let mut usage = WebGpuUsage {
                processes: hosts.len(),
                ..WebGpuUsage::default()
            };
            for host in hosts {
                let (devices, objects, gpu_memory_bytes) = host.usage();
                usage.devices += devices;
                usage.objects += objects;
                usage.gpu_memory_bytes += gpu_memory_bytes;
            }
            usage
        };
        usage
    }

    /// A handle for stopping every guest of this context.
    pub fn runtime_control(&self) -> WebGpuRuntimeControl {
        WebGpuRuntimeControl {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Stops the GPU work of every guest process of a context.
#[derive(Clone, Debug)]
pub struct WebGpuRuntimeControl {
    inner: Arc<CtxInner>,
}

impl WebGpuRuntimeControl {
    /// Wakes every guest thread blocked in a WebGPU wait and destroys every
    /// device, so killing an instance never has to wait for the GPU. Sticky:
    /// guests started afterwards get no adapters.
    pub fn terminate_all(&self) {
        self.inner.stopped.store(true, Ordering::Release);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let hosts = self
                .inner
                .hosts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .live();
            for host in hosts {
                host.terminate();
            }
        }
    }
}

/// Opaque per-instantiation state: returned by
/// [`WebGpuRuntimeHooks::add_imports`], handed back unmodified to
/// [`WebGpuRuntimeHooks::configure_instance`] for the instance created with
/// those imports.
pub struct WebGpuInstantiationState {
    env: Option<FunctionEnv<WebGpuEnv>>,
}

/// Runtime hooks that provide the WebGPU imports to WASIX guests.
///
/// Stateless: everything an instantiation needs travels in its
/// [`WebGpuInstantiationState`], so concurrent instantiations cannot receive
/// each other's state.
#[derive(Clone, Debug)]
pub struct WebGpuRuntimeHooks {
    ctx: WebGpuCtx,
}

impl WebGpuRuntimeHooks {
    /// Creates the WebGPU imports when `module` asks for them.
    pub fn additional_imports(
        &self,
        module: &Module,
        store: &mut StoreMut<'_>,
    ) -> Result<(Imports, WebGpuInstantiationState)> {
        let mut imports = Imports::new();
        let state = self.add_imports(module, store, &mut imports)?;
        Ok((imports, state))
    }

    /// Adds the WebGPU imports to an existing import object when `module` asks
    /// for them. Does nothing for every other module.
    pub fn add_imports(
        &self,
        module: &Module,
        store: &mut StoreMut<'_>,
        imports: &mut Imports,
    ) -> Result<WebGpuInstantiationState> {
        match module_needs_webgpu(module) {
            None => return Ok(WebGpuInstantiationState { env: None }),
            Some(WebGpuAbi::V0) => {}
            Some(WebGpuAbi::Unknown) => bail!(
                "this module imports a WebGPU ABI this build of wasmer-webgpu does not provide \
                 (it supports `{}`); rebuild the guest against the matching libwebgpu.a, or \
                 update the runtime",
                crate::WEBGPU_MODULE_NAME
            ),
        }

        let env = FunctionEnv::new(store, WebGpuEnv::new());
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut exports = wasmer::Exports::new();
            crate::abi::api::register(store, &env, &mut exports);
            imports.register_namespace(crate::WEBGPU_MODULE_NAME, exports);
        }
        #[cfg(all(target_arch = "wasm32", feature = "js"))]
        crate::web::add_imports(&self.ctx, module, store, &env, imports)?;
        #[cfg(all(target_arch = "wasm32", not(feature = "js")))]
        bail!("wasmer-webgpu needs the `js` feature on wasm32");

        Ok(WebGpuInstantiationState { env: Some(env) })
    }

    /// Binds the new instance's memory to the imports created for it.
    pub fn configure_instance(
        &self,
        module: &Module,
        store: &mut StoreMut<'_>,
        instance: &Instance,
        imported_memory: Option<&wasmer::Memory>,
        state: WebGpuInstantiationState,
    ) -> Result<()> {
        if module_needs_webgpu(module).is_none() {
            return Ok(());
        }
        let env = state.env.context(
            "missing WebGPU state for module instance setup \
             (the state was not created for this module's imports)",
        )?;
        let memory = match imported_memory {
            Some(memory) => memory.clone(),
            None => instance
                .exports
                .get_memory("memory")
                .context("WebGPU imports require the guest to have linear memory")?
                .clone(),
        };

        #[cfg(not(target_arch = "wasm32"))]
        {
            let host = crate::host::Host::for_memory(&self.ctx, store, &memory);
            let kill_probe = memory.as_shared(store).map(|shared| shared.ops());
            let data = env.as_mut(store);
            data.memory = Some(memory);
            data.host = Some(host);
            data.kill_probe = kill_probe;
        }
        #[cfg(all(target_arch = "wasm32", feature = "js"))]
        {
            crate::web::configure_instance(store, &env, &memory)?;
            env.as_mut(store).memory = Some(memory);
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "js")))]
        {
            let _ = (env, memory);
        }
        Ok(())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use wasmer::{AsStoreMut, Store};

    fn compile(store: &Store, wat: &str) -> Module {
        Module::new(store, wat::parse_str(wat).expect("wat parses")).expect("module compiles")
    }

    #[test]
    fn modules_without_webgpu_get_no_imports() {
        let mut store = Store::default();
        let module = compile(&store, r#"(module (memory (export "memory") 1))"#);
        let hooks = WebGpuCtx::default().runtime_hooks();
        let (imports, state) = hooks
            .additional_imports(&module, &mut store.as_store_mut())
            .expect("hook runs");
        assert_eq!(imports.iter().count(), 0);
        assert!(state.env.is_none());
    }

    #[test]
    fn every_declared_import_is_registered_with_its_signature() {
        // A module importing the entire ABI must instantiate: this is what
        // catches a signature drifting between the import table (shared with
        // the JavaScript backend) and the generated glue.
        let mut wat = String::from("(module\n");
        let ty = |t: &crate::abi::ValType| match t {
            crate::abi::ValType::I32 => "i32",
            crate::abi::ValType::I64 => "i64",
            crate::abi::ValType::F32 => "f32",
            crate::abi::ValType::F64 => "f64",
        };
        for sig in crate::abi::IMPORTS {
            let params: Vec<_> = sig.params.iter().map(ty).collect();
            let results: Vec<_> = sig.results.iter().map(ty).collect();
            wat.push_str(&format!(
                "(import \"{}\" \"{}\" (func (param {}) (result {})))\n",
                crate::WEBGPU_MODULE_NAME,
                sig.name,
                params.join(" "),
                results.join(" ")
            ));
        }
        wat.push_str("(memory (export \"memory\") 1))");

        let mut store = Store::default();
        let module = compile(&store, &wat);
        let hooks = WebGpuCtx::default().runtime_hooks();
        let (imports, state) = hooks
            .additional_imports(&module, &mut store.as_store_mut())
            .expect("imports register");
        let instance = Instance::new(&mut store, &module, &imports).expect("instantiates");
        hooks
            .configure_instance(&module, &mut store.as_store_mut(), &instance, None, state)
            .expect("configures");
    }

    #[test]
    fn unknown_abi_revisions_are_refused() {
        let mut store = Store::default();
        let module = compile(
            &store,
            r#"(module (import "wasmer_webgpu_v9" "wgpuCreateInstance" (func)))"#,
        );
        let hooks = WebGpuCtx::default().runtime_hooks();
        let error = hooks
            .additional_imports(&module, &mut store.as_store_mut())
            .err()
            .expect("unknown revision is refused");
        assert!(error.to_string().contains("wasmer_webgpu_v0"));
    }

    #[test]
    fn configure_instance_rejects_missing_state() {
        let mut store = Store::default();
        let module = compile(
            &store,
            r#"(module
                (import "wasmer_webgpu_v0" "wasmer_wgpu_abi_version" (func (result i32)))
                (memory (export "memory") 1))"#,
        );
        let hooks = WebGpuCtx::default().runtime_hooks();
        let (imports, _state) = hooks
            .additional_imports(&module, &mut store.as_store_mut())
            .expect("imports register");
        let instance = Instance::new(&mut store, &module, &imports).expect("instantiates");
        let error = hooks
            .configure_instance(
                &module,
                &mut store.as_store_mut(),
                &instance,
                None,
                WebGpuInstantiationState { env: None },
            )
            .expect_err("missing state is refused");
        assert!(error.to_string().contains("missing WebGPU state"));
    }
}
