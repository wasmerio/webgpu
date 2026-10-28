//! Per-instance state behind the WebGPU imports.

#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

/// One of these backs the imports of each guest instance (the process's main
/// module and every thread it spawns each get their own).
pub struct WebGpuEnv {
    /// The instance's linear memory, known once the instance exists.
    pub(crate) memory: Option<wasmer::Memory>,
    /// The process-wide host state. Threads of one process share it, which is
    /// what lets a handle created on one guest thread work on another.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) host: Option<Arc<crate::host::Host>>,
    /// How to tell that the process was killed, where the runtime lets that
    /// be seen on its memory (see `host::killed`).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) kill_probe: Option<wasmer::MemoryOps>,
    /// The JavaScript host whose functions are this instance's imports. Held
    /// here so that it lives exactly as long as the instance.
    #[cfg(all(target_arch = "wasm32", feature = "js"))]
    pub(crate) js_host: Option<crate::web::JsHost>,
}

impl WebGpuEnv {
    pub(crate) fn new() -> Self {
        Self {
            memory: None,
            #[cfg(not(target_arch = "wasm32"))]
            host: None,
            #[cfg(not(target_arch = "wasm32"))]
            kill_probe: None,
            #[cfg(all(target_arch = "wasm32", feature = "js"))]
            js_host: None,
        }
    }
}
