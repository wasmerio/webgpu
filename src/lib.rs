//! WebGPU (`webgpu.h`) for WASIX guests running on Wasmer.
//!
//! A guest links the small static shim in `guest/` (it defines every `wgpu*`
//! symbol of the standard header) and ends up importing raw functions from the
//! [`WEBGPU_MODULE_NAME`] module. This crate provides those imports:
//!
//! - natively, on top of [`wgpu`], for the Wasmer CLI and the Rust-based SDKs;
//! - in a browser (`js` feature on wasm32), by handing the guest JavaScript
//!   functions that talk to `navigator.gpu` directly.
//!
//! Embedders install [`WebGpuRuntimeHooks`] on their WASIX runtime, exactly as
//! they do with the N-API hooks:
//!
//! ```ignore
//! let ctx = wasmer_webgpu::WebGpuCtx::default();
//! runtime.with_instantiation_hook(ctx.runtime_hooks());
//! ```
//!
//! The host never calls into the guest. Callbacks, future bookkeeping and
//! mapped-range pointers live in the shim; the host only reports completed
//! events when the shim polls for them.

// Proving that the host state is `Send` walks deep into wgpu's object graph.
#![recursion_limit = "512"]

pub mod abi;
#[cfg(all(feature = "cli", not(target_arch = "wasm32")))]
pub mod cli;
mod ctx;
mod env;
// The native host: wgpu by default, Dawn with the `dawn` feature. Both are
// `crate::host` to the rest of the crate.
#[cfg(all(not(target_arch = "wasm32"), not(feature = "dawn")))]
mod host;
#[cfg(all(not(target_arch = "wasm32"), feature = "dawn"))]
#[path = "dawn/mod.rs"]
mod host;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub mod surface;
#[cfg(feature = "wasix")]
mod wasix;
#[cfg(all(target_arch = "wasm32", feature = "js"))]
mod web;
#[cfg(all(feature = "window", not(target_arch = "wasm32")))]
pub mod window;

pub use ctx::{
    WebGpuCtx, WebGpuCtxBuilder, WebGpuInstantiationState, WebGpuLimits, WebGpuRuntimeControl,
    WebGpuRuntimeHooks, WebGpuUsage,
};

/// Name of the import module every WebGPU import lives in.
pub const WEBGPU_MODULE_NAME: &str = abi::IMPORT_MODULE;
/// Import modules named `wasmer_webgpu_v<N>` are WebGPU ABI revisions.
pub const WEBGPU_MODULE_PREFIX: &str = "wasmer_webgpu_v";
/// Revision reported to the shim by `wasmer_wgpu_abi_version`: the major half
/// must match, the minor half only grows.
pub const WEBGPU_ABI_VERSION: u32 = 0x0000_0001;

/// Which revision of the WebGPU import ABI a module was built against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WebGpuAbi {
    /// `wasmer_webgpu_v0`, with only imports this build knows.
    V0,
    /// A newer revision, or a `v0` import this build does not provide.
    Unknown,
}

/// Reports whether `module` imports the WebGPU ABI, and which revision.
pub fn module_needs_webgpu(module: &wasmer::Module) -> Option<WebGpuAbi> {
    let mut found = None;
    for import in module.imports() {
        let namespace = import.module();
        if !namespace.starts_with(WEBGPU_MODULE_PREFIX) {
            continue;
        }
        let known = namespace == WEBGPU_MODULE_NAME && abi::import_sig(import.name()).is_some();
        found = Some(match (found, known) {
            (Some(WebGpuAbi::Unknown), _) | (_, false) => WebGpuAbi::Unknown,
            (_, true) => WebGpuAbi::V0,
        });
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasmer::{Module, Store};

    fn compile(wat: &str) -> Module {
        let store = Store::default();
        Module::new(&store, wat::parse_str(wat).expect("wat parses")).expect("module compiles")
    }

    #[test]
    fn modules_without_webgpu_imports_are_left_alone() {
        let module = compile(r#"(module (import "env" "f" (func)))"#);
        assert_eq!(module_needs_webgpu(&module), None);
    }

    #[test]
    fn known_imports_select_v0() {
        let module = compile(
            r#"(module
                (import "wasmer_webgpu_v0" "wgpuCreateInstance" (func (param i32) (result i32)))
                (import "wasmer_webgpu_v0" "wasmer_wgpu_poll" (func (param i32 i32) (result i32))))"#,
        );
        assert_eq!(module_needs_webgpu(&module), Some(WebGpuAbi::V0));
    }

    #[test]
    fn unknown_imports_and_revisions_are_reported() {
        let unknown_import =
            compile(r#"(module (import "wasmer_webgpu_v0" "wgpuFromTheFuture" (func)))"#);
        assert_eq!(
            module_needs_webgpu(&unknown_import),
            Some(WebGpuAbi::Unknown)
        );
        let unknown_revision =
            compile(r#"(module (import "wasmer_webgpu_v9" "wgpuCreateInstance" (func)))"#);
        assert_eq!(
            module_needs_webgpu(&unknown_revision),
            Some(WebGpuAbi::Unknown)
        );
    }

    #[test]
    fn every_import_has_a_unique_name() {
        let mut names: Vec<_> = abi::IMPORTS.iter().map(|sig| sig.name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total);
    }
}
