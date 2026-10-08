//! Runs a WASIX module with the WebGPU imports installed.
//!
//! This is the standalone runner behind the `webgpu_wasmer` binary and the
//! manifest tests. Embedders do the same thing with their own runtime: build a
//! [`WebGpuCtx`] and register its hooks.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use virtual_fs::AsyncReadExt;
use wasmer::sys::{Cranelift, EngineBuilder, Features};
use wasmer::{Module, Store};
use wasmer_types::ModuleHash;
use wasmer_wasix::{
    Pipe, PluggableRuntime, WasiError, WasiRuntimeError,
    runners::wasi::{RuntimeOrEngine, WasiRunner},
    runtime::task_manager::tokio::TokioTaskManager,
};

use crate::WebGpuCtx;

/// What a captured run produced.
#[derive(Debug, Clone)]
pub struct RunOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

fn create_store() -> Store {
    // wasixcc builds use the exception-handling proposal for setjmp/longjmp
    // and C++ exceptions.
    let mut features = Features::default();
    features.exceptions(true);
    let engine = EngineBuilder::new(Cranelift::default())
        .set_features(Some(features))
        .engine();
    Store::new(engine)
}

fn drain(
    mut pipe: Pipe,
    mut echo: Option<Box<dyn Write + Send>>,
) -> std::thread::JoinHandle<Result<String>> {
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to create the stdio drain runtime")?;
        let mut captured = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            let n = runtime
                .block_on(pipe.read(&mut chunk))
                .context("failed reading a WASIX stdio pipe")?;
            if n == 0 {
                break;
            }
            if let Some(echo) = echo.as_mut() {
                echo.write_all(&chunk[..n]).ok();
                echo.flush().ok();
            }
            captured.extend_from_slice(&chunk[..n]);
        }
        Ok(String::from_utf8_lossy(&captured).into_owned())
    })
}

fn run(
    ctx: &WebGpuCtx,
    wasm_path: &Path,
    args: &[String],
    configure: impl FnOnce(&mut WasiRunner),
) -> Result<i32> {
    let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to create the tokio runtime for WASIX")?;
    let _guard = tokio_runtime.enter();

    let wasm = std::fs::read(wasm_path)
        .with_context(|| format!("failed to read {}", wasm_path.display()))?;
    let store = create_store();
    let module = Module::new(&store, &wasm).context("failed to compile the wasm module")?;
    let module_hash = ModuleHash::sha256(&wasm);

    let mut runner = WasiRunner::new();
    runner.with_args(args.iter().cloned());
    configure(&mut runner);

    let task_manager = Arc::new(TokioTaskManager::new(tokio::runtime::Handle::current()));
    let mut runtime = PluggableRuntime::new(task_manager);
    runtime.set_engine(store.engine().clone());
    runtime.with_instantiation_hook(ctx.runtime_hooks());

    let program = wasm_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("guest");
    match runner.run_wasm(
        RuntimeOrEngine::Runtime(Arc::new(runtime)),
        program,
        module,
        module_hash,
    ) {
        Ok(()) => Ok(0),
        Err(error) => match exit_code(&error) {
            Some(code) => Ok(code),
            None => Err(error).context("failed to run the WASIX module"),
        },
    }
}

/// The guest's exit code, if `error` is a guest exiting rather than the run
/// failing. The runner wraps it at different depths depending on how the
/// process ended.
fn exit_code(error: &anyhow::Error) -> Option<i32> {
    error.chain().find_map(|cause| {
        if let Some(WasiError::Exit(code)) = cause.downcast_ref::<WasiError>() {
            return Some(i32::from(*code));
        }
        cause
            .downcast_ref::<WasiRuntimeError>()
            .and_then(WasiRuntimeError::as_exit_code)
            .map(i32::from)
    })
}

/// Runs `wasm_path` with the process's own stdio. Returns the exit code.
pub fn run_wasix_main(ctx: &WebGpuCtx, wasm_path: &Path, args: &[String]) -> Result<i32> {
    run(ctx, wasm_path, args, |runner| {
        runner
            .with_stdin(Box::new(virtual_fs::host_fs::Stdin::default()))
            .with_stdout(Box::new(virtual_fs::host_fs::Stdout::default()))
            .with_stderr(Box::new(virtual_fs::host_fs::Stderr::default()));
    })
}

/// Runs `wasm_path` and captures what it prints. With `echo`, the output is
/// also forwarded to this process's stdout and stderr as it arrives.
pub fn run_wasix_main_capture(
    ctx: &WebGpuCtx,
    wasm_path: &Path,
    args: &[String],
    echo: bool,
) -> Result<RunOutput> {
    let (stdout_tx, stdout_rx) = Pipe::channel();
    let (stderr_tx, stderr_rx) = Pipe::channel();
    let stdout = drain(
        stdout_rx,
        echo.then(|| Box::new(std::io::stdout()) as Box<dyn Write + Send>),
    );
    let stderr = drain(
        stderr_rx,
        echo.then(|| Box::new(std::io::stderr()) as Box<dyn Write + Send>),
    );
    let exit_code = run(ctx, wasm_path, args, |runner| {
        runner
            .with_stdout(Box::new(stdout_tx))
            .with_stderr(Box::new(stderr_tx));
    })?;
    let join = |handle: std::thread::JoinHandle<Result<String>>| {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("a stdio drain thread panicked"))?
    };
    Ok(RunOutput {
        exit_code,
        stdout: join(stdout)?,
        stderr: join(stderr)?,
    })
}
