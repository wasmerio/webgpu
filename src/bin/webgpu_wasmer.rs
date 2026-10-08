//! `webgpu_wasmer <module.wasm> [args...]`: runs a WASIX module with WebGPU.
//!
//! A small stand-in for `wasmer run` that this repository can build and test
//! on its own. Surfaces are headless by default: a guest can configure and
//! present to one, and `--frames <dir>` saves what it presented. Built with
//! the `window` feature, `--window` shows them in windows instead.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use wasmer_webgpu::WebGpuCtx;
use wasmer_webgpu::surface::{Frame, FrameFormat, FrameSink, SingleSurface, SurfaceTarget};

struct Options {
    wasm: PathBuf,
    args: Vec<String>,
    frames: Option<PathBuf>,
    size: (u32, u32),
    fallback_adapter: bool,
    #[cfg_attr(not(feature = "window"), allow(dead_code))]
    window: bool,
}

const USAGE: &str = "usage: webgpu_wasmer [--window | --frames <dir>] [--size <width>x<height>] \
                     [--fallback-adapter] <module.wasm> [args...]";

fn parse() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut frames = None;
    let mut size = (640, 480);
    let mut fallback_adapter = false;
    let mut window = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--frames" => frames = Some(PathBuf::from(args.next().ok_or(USAGE)?)),
            "--size" => {
                let value = args.next().ok_or(USAGE)?;
                let (w, h) = value.split_once('x').ok_or(USAGE)?;
                size = (
                    w.parse().map_err(|_| USAGE.to_owned())?,
                    h.parse().map_err(|_| USAGE.to_owned())?,
                );
            }
            "--fallback-adapter" => fallback_adapter = true,
            "--window" if cfg!(feature = "window") => window = true,
            "--window" => {
                return Err("webgpu_wasmer was built without windows; \
                            rebuild with `--features window`"
                    .to_owned());
            }
            "-h" | "--help" => return Err(USAGE.to_owned()),
            _ => {
                return Ok(Options {
                    wasm: PathBuf::from(arg),
                    args: args.collect(),
                    frames,
                    size,
                    fallback_adapter,
                    window,
                });
            }
        }
    }
    Err(USAGE.to_owned())
}

/// Writes every presented frame as a binary PPM, a format any image tool
/// reads and that needs no encoder here.
struct SaveFrames {
    directory: PathBuf,
    size: (u32, u32),
}

impl FrameSink for SaveFrames {
    fn size(&self) -> (u32, u32) {
        self.size
    }

    fn wants_pixels(&self) -> bool {
        true
    }

    fn present(&self, frame: Frame<'_>) {
        let Some(pixels) = frame.pixels else {
            return;
        };
        let mut out = format!("P6\n{} {}\n255\n", frame.width, frame.height).into_bytes();
        for pixel in pixels.chunks_exact(4) {
            match frame.format {
                FrameFormat::Rgba8 => out.extend_from_slice(&pixel[..3]),
                FrameFormat::Bgra8 => out.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]),
            }
        }
        let path = self.directory.join(format!("frame-{:05}.ppm", frame.index));
        if let Err(error) = std::fs::write(&path, out) {
            eprintln!("webgpu_wasmer: cannot write {}: {error}", path.display());
        }
    }
}

fn main() -> ExitCode {
    let options = match parse() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(2);
        }
    };

    let builder = WebGpuCtx::builder().force_fallback_adapter(options.fallback_adapter);
    #[cfg(feature = "window")]
    if options.window {
        return run_in_windows(builder, options);
    }
    let builder = match &options.frames {
        Some(directory) => {
            if let Err(error) = std::fs::create_dir_all(directory) {
                eprintln!(
                    "webgpu_wasmer: cannot create {}: {error}",
                    directory.display()
                );
                return ExitCode::from(2);
            }
            builder.surface_provider(Arc::new(SingleSurface(SurfaceTarget::Headless(Arc::new(
                SaveFrames {
                    directory: directory.clone(),
                    size: options.size,
                },
            )))))
        }
        None => builder.surface_provider(Arc::new(wasmer_webgpu::surface::HeadlessSurfaces::new(
            options.size.0,
            options.size.1,
        ))),
    };
    exit_code(wasmer_webgpu::cli::run_wasix_main(
        &builder.build(),
        &options.wasm,
        &options.args,
    ))
}

fn exit_code(outcome: anyhow::Result<i32>) -> ExitCode {
    match outcome {
        Ok(code) => ExitCode::from((code & 0xFF) as u8),
        Err(error) => {
            eprintln!("webgpu_wasmer: {error:#}");
            ExitCode::from(1)
        }
    }
}

/// Shows the guest's surfaces in windows. The event loop needs this (the
/// main) thread, so the guest runs on another one.
#[cfg(feature = "window")]
fn run_in_windows(builder: wasmer_webgpu::WebGpuCtxBuilder, options: Options) -> ExitCode {
    use wasmer_webgpu::window::{WindowExit, WindowHost, WindowOptions};

    let title = options.wasm.file_stem().map_or_else(
        || "Wasmer".to_owned(),
        |name| name.to_string_lossy().into_owned(),
    );
    let host = WindowHost::with_options(WindowOptions {
        title,
        size: options.size,
        ..WindowOptions::default()
    });
    let ctx = builder.surface_provider(host.surfaces()).build();
    let control = ctx.runtime_control();
    let guest = move || wasmer_webgpu::cli::run_wasix_main(&ctx, &options.wasm, &options.args);
    match host.run(guest) {
        Ok(WindowExit::Finished(outcome)) => exit_code(outcome),
        // Closing the window is how the user quits a program that draws.
        Ok(WindowExit::Closed) => {
            control.terminate_all();
            std::process::exit(0)
        }
        Err(error) => {
            eprintln!("webgpu_wasmer: {error:#}");
            ExitCode::from(1)
        }
    }
}
