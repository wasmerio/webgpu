//! What an embedder controls: where surfaces present, how much a guest may
//! hold, and stopping a guest that is blocked on the GPU.
//!
//! These run the conformance programs under non-default contexts, so they
//! need `wasixcc` and a GPU adapter like `manifest_tests`.

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use wasmer_webgpu::cli::{RunOutput, run_wasix_main_capture};
use wasmer_webgpu::surface::{
    CaptureFrames, FrameColorSpace, FrameFormat, FrameToneMapping, SingleSurface, SurfaceTarget,
};
use wasmer_webgpu::{WebGpuCtx, WebGpuCtxBuilder};

fn program(name: &str) -> PathBuf {
    static BUILD: Once = Once::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    BUILD.call_once(|| {
        let status = Command::new(root.join("tests/build-test-wasix.sh"))
            .status()
            .expect("tests/build-test-wasix.sh runs (is wasixcc installed?)");
        assert!(status.success(), "building the test programs failed");
    });
    root.join(format!("tests/programs/out/{name}.wasm"))
}

fn builder() -> WebGpuCtxBuilder {
    WebGpuCtx::builder()
        .force_fallback_adapter(std::env::var_os("WEBGPU_TEST_FALLBACK_ADAPTER").is_some())
}

fn run(ctx: &WebGpuCtx, name: &str) -> RunOutput {
    run_wasix_main_capture(ctx, &program(name), &[], false).expect("the module runs")
}

#[test]
fn headless_surfaces_hand_presented_frames_to_the_sink() {
    let sink = CaptureFrames::new(32, 24);
    let ctx = builder()
        .surface_provider(Arc::new(SingleSurface(SurfaceTarget::Headless(
            sink.clone(),
        ))))
        .build();
    let output = run(&ctx, "surface");
    assert_eq!(output.exit_code, 0, "{}{}", output.stdout, output.stderr);

    // The program presents red, green, then blue, at the size the sink asked for.
    let frame = sink.last_frame().expect("a frame was presented");
    assert_eq!((frame.index, frame.width, frame.height), (2, 32, 24));
    assert_eq!(frame.pixels.len(), 32 * 24 * 4);
    let blue: [u8; 4] = match frame.format {
        FrameFormat::Rgba8 => [0, 0, 255, 255],
        FrameFormat::Bgra8 => [255, 0, 0, 255],
    };
    assert!(
        frame
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| *pixel == blue),
        "the last frame is solid blue"
    );
    // The program asked for nothing special about its colours.
    assert_eq!(frame.color_space, FrameColorSpace::Srgb);
    assert_eq!(frame.tone_mapping, FrameToneMapping::Standard);
}

#[test]
fn a_frame_sink_is_told_how_the_guest_means_its_colours() {
    let sink = CaptureFrames::new(16, 16);
    let ctx = builder()
        .surface_provider(Arc::new(SingleSurface(SurfaceTarget::Headless(
            sink.clone(),
        ))))
        .build();
    // Configures its surface for display-p3 with extended tone mapping.
    let output = run(&ctx, "surface_colors");
    assert_eq!(output.exit_code, 0, "{}{}", output.stdout, output.stderr);
    let frame = sink.last_frame().expect("a frame was presented");
    assert_eq!(frame.color_space, FrameColorSpace::DisplayP3);
    assert_eq!(frame.tone_mapping, FrameToneMapping::Extended);
}

/// Whether the native host's backend writes a timestamp from an encoder.
/// wgpu does wherever the adapter can.
#[cfg(not(feature = "dawn"))]
fn encoder_timestamps_written() -> bool {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        force_fallback_adapter: std::env::var_os("WEBGPU_TEST_FALLBACK_ADAPTER").is_some(),
        ..Default::default()
    }))
    .expect("an adapter for the tests");
    adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
}

/// Dawn never does for a guest: it keeps them behind its unsafe APIs, as a
/// browser does, since the WebGPU specification dropped them.
#[cfg(feature = "dawn")]
fn encoder_timestamps_written() -> bool {
    false
}

/// `wgpuCommandEncoderWriteTimestamp` goes through wherever the backend
/// writes a timestamp from an encoder, and is a validation error elsewhere.
#[test]
fn encoder_timestamps_are_written_where_the_backend_can() {
    let able = encoder_timestamps_written();

    let output = run(&builder().build(), "timestamps");
    assert_eq!(output.exit_code, 0, "{}{}", output.stdout, output.stderr);
    let expected = if able { "written" } else { "refused" };
    assert!(
        output
            .stderr
            .contains(&format!("encoder timestamp: {expected}")),
        "an adapter that {} write encoder timestamps:\n{}",
        if able { "can" } else { "cannot" },
        output.stderr
    );
}

#[test]
fn the_object_limit_fails_creation_instead_of_the_host() {
    // Instance, adapter and device fit; the queue and everything after do not.
    let ctx = builder().max_objects(3).build();
    let output = run(&ctx, "compute_double");
    assert_eq!(output.exit_code, 1);
    assert!(output.stdout.contains("FAILED"), "{}", output.stdout);
}

#[test]
fn the_device_limit_fails_the_second_request() {
    let ctx = builder().max_devices(1).build();
    let output = run(&ctx, "instance_info");
    assert_eq!(output.exit_code, 1);
    assert!(
        output.stdout.contains("device: ok") && output.stdout.contains("FAILED"),
        "the first device works, the second request fails:\n{}",
        output.stdout
    );
}

#[test]
fn the_gpu_memory_budget_is_an_out_of_memory_error() {
    // A 64x64 RGBA8 render target is 16 KiB; allow one.
    let ctx = builder().max_gpu_memory_bytes(1024).build();
    let output = run(&ctx, "render_triangle");
    // The texture is refused (as an out-of-memory error on the device), so
    // the program's own check on the handle fails.
    assert_eq!(output.exit_code, 1);
    assert!(
        output.stdout.contains("texture != NULL"),
        "{}",
        output.stdout
    );

    let roomy = builder().max_gpu_memory_bytes(64 << 20).build();
    assert_eq!(run(&roomy, "render_triangle").exit_code, 0);
}

#[test]
fn destroyed_and_released_storage_returns_to_the_budget() {
    let ctx = builder().max_gpu_memory_bytes(1 << 20).build();
    let output = run(&ctx, "budget");
    assert_eq!(
        output.stdout,
        "768 KiB: fits\n\
         768 KiB more: out of memory\n\
         after destroying the first: fits\n\
         after releasing it: fits\n\
         textures: counted, and given back\n\
         ok\n",
        "{}",
        output.stderr
    );
    assert_eq!(output.exit_code, 0);
    // The guest released everything, so nothing is charged any more.
    assert_eq!(ctx.usage().gpu_memory_bytes, 0);
}

#[test]
fn a_guest_killed_while_it_waits_lets_go() {
    // The guest kills itself from a second thread while its first one waits
    // on the GPU for something that never happens, without a system call.
    let ctx = builder().build();
    program("killed"); // build before the clock starts
    let outcome = run_wasix_main_capture(&ctx, &program("killed"), &[], false);
    match outcome {
        Ok(output) => {
            assert!(output.stdout.contains("killing"), "{}", output.stdout);
            assert!(
                !output.stdout.contains("the wait returned"),
                "{}",
                output.stdout
            );
            assert_ne!(output.exit_code, 0, "a killed program did not succeed");
        }
        Err(error) => {
            let error = format!("{error:#}");
            assert!(error.contains("stopped"), "{error}");
        }
    }
    // The run is over either way: a killed process is reported as ended
    // whatever its threads are doing. What tells is whether the thread that
    // was waiting came out of the host, and with it everything it held.
    let ended = Instant::now();
    while ctx.usage().processes != 0 {
        assert!(
            ended.elapsed() < Duration::from_secs(5),
            "the killed guest still holds {:?}",
            ctx.usage()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn terminate_all_gets_a_guest_out_of_a_wait_and_stays_in_effect() {
    let ctx = builder().build();
    let control = ctx.runtime_control();
    program("wait_forever"); // build before the clock starts
    let guest = {
        let ctx = ctx.clone();
        std::thread::spawn(move || run(&ctx, "wait_forever"))
    };
    // The guest waits right after it gets its device: watch for the device,
    // then give it a moment to get into the wait.
    let started = Instant::now();
    while ctx.usage().devices == 0 {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "the guest never got a device"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(ctx.usage().processes, 1);
    std::thread::sleep(Duration::from_millis(500));
    assert!(!guest.is_finished(), "the guest is blocked in its wait");
    let stopped = Instant::now();
    control.terminate_all();
    let output = guest.join().expect("the guest thread does not panic");
    assert!(
        stopped.elapsed() < Duration::from_secs(5),
        "the wait ended promptly"
    );
    // Destroying its device is how the guest learns: the lost callback runs
    // (reason 2 is WGPUDeviceLostReason_Destroyed) and the wait returns.
    assert_eq!(output.exit_code, 0, "{}{}", output.stdout, output.stderr);
    assert!(
        output.stdout.contains("device lost: reason 2"),
        "{}",
        output.stdout
    );

    // Sticky: a guest started afterwards gets no adapter.
    let later = run(&ctx, "compute_double");
    assert_eq!(later.exit_code, 1);
    assert!(
        later.stdout.contains("request adapter failed"),
        "{}",
        later.stdout
    );
}
