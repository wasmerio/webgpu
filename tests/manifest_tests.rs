//! Runs every program of `tests/manifest.json` on the native host and
//! compares what it prints with the manifest.
//!
//! The programs are C, built for WASIX with `tests/build-test-wasix.sh`
//! (needs `wasixcc`). They need a GPU adapter; on a machine without one, set
//! `WEBGPU_TEST_FALLBACK_ADAPTER=1` to ask for a software adapter (lavapipe,
//! WARP) instead (empty or `0` is the same as unset). The Dawn host has no
//! such fallback; it picks a software adapter when that is the only one (see
//! docs/dawn.md). `WEBGPU_TEST_FILTER=<substring>` runs a subset.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;
use wasmer_webgpu::WebGpuCtx;

#[derive(Deserialize)]
struct Manifest {
    tests: Vec<Test>,
}

#[derive(Deserialize)]
struct Test {
    name: String,
    stdout: String,
    #[serde(default = "enabled")]
    native: bool,
    #[serde(default)]
    args: Vec<String>,
}

fn enabled() -> bool {
    true
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn build_programs() {
    let status = Command::new(root().join("tests/build-test-wasix.sh"))
        .status()
        .expect("tests/build-test-wasix.sh runs (is wasixcc installed?)");
    assert!(status.success(), "building the test programs failed");
}

#[test]
fn manifest_programs_print_what_the_manifest_says() {
    build_programs();
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(root().join("tests/manifest.json")).expect("manifest is readable"),
    )
    .expect("manifest parses");
    let filter = std::env::var("WEBGPU_TEST_FILTER").ok();
    let fallback = std::env::var_os("WEBGPU_TEST_FALLBACK_ADAPTER")
        .is_some_and(|value| !value.is_empty() && value != "0");

    let mut failures = Vec::new();
    let mut ran = 0;
    for test in &manifest.tests {
        if !test.native
            || filter
                .as_ref()
                .is_some_and(|f| !test.name.contains(f.as_str()))
        {
            continue;
        }
        ran += 1;
        let wasm = root().join(format!("tests/programs/out/{}.wasm", test.name));
        let ctx = WebGpuCtx::builder()
            .force_fallback_adapter(fallback)
            .build();
        match wasmer_webgpu::cli::run_wasix_main_capture(&ctx, &wasm, &test.args, false) {
            Ok(output) if output.exit_code == 0 && output.stdout == test.stdout => {
                println!("ok       {}", test.name);
            }
            Ok(output) => {
                println!("FAILED   {}", test.name);
                failures.push(format!(
                    "{}: exit code {}\n--- expected stdout ---\n{}--- actual stdout ---\n{}--- stderr ---\n{}",
                    test.name, output.exit_code, test.stdout, output.stdout, output.stderr
                ));
            }
            Err(error) => {
                println!("FAILED   {}", test.name);
                failures.push(format!("{}: {error:#}", test.name));
            }
        }
    }
    assert!(ran > 0, "no test matched");
    assert!(
        failures.is_empty(),
        "{} of {ran} programs failed:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

/// Every function of webgpu.h must be exercised by at least one program:
/// this is what "all of webgpu.h works" is measured by. A function counts
/// as covered when a program's source mentions it.
#[test]
fn every_webgpu_function_is_covered_by_a_program() {
    let header = std::fs::read_to_string(root().join("include/webgpu/webgpu.h")).expect("header");
    let extension =
        std::fs::read_to_string(root().join("include/webgpu/webgpu_wasix.h")).expect("header");
    let mut functions = Vec::new();
    for line in header.lines().chain(extension.lines()) {
        let Some(rest) = line.strip_prefix("WGPU_EXPORT ") else {
            continue;
        };
        let Some(open) = rest.find('(') else { continue };
        let name = rest[..open].rsplit(' ').next().expect("a name");
        functions.push(name.to_owned());
    }
    assert!(
        functions.len() >= 203,
        "found only {} functions",
        functions.len()
    );

    let mut sources = String::new();
    for entry in std::fs::read_dir(root().join("tests/programs")).expect("programs dir") {
        let path = entry.expect("entry").path();
        if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("c" | "cc" | "h")
        ) {
            sources.push_str(&std::fs::read_to_string(&path).expect("source"));
        }
    }
    let missing: Vec<_> = functions
        .iter()
        .filter(|name| {
            // Match whole identifiers: wgpuDeviceCreateBuffer must not be
            // satisfied by wgpuDeviceCreateBufferFoo.
            !sources.match_indices(name.as_str()).any(|(at, _)| {
                sources[at + name.len()..]
                    .chars()
                    .next()
                    .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
            })
        })
        .collect();
    if std::env::var_os("WEBGPU_COVERAGE_REPORT").is_some() {
        println!(
            "{} of {} functions uncovered:",
            missing.len(),
            functions.len()
        );
        for name in &missing {
            println!("  {name}");
        }
        return;
    }
    assert!(
        missing.is_empty(),
        "{} of {} webgpu.h functions are not used by any test program:\n  {}",
        missing.len(),
        functions.len(),
        missing
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}
