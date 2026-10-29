//! Links Dawn for the `dawn` feature; nothing to do otherwise.
//!
//! Dawn is found through `WASMER_WEBGPU_DAWN_DIR`, a Dawn install (`include/`
//! and `lib/` with the monolithic `webgpu_dawn` library), or else in
//! `third_party/dawn/install`, where `make dawn` (scripts/build-dawn.sh)
//! builds the pinned revision. Before linking, the layouts the Rust bindings
//! assume are checked against that Dawn's headers by a C compiler.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=WASMER_WEBGPU_DAWN_DIR");
    #[cfg(feature = "dawn")]
    dawn::link();
}

#[cfg(feature = "dawn")]
mod dawn {
    use std::env;
    use std::path::PathBuf;

    pub fn link() {
        let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
            // The browser host is JavaScript; Dawn is only for native hosts.
            return;
        }
        let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("set by cargo"));
        let dir = env::var_os("WASMER_WEBGPU_DAWN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest.join("third_party/dawn/install"));
        let include = dir.join("include");
        let lib = dir.join("lib");
        let library = lib.join(if os == "windows" {
            "webgpu_dawn.lib"
        } else {
            "libwebgpu_dawn.a"
        });
        let header = include.join("webgpu/webgpu.h");
        if !header.exists() || !library.exists() {
            panic!(
                "the `dawn` feature needs a Dawn build: {} has no {} or no {}.\n\
                 Run `make dawn` (scripts/build-dawn.sh builds the pinned revision into \
                 third_party/dawn/install), or set WASMER_WEBGPU_DAWN_DIR to a Dawn install.",
                dir.display(),
                header.display(),
                library.display(),
            );
        }
        println!("cargo::rerun-if-changed={}", library.display());
        println!("cargo::rerun-if-changed={}", header.display());
        println!("cargo::rerun-if-changed=src/dawn/layout_check.gen.c");
        println!("cargo::rerun-if-changed=src/dawn/dawn_check.c");

        // Fails the build if a struct the bindings declare is laid out
        // differently by this Dawn's headers, or a value they spell out moved.
        cc::Build::new()
            .file("src/dawn/layout_check.gen.c")
            .file("src/dawn/dawn_check.c")
            .include(&include)
            .warnings(false)
            .compile("wasmer_webgpu_dawn_checks");

        println!("cargo::rustc-link-search=native={}", lib.display());
        println!("cargo::rustc-link-lib=static=webgpu_dawn");
        match os.as_str() {
            "macos" | "ios" => {
                for framework in [
                    "Metal",
                    "QuartzCore",
                    "IOSurface",
                    "IOKit",
                    "Foundation",
                    "CoreFoundation",
                    "CoreGraphics",
                    "Cocoa",
                ] {
                    println!("cargo::rustc-link-lib=framework={framework}");
                }
                println!("cargo::rustc-link-lib=c++");
            }
            "linux" | "android" | "freebsd" => {
                println!("cargo::rustc-link-lib=stdc++");
                println!("cargo::rustc-link-lib=dl");
                println!("cargo::rustc-link-lib=pthread");
            }
            "windows" => {
                for library in ["dxgi", "dxguid", "d3d12", "user32", "onecore"] {
                    println!("cargo::rustc-link-lib={library}");
                }
            }
            _ => {}
        }
    }
}
