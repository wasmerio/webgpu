# Integrations

Patches for the repositories that embed this one. Each applies with
`git apply` at the root of its target.

## `wasmer-cli.patch`: `wasmer run --experimental-webgpu`

Made against wasmerio/wasmer at `0df5d2ee6c6` (7.4.2).

1. Check this repository out as `lib/webgpu`:

   ```sh
   git submodule add https://github.com/wasmerio/webgpu lib/webgpu
   ```

2. Apply the patch and let Cargo record the new dependencies:

   ```sh
   git apply lib/webgpu/integrations/wasmer-cli.patch
   cargo check -p wasmer-cli --features webgpu-window
   ```

What it changes:

- **`Cargo.toml`** excludes `lib/webgpu` from the workspace (it is a
  workspace of its own) and adds a `[patch.crates-io]` section that points
  `wasmer` and `wasmer-wasix` at this checkout's `lib/api` and `lib/wasix`.
  This crate names its Wasmer dependencies by version; the patch is what makes
  those the crates being built rather than the ones on crates.io.
- **`lib/cli/Cargo.toml`** adds the optional dependency and two features:
  `webgpu` (guests get the imports; surfaces are headless) and
  `webgpu-window` (surfaces open windows).
- **`lib/cli/src/commands/run/mod.rs`** adds `--experimental-webgpu`. A module
  that imports WebGPU without the flag is refused with a message that names
  it; with `webgpu-window` the command runs on a second thread so the main
  one can own the window system, which is only started if a guest creates a
  surface. Closing a window ends the run. A smoke test runs a guest through
  the wrapped runtime.

The features are opt-in: add `webgpu-window` to the features a release build
enables to ship it.
