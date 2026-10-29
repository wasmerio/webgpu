# Browser tests

These run the WebGPU conformance programs of `tests/programs` in a real
browser, against the JavaScript host (`js/webgpu_host.js`). The same wasm
binaries run on the native host through `tests/manifest_tests.rs`; a program
passes when it prints exactly what `tests/manifest.json` says, on both.

The harness is standalone: it does not use the Wasmer SDK. That keeps a
failure here about the host and nothing else. Only `runtime_hooks` goes
through the crate's Rust side, as the SDK does.

## Running

```sh
cd tests/browser
npm install            # once: playwright-core, which downloads no browser
node run.mjs           # builds the programs and the fixture, then runs everything
```

`run.mjs` exits non-zero if anything fails and prints one line per test.

```
node run.mjs [name...] [--headed] [--no-build] [--verbose] [--repeat <n>] [--timeout <seconds>] [--serve]
```

| | |
|---|---|
| `name...` | Run only these tests (program names, `embedder_*` names, `abi_sweep`, `host_tests`, `runtime_hooks`). |
| `WEBGPU_TEST_FILTER=<substring>` | Run the tests whose name contains it, as for the native tests. |
| `--no-build` | Do not run `tests/build-test-wasix.sh` and `build-runtime-hooks.sh` first. |
| `WEBGPU_TEST_OUT_DIR=<dir>` | Where the programs are built and served from (the build script's own variable). |
| `--verbose` | Also print what each program wrote and what the host logged. |
| `--repeat <n>` | Run each program `n` times. Anything that depends on timing only shows this way. |
| `--timeout <seconds>` | How long one program may run (default 60). A hang is reported with what was printed so far. |
| `--headed` | Show the browser window. |
| `WEBGPU_CHROME_ARGS="..."` | Extra command-line switches for Chrome. |
| `--serve` | Only serve the repository, to open `tests/browser/index.html` by hand. |

## What it needs

- **Node.js** 20 or later, and `wasixcc` to build the programs (not needed
  with `--no-build`).
- For `runtime_hooks`, Rust's `wasm32-unknown-unknown` target and the
  `wasm-bindgen` CLI of the version in `Cargo.lock`
  (`cargo install wasm-bindgen-cli --locked --version <version>`;
  `WASM_BINDGEN` names another binary).
- **Google Chrome**, installed normally. Playwright drives it through
  `channel: 'chrome'`; nothing is downloaded, and a Chrome that is already
  open is left alone (the tests run in a fresh profile of their own).
- A Chrome with **JavaScript Promise Integration** (`WebAssembly.Suspending`,
  Chrome 137 and later). The host's blocking imports suspend the guest with
  it.
- A **WebGPU adapter**.

### Chrome flags

None are needed for WebGPU. Checked with Chrome 154 on macOS 27 (Apple
silicon): the default headless mode (`--headless=new`, which is what
Playwright's `headless: true` gives a branded Chrome) has `navigator.gpu`, a
hardware adapter (`vendor: apple`, `architecture: metal-3`) on the page and in
dedicated workers, `WebAssembly.Suspending`, and `requestAnimationFrame` in
workers at 60 Hz. `--enable-unsafe-webgpu` and `--enable-features=WebGPU` are
not required; the first only adds experimental features to the adapter, and
the suite gives the same results with it (`WEBGPU_CHROME_ARGS=--enable-unsafe-webgpu`).

`run.mjs` passes three switches for a different reason:
`--disable-backgrounding-occluded-windows`, `--disable-renderer-backgrounding`
and `--disable-background-timer-throttling`. A window that is covered or in
the background gets no animation frames, and `wgpuSurfacePresent` waits for
one; without the switches a `--headed` run stalls when its window is hidden.

A fourth, `--js-flags=--expose-gc`, lets `embedder_thread_reuse` collect
garbage at the moment that matters to it. Without the switch (a browser
opened by hand) that test still runs, and can only pass.

If a machine's headless Chrome has no adapter, `run.mjs` says so; `--headed`
is the thing to try there.

The page has to be cross-origin isolated, because the guests use a shared
memory. `run.mjs` serves the repository on a free localhost port with

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

## What runs

1. **The manifest programs.** Every entry of `tests/manifest.json` that is
   not marked `"browser": false`. Each runs in a dedicated worker of its own,
   which is where the Wasmer SDK runs guests too. Besides the transcript, a
   run fails if a surface target the program opened was not released.
2. **Embedder tests** (`embedder_*`), the browser twin of
   `tests/embedder_tests.rs`: the same programs under the options an embedder
   sets on the host. The object, device and GPU memory limits; the kill
   switch getting a guest out of a wait; a surface target of the embedder's
   own, whose presented frames are read back; `forceFallbackAdapter`; and a
   host that may not suspend the guest. One has no native twin:
   `embedder_thread_reuse` starts a second program on the worker of one that
   exited with GPU work still running, which is what the Wasmer SDK does with
   its workers and what deadlocks Chrome 154 unless the host prevents it.
3. **`abi_sweep`**, the twin of `tests/abi_sweep.rs`: every import called
   with all-zero and all-ones arguments on an empty host, then with every live
   handle in every argument position. Nothing but a trap naming the import
   may come out.
4. **`host_tests`**: the imports called by hand on the page's main thread,
   with a spy on the browser's WebGPU objects. They pin down what a guest
   cannot observe: which descriptor a C struct becomes (the "undefined"
   values of `webgpu.h`, strings with an explicit length, extension structs
   that do not belong), how event records are handed over, timeouts, and a
   memory that grows.
5. **`runtime_hooks`**: the crate's browser backend, built with wasm-bindgen,
   instantiating small guests through `WebGpuRuntimeHooks` in a worker. It
   covers what the backend adds to the host for an embedder: the
   `event_waker` of the context, woken by an event a call makes ready and by
   one the browser settles; `__wasmerWebGpuHostCreated`, called once the
   guest's memory is bound, never for an instance that failed, and failing
   the instance when it throws; and every host disposed of with its store.

## How a program runs

| file | |
|---|---|
| `run.mjs` | Node: serves the repository, launches Chrome, compares results. |
| `index.html`, `runner.js` | The page: starts a worker per program, gives it a canvas and collects its output. Opened by hand, the page runs the manifest and shows the transcripts. |
| `worker.js` | The guest's worker: creates the shared `WebAssembly.Memory` the module asks for, the WASIX stub and the WebGPU host, wraps the imports flagged `suspending` in `WebAssembly.Suspending`, and calls `WebAssembly.promising(_start)`. On request it starts a second program afterwards, with a host of its own. |
| `wasix_stub.js` | The WASI and WASIX imports a single-threaded C program needs to print and exit. Anything else a program imports throws when called. |
| `sweep.js`, `host_tests.js` | The tests that need no program. |
| `runtime_hooks.rs` | The `runtime_hooks` tests, built by `build-runtime-hooks.sh` into `out/` (a Cargo example of the crate). `runtime_hooks.js` runs them in a worker (`runtime_hooks_worker.js`). |

Every surface of a program presents to one canvas of 640 by 480, the size of
the native runner's default target. On the page it is a real `<canvas>`,
transferred to the worker, so what a program draws can be seen.

Programs that start threads cannot run here (the stub has no threads); they
are marked `"browser": false` in the manifest. In a browser, WebGPU objects
belong to the worker that created them in any case.
