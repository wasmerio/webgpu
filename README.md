# wasmer-webgpu

WebGPU for WebAssembly programs running on [Wasmer](https://wasmer.io).

A C or C++ program written against the standard [`webgpu.h`] compiles to
WASIX, links one small static library, and runs on the GPU of whatever hosts
it:

| Host | GPU backend | Surfaces present to |
| --- | --- | --- |
| `wasmer run` and Rust embedders | [wgpu] (Metal, Vulkan, D3D12) | a window, or a headless target |
| A browser, through the [Wasmer SDK] | the page's `navigator.gpu` | a canvas of the page |

The program is the same binary in every case. It uses the header exactly as
upstream ships it: no macros to set, no Wasmer-specific calls except to say
which canvas or window a surface belongs to.

```c
#include <webgpu/webgpu.h>

WGPUInstance instance = wgpuCreateInstance(NULL);
/* ...request an adapter and a device, build pipelines, submit work... */
```

## Trying it

You need [wasixcc] for the guest side and a Rust toolchain for the host.

```sh
make guest      # lib/wasm32-wasix/libwebgpu.a
make examples   # examples/out/triangle.wasm
cargo run --features cli,window --bin webgpu_wasmer -- --window examples/out/triangle.wasm
```

`webgpu_wasmer` is a small stand-in for `wasmer run` that this repository
builds on its own. Without `--window` surfaces are headless, and
`--frames <dir>` saves what the program presents.

Building your own program is one compiler invocation:

```sh
wasixcc app.c -Iinclude -Llib/wasm32-wasix -lwebgpu -o app.wasm
```

## Where it runs

### The Wasmer CLI

`integrations/wasmer-cli.patch` adds this crate to the Wasmer CLI:

```sh
wasmer run --experimental-webgpu app.wasm
```

A program that creates a surface gets a window; closing it ends the program.
One that only computes never touches the window system. See
[Integrating](#integrating) for how the patch is applied.

### Rust

```rust
use wasmer_webgpu::WebGpuCtx;

let ctx = WebGpuCtx::builder()
    .max_gpu_memory_bytes(512 << 20)
    .build();
// With the `wasix` feature the hooks are a `wasmer_wasix` InstantiationHook:
let runtime = OverriddenRuntime::new(runtime).with_instantiation_hook(ctx.runtime_hooks());
```

A `WebGpuCtx` is what an embedder grants its guests: limits on devices,
objects and GPU memory, a software-adapter-only switch, and a
`SurfaceProvider` that decides what a guest's surface presents to. The hooks
add imports only to modules that ask for WebGPU and are inert for everything
else. `ctx.runtime_control().terminate_all()` stops the GPU work of every
guest of the context.

The Wasmer SDK wraps this as `SandboxBuilder::webgpu(ctx)`.

### JavaScript, in a browser

```js
import { Wasmer } from "@wasmer/sdk";

const wasmer = new Wasmer();
const sandbox = await wasmer.sandboxes.create({
  packages: [app],
  webgpu: { canvas: document.querySelector("canvas") },
});
await sandbox.command(app).run();
```

The guest runs in a worker. For an `HTMLCanvasElement` each frame it presents
is shown in the element, which stays with the page, so any number of commands
can draw to it. An `OffscreenCanvas` (from `transferControlToOffscreen()`)
moves to the guest instead, which then presents without involving the page's
thread.

Browsers deliver everything WebGPU reports through the event loop, so a guest
has to be suspended while it waits. That takes WebAssembly [JSPI]: the SDK
refuses the `webgpu` option where the engine lacks it.

## What a guest can rely on

Every function of the pinned `webgpu.h` is provided: 203 of 203, a number
the test suite enforces. Behaviour follows the header's documentation:

- **Futures and callbacks.** All three callback modes work, as do
  `wgpuInstanceProcessEvents` and `wgpuInstanceWaitAny` with any timeout
  (the `TimedWaitAny` instance feature is always available natively, and in
  a browser whenever the guest can be suspended).
- **Blocking is fine.** A native-style loop of `wgpuSurfaceGetCurrentTexture`,
  draw, `wgpuSurfacePresent` paces itself: presenting waits for the display.
- **Errors, not crashes.** A mistake is a WebGPU validation error delivered
  to the error scope or the uncaptured-error callback. A pointer outside the
  guest's memory traps the guest. Nothing a guest does may panic the host.
- **Threads.** Natively a handle works on any thread of the process. In a
  browser GPU objects belong to the worker that created them.

One thing `webgpu.h` leaves to the platform is where a presentable surface
comes from. A WASIX guest has no window handles, so it names its target and
the host supplies it (`include/webgpu/webgpu_wasix.h`):

```c
WGPUWasixSurfaceSourceCanvas source = WGPU_WASIX_SURFACE_SOURCE_CANVAS_INIT;
source.selector = (WGPUStringView){ "#canvas", WGPU_STRLEN };
WGPUSurfaceDescriptor descriptor = WGPU_SURFACE_DESCRIPTOR_INIT;
descriptor.nextInChain = &source.chain;
WGPUSurface surface = wgpuInstanceCreateSurface(instance, &descriptor);
```

Programs written for Emscripten's
`WGPUEmscriptenSurfaceSourceCanvasHTMLSelector` work unchanged: its struct
type is accepted for the same layout. `wgpuWasixSurfaceGetSize` reports the
size the target wants, for configuring and for reacting to a resize.

`examples/triangle.c` is a complete program with the frame loop,
resize handling and device-loss handling a real one needs.

## How it is built

```
guest (wasm32-wasix)                         host
+--------------------------+   imports   +--------------------------------+
| app.c  -> webgpu.h       |  wasmer_    | native: src/host/*.rs on wgpu  |
| libwebgpu.a              |  webgpu_v0  | browser: js/webgpu_host.js on  |
|   callbacks, futures,    | ----------> |          navigator.gpu         |
|   mapped ranges, memory  |             | handles, validation, limits    |
+--------------------------+             +--------------------------------+
```

- The guest library defines every `wgpu*` symbol. Most forward to an import
  of the same name; the ones that involve a callback, a pointer the guest
  owns, or waiting are implemented in the library itself.
- The host never calls into the guest. Work that completes later becomes an
  event record the library collects and turns into the callback.
- Guest pointers are only ever copied from and to, with bounds checks.
  Objects are 32-bit handles checked for liveness and type on every use.
- Layouts, enum values, import signatures and the JavaScript tables are
  generated from the pinned `spec/webgpu.json`, and the C compiler confirms
  the generator's model of every struct.

[docs/architecture.md](docs/architecture.md) has the details: the import
ABI, the event wire format, how waiting works on each host, and what a
browser cannot do.

## Repository layout

| Path | What |
| --- | --- |
| `spec/` | Pinned upstream `webgpu.h` / `webgpu.json`, and `abi.json`: the hand-written facts the generator needs |
| `codegen/` | The generator (`python3 codegen/gen.py`, `--check` in CI) |
| `include/` | What guests compile against: `webgpu/webgpu.h` (unmodified) and `webgpu/webgpu_wasix.h` |
| `guest/` | Sources of `libwebgpu.a` |
| `src/` | The `wasmer-webgpu` crate: embedder API, native host, browser glue |
| `js/` | The browser host |
| `examples/` | Programs to start from |
| `tests/` | Conformance programs with their expected output, run natively and in a browser |
| `integrations/` | Patches for repositories that embed this one |

## Testing

```sh
make test           # native: unit tests, ABI sweep, conformance programs, embedder tests
make test-browser   # the same programs in the installed Chrome
make check          # generated files up to date, formatting, lints, wasm32 build
```

`tests/manifest.json` records what each conformance program must print; the
same binaries run on both hosts. The native run also fails if any function of
`webgpu.h` is used by no program. Machines without a GPU can run the native
suite on a software adapter with `WEBGPU_TEST_FALLBACK_ADAPTER=1`.

## Integrating

The crate names its Wasmer dependencies by version and leaves resolving them
to the workspace that embeds it:

- **Standalone**, this repository pins one Wasmer revision in its own
  `[patch.crates-io]`.
- **The Wasmer SDK** depends on the crate and patches Wasmer to the revision
  it ships.
- **Wasmer itself** checks this repository out as `lib/webgpu`, and
  `integrations/wasmer-cli.patch` (apply with `git apply`) excludes it from
  the workspace, points the two Wasmer crates it needs at the checkout, and
  adds the `webgpu` / `webgpu-window` features and the
  `--experimental-webgpu` flag to the CLI.

The import module is `wasmer_webgpu_v0`. An incompatible change gets a new
module name, so a binary built for another revision fails to instantiate
with a message that says why, instead of misbehaving.

## Status

Experimental.

**In a browser** a program's requests are passed on to the browser's WebGPU,
the way Dawn's Emscripten binding (emdawnwebgpu) passes them on, and what the
browser cannot do comes back as its validation error. External textures are
the exception: neither host provides them yet. A guest's threads cannot share
GPU objects there, and a guest needs JSPI.

**Natively** the backend is wgpu, which does not implement all of WebGPU.
Where it cannot do what a program asked, the program gets a validation error
or an absent feature:

- Shaders are WGSL, as in a browser. A SPIR-V source is refused.
- External textures, the `Snorm10_10_10_2` vertex format, compatibility mode,
  and a few texture formats and limits are missing.
- A window's surface presents in sRGB with standard tone mapping. A headless
  target is told the colour space and tone mapping the program configured.
- `wgpuCommandEncoderWriteTimestamp` needs an adapter that can write
  timestamps outside a pass. (A browser needs an experimental flag for it.)

A label given to an object after it was created, and the queue's label, are
accepted natively and go nowhere: wgpu takes a label at creation only.

Window input (keyboard, pointer) is outside `webgpu.h` and not provided.

## License

MIT. Third-party material is listed in [NOTICE.md](NOTICE.md).

[`webgpu.h`]: https://github.com/webgpu-native/webgpu-headers
[wgpu]: https://github.com/gfx-rs/wgpu
[wasixcc]: https://github.com/wasix-org/wasixcc
[Wasmer SDK]: https://github.com/wasmerio/wasmer-sdk
[JSPI]: https://github.com/WebAssembly/js-promise-integration
