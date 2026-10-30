# The Dawn host

With the `dawn` feature, the native host runs on [Dawn], Chromium's WebGPU
implementation, instead of [wgpu]. Guests do not change: they import the same
`wasmer_webgpu_v0` functions and link the same `libwebgpu.a`.

## Why a second native host

- **It is `webgpu.h`.** Dawn implements the C API itself, at the
  webgpu-headers revision `spec/` pins. The host forwards a call to the C
  function of the same name instead of translating it into another API, and
  semantics come from the implementation the header is written against:
  error scopes, render bundles, the map state machine, callback statuses.
- **It is what browsers run.** A program that works on the Dawn host meets
  the validation, limits and shader compiler Chrome gives it.
- **It costs less CPU per frame.** Dawn records a queue submission into one
  Metal command buffer; wgpu commits one per render pass, plus an empty one
  before it. On a frame of a browser compositor (18 tile passes, a composite,
  two readbacks), measured by replaying the same calls on both libraries at
  60 Hz on an M5 Max: about 1.5 ms of CPU per frame on Dawn, 2.9 ms on wgpu.
  Each extra render pass cost about 22 µs on Dawn and 70 µs on wgpu, most of
  it the kernel submitting command buffers.

wgpu stays the default. It builds with cargo alone, and runs where Dawn has
not been built.

## Building

```sh
make dawn                         # the pinned Dawn, into third_party/dawn/install
cargo build --features dawn       # or: make test-dawn
```

`scripts/build-dawn.sh` fetches the revision it pins, checks that its
`webgpu.h` is `spec/webgpu.h`, and builds and installs Dawn's monolithic
static library with one backend per platform (Metal, Vulkan, D3D12). It
needs git, cmake, ninja and a C++20 compiler; the first build takes a few
minutes.

`build.rs` links whatever `WASMER_WEBGPU_DAWN_DIR` names (a Dawn install:
`include/` and `lib/libwebgpu_dawn.a`), or `third_party/dawn/install`. An
embedder that builds Dawn itself points it there.

## How it is made

Most of the host is generated from `spec/webgpu.json` by
`codegen/emit_native.py`, like everything else derived from the header:

| File | What |
| --- | --- |
| `src/dawn/ffi.gen.rs` | The native C API: objects, enums, structs, callbacks, functions |
| `src/dawn/layout_check.gen.c` | The same struct layouts, every enum value and every function prototype, as C static assertions |
| `src/dawn/lower.gen.rs` | Decoded guest structs to native ones, and the few native structs written back |
| `src/dawn/forward.gen.rs` | Every import: a forward, or a call to the hand-written `manual_*` method |

What the bindings assume is checked against the Dawn being linked:
`ffi.gen.rs` asserts at compile time the struct layouts the generator
computed for a 64-bit target, and `build.rs` compiles `layout_check.gen.c`,
which asserts those layouts, every enum and bitflag value, and every
function's prototype, against that Dawn's headers. A Dawn whose `webgpu.h`
differs from `spec/` fails the build instead of linking.

A forward looks the receiver up, lowers the arguments, calls Dawn and
registers what it returns. Lowering is where the guest is kept honest:

- Every handle is checked for liveness and type before Dawn sees it. A bad or
  missing one is a validation error on the device the call is about, injected
  into Dawn's own error scopes, and the call is not made.
- Strings, arrays and nested structs are copied out of guest memory; the
  native struct points at host memory that lives until the call returns.
- Extension structs a guest must not pass are refused: native window handles
  (a pointer from guest memory would be an address in the host), SPIR-V
  (refused on every host, so a program meets the same shading language
  everywhere), and the WASIX canvas source outside surface creation.

27 imports are written by hand (`src/dawn/*.rs`), as are the shim's:

- **Futures.** Every future the host starts in Dawn uses `AllowSpontaneous`;
  its callback queues an event record and does nothing else. A guest that
  polls drives Dawn with `wgpuInstanceProcessEvents`; one that waits blocks
  inside `wgpuInstanceWaitAny` on the Dawn futures behind the ones it named.
  Adapter and device requests complete before the import returns.
- **Guest memory.** Dawn never touches guest memory, which other guest
  threads may be writing: every copy to or from it is Wasmer's, which allows
  that. Mapped ranges go between Dawn's mapping and guest memory in that one
  copy; uploads and immediate data reach Dawn as a host copy
  (`GuestMemory::with_copy`, through a buffer each thread keeps).
- **Policy.** The GPU memory budget, the device and object limits, stable
  queue handles, and termination.
- **Surfaces.** A headless target is a swap chain the host runs itself; a
  window gets a Dawn surface made on the window's thread (a `CAMetalLayer` on
  macOS, Xlib, XCB or Wayland on Linux, an HWND on Windows).

One Dawn instance serves a guest process; every device is created with
Dawn's implicit device synchronization, because a guest's threads share
devices.

## Differences from the wgpu host

What a program can observe differs where the two libraries do, never in the
ABI:

- `wgpuCommandEncoderWriteTimestamp` is a validation error: Dawn keeps it
  behind its unsafe APIs, as Chrome does, since the WebGPU specification
  dropped it. (wgpu writes it where the adapter can.)
- Errors carry Dawn's messages, objects a call failed to create are Dawn's
  error objects rather than null, and limits and features are Dawn's.
- `WGPU_WHOLE_SIZE` as the size of `wgpuCommandEncoderCopyBufferToBuffer`
  means the rest of the source, as in the browser host, although Dawn's C
  API takes only real sizes.
- `WebGpuCtxBuilder::renderer_mapped_draw_buffers` lets one device of each
  guest process have Dawn's `BufferMapExtendedUsages`: buffers that are
  mappable and also uniform, vertex, index or storage buffers, which a
  renderer writes its draw data straight into instead of staging it and
  copying it on the GPU. Dawn calls the feature experimental and grants it
  only with its `allow_unsafe_apis` toggle, so the option is off by default
  and is for an embedder whose guest is its own renderer; no other device of
  the process gets the toggle. A guest that asks for the feature without it
  is refused, as it is for any other feature outside `webgpu.h`. The wgpu
  host has nothing like it.
- `WebGpuCtxBuilder::force_fallback_adapter` (`WEBGPU_TEST_FALLBACK_ADAPTER`
  for the tests) asks Dawn for its fallback adapter, and Dawn has one only
  with SwiftShader, which this build leaves out: Mesa's lavapipe or WARP are
  ordinary adapters to it. Where the software one is the only adapter (a CI
  runner), leave the option off; Dawn picks it anyway.

[Dawn]: https://dawn.googlesource.com/dawn
[wgpu]: https://github.com/gfx-rs/wgpu
