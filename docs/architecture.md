# Architecture

This document explains how a `webgpu.h` call made by a WASIX guest reaches a
GPU, and why the pieces are shaped the way they are. It is for people who
change this repository or embed it.

## The problem

`webgpu.h` is a C API with by-value structs, chained extension structs,
callbacks, futures, and pointers into mapped GPU memory. A guest calls it
from inside a 32-bit linear memory; the implementation lives outside, in one
of two very different hosts:

- natively, Rust code on [wgpu], with real threads and blocking calls;
- in a browser, JavaScript on `navigator.gpu`, where every result arrives
  through an event loop that a running guest blocks.

The design has one rule that makes both possible:

> **The host never calls into the guest.**

Everything that would be a host-to-guest call in a native WebGPU library
(a callback, an allocation for a returned string, a pointer to mapped memory)
is done by code that already runs in the guest. The host only answers calls.
That removes re-entrancy from the host, makes the import surface plain
functions that JavaScript can provide directly, and lets a guest be
suspended at an import without the host holding guest state on its stack.

## The two halves

```
guest                                     host
----------------------------------------  -------------------------------------
app code
  | webgpu.h (unmodified upstream)
libwebgpu.a
  | forwarders:  wgpuX(...) -> import X   wgpuX(...) implemented on the backend
  | shim:        futures, callbacks,      wasmer_wgpu_*: start work, hand over
  |              mapped ranges, strings   completed events and result blobs
  +---- import module wasmer_webgpu_v0 -->
```

### The guest library (`guest/`)

`libwebgpu.a` defines every `wgpu*` symbol of the header.

- **Forwarders** (`wgpu_forwarders.gen.c`, generated). 172 functions need
  nothing but to be passed on: a function that takes handles, numbers and
  pointers to structs has the same shape as a wasm import. Each forwarder is
  one line calling the raw import of the same name.
- **The shim** (`webgpu_shim.c`, hand-written). The 31 functions that cannot
  simply be forwarded: those that take a callback, return memory the guest
  must free, hand out pointers into mapped buffers, or wait. The shim also
  owns the reference counts whose final release has guest-side consequences
  (an instance, a device, a buffer).
- **The proc table** (`wgpu_proc_table.gen.c`), for `wgpuGetProcAddress`.

Only what a program references is linked, so a program's import list is the
list of WebGPU functions it uses.

### The import ABI

The import module is `wasmer_webgpu_v0`. It has two groups of functions.

**Raw imports** carry the name and the C signature of the `webgpu.h` function
they implement, lowered the way the wasm32 C ABI lowers it:

| C | wasm |
| --- | --- |
| object handle, enum, `uint32_t`, `size_t`, `WGPUBool`, any pointer | `i32` |
| flags (`WGPUBufferUsage`, ...), `uint64_t` | `i64` |
| `float` / `double` | `f32` / `f64` |
| a struct passed by value (`WGPUStringView`, a callback info) | `i32` pointer to a copy |
| `WGPUFuture` returned by value | `i64` |

**Special imports** (`wasmer_wgpu_*`) exist for the shim:

| Import | Purpose |
| --- | --- |
| `abi_version` | Checked once; the shim refuses a host that speaks another major revision |
| `instance_request_adapter`, `adapter_request_device`, `buffer_map_async`, `device_pop_error_scope`, `device_create_*_pipeline_async`, `queue_on_submitted_work_done`, `shader_module_get_compilation_info` | Start asynchronous work under a future id the shim chose |
| `poll(buf, cap)` | Copy completed event records to the guest |
| `epoch()`, `wait(futures, count, timeout_ns, epoch, buf, cap)` | Block until something listed completes |
| `yield()` | Give the host a chance to make progress |
| `instance_release`, `device_release`, `buffer_release` | Release, and say whether it was the last reference |
| `buffer_map_info`, `adapter_get_info`, `*_get_features`, `instance_get_wgsl_language_features`, `surface_get_capabilities`, `object_instance` | Results the shim has to own, as serialized blobs |

Blob getters return the number of bytes the result needs and write it only
if it fits, so the shim can retry with a buffer that is large enough.

Everything layout-shaped is generated from the pinned `spec/webgpu.json` by
`codegen/gen.py`: the Rust struct readers, the JavaScript offset tables, the
import signatures, the forwarders. `spec/abi.json` holds what the upstream
description cannot say (which functions are the shim's, the extension
struct, which enums are open-ended). `guest/tests/abi_check.gen.c` is
compiled with every build of the guest library and consists of static
assertions: every size, offset and enum value the generator computed is
checked against what the C compiler does with the real header.

### Handles

A guest holds 32-bit handles: `(generation << 24) | (index + 1)`, zero being
null. Each table slot records the object's type, so a handle of the wrong
kind, a stale handle, or a made-up number is found before any GPU object is
touched, and reported as a WebGPU error or a null result. Releasing a slot
bumps its generation.

### Guest memory

The host reads structs out of guest memory and writes results back with
bounds-checked copies; it never keeps a pointer. An address range outside
the memory traps the guest, as a wild pointer would natively. Counts are
guest-controlled, so arrays, strings and chains have ceilings
(`MAX_ARRAY_LENGTH`, `MAX_STRING_BYTES`, the chain length limit).

## Asynchronous work

`webgpu.h` reports results through callbacks tied to futures. Here:

1. The shim assigns the future id, records the user's callback, and calls a
   special import to start the work.
2. When the work completes, the host queues an **event record**:

   ```
   u64 future;       // 0 for events that belong to no future
   u32 kind;         // request adapter, buffer map, device lost, ...
   u32 status;       // the callback's status enum
   u32 handle;       // result object, or the device an error is about
   u32 aux;          // per kind (an error type, for instance)
   u32 message_len;
   u32 extra_len;
   u8  message[], padded to 8 bytes
   u8  extra[],   padded to 8 bytes
   ```

3. The shim collects records (`poll`, or as the result of `wait`), finds
   each one's callback and invokes it, at the points the callback mode
   allows: during `wgpuInstanceProcessEvents`, inside `wgpuInstanceWaitAny`,
   or as soon as the record is seen for `AllowSpontaneous`.

Uncaptured errors and lost devices use the same queue with future id 0 (or
the device's lost future).

### Waiting

`wgpuInstanceWaitAny` is a futex-style wait. The host keeps an event
**epoch** that changes whenever a record is queued, or a compiled pipeline
is ready to be published (see below). The shim reads the
epoch, looks at what it already has, and only then waits, passing the epoch
it saw: the host returns immediately if the epoch has moved. That closes the
window in which one guest thread collects a record another thread is
waiting for.

Natively `wait` blocks the calling thread on a condition variable, in slices
so that termination is noticed. In a browser `wait`, `yield`,
`wgpuSurfacePresent` and `wgpuInstanceCreateSurface` return promises, and the
runtime suspends the guest with JSPI until they settle; the worker's event
loop is free meanwhile, which is the only time a browser delivers anything.
For the same reason `wgpuInstanceProcessEvents` and a zero-timeout
`wgpuInstanceWaitAny` yield once when work is outstanding: a program that
polls in a loop still makes progress.

### Async pipelines

Natively, `wgpuDeviceCreate{Compute,Render}PipelineAsync` resolves the
objects its descriptor names into an owned job and hands it to a compiler
thread, one per guest process. The result is published (the pipeline
registered, its record queued) the next time a guest thread polls or waits,
and the epoch changes as soon as it is ready, so that a waiting thread looks
again. While a pipeline compiles, a waiting thread that would otherwise block
inside a busy GPU polls it in short slices instead. A lane at capacity hands
the job back, and the call compiles it on the guest's thread. The browser
compiles with the browser's own asynchronous calls.

## Buffer mapping

`wgpuBufferGetMappedRange` returns a pointer, and guest pointers can only
point into guest memory. The shim allocates a shadow of the requested range
in the guest heap, fills it from the host for a readable mapping, hands the
pointer to the program, and writes it back to the host at `wgpuBufferUnmap`
for a writable one. `wgpuBufferReadMappedRange` and
`wgpuBufferWriteMappedRange` are the copy primitives underneath, and are also
what the header offers programs that do not need a pointer.

## Errors

Mistakes a native WebGPU implementation reports as validation errors are
reported the same way here, even when the host finds them itself (an
unsupported enum value, a struct chained where it does not belong). Each
host keeps its own stack of error scopes per device and, natively, per guest
thread, so an error it raises reaches the scope a backend error would.

Three kinds of failure exist, and they do not blur:

| What happened | Result |
| --- | --- |
| The program misused WebGPU | Validation error: scope, or uncaptured-error callback |
| The program passed a pointer outside its memory | The guest traps |
| The embedder's limit was reached | The call fails the way running out of that resource does (a null object, an out-of-memory error) |

A panic in the native host while serving a guest is treated as a host bug
and converted into a guest trap; `tests/abi_sweep.rs` calls every import
with hostile arguments to keep it from happening.

## Surfaces

`webgpu.h` leaves the origin of a presentable surface to the platform. A
WASIX guest names a target with `WGPUWasixSurfaceSourceCanvas { selector }`,
and the embedder decides what the name means.

**Natively** a `SurfaceProvider` maps selectors to a `SurfaceTarget`:

- `Window`: anything with raw window and display handles. The `window`
  feature provides a winit-based `WindowHost` that owns the event loop on
  the main thread and runs the guest on another; the window system is only
  started when a guest first asks for a surface.
- `Headless`: the guest renders into an offscreen texture and each presented
  frame is offered to a `FrameSink` (optionally with pixels). This is what
  servers, tests and screenshot tools use, and the default.

**In a browser** the JavaScript host asks its embedder for a canvas. In the
Wasmer SDK the guest's worker asks the page over a `MessagePort`: an
`OffscreenCanvas` is transferred to the worker and presented to directly; for
an `HTMLCanvasElement` the worker renders into a canvas of its own and sends
each frame to the page as an `ImageBitmap`.

`wgpuSurfacePresent` blocks until the frame is on its way to the screen
(vsync natively, the next animation frame in a browser), which is what paces
a frame loop.

## The two hosts

### Native (`src/host/`)

One `Host` per guest process, found through the process's linear memory, so
every thread of a process sees the same handle table. Objects are `Arc`s of
wgpu objects plus the state WebGPU needs that wgpu does not keep (map state,
error scopes, the configured surface). `WebGpuCtx` holds the embedder's
policy and a registry of live hosts, which is what `terminate_all` and
`usage` walk.

Where wgpu and `webgpu.h` disagree, the host does the translating: the
error-scope stack, render bundles (recorded and replayed, because wgpu
validates them eagerly), the map-state machine, the callback status codes,
and the rules wgpu does not enforce itself (a render pass's `maxDrawCount`,
which the host counts; what may be done with a transient attachment). What
wgpu cannot do at all is a validation error, as it would be in a browser
that lacks it: SPIR-V shader sources are in that group on purpose, so that a
program meets the same shading language on both hosts.

### Browser (`js/webgpu_host.js`, `src/web/`)

The same imports, implemented in JavaScript against `navigator.gpu` with the
generated offset tables. `src/web/mod.rs` only wires them up: synchronous
imports are handed to the engine as plain JavaScript functions, so a call
goes from wasm to JavaScript to the browser's WebGPU with no Rust in
between; the imports that may wait go through Wasmer's asynchronous host
functions.

What a browser changes, and cannot hide:

- Results arrive only while the guest is suspended, which requires JSPI.
- GPU objects belong to the worker that created them, so each guest thread
  has its own host and handles do not cross threads.
- Releasing an object drops a reference; its GPU memory goes when the
  garbage collector gets to it, unless the program called `Destroy`.
- An embedder's kill switch is polled, because nothing can interrupt a
  worker from outside.

One thing the host does only because of a browser bug. Chrome answers every
submit from its GPU process and keeps the queue alive by itself until then.
If nothing else refers to the queue when the answer arrives, Chrome 154 frees
it from inside the answer's handler, and when that follows a flush the thread
deadlocks for good. A device that was destroyed can be collected at once, so
a guest that exits right after a submit leaves exactly that behind for the
next guest on its worker. The host therefore keeps every queue reachable
until the browser reports its submitted work done
(`tests/browser`: `embedder_thread_reuse`).

## Versioning

- The import module name (`wasmer_webgpu_v0`) changes when the ABI breaks. A
  guest built for another module fails to instantiate, with a message that
  says why.
- `wasmer_wgpu_abi_version` returns `major << 16 | minor`, and the shim
  checks the major part before it does anything else. Additions bump the
  minor version. A guest that uses an addition imports it, so on a host
  that predates it the guest fails to instantiate rather than at some later
  call.
- `spec/` pins the upstream header. Updating it is a codegen run, a rebuild
  of the guest library, and whatever the new functions need in the two
  hosts; the coverage test fails until every new function is exercised.

[wgpu]: https://github.com/gfx-rs/wgpu
