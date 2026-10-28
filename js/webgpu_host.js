// The browser half of wasmer-webgpu: every import of the `wasmer_webgpu_v0`
// module, implemented on the JavaScript WebGPU API (`navigator.gpu`).
//
// A guest links `lib/wasm32-wasix/libwebgpu.a` and ends up importing the raw
// functions listed in `IMPORTS` (webgpu_abi.gen.js). This file provides them:
//
//     const host = createWebGpuHost({ getCanvas });
//     // Wrap the imports flagged `suspending` with `new WebAssembly.Suspending`,
//     // instantiate with { [IMPORT_MODULE]: host.imports, ... }, then:
//     host.setMemory(memory);
//
// It mirrors the native host (src/host/*.rs) wherever a browser allows:
//
//  - The host never calls into the guest. Asynchronous work completes as
//    event records the guest shim collects with `wasmer_wgpu_poll` and
//    `wasmer_wgpu_wait`; the shim invokes the callbacks.
//  - The guest only ever holds 32-bit handles. Each is checked for liveness
//    and type before a GPU object is touched, so a confused guest gets a
//    WebGPU error or a null result, never an exception.
//  - Guest memory is only ever copied from and to: `wgpuBufferGetMappedRange`
//    pointers live in the shim, which fills and flushes them through
//    `wgpuBufferReadMappedRange` / `wgpuBufferWriteMappedRange`.
//
// What a browser does differently, and cannot be hidden:
//
//  - Results arrive through the event loop. Nothing completes (not even an
//    uncaptured error) until the guest suspends in `wasmer_wgpu_wait`,
//    `wasmer_wgpu_yield` or `wgpuSurfacePresent`, which needs JavaScript
//    Promise Integration. Without it those imports cannot block, and the
//    TimedWaitAny instance feature is not offered.
//  - GPU objects are not shared between workers, so a host (and its handles)
//    belongs to the thread that created it.
//  - Releasing a handle drops a reference; the GPU memory behind it goes when
//    the garbage collector gets to it, unless the guest called Destroy.
//
// The descriptor decoding follows Dawn's Emscripten binding
// (third_party/emdawnwebgpu/pkg/webgpu/src/library_webgpu.js,
// Copyright 2019 The Emscripten Authors, MIT license), adapted to this ABI.

import { E, EVENT, F, IMPORTS, S, STYPE, WEBIDL } from './webgpu_abi.gen.js';

export { IMPORT_MODULE } from './webgpu_abi.gen.js';

// ---------------------------------------------------------------------------
// Constants the generated tables do not carry
// ---------------------------------------------------------------------------

/** `WEBGPU_ABI_VERSION` of src/lib.rs: major << 16 | minor. */
const ABI_VERSION = 0x0000_0001;

const U32_MAX = 0xffff_ffff;
const U64_MAX = 0xffff_ffff_ffff_ffffn;

// The "undefined" values of webgpu.h.
const WGPU_STRLEN = U32_MAX;
const WGPU_WHOLE_SIZE = U64_MAX;
const WGPU_WHOLE_MAP_SIZE = U32_MAX;
const WGPU_LIMIT_U32_UNDEFINED = U32_MAX;
const WGPU_LIMIT_U64_UNDEFINED = U64_MAX;
const WGPU_COPY_STRIDE_UNDEFINED = U32_MAX;
const WGPU_DEPTH_SLICE_UNDEFINED = U32_MAX;
const WGPU_MIP_LEVEL_COUNT_UNDEFINED = U32_MAX;
const WGPU_ARRAY_LAYER_COUNT_UNDEFINED = U32_MAX;
const WGPU_QUERY_SET_INDEX_UNDEFINED = U32_MAX;

// Ceilings on what one call may make the host copy out of guest memory
// (src/abi/mod.rs and structs.gen.rs).
const MAX_ARRAY_LENGTH = 1 << 20;
const MAX_STRING_BYTES = 64 << 20;
const MAX_CHAIN_LENGTH = 16;

/** Most futures one timed `wgpuInstanceWaitAny` may wait on. */
const TIMED_WAIT_ANY_MAX_COUNT = 64;
/** What `wasmer_wgpu_wait` returns on a host that cannot block. */
const CANNOT_BLOCK = -0x8000_0000;
/** Futureless events kept for a guest that does not collect them. */
const DEFAULT_MAX_PENDING_EVENTS = 4096;
/** Longest delay a timer takes; anything longer is "no timeout". */
const MAX_TIMER_MS = 0x7fff_ffff;
/** How often a suspended guest's host asks the embedder whether to stop. */
const TERMINATION_POLL_MS = 50;

// Layout of an event record (src/host/events.rs): a 32-byte header, then the
// message and the extra payload, each padded to 8 bytes.
const RECORD = { future: 0, kind: 8, status: 12, handle: 16, aux: 20, messageLength: 24, extraLength: 28, size: 32 };

// The members of WGPULimits that are 64-bit; `S` has offsets, not types.
const LIMITS_64 = new Set(['maxUniformBufferBindingSize', 'maxStorageBufferBindingSize', 'maxBufferSize']);
const LIMIT_NAMES = Object.keys(S.WGPULimits).filter((name) => !name.startsWith('__') && name !== 'nextInChain');
const COMPATIBILITY_LIMIT_NAMES = Object.keys(S.WGPUCompatibilityModeLimits).filter(
  (name) => !name.startsWith('__') && name !== 'chain',
);
/** The core limit a compatibility-mode limit refines. */
const coreLimitOf = (name) => name.replace(/In(?:Vertex|Fragment)Stage$/, 'PerShaderStage');

const CORE_FEATURE = WEBIDL.FeatureName[E.FeatureName.CoreFeaturesAndLimits];
const FEATURE_VALUES = Object.values(E.FeatureName).sort((a, b) => a - b);
const WGSL_FEATURE_VALUES = Object.values(E.WGSLLanguageFeatureName).sort((a, b) => a - b);

const VALID_ENUM_VALUES = Object.fromEntries(
  Object.entries(E).map(([name, entries]) => [name, new Set(Object.values(entries))]),
);
const reversed = (table) => Object.fromEntries(Object.entries(table).map(([value, text]) => [text, Number(value)]));
const TEXTURE_FORMAT_VALUES = reversed(WEBIDL.TextureFormat);
const TEXTURE_DIMENSION_VALUES = reversed(WEBIDL.TextureDimension);
const TEXTURE_VIEW_DIMENSION_VALUES = reversed(WEBIDL.TextureViewDimension);

const allBits = (flags) => BigInt(Object.values(flags).reduce((bits, bit) => bits | bit, 0));
const BUFFER_USAGES = allBits(F.BufferUsage);
const TEXTURE_USAGES = allBits(F.TextureUsage);
const SHADER_STAGES = allBits(F.ShaderStage);
const COLOR_WRITES = allBits(F.ColorWriteMask);
const MAP_READ = BigInt(F.MapMode.Read);
const MAP_WRITE = BigInt(F.MapMode.Write);

const NO_EXTENSIONS = [];
const SURFACE_SOURCES = Object.entries(STYPE)
  .filter(([name]) => name.includes('SurfaceSource'))
  .map(([, value]) => value);

const Status = E.Status;
const MapState = E.BufferMapState;
const ErrorType = E.ErrorType;

const EMPTY = new Uint8Array(0);
const NON_ASCII = /[^\0-\x7f]/;
const utf8 = new TextEncoder();
// `ignoreBOM` keeps a leading U+FEFF, as String::from_utf8_lossy does.
const utf8Decoder = new TextDecoder('utf-8', { ignoreBOM: true });

const CHANNELS = { r: 1, rg: 2, rgba: 4, bgra: 4 };

/**
 * Texels and bytes of one block of a texture format: what wgpu's
 * `block_dimensions` and `block_copy_size` say, which is what the native host
 * estimates a texture's storage from.
 */
function texelBlock(format) {
  const astc = /^astc-(\d+)x(\d+)-/.exec(format);
  if (astc) return { width: Number(astc[1]), height: Number(astc[2]), bytes: 16 };
  if (/^(bc|etc2|eac)/.test(format)) {
    return { width: 4, height: 4, bytes: /^(bc1|bc4|etc2-rgb8|eac-r11)/.test(format) ? 8 : 16 };
  }
  const plain = /^(r|rg|rgba|bgra)(8|16|32)/.exec(format);
  if (plain) return { width: 1, height: 1, bytes: (CHANNELS[plain[1]] * Number(plain[2])) / 8 };
  // Formats with no copy size of their own count as eight bytes natively;
  // what is left after these are the packed 32-bit formats.
  const bytes = {
    stencil8: 1, depth16unorm: 2, depth32float: 4, depth24plus: 8, 'depth24plus-stencil8': 8,
    'depth32float-stencil8': 8,
  }[format] ?? 4;
  return { width: 1, height: 1, bytes };
}

/** A rough size for accounting: every mip of every layer and sample. */
function textureBytes({ format, size, sampleCount, mipLevelCount }) {
  const block = texelBlock(format);
  const base = Math.ceil(size.width / block.width) * Math.ceil(size.height / block.height) *
    size.depthOrArrayLayers * block.bytes * Math.max(sampleCount, 1);
  // A full mip chain adds a third.
  return mipLevelCount > 1 ? base + Math.floor(base / 3) : base;
}

const hex = (value) => `0x${value.toString(16)}`;
const pad8 = (length) => (length + 7) & ~7;
const describe = (error) => (typeof error?.message === 'string' ? error.message : String(error));
const toU32 = (value) => value >>> 0;
const toU64 = (value) => BigInt.asUintN(64, value);
const unchanged = (value) => value;

/** The size of an image as `copyExternalImageToTexture` copies it. */
const imageSize = (source) => [
  source.displayWidth ?? source.videoWidth ?? source.naturalWidth ?? source.width,
  source.displayHeight ?? source.videoHeight ?? source.naturalHeight ?? source.height,
];

// ---------------------------------------------------------------------------
// Queues the browser still owes an answer
// ---------------------------------------------------------------------------

/**
 * The queues with submitted work the browser has not reported done, each with
 * whether more was submitted since it was last asked.
 *
 * Chrome answers every submit from its GPU process, and keeps the queue alive
 * by itself until the answer is in. A queue nothing else refers to by then is
 * freed from inside the answer's handler, and when that follows a flush the
 * thread deadlocks for good (Chrome 154). It takes a queue that is collected
 * within milliseconds of its last submit, which is what a guest that exits
 * after a submit leaves behind for the next guest on its thread. So a queue
 * stays reachable from here, whatever becomes of the host that used it, until
 * its work is done.
 */
const queuesAtWork = new Map();

function keepUntilWorkIsDone(queue) {
  const known = queuesAtWork.get(queue);
  if (known) {
    known.more = true;
    return;
  }
  const entry = { more: false };
  queuesAtWork.set(queue, entry);
  const settled = () => {
    if (entry.more) ask();
    else queuesAtWork.delete(queue);
  };
  // Settles after the answers to everything submitted before it, and also
  // when the device is destroyed or lost.
  const ask = () => {
    entry.more = false;
    queue.onSubmittedWorkDone().then(settled, settled);
  };
  ask();
}

// ---------------------------------------------------------------------------
// How a call can fail
// ---------------------------------------------------------------------------

/** A guest address outside linear memory: the guest's segfault. It traps. */
class GuestFault extends WebAssembly.RuntimeError {}

/**
 * A decoding failure that is the guest's bug (an out-of-range enum, an sType
 * that does not belong in a chain). As on the native host there is no device
 * to blame in general, so it is logged and the call returns its zero value.
 */
class AbiError extends Error {}

/** A WebGPU validation error this host found before the browser could. */
class Invalid extends Error {}

// ---------------------------------------------------------------------------
// Guest memory
// ---------------------------------------------------------------------------

/**
 * Bounds-checked access to the guest's linear memory. The memory is shared
 * and can grow, which gives it a new buffer object, so views are re-derived
 * whenever that changes.
 */
function guestMemory() {
  let memory = null;
  let buffer = null;
  let view = null;
  let bytes = null;

  const check = (ptr, length) => {
    if (memory === null) throw new GuestFault('WebGPU was called before the instance memory was set');
    if (memory.buffer !== buffer) {
      buffer = memory.buffer;
      view = new DataView(buffer);
      bytes = new Uint8Array(buffer);
    }
    if (ptr + length > bytes.length) {
      throw new GuestFault(`${length} bytes at ${hex(ptr)} are outside guest memory`);
    }
  };

  return {
    set(next) {
      memory = next;
      buffer = null;
    },
    u16(ptr) {
      check(ptr, 2);
      return view.getUint16(ptr, true);
    },
    u32(ptr) {
      check(ptr, 4);
      return view.getUint32(ptr, true);
    },
    i32(ptr) {
      check(ptr, 4);
      return view.getInt32(ptr, true);
    },
    u64(ptr) {
      check(ptr, 8);
      return view.getBigUint64(ptr, true);
    },
    f32(ptr) {
      check(ptr, 4);
      return view.getFloat32(ptr, true);
    },
    f64(ptr) {
      check(ptr, 8);
      return view.getFloat64(ptr, true);
    },
    setU32(ptr, value) {
      check(ptr, 4);
      view.setUint32(ptr, value, true);
    },
    setU64(ptr, value) {
      check(ptr, 8);
      view.setBigUint64(ptr, value, true);
    },
    /** The guest's own bytes: only valid until the guest runs again. */
    shared(ptr, length) {
      if (length === 0) return EMPTY;
      check(ptr, length);
      return bytes.subarray(ptr, ptr + length);
    },
    /** A private copy, for APIs that do not take shared memory. */
    copy(ptr, length) {
      if (length === 0) return EMPTY;
      check(ptr, length);
      return bytes.slice(ptr, ptr + length);
    },
    write(ptr, data) {
      if (data.length === 0) return;
      check(ptr, data.length);
      bytes.set(data, ptr);
    },
    /** Length of the NUL-terminated string at `ptr`. */
    stringLength(ptr) {
      check(ptr, 1);
      const end = bytes.indexOf(0, ptr);
      if (end < 0) throw new GuestFault(`the string at ${hex(ptr)} runs past the end of guest memory`);
      return end - ptr;
    },
  };
}

// ---------------------------------------------------------------------------
// Handle table
// ---------------------------------------------------------------------------

const INDEX_BITS = 24;
const INDEX_MASK = (1 << INDEX_BITS) - 1;

/**
 * The guest's opaque `WGPU*` values and what they refer to (src/host/table.rs).
 * A handle is a slot index plus the slot's generation, so a stale handle is
 * detected instead of aliasing the slot's next occupant.
 */
function handleTable(maxLive, log) {
  const slots = [];
  const free = [];
  let live = 0;

  const entryOf = (handle) => {
    const slot = slots[(handle & INDEX_MASK) - 1];
    return slot !== undefined && slot.generation === handle >>> INDEX_BITS ? slot.entry : null;
  };

  return {
    /** Stores `state` and returns the guest's single reference to it, or 0. */
    insert(kind, state, instance) {
      if (live >= maxLive) {
        log(`object limit reached; refusing to create a ${kind}`);
        return 0;
      }
      let index = free.pop();
      if (index === undefined) {
        if (slots.length >= INDEX_MASK) return 0;
        index = slots.push({ generation: 0, entry: null }) - 1;
      }
      const slot = slots[index];
      const handle = ((slot.generation << INDEX_BITS) | (index + 1)) >>> 0;
      // An instance descends from itself.
      slot.entry = { kind, state, refs: 1, instance: kind === 'Instance' ? handle : instance };
      live += 1;
      return handle;
    },
    /** The state behind `handle` if it is live and of this kind. */
    peek(handle, kind) {
      const entry = entryOf(handle);
      return entry !== null && entry.kind === kind ? entry.state : null;
    },
    get(handle, kind) {
      const entry = entryOf(handle);
      if (entry !== null && entry.kind === kind) return entry.state;
      const found = entry !== null ? `a ${entry.kind}` : handle === 0 ? 'null' : 'not a live handle';
      log(`expected a ${kind} handle, got ${hex(handle)} (${found})`);
      return null;
    },
    /** The instance an object descends from (itself, for an instance). */
    instanceOf(handle) {
      return entryOf(handle)?.instance ?? 0;
    },
    addRef(handle, kind) {
      const entry = entryOf(handle);
      if (entry === null || entry.kind !== kind) {
        log(`AddRef on an invalid ${kind} handle ${hex(handle)}`);
        return;
      }
      entry.refs += 1;
    },
    /** Drops one guest reference. Returns the state when it was the last. */
    release(handle, kind) {
      const entry = entryOf(handle);
      if (entry === null || entry.kind !== kind) {
        log(`Release on an invalid ${kind} handle ${hex(handle)}`);
        return null;
      }
      entry.refs -= 1;
      if (entry.refs > 0) return null;
      const index = (handle & INDEX_MASK) - 1;
      const slot = slots[index];
      slot.entry = null;
      slot.generation = (slot.generation + 1) & 0xff;
      free.push(index);
      live -= 1;
      return entry.state;
    },
  };
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

function encodeRecord({ future = 0n, kind, status = 0, handle = 0, aux = 0, message = '', extra = EMPTY }) {
  const text = utf8.encode(message);
  const bytes = new Uint8Array(RECORD.size + pad8(text.length) + pad8(extra.length));
  const view = new DataView(bytes.buffer);
  view.setBigUint64(RECORD.future, future, true);
  view.setUint32(RECORD.kind, kind, true);
  view.setUint32(RECORD.status, status, true);
  view.setUint32(RECORD.handle, handle, true);
  view.setUint32(RECORD.aux, aux, true);
  view.setUint32(RECORD.messageLength, text.length, true);
  view.setUint32(RECORD.extraLength, extra.length, true);
  bytes.set(text, RECORD.size);
  bytes.set(extra, RECORD.size + pad8(text.length));
  return bytes;
}

/**
 * Completed asynchronous work, waiting for the guest to collect it
 * (src/host/events.rs).
 */
function eventQueue(maxPending) {
  const ready = [];
  const waiters = new Set();
  let epoch = 0n;

  return {
    /** Records queued so far; the shim compares it before it blocks. */
    get epoch() {
      return epoch;
    },
    /**
     * Queues a completed event. Events of a future are never dropped: the
     * guest is owed exactly one callback per future. Futureless events
     * (uncaptured errors) are dropped once the queue is over its limit, so a
     * guest that never polls cannot grow host memory without bound.
     */
    push(record) {
      const future = record.future ?? 0n;
      if (future === 0n && ready.length >= maxPending) return;
      ready.push({ future, bytes: encodeRecord(record) });
      epoch += 1n;
      for (const wake of [...waiters]) wake();
    },
    isReady(futures) {
      return ready.some((record) => record.future !== 0n && futures.includes(record.future));
    },
    /**
     * Calls `wake` when the next event is queued or after `milliseconds`,
     * whichever comes first. Returns how to call that off.
     */
    onNextPush(wake, milliseconds) {
      const timer = milliseconds <= MAX_TIMER_MS ? setTimeout(wake, milliseconds) : undefined;
      waiters.add(wake);
      return () => {
        clearTimeout(timer);
        waiters.delete(wake);
      };
    },
    /**
     * Moves as many whole records as fit into the guest buffer. Returns the
     * bytes written, or the negated size of the first record when not even
     * that one fits (the shim grows its buffer and asks again).
     */
    drainInto(memory, buf, cap) {
      let written = 0;
      let taken = 0;
      for (const { bytes } of ready) {
        if (written + bytes.length > cap) {
          if (written === 0) return -Math.min(bytes.length, 0x7fff_ffff);
          break;
        }
        memory.write(buf + written, bytes);
        written += bytes.length;
        taken += 1;
      }
      ready.splice(0, taken);
      return written;
    },
  };
}

// ---------------------------------------------------------------------------
// The host
// ---------------------------------------------------------------------------

/** The default target lookup: canvases the embedder registered by selector. */
const registeredCanvas = (selector) => globalThis.__wasmerWebGpuCanvases?.get(selector) ?? null;

/**
 * Creates the WebGPU imports of one guest process.
 *
 * - `gpu`: the `GPU` object to use; none means no adapter is ever available.
 * - `getCanvas(selector)`: resolves the selector of a
 *   `WGPUWasixSurfaceSourceCanvas` to the target a surface presents to, or to
 *   a promise of it; nullish means "no such target". The target is an
 *   `OffscreenCanvas` or `HTMLCanvasElement`, or a binding
 *   `{ canvas, getSize?, present?, release? }` of one:
 *     - `getSize()` returns `[width, height]`, the size the embedder would
 *       like the guest to render at (default: the canvas's own size);
 *     - `present()` is called each time the guest presents a frame, before
 *       the guest is held until the next animation frame;
 *     - `release()` is called once, when the surface is gone.
 * - `log(message)`: diagnostics for the embedder. Guests learn about their
 *   mistakes through WebGPU's own error reporting, not from this.
 * - `suspending`: whether the embedder wraps the imports flagged suspending
 *   with `WebAssembly.Suspending`, so that they may block the guest.
 * - `forceFallbackAdapter`: only ever hand the guest a software adapter,
 *   whatever it asks for.
 * - `isTerminated()`: the embedder's kill switch. Once it returns true the
 *   guest's GPU work is stopped, as by `dispose`, but the imports keep
 *   answering the calls a dying guest still makes.
 * - `limits`: `{ maxObjects, maxDevices, maxPendingEvents, maxGpuMemoryBytes }`,
 *   the ceilings of `WebGpuLimits` on the native host.
 *
 * Returns `{ imports, setMemory, copyExternalImage, dispose }`. `imports` has
 * one function per entry of `IMPORTS`; the suspending ones are `async` when
 * `suspending`.
 */
export function createWebGpuHost({
  gpu = globalThis.navigator?.gpu,
  getCanvas = registeredCanvas,
  log = () => {},
  suspending = typeof WebAssembly.Suspending === 'function',
  forceFallbackAdapter = false,
  isTerminated = () => false,
  limits: {
    maxObjects = Infinity,
    maxDevices = Infinity,
    maxPendingEvents = DEFAULT_MAX_PENDING_EVENTS,
    maxGpuMemoryBytes = Infinity,
  } = {},
} = {}) {
  const mem = guestMemory();
  const table = handleTable(maxObjects, log);
  const events = eventQueue(maxPendingEvents);
  /** Devices that are not lost, for termination and the device limit. */
  const devices = new Set();
  /** Surfaces whose target has not been released yet. */
  const surfaces = new Set();
  /** How to resume each guest that is suspended in a blocking import. */
  const resumers = new Set();
  /** Runs while a guest is suspended, to notice the embedder's kill switch. */
  let watchdog;
  /** Carries the turns of the event loop `wasmer_wgpu_yield` takes. */
  let channel = null;
  const yields = [];
  /** Bytes of buffer and texture storage currently allocated. */
  let gpuMemory = 0;
  /** The guest's GPU work was stopped; see `terminate`. */
  let terminated = false;
  let disposed = false;
  /**
   * The device the current call acts on: where an error the call raises is
   * reported. It is the device of the first object the call resolves.
   */
  let blame = null;

  // -- handles ----------------------------------------------------------------

  const get = (handle, kind) => {
    const state = table.get(handle, kind);
    if (state !== null) blame ??= state.device ?? null;
    return state;
  };

  /**
   * A handle webgpu.h allows to be null: `null` for the null handle, the
   * state for a live one, `undefined` (after logging) for anything else.
   */
  const optional = (handle, kind) => (handle === 0 ? null : (get(handle, kind) ?? undefined));

  const insert = (kind, state, parent) => table.insert(kind, state, table.instanceOf(parent));

  // -- errors -----------------------------------------------------------------

  const filterMatches = (filter, type) =>
    (filter === E.ErrorFilter.Validation && type === ErrorType.Validation) ||
    (filter === E.ErrorFilter.OutOfMemory && type === ErrorType.OutOfMemory) ||
    (filter === E.ErrorFilter.Internal && type === ErrorType.Internal);

  /**
   * Reports an error this host found itself. The browser's own errors go
   * through the device's real error scopes; this keeps a mirror of that
   * stack so that an error raised here reaches the same scope: the innermost
   * one whose filter matches, or else the uncaptured-error callback.
   */
  const reportError = (device, type, message) => {
    for (let i = device.scopes.length - 1; i >= 0; i--) {
      const scope = device.scopes[i];
      if (filterMatches(scope.filter, type)) {
        scope.error ??= { type, message };
        return;
      }
    }
    // "The uncaptured error callback never fires after the device is lost."
    if (device.lost) return;
    events.push({ kind: EVENT.UncapturedError, handle: device.handle, aux: type, message });
  };

  const errorTypeOf = (error) => {
    if (error instanceof globalThis.GPUValidationError) return ErrorType.Validation;
    if (error instanceof globalThis.GPUOutOfMemoryError) return ErrorType.OutOfMemory;
    if (error instanceof globalThis.GPUInternalError) return ErrorType.Internal;
    return ErrorType.Unknown;
  };

  /**
   * Wraps the implementation of one import: arguments arrive as wasm passes
   * them (signed), and nothing but a guest fault may leave as an exception.
   */
  const guard = ([name, params, result, mayBlock], implementation) => {
    const zero = result === 'j' ? 0n : result === 'i' ? 0 : undefined;
    const convert = Array.from(params, (type) => (type === 'i' ? toU32 : type === 'j' ? toU64 : unchanged));
    const enter = (args) => {
      blame = null;
      for (let i = 0; i < args.length; i++) args[i] = convert[i](args[i]);
      if (mayBlock) checkTerminated();
    };
    const recover = (error) => {
      if (error instanceof GuestFault) {
        // The trap names the call the guest made, as natively.
        error.message = `${name}: ${error.message}`;
        throw error;
      }
      // The browser's own argument checks throw; webgpu.h has no exceptions,
      // and on a native host the same mistakes are validation errors.
      const message = error instanceof Invalid ? error.message : `${name}: ${describe(error)}`;
      if (error instanceof AbiError || blame === null) log(message);
      else reportError(blame, ErrorType.Validation, message);
      return zero;
    };
    if (mayBlock && suspending) {
      return async (...args) => {
        try {
          enter(args);
          return await implementation(...args);
        } catch (error) {
          return recover(error);
        }
      };
    }
    return (...args) => {
      try {
        enter(args);
        return implementation(...args);
      } catch (error) {
        return recover(error);
      } finally {
        // Whatever the call asked of a device may still be waiting to be
        // sent; see `flush`.
        if (blame !== null) blame.unflushed = true;
      }
    };
  };

  /**
   * Starts the work behind a future and makes sure the future gets exactly
   * one record whatever happens. `start` returns the record's fields, or a
   * promise of them; what is known at once is queued at once, as natively.
   */
  const settle = (future, kind, failed, start) => {
    const push = (record) => events.push({ ...record, future, kind });
    const fail = (error) => push({ status: failed, message: describe(error) });
    let outcome;
    try {
      outcome = start();
    } catch (error) {
      if (error instanceof GuestFault) throw error;
      fail(error);
      return;
    }
    if (typeof outcome?.then === 'function') outcome.then(push).catch(fail);
    else push(outcome);
  };

  // -- suspending and termination ---------------------------------------------

  /**
   * Suspends the guest until `arm` calls back, or until its GPU work is
   * terminated. `arm` may return how to call the wait off.
   */
  const suspended = (arm) =>
    new Promise((resolve) => {
      let disarm;
      const resume = () => {
        if (!resumers.delete(resume)) return;
        if (resumers.size === 0) clearInterval(watchdog);
        disarm?.();
        resolve();
      };
      // One shared timer, and only while something is suspended.
      if (resumers.size === 0) watchdog = setInterval(checkTerminated, TERMINATION_POLL_MS);
      resumers.add(resume);
      disarm = arm(resume);
    });

  /**
   * Sends what the guest asked of its devices on its way, and returns a
   * promise of the GPU process having caught up with it (`null` when nothing
   * was waiting).
   *
   * A browser may batch the calls made on a device and send them only when a
   * call needs an answer; Chrome does. Until then nothing comes back, not
   * even the uncaptured error of a call that was invalid. Natively the
   * backend is driven every time the guest polls, so a guest may well wait
   * for such an error without asking for anything else. An empty error scope
   * is the call with an answer used here: it captures nothing, and popping it
   * is a round trip that does not wait for the GPU itself.
   */
  const flush = () => {
    const answers = [];
    for (const device of devices) {
      if (!device.unflushed) continue;
      device.unflushed = false;
      device.object.pushErrorScope('validation');
      answers.push(device.object.popErrorScope());
    }
    return answers.length === 0 ? null : Promise.allSettled(answers);
  };

  /**
   * One turn of the event loop. A message channel, not a timer: timers are
   * clamped to several milliseconds once they nest.
   */
  const macrotask = () => {
    if (disposed) return Promise.resolve();
    if (channel === null) {
      channel = new MessageChannel();
      channel.port1.onmessage = () => yields.shift()?.();
    }
    return suspended((resume) => {
      yields.push(resume);
      channel.port2.postMessage(0);
    });
  };

  /**
   * `wasmer_wgpu_yield`: what a guest that spins on
   * `wgpuInstanceProcessEvents` does between two looks. Natively polling
   * drives the backend; here the guest lets the event loop turn once, and
   * waits for the answers to what it asked since it last yielded.
   */
  const yieldToBrowser = async () => {
    const caughtUp = flush();
    await macrotask();
    if (caughtUp !== null) await suspended((resume) => void caughtUp.then(resume));
  };

  /** The next animation frame: what presenting to a canvas waits for. */
  const nextFrame = () => {
    flush();
    // A page that is not visible has no frames; a guest being stopped must
    // not be left waiting for one.
    if (terminated || typeof globalThis.requestAnimationFrame !== 'function') return macrotask();
    return suspended((resume) => {
      const frame = globalThis.requestAnimationFrame(resume);
      return () => globalThis.cancelAnimationFrame(frame);
    });
  };

  /**
   * Stops the guest's GPU work, as `Host::terminate` does natively: every
   * device is destroyed, a suspended guest is resumed, and no adapter or
   * device can be requested any more. The handles stay valid for the calls a
   * dying guest still makes.
   */
  const terminate = () => {
    if (terminated) return;
    terminated = true;
    for (const device of [...devices]) destroyDevice(device);
    for (const resume of [...resumers]) resume();
  };

  const checkTerminated = () => {
    if (!terminated && isTerminated()) terminate();
  };

  // -- reading guest structs --------------------------------------------------

  const SV = S.WGPUStringView;
  const CHAINED = S.WGPUChainedStruct;

  const bool = (ptr) => mem.u32(ptr) !== 0;

  const required = (ptr, name) => {
    if (ptr === 0) throw new AbiError(`\`${name}\` must not be null`);
  };

  /** Checks that `value` is one webgpu.h defines for the enum. */
  const checkEnum = (name, value) => {
    if (!VALID_ENUM_VALUES[name].has(value)) throw new AbiError(`${hex(value)} is not a valid WGPU${name}`);
    return value;
  };

  const enumValue = (name, ptr) => checkEnum(name, mem.u32(ptr));

  /** The WebIDL string of an enum member; `undefined` for its "undefined". */
  const idl = (name, ptr) => WEBIDL[name][enumValue(name, ptr)];

  const checkArray = (ptr, count, elementSize) => {
    if (count > MAX_ARRAY_LENGTH) throw new AbiError(`array of ${count} elements is too long`);
    // Touch both ends so that an array outside memory faults before any use.
    if (count !== 0) mem.shared(ptr, count * elementSize);
  };

  const array = (ptr, count, elementSize, read) => {
    checkArray(ptr, count, elementSize);
    return Array.from({ length: count }, (_, i) => read(ptr + i * elementSize));
  };

  const u32Array = (ptr, count) => array(ptr, count, 4, mem.u32);

  const stringBytes = (data, length) => {
    // An explicit length is the whole string, NULs included.
    const byteLength = length === WGPU_STRLEN ? mem.stringLength(data) : length;
    if (byteLength > MAX_STRING_BYTES) throw new AbiError('string is too long');
    // TextDecoder does not take views of shared memory.
    return mem.copy(data, byteLength);
  };

  /** A string view that cannot be null: null `data` reads as "". */
  const text = (ptr) => {
    const data = mem.u32(ptr + SV.data);
    return data === 0 ? '' : utf8Decoder.decode(stringBytes(data, mem.u32(ptr + SV.length)));
  };

  /** A nullable string view: `{NULL, WGPU_STRLEN}` is the null value. */
  const optionalText = (ptr) => {
    const data = mem.u32(ptr + SV.data);
    const length = mem.u32(ptr + SV.length);
    if (data === 0) return length === WGPU_STRLEN ? undefined : '';
    return utf8Decoder.decode(stringBytes(data, length));
  };

  /**
   * Follows the `nextInChain` list of the struct at `ptr`. Returns
   * `[sType, address]` pairs; an sType outside `allowed` is the guest's bug.
   */
  const chainOf = (ptr, layout, allowed = NO_EXTENSIONS) => {
    const links = [];
    for (let next = mem.u32(ptr + layout.nextInChain); next !== 0; next = mem.u32(next + CHAINED.next)) {
      if (links.length >= MAX_CHAIN_LENGTH) throw new AbiError('nextInChain list is too long');
      const sType = mem.u32(next + CHAINED.sType);
      if (!allowed.includes(sType)) throw new AbiError(`sType ${hex(sType)} is not allowed in this chain`);
      // Emscripten's sType for the same struct.
      links.push([sType === STYPE.WasixSurfaceSourceCanvasAlias ? STYPE.WasixSurfaceSourceCanvas : sType, next]);
    }
    return links;
  };

  const linked = (links, sType) => links.find((link) => link[0] === sType)?.[1];

  /** Flag bits as a number, after checking the guest set only known ones. */
  const flags = (bits, known, message) => {
    if ((bits & ~known) !== 0n) throw new Invalid(message);
    return Number(bits);
  };

  const color = (ptr) => {
    const C = S.WGPUColor;
    return { r: mem.f64(ptr + C.r), g: mem.f64(ptr + C.g), b: mem.f64(ptr + C.b), a: mem.f64(ptr + C.a) };
  };

  const extent3D = (ptr) => {
    const X = S.WGPUExtent3D;
    return {
      width: mem.u32(ptr + X.width),
      height: mem.u32(ptr + X.height),
      depthOrArrayLayers: mem.u32(ptr + X.depthOrArrayLayers),
    };
  };

  const origin3D = (ptr) => {
    const O = S.WGPUOrigin3D;
    return { x: mem.u32(ptr + O.x), y: mem.u32(ptr + O.y), z: mem.u32(ptr + O.z) };
  };

  /** `WGPU_COPY_STRIDE_UNDEFINED` means "not specified". */
  const copyStride = (ptr) => {
    const value = mem.u32(ptr);
    return value === WGPU_COPY_STRIDE_UNDEFINED ? undefined : value;
  };

  const texelCopyBufferLayout = (ptr) => {
    const L = S.WGPUTexelCopyBufferLayout;
    return {
      offset: Number(mem.u64(ptr + L.offset)),
      bytesPerRow: copyStride(ptr + L.bytesPerRow),
      rowsPerImage: copyStride(ptr + L.rowsPerImage),
    };
  };

  /** A buffer copy view, or `null` when its buffer handle is not valid. */
  const texelCopyBufferInfo = (ptr, name) => {
    required(ptr, name);
    const I = S.WGPUTexelCopyBufferInfo;
    const layout = texelCopyBufferLayout(ptr + I.layout);
    const buffer = get(mem.u32(ptr + I.buffer), 'Buffer');
    return buffer && { ...layout, buffer: buffer.object };
  };

  /** A texture copy view, or `null` when its texture handle is not valid. */
  const texelCopyTextureInfo = (ptr, name) => {
    required(ptr, name);
    const I = S.WGPUTexelCopyTextureInfo;
    const info = {
      mipLevel: mem.u32(ptr + I.mipLevel),
      origin: origin3D(ptr + I.origin),
      aspect: idl('TextureAspect', ptr + I.aspect),
    };
    const texture = get(mem.u32(ptr + I.texture), 'Texture');
    return texture && { ...info, texture: texture.object };
  };

  const copySize = (ptr) => {
    required(ptr, 'copy_size');
    return extent3D(ptr);
  };

  /** `WGPU_QUERY_SET_INDEX_UNDEFINED` means "do not write". */
  const queryIndex = (ptr) => {
    const value = mem.u32(ptr);
    return value === WGPU_QUERY_SET_INDEX_UNDEFINED ? undefined : value;
  };

  const passTimestampWrites = (ptr, call) => {
    if (ptr === 0) return undefined;
    const W = S.WGPUPassTimestampWrites;
    chainOf(ptr, W);
    const querySet = get(mem.u32(ptr + W.querySet), 'QuerySet');
    if (!querySet) throw new Invalid(`${call}: invalid timestamp query set.`);
    return {
      querySet: querySet.object,
      beginningOfPassWriteIndex: queryIndex(ptr + W.beginningOfPassWriteIndex),
      endOfPassWriteIndex: queryIndex(ptr + W.endOfPassWriteIndex),
    };
  };

  const pipelineConstants = (ptr, count) => {
    const C = S.WGPUConstantEntry;
    const entries = array(ptr, count, C.__size, (entry) => {
      chainOf(entry, C);
      return [text(entry + C.key), mem.f64(entry + C.value)];
    });
    return count === 0 ? undefined : Object.fromEntries(entries);
  };

  /** The members every programmable stage shares. */
  const stage = (ptr, layout, call, what) => {
    chainOf(ptr, layout);
    const constants = pipelineConstants(mem.u32(ptr + layout.constants), mem.u32(ptr + layout.constantCount));
    const module = get(mem.u32(ptr + layout.module), 'ShaderModule');
    if (!module) throw new Invalid(`${call}: invalid ${what}.`);
    return { module: module.object, entryPoint: optionalText(ptr + layout.entryPoint), constants };
  };

  /** A null layout asks for the layout to be derived from the shader. */
  const pipelineLayout = (handle, call) => {
    if (handle === 0) return 'auto';
    const layout = get(handle, 'PipelineLayout');
    if (!layout) throw new Invalid(`${call}: invalid layout.`);
    return layout.object;
  };

  const computePipelineDescriptor = (ptr) => {
    const D = S.WGPUComputePipelineDescriptor;
    const call = 'wgpuDeviceCreateComputePipeline';
    chainOf(ptr, D);
    return {
      label: text(ptr + D.label),
      compute: stage(ptr + D.compute, S.WGPUComputeState, call, 'shader module'),
      layout: pipelineLayout(mem.u32(ptr + D.layout), call),
    };
  };

  const vertexBufferLayout = (ptr) => {
    const B = S.WGPUVertexBufferLayout;
    const A = S.WGPUVertexAttribute;
    chainOf(ptr, B);
    const stepMode = idl('VertexStepMode', ptr + B.stepMode);
    const vertexAttribute = (attribute) => {
      chainOf(attribute, A);
      return {
        format: idl('VertexFormat', attribute + A.format),
        offset: Number(mem.u64(attribute + A.offset)),
        shaderLocation: mem.u32(attribute + A.shaderLocation),
      };
    };
    const attributes = array(
      mem.u32(ptr + B.attributes), mem.u32(ptr + B.attributeCount), A.__size, vertexAttribute,
    );
    // A slot with no step mode and no attributes is a hole.
    if (stepMode === undefined && attributes.length === 0) return null;
    return { arrayStride: Number(mem.u64(ptr + B.arrayStride)), stepMode, attributes };
  };

  const blendComponent = (ptr) => {
    const C = S.WGPUBlendComponent;
    return {
      operation: idl('BlendOperation', ptr + C.operation),
      srcFactor: idl('BlendFactor', ptr + C.srcFactor),
      dstFactor: idl('BlendFactor', ptr + C.dstFactor),
    };
  };

  const colorTarget = (ptr, call) => {
    const T = S.WGPUColorTargetState;
    chainOf(ptr, T);
    const format = idl('TextureFormat', ptr + T.format);
    const blendPtr = mem.u32(ptr + T.blend);
    const blend = blendPtr === 0 ? undefined : {
      color: blendComponent(blendPtr + S.WGPUBlendState.color),
      alpha: blendComponent(blendPtr + S.WGPUBlendState.alpha),
    };
    // A target with an undefined format is a hole.
    if (format === undefined) return null;
    const writeMask = flags(mem.u64(ptr + T.writeMask), COLOR_WRITES, `${call}: invalid color write mask.`);
    return { format, blend, writeMask };
  };

  const stencilFace = (ptr) => {
    const FACE = S.WGPUStencilFaceState;
    return {
      compare: idl('CompareFunction', ptr + FACE.compare),
      failOp: idl('StencilOperation', ptr + FACE.failOp),
      depthFailOp: idl('StencilOperation', ptr + FACE.depthFailOp),
      passOp: idl('StencilOperation', ptr + FACE.passOp),
    };
  };

  const depthStencilState = (ptr, call) => {
    if (ptr === 0) return undefined;
    const D = S.WGPUDepthStencilState;
    chainOf(ptr, D);
    const depthWrite = enumValue('OptionalBool', ptr + D.depthWriteEnabled);
    const state = {
      format: idl('TextureFormat', ptr + D.format),
      // WGPUOptionalBool_Undefined leaves the member out.
      depthWriteEnabled: depthWrite === E.OptionalBool.Undefined ? undefined : depthWrite === E.OptionalBool.True,
      depthCompare: idl('CompareFunction', ptr + D.depthCompare),
      stencilFront: stencilFace(ptr + D.stencilFront),
      stencilBack: stencilFace(ptr + D.stencilBack),
      stencilReadMask: mem.u32(ptr + D.stencilReadMask),
      stencilWriteMask: mem.u32(ptr + D.stencilWriteMask),
      depthBias: mem.i32(ptr + D.depthBias),
      depthBiasSlopeScale: mem.f32(ptr + D.depthBiasSlopeScale),
      depthBiasClamp: mem.f32(ptr + D.depthBiasClamp),
    };
    if (state.format === undefined) throw new Invalid(`${call}: a depth-stencil state needs a format.`);
    return state;
  };

  const renderPipelineDescriptor = (ptr) => {
    const D = S.WGPURenderPipelineDescriptor;
    const V = S.WGPUVertexState;
    const P = S.WGPUPrimitiveState;
    const M = S.WGPUMultisampleState;
    const FS = S.WGPUFragmentState;
    const call = 'wgpuDeviceCreateRenderPipeline';
    chainOf(ptr, D);

    const vertexPtr = ptr + D.vertex;
    const buffers = array(
      mem.u32(vertexPtr + V.buffers), mem.u32(vertexPtr + V.bufferCount), S.WGPUVertexBufferLayout.__size,
      vertexBufferLayout,
    );
    const vertex = { ...stage(vertexPtr, V, call, 'vertex shader module'), buffers };

    const primitivePtr = ptr + D.primitive;
    chainOf(primitivePtr, P);
    const primitive = {
      topology: idl('PrimitiveTopology', primitivePtr + P.topology),
      stripIndexFormat: idl('IndexFormat', primitivePtr + P.stripIndexFormat),
      frontFace: idl('FrontFace', primitivePtr + P.frontFace),
      cullMode: idl('CullMode', primitivePtr + P.cullMode),
      unclippedDepth: bool(primitivePtr + P.unclippedDepth),
    };

    const multisamplePtr = ptr + D.multisample;
    chainOf(multisamplePtr, M);
    const multisample = {
      // Zero is what a zero-initialized struct says; it means one.
      count: Math.max(mem.u32(multisamplePtr + M.count), 1),
      mask: mem.u32(multisamplePtr + M.mask),
      alphaToCoverageEnabled: bool(multisamplePtr + M.alphaToCoverageEnabled),
    };

    const fragmentPtr = mem.u32(ptr + D.fragment);
    let fragment;
    if (fragmentPtr !== 0) {
      const targets = array(
        mem.u32(fragmentPtr + FS.targets), mem.u32(fragmentPtr + FS.targetCount), S.WGPUColorTargetState.__size,
        (target) => colorTarget(target, call),
      );
      fragment = { ...stage(fragmentPtr, FS, call, 'fragment shader module'), targets };
    }

    return {
      label: text(ptr + D.label),
      layout: pipelineLayout(mem.u32(ptr + D.layout), call),
      vertex,
      primitive,
      depthStencil: depthStencilState(mem.u32(ptr + D.depthStencil), call),
      multisample,
      fragment,
    };
  };

  const colorAttachment = (ptr, call) => {
    const A = S.WGPURenderPassColorAttachment;
    chainOf(ptr, A);
    const depthSlice = mem.u32(ptr + A.depthSlice);
    const attachment = {
      depthSlice: depthSlice === WGPU_DEPTH_SLICE_UNDEFINED ? undefined : depthSlice,
      loadOp: idl('LoadOp', ptr + A.loadOp),
      storeOp: idl('StoreOp', ptr + A.storeOp) ?? 'store',
      clearValue: color(ptr + A.clearValue),
    };
    // A null view is a hole in the attachment list.
    const viewHandle = mem.u32(ptr + A.view);
    if (viewHandle === 0) return null;
    const view = get(viewHandle, 'TextureView');
    if (!view) throw new Invalid(`${call}: invalid color attachment view.`);
    const resolveTarget = optional(mem.u32(ptr + A.resolveTarget), 'TextureView');
    if (resolveTarget === undefined) throw new Invalid(`${call}: invalid resolve target.`);
    if (attachment.loadOp === undefined) throw new Invalid(`${call}: a color attachment needs a load op.`);
    return { ...attachment, view: view.object, resolveTarget: resolveTarget?.object };
  };

  /**
   * The load and store ops of one aspect. An aspect with neither is one the
   * pass does not touch (absent from the format, or read-only); with one,
   * the other takes its default.
   */
  const aspectOps = (readOnly, load, store) =>
    readOnly || (load === undefined && store === undefined) ? [] : [load ?? 'load', store ?? 'store'];

  const depthStencilAttachment = (ptr, call) => {
    if (ptr === 0) return undefined;
    const A = S.WGPURenderPassDepthStencilAttachment;
    chainOf(ptr, A);
    const depthReadOnly = bool(ptr + A.depthReadOnly);
    const stencilReadOnly = bool(ptr + A.stencilReadOnly);
    const [depthLoadOp, depthStoreOp] = aspectOps(
      depthReadOnly, idl('LoadOp', ptr + A.depthLoadOp), idl('StoreOp', ptr + A.depthStoreOp),
    );
    const [stencilLoadOp, stencilStoreOp] = aspectOps(
      stencilReadOnly, idl('LoadOp', ptr + A.stencilLoadOp), idl('StoreOp', ptr + A.stencilStoreOp),
    );
    const depthClearValue = mem.f32(ptr + A.depthClearValue);
    const view = get(mem.u32(ptr + A.view), 'TextureView');
    if (!view) throw new Invalid(`${call}: invalid depth-stencil attachment view.`);
    return {
      view: view.object,
      // WGPU_DEPTH_CLEAR_VALUE_UNDEFINED is NaN.
      depthClearValue: Number.isNaN(depthClearValue) ? undefined : depthClearValue,
      depthLoadOp,
      depthStoreOp,
      depthReadOnly,
      stencilClearValue: mem.u32(ptr + A.stencilClearValue),
      stencilLoadOp,
      stencilStoreOp,
      stencilReadOnly,
    };
  };

  const renderPassDescriptor = (ptr) => {
    const D = S.WGPURenderPassDescriptor;
    const call = 'wgpuCommandEncoderBeginRenderPass';
    const links = chainOf(ptr, D, [STYPE.RenderPassMaxDrawCount]);
    const maxDrawCount = linked(links, STYPE.RenderPassMaxDrawCount);
    const colorAttachments = array(
      mem.u32(ptr + D.colorAttachments), mem.u32(ptr + D.colorAttachmentCount),
      S.WGPURenderPassColorAttachment.__size, (attachment) => colorAttachment(attachment, call),
    );
    const depthStencil = depthStencilAttachment(mem.u32(ptr + D.depthStencilAttachment), call);
    const occlusionQuerySet = optional(mem.u32(ptr + D.occlusionQuerySet), 'QuerySet');
    if (occlusionQuerySet === undefined) throw new Invalid(`${call}: invalid occlusion query set.`);
    return {
      label: text(ptr + D.label),
      colorAttachments,
      depthStencilAttachment: depthStencil,
      occlusionQuerySet: occlusionQuerySet?.object,
      timestampWrites: passTimestampWrites(mem.u32(ptr + D.timestampWrites), call),
      maxDrawCount: maxDrawCount === undefined
        ? undefined
        : Number(mem.u64(maxDrawCount + S.WGPURenderPassMaxDrawCount.maxDrawCount)),
    };
  };

  /** One layout entry. Exactly one of its four sub-layouts must be used. */
  const bindGroupLayoutEntry = (ptr) => {
    const L = S.WGPUBindGroupLayoutEntry;
    const B = S.WGPUBufferBindingLayout;
    const SL = S.WGPUSamplerBindingLayout;
    const T = S.WGPUTextureBindingLayout;
    const ST = S.WGPUStorageTextureBindingLayout;
    const links = chainOf(ptr, L, [STYPE.ExternalTextureBindingLayout]);
    const binding = mem.u32(ptr + L.binding);
    const entry = { binding };
    let used = 0;
    let missingStorageFormat = false;

    const bufferPtr = ptr + L.buffer;
    chainOf(bufferPtr, B);
    if (enumValue('BufferBindingType', bufferPtr + B.type) !== E.BufferBindingType.BindingNotUsed) {
      used += 1;
      entry.buffer = {
        type: idl('BufferBindingType', bufferPtr + B.type),
        hasDynamicOffset: bool(bufferPtr + B.hasDynamicOffset),
        minBindingSize: Number(mem.u64(bufferPtr + B.minBindingSize)),
      };
    }

    const samplerPtr = ptr + L.sampler;
    chainOf(samplerPtr, SL);
    if (enumValue('SamplerBindingType', samplerPtr + SL.type) !== E.SamplerBindingType.BindingNotUsed) {
      used += 1;
      entry.sampler = { type: idl('SamplerBindingType', samplerPtr + SL.type) };
    }

    const texturePtr = ptr + L.texture;
    chainOf(texturePtr, T);
    const viewDimension = idl('TextureViewDimension', texturePtr + T.viewDimension);
    if (enumValue('TextureSampleType', texturePtr + T.sampleType) !== E.TextureSampleType.BindingNotUsed) {
      used += 1;
      entry.texture = {
        sampleType: idl('TextureSampleType', texturePtr + T.sampleType),
        viewDimension,
        multisampled: bool(texturePtr + T.multisampled),
      };
    }

    const storagePtr = ptr + L.storageTexture;
    chainOf(storagePtr, ST);
    const storageFormat = idl('TextureFormat', storagePtr + ST.format);
    const storageDimension = idl('TextureViewDimension', storagePtr + ST.viewDimension);
    if (enumValue('StorageTextureAccess', storagePtr + ST.access) !== E.StorageTextureAccess.BindingNotUsed) {
      used += 1;
      missingStorageFormat = storageFormat === undefined;
      entry.storageTexture = {
        access: idl('StorageTextureAccess', storagePtr + ST.access),
        format: storageFormat,
        viewDimension: storageDimension,
      };
    }

    const fail = (message) => {
      throw new Invalid(`binding ${binding}: ${message}`);
    };
    if (missingStorageFormat) fail('a storage texture binding needs a format');
    // webgpu.h has the external texture type but no way to create one.
    if (linked(links, STYPE.ExternalTextureBindingLayout) !== undefined) {
      fail('external texture bindings are not supported');
    }
    if (used === 0) {
      fail('a bind group layout entry must use exactly one of buffer, sampler, texture or storageTexture');
    }
    if (used > 1) {
      fail('a bind group layout entry uses more than one of buffer, sampler, texture and storageTexture');
    }
    entry.visibility = flags(
      mem.u64(ptr + L.visibility), SHADER_STAGES, `binding ${binding}: invalid shader stage visibility`,
    );
    if (mem.u32(ptr + L.bindingArraySize) > 1) fail('binding arrays are not supported by a browser');
    return entry;
  };

  const bindGroupEntry = (ptr) => {
    const B = S.WGPUBindGroupEntry;
    const links = chainOf(ptr, B, [STYPE.ExternalTextureBindingEntry]);
    if (linked(links, STYPE.ExternalTextureBindingEntry) !== undefined) {
      throw new Invalid('External texture bindings are not supported.');
    }
    const binding = mem.u32(ptr + B.binding);
    const bufferHandle = mem.u32(ptr + B.buffer);
    const samplerHandle = mem.u32(ptr + B.sampler);
    const viewHandle = mem.u32(ptr + B.textureView);
    let resource;
    if (bufferHandle !== 0) {
      const buffer = get(bufferHandle, 'Buffer');
      const size = mem.u64(ptr + B.size);
      resource = buffer && {
        buffer: buffer.object,
        offset: Number(mem.u64(ptr + B.offset)),
        // WGPU_WHOLE_SIZE, and the zero of a zero-initialized entry: to the
        // end of the buffer.
        size: size === WGPU_WHOLE_SIZE || size === 0n ? undefined : Number(size),
      };
    } else if (samplerHandle !== 0) {
      resource = get(samplerHandle, 'Sampler')?.object;
    } else if (viewHandle !== 0) {
      resource = get(viewHandle, 'TextureView')?.object;
    }
    if (!resource) {
      throw new Invalid(`binding ${binding}: a bind group entry needs a valid buffer, sampler or texture view`);
    }
    return { binding, resource };
  };

  // -- instance, adapter and device state -------------------------------------

  const timedWaitAnyMaxCount = suspending ? TIMED_WAIT_ANY_MAX_COUNT : 0;

  /**
   * Timed waits need the guest to be suspendable. SPIR-V shader sources and
   * several devices per adapter are things no browser offers.
   */
  const hasInstanceFeature = (feature) => feature === E.InstanceFeatureName.TimedWaitAny && suspending;

  const hasFeature = (features, feature) => {
    const name = WEBIDL.FeatureName[feature];
    return name !== undefined && features.has(name);
  };

  /** Features as C values. Ones webgpu.h has no name for are skipped. */
  const featureList = (features) => FEATURE_VALUES.filter((feature) => hasFeature(features, feature));

  const wgslFeatures = () => gpu?.wgslLanguageFeatures ?? new Set();
  const hasWgslFeature = (feature) => {
    const name = WEBIDL.WGSLLanguageFeatureName[feature];
    return name !== undefined && wgslFeatures().has(name);
  };

  const u32Blob = (values) => new Uint8Array(Uint32Array.from(values).buffer);

  /**
   * Copies `bytes` to the guest if they fit. Returns the size either way, so
   * the shim can retry with a buffer that is large enough.
   */
  const writeBlob = (buf, cap, bytes) => {
    if (bytes.length <= cap) mem.write(buf, bytes);
    return bytes.length;
  };

  /**
   * Serializes adapter info for the shim, which owns the strings it hands
   * out (src/host/adapter.rs):
   *
   *     u32 backendType, adapterType, vendorID, deviceID, subgroupMinSize, subgroupMaxSize;
   *     u32 vendorLength, architectureLength, deviceLength, descriptionLength;
   *     u8  strings[] (in that order, no terminators)
   */
  const encodeAdapterInfo = (adapter) => {
    const info = adapter.info;
    const strings = [info?.vendor, info?.architecture, info?.device, info?.description].map((string) =>
      utf8.encode(string ?? ''),
    );
    const header = Uint32Array.from([
      E.BackendType.WebGPU,
      // All a browser tells about the kind of adapter.
      (info?.isFallbackAdapter ?? adapter.isFallbackAdapter) ? E.AdapterType.CPU : E.AdapterType.Unknown,
      0, // A browser does not disclose PCI ids.
      0,
      info?.subgroupMinSize ?? 0,
      info?.subgroupMaxSize ?? 0,
      ...strings.map((string) => string.length),
    ]);
    const out = new Uint8Array(header.byteLength + strings.reduce((total, string) => total + string.length, 0));
    out.set(new Uint8Array(header.buffer));
    let at = header.byteLength;
    for (const string of strings) {
      out.set(string, at);
      at += string.length;
    }
    return out;
  };

  /** Fills a guest `WGPULimits` and the limit structs chained to it. */
  const writeLimits = (ptr, supported) => {
    if (ptr === 0) return Status.Error;
    const L = S.WGPULimits;
    let links;
    try {
      // Reading first validates the chain the guest hung off the struct.
      links = chainOf(ptr, L, [STYPE.CompatibilityModeLimits]);
    } catch (error) {
      if (error instanceof AbiError) return Status.Error;
      throw error;
    }
    for (const name of LIMIT_NAMES) {
      // A limit this browser does not know reads as zero.
      const value = supported[name] ?? 0;
      if (LIMITS_64.has(name)) mem.setU64(ptr + L[name], BigInt(value));
      else mem.setU32(ptr + L[name], value);
    }
    const compatibility = linked(links, STYPE.CompatibilityModeLimits);
    if (compatibility !== undefined) {
      // Outside compatibility mode these take the values of the core limits
      // they refine.
      for (const name of COMPATIBILITY_LIMIT_NAMES) {
        const value = supported[name] ?? supported[coreLimitOf(name)] ?? 0;
        mem.setU32(compatibility + S.WGPUCompatibilityModeLimits[name], value);
      }
    }
    return Status.Success;
  };

  /**
   * The limits a guest requires, as a `requiredLimits` dictionary. A member
   * left at its "undefined" sentinel is left out, which keeps the default.
   */
  const requiredLimits = (ptr, supported) => {
    const L = S.WGPULimits;
    const links = chainOf(ptr, L, [STYPE.CompatibilityModeLimits]);
    const limits = {};
    const require = (name, value) => {
      // A limit this browser does not know read as zero when the guest asked
      // for the adapter's limits; echoing that back must not fail.
      if (!(name in supported) && value === 0) return;
      limits[name] = value;
    };
    for (const name of LIMIT_NAMES) {
      if (LIMITS_64.has(name)) {
        const value = mem.u64(ptr + L[name]);
        if (value !== WGPU_LIMIT_U64_UNDEFINED) require(name, Number(value));
      } else {
        const value = mem.u32(ptr + L[name]);
        if (value !== WGPU_LIMIT_U32_UNDEFINED) require(name, value);
      }
    }
    const compatibility = linked(links, STYPE.CompatibilityModeLimits);
    if (compatibility !== undefined) {
      for (const name of COMPATIBILITY_LIMIT_NAMES) {
        const value = mem.u32(compatibility + S.WGPUCompatibilityModeLimits[name]);
        // Asking a browser for a limit it does not know fails the request.
        if (value !== WGPU_LIMIT_U32_UNDEFINED && name in supported) limits[name] = value;
      }
    }
    return limits;
  };

  const loseDevice = (device, reason, message) => {
    if (device.lost) return;
    device.lost = true;
    devices.delete(device);
    device.object.removeEventListener('uncapturederror', device.onUncapturedError);
    events.push({
      future: device.lostFuture, kind: EVENT.DeviceLost, status: reason, handle: device.handle, message,
    });
  };

  const destroyDevice = (device) => {
    device.object.destroy();
    // The browser reports the loss a turn of the event loop later. Make it
    // immediate, as natively, so a guest waiting on the lost future does not
    // depend on anything else yielding.
    loseDevice(device, E.DeviceLostReason.Destroyed, 'Device was destroyed.');
  };

  /** Wraps a new device and installs its error and loss handlers. */
  const adoptDevice = (gpuDevice, adapter, lostFuture, instance) => {
    const features = new Set(gpuDevice.features);
    if (adapter.assumesCore) features.add(CORE_FEATURE);
    const device = {
      object: gpuDevice,
      /** The guest's handle to the device, or 0 once the guest dropped it. */
      handle: 0,
      lost: false,
      lostFuture,
      /** Mirrors the device's error scope stack; see `reportError`. */
      scopes: [],
      /** The guest's queue handle while it holds one. */
      queueHandle: 0,
      /** The guest used the device since it was last flushed. */
      unflushed: false,
      info: adapter.info,
      features,
      onUncapturedError: (event) => {
        if (device.lost) return;
        events.push({
          kind: EVENT.UncapturedError,
          handle: device.handle,
          aux: errorTypeOf(event.error),
          message: describe(event.error),
        });
      },
    };
    device.device = device;
    devices.add(device);
    gpuDevice.addEventListener('uncapturederror', device.onUncapturedError);
    gpuDevice.lost.then((info) => {
      const reason = info.reason === 'destroyed' ? E.DeviceLostReason.Destroyed : E.DeviceLostReason.Unknown;
      loseDevice(device, reason, info.message);
    });
    device.handle = table.insert('Device', device, instance);
    if (device.handle === 0) destroyDevice(device);
    return device;
  };

  /** Validates a device request and starts it. Throws what makes it fail. */
  const requestDevice = (adapterHandle, descriptorPtr) => {
    const D = S.WGPUDeviceDescriptor;
    let label;
    let featureValues = [];
    let limitsPtr = 0;
    let queueLabel;
    if (descriptorPtr !== 0) {
      chainOf(descriptorPtr, D);
      label = text(descriptorPtr + D.label);
      featureValues = u32Array(
        mem.u32(descriptorPtr + D.requiredFeatures), mem.u32(descriptorPtr + D.requiredFeatureCount),
      );
      limitsPtr = mem.u32(descriptorPtr + D.requiredLimits);
      chainOf(descriptorPtr + D.defaultQueue, S.WGPUQueueDescriptor);
      queueLabel = text(descriptorPtr + D.defaultQueue + S.WGPUQueueDescriptor.label);
    }
    checkTerminated();
    if (terminated) throw new Error('The WebGPU context was shut down.');
    const adapter = get(adapterHandle, 'Adapter');
    if (!adapter) throw new Error('Invalid adapter.');
    if (devices.size >= maxDevices) throw new Error(`This guest may have at most ${maxDevices} devices.`);
    for (const feature of featureValues) {
      if (!hasFeature(adapter.features, feature)) {
        throw new Error(`Required feature ${hex(feature)} is not supported.`);
      }
    }
    const requiredFeatures = featureValues
      .map((feature) => WEBIDL.FeatureName[feature])
      // Not a name this browser knows: every device it makes is a core one.
      .filter((name) => !(name === CORE_FEATURE && adapter.assumesCore));
    return {
      adapter,
      instance: table.instanceOf(adapterHandle),
      promise: adapter.object.requestDevice({
        label,
        requiredFeatures,
        requiredLimits: limitsPtr === 0 ? undefined : requiredLimits(limitsPtr, adapter.object.limits),
        defaultQueue: { label: queueLabel },
      }),
    };
  };

  // -- buffer and texture storage ---------------------------------------------

  /** Whether `bytes` more of storage would exceed the guest's budget. */
  const overBudget = (bytes) => gpuMemory + bytes > maxGpuMemoryBytes;

  /**
   * Hands the guest a buffer or texture and counts its storage against the
   * budget until `returnStorage`.
   */
  const insertStorage = (kind, state, bytes, parent) => {
    state.accounted = bytes;
    gpuMemory += bytes;
    const handle = insert(kind, state, parent);
    if (handle === 0) returnStorage(state);
    return handle;
  };

  /** The object's storage is gone: destroyed, or no longer referenced. */
  const returnStorage = (state) => {
    gpuMemory -= state.accounted;
    state.accounted = 0;
  };

  // -- buffers ----------------------------------------------------------------

  const unmapped = () => ({ state: MapState.Unmapped, mode: 0n, offset: 0, size: 0, request: null });

  /** Marks the buffer unmapped, noting a cancelled pending map. */
  const endMapping = (buffer) => {
    if (buffer.map.state === MapState.Pending) buffer.map.request.aborted = true;
    buffer.map = unmapped();
    buffer.mapped = null;
  };

  /**
   * The bytes of `[offset, offset + size)` of the current mapping, or `null`
   * when that is not something the mapping covers.
   *
   * JavaScript hands out each mapped byte once, as an ArrayBuffer that lives
   * until unmap. The copying calls of the C API have no such rule (nor the
   * alignment rules of getMappedRange), so the whole mapping is taken once
   * and kept.
   */
  const mappedBytes = (buffer, offset, size, write) => {
    const { state, mode, offset: start, size: length } = buffer.map;
    if (state !== MapState.Mapped) return null;
    if (write && (mode & MAP_WRITE) === 0n) return null;
    if (offset < start || offset + size > start + length) return null;
    if (size === 0) return EMPTY;
    try {
      buffer.mapped ??= new Uint8Array(buffer.object.getMappedRange(start, length));
    } catch (error) {
      log(`getMappedRange failed: ${describe(error)}`);
      return null;
    }
    return buffer.mapped.subarray(offset - start, offset - start + size);
  };

  // -- surfaces ---------------------------------------------------------------

  /**
   * The checks of surface creation that need no canvas. Returns what is left
   * to do, or `null` (after logging) when the surface cannot be created.
   */
  const surfaceRequest = (instance, descriptorPtr) => {
    const D = S.WGPUSurfaceDescriptor;
    required(descriptorPtr, 'descriptor');
    const links = chainOf(descriptorPtr, D, SURFACE_SOURCES);
    // A canvas context has no label, but a bad string is still the guest's bug.
    text(descriptorPtr + D.label);
    const source = linked(links, STYPE.WasixSurfaceSourceCanvas);
    const selector = source === undefined ? '' : text(source + S.WGPUWasixSurfaceSourceCanvas.selector);
    if (!get(instance, 'Instance')) return null;
    // A WASIX guest has no native window handles to offer, so the only
    // surface source that means anything here is the host-provided one.
    if (source === undefined) {
      log(
        'wgpuInstanceCreateSurface: chain WGPUWasixSurfaceSourceCanvas (webgpu_wasix.h); ' +
          'native window sources are not available to WASIX guests',
      );
      return null;
    }
    if (links.length !== 1) {
      log('wgpuInstanceCreateSurface: exactly one surface source is allowed');
      return null;
    }
    return { instance, selector };
  };

  /** What `getCanvas` found, as a binding: a bare canvas is one with nothing else to say. */
  const targetOf = (found) => (typeof found?.getContext === 'function' ? { canvas: found } : found);

  /** Tells the embedder that a target it handed out is no longer used. */
  const releaseTarget = (target) => {
    try {
      target?.release?.();
    } catch (error) {
      log(`releasing a surface target failed: ${describe(error)}`);
    }
  };

  const openSurface = ({ instance, selector }, found) => {
    const target = targetOf(found);
    if (!target?.canvas) {
      log(`wgpuInstanceCreateSurface: the embedder provides no target for ${JSON.stringify(selector)}`);
      return 0;
    }
    if (disposed) {
      // `dispose` released every target then in use. One the embedder was
      // still looking for has nobody left to release it later.
      releaseTarget(target);
      return 0;
    }
    const surface = { target, canvas: target.canvas, context: null, configured: null, hasFrame: false };
    let handle = 0;
    try {
      surface.context = surface.canvas.getContext('webgpu');
      if (surface.context) handle = table.insert('Surface', surface, instance);
      else log(`wgpuInstanceCreateSurface: the target for ${JSON.stringify(selector)} has no WebGPU context`);
    } finally {
      if (handle === 0) releaseTarget(target);
      else surfaces.add(surface);
    }
    return handle;
  };

  const closeSurface = (surface) => {
    // `dispose` may have closed it before the guest released its handle.
    if (!surfaces.delete(surface)) return;
    surface.context.unconfigure();
    releaseTarget(surface.target);
  };

  /**
   * The size the embedder would like the guest to render at, or `null` (after
   * logging) when its `getSize` fails. Like `present` and `release`, that is
   * the embedder's failure: it must not leave the guest's call half done.
   */
  const surfaceSize = ({ target, canvas }) => {
    try {
      const [width, height] = target.getSize?.() ?? [canvas.width, canvas.height];
      return [width, height];
    } catch (error) {
      log(`asking the embedder for the size of a surface failed: ${describe(error)}`);
      return null;
    }
  };

  /**
   * Takes the surface's current frame and hands it to the embedder. Returns
   * whether a frame was presented.
   */
  const presentFrame = (handle) => {
    const surface = get(handle, 'Surface');
    if (!surface?.hasFrame) return false;
    surface.hasFrame = false;
    try {
      surface.target.present?.();
    } catch (error) {
      log(`wgpuSurfacePresent: ${describe(error)}`);
      return false;
    }
    return true;
  };

  // -- shader modules ---------------------------------------------------------

  const COMPILATION_MESSAGE_TYPES = {
    error: E.CompilationMessageType.Error,
    warning: E.CompilationMessageType.Warning,
    info: E.CompilationMessageType.Info,
  };

  /**
   * The messages of a compilation info, as the event record's extra payload
   * (src/host/shader.rs):
   *
   *     u32 count; u32 reserved;
   *     per message:
   *       u32 type; u32 messageLength;
   *       u64 lineNum; u64 linePos; u64 offset; u64 length;
   *       u8  message[messageLength], padded to 8 bytes
   *
   * JavaScript counts positions in UTF-16 code units and webgpu.h in UTF-8
   * ones. `source` is the module's code when the two differ for it.
   */
  const encodeCompilationMessages = (messages, source) => {
    const utf8Units = (from, to) => (source === null ? to - from : utf8.encode(source.slice(from, to)).length);
    const parts = messages.map((message) => {
      const body = utf8.encode(message.message);
      const part = new Uint8Array(40 + pad8(body.length));
      const view = new DataView(part.buffer);
      const { offset, length, linePos } = message;
      view.setUint32(0, COMPILATION_MESSAGE_TYPES[message.type] ?? E.CompilationMessageType.Info, true);
      view.setUint32(4, body.length, true);
      view.setBigUint64(8, BigInt(message.lineNum), true);
      // A position of zero means the message has none.
      view.setBigUint64(16, BigInt(linePos === 0 ? 0 : utf8Units(offset - (linePos - 1), offset) + 1), true);
      view.setBigUint64(24, BigInt(utf8Units(0, offset)), true);
      view.setBigUint64(32, BigInt(utf8Units(offset, offset + length)), true);
      part.set(body, 40);
      return part;
    });
    const out = new Uint8Array(8 + parts.reduce((total, part) => total + part.length, 0));
    new DataView(out.buffer).setUint32(0, parts.length, true);
    let at = 8;
    for (const part of parts) {
      out.set(part, at);
      at += part.length;
    }
    return out;
  };

  // -- the imports ------------------------------------------------------------

  /** Starts `wgpuDeviceCreate{Compute,Render}PipelineAsync`. */
  const createPipelineAsync = (event, kind, method, describeDescriptor) => (deviceHandle, descriptorPtr, future) => {
    const PipelineStatus = E.CreatePipelineAsyncStatus;
    // A failure is the callback's to report: it raises no device error.
    settle(future, event, PipelineStatus.ValidationError, () => {
      required(descriptorPtr, 'descriptor');
      const device = get(deviceHandle, 'Device');
      if (!device) throw new Invalid('Invalid device.');
      const descriptor = describeDescriptor(descriptorPtr);
      const instance = table.instanceOf(deviceHandle);
      return device.object[method](descriptor).then(
        (pipeline) => {
          const handle = table.insert(kind, { object: pipeline, device }, instance);
          return handle !== 0
            ? { status: PipelineStatus.Success, handle }
            : { status: PipelineStatus.ValidationError, message: 'The pipeline could not be created.' };
        },
        (error) => ({
          status: error?.reason === 'internal' ? PipelineStatus.InternalError : PipelineStatus.ValidationError,
          message: describe(error),
        }),
      );
    });
  };

  /**
   * `wasmer_wgpu_wait`: suspends the guest until one of the listed futures
   * is ready, any event was queued since `epoch`, or the timeout elapses;
   * then behaves like poll.
   */
  const wait = async (futuresPtr, count, timeoutNs, epoch, buf, cap) => {
    checkArray(futuresPtr, count, 8);
    const futures = Array.from({ length: count }, (_, i) => mem.u64(futuresPtr + i * 8));
    if (!events.isReady(futures) && events.epoch === epoch && timeoutNs !== 0n) {
      // Rounded up, so that the shim does not find itself woken a fraction
      // of a millisecond early and have to come back.
      const milliseconds = timeoutNs === U64_MAX ? Infinity : Math.ceil(Number(timeoutNs) / 1e6);
      flush();
      // Nothing will arrive for a guest that is being stopped. It still takes
      // a turn of the event loop, so that one that keeps waiting cannot keep
      // its thread from hearing anything else.
      if (terminated) await macrotask();
      else await suspended((resume) => events.onNextPush(resume, milliseconds));
    }
    return events.drainInto(mem, buf, cap);
  };

  const implementations = {
    // -- global ---------------------------------------------------------------

    wgpuCreateInstance: (descriptorPtr) => {
      if (descriptorPtr !== 0) {
        const D = S.WGPUInstanceDescriptor;
        chainOf(descriptorPtr, D);
        const features = u32Array(
          mem.u32(descriptorPtr + D.requiredFeatures), mem.u32(descriptorPtr + D.requiredFeatureCount),
        );
        const limitsPtr = mem.u32(descriptorPtr + D.requiredLimits);
        const missing = features.find((feature) => !hasInstanceFeature(feature));
        if (missing !== undefined) {
          log(`wgpuCreateInstance: instance feature ${hex(missing)} is not available`);
          return 0;
        }
        if (limitsPtr !== 0) {
          chainOf(limitsPtr, S.WGPUInstanceLimits);
          if (mem.u32(limitsPtr + S.WGPUInstanceLimits.timedWaitAnyMaxCount) > timedWaitAnyMaxCount) return 0;
        }
      }
      return table.insert('Instance', {}, 0);
    },

    wgpuGetInstanceLimits: (limitsPtr) => {
      if (limitsPtr === 0) return Status.Error;
      mem.setU32(limitsPtr + S.WGPUInstanceLimits.timedWaitAnyMaxCount, timedWaitAnyMaxCount);
      return Status.Success;
    },

    wgpuHasInstanceFeature: (feature) => Number(hasInstanceFeature(feature)),

    // -- adapter --------------------------------------------------------------

    wgpuAdapterGetLimits: (handle, limitsPtr) => {
      const adapter = get(handle, 'Adapter');
      return adapter ? writeLimits(limitsPtr, adapter.object.limits) : Status.Error;
    },

    wgpuAdapterHasFeature: (handle, feature) => {
      const adapter = get(handle, 'Adapter');
      return Number(adapter !== null && hasFeature(adapter.features, feature));
    },

    // -- buffer ---------------------------------------------------------------

    wgpuBufferReadMappedRange: (handle, offset, data, size) => {
      const buffer = get(handle, 'Buffer');
      const bytes = buffer && mappedBytes(buffer, offset, size, false);
      if (!bytes) return Status.Error;
      mem.write(data, bytes);
      return Status.Success;
    },

    wgpuBufferWriteMappedRange: (handle, offset, data, size) => {
      const buffer = get(handle, 'Buffer');
      const bytes = buffer && mappedBytes(buffer, offset, size, true);
      if (!bytes) return Status.Error;
      bytes.set(mem.shared(data, size));
      return Status.Success;
    },

    wgpuBufferGetUsage: (handle) => get(handle, 'Buffer')?.usage ?? 0n,

    wgpuBufferGetSize: (handle) => get(handle, 'Buffer')?.size ?? 0n,

    wgpuBufferGetMapState: (handle) => get(handle, 'Buffer')?.map.state ?? MapState.Unmapped,

    wgpuBufferUnmap: (handle) => {
      const buffer = get(handle, 'Buffer');
      if (!buffer) return;
      endMapping(buffer);
      buffer.object.unmap();
    },

    wgpuBufferDestroy: (handle) => {
      const buffer = get(handle, 'Buffer');
      if (!buffer) return;
      endMapping(buffer);
      buffer.object.destroy();
      returnStorage(buffer);
    },

    // -- command encoder ------------------------------------------------------

    wgpuCommandEncoderFinish: (handle, descriptorPtr) => {
      let descriptor;
      if (descriptorPtr !== 0) {
        chainOf(descriptorPtr, S.WGPUCommandBufferDescriptor);
        descriptor = { label: text(descriptorPtr + S.WGPUCommandBufferDescriptor.label) };
      }
      const encoder = get(handle, 'CommandEncoder');
      if (!encoder) return 0;
      const commands = { object: encoder.object.finish(descriptor), device: encoder.device };
      return insert('CommandBuffer', commands, handle);
    },

    wgpuCommandEncoderBeginComputePass: (handle, descriptorPtr) => {
      const D = S.WGPUComputePassDescriptor;
      const call = 'wgpuCommandEncoderBeginComputePass';
      const encoder = get(handle, 'CommandEncoder');
      if (!encoder) return 0;
      let descriptor;
      if (descriptorPtr !== 0) {
        chainOf(descriptorPtr, D);
        descriptor = {
          label: text(descriptorPtr + D.label),
          timestampWrites: passTimestampWrites(mem.u32(descriptorPtr + D.timestampWrites), call),
        };
      }
      const pass = { object: encoder.object.beginComputePass(descriptor), device: encoder.device };
      return insert('ComputePassEncoder', pass, handle);
    },

    wgpuCommandEncoderBeginRenderPass: (handle, descriptorPtr) => {
      required(descriptorPtr, 'descriptor');
      const encoder = get(handle, 'CommandEncoder');
      if (!encoder) return 0;
      const descriptor = renderPassDescriptor(descriptorPtr);
      const pass = { object: encoder.object.beginRenderPass(descriptor), device: encoder.device };
      return insert('RenderPassEncoder', pass, handle);
    },

    wgpuCommandEncoderCopyBufferToBuffer: (handle, sourceHandle, sourceOffset, targetHandle, targetOffset, size) => {
      const encoder = get(handle, 'CommandEncoder');
      const source = get(sourceHandle, 'Buffer');
      const target = get(targetHandle, 'Buffer');
      if (!encoder || !source || !target) return;
      // WGPU_WHOLE_SIZE: the rest of the source. Spelled out, because not
      // every browser takes the size as optional yet.
      const length = size === WGPU_WHOLE_SIZE ? source.size - sourceOffset : size;
      encoder.object.copyBufferToBuffer(
        source.object, Number(sourceOffset), target.object, Number(targetOffset), Number(length),
      );
    },

    wgpuCommandEncoderCopyBufferToTexture: (handle, sourcePtr, destinationPtr, copySizePtr) => {
      const encoder = get(handle, 'CommandEncoder');
      const source = texelCopyBufferInfo(sourcePtr, 'source');
      const destination = texelCopyTextureInfo(destinationPtr, 'destination');
      const size = copySize(copySizePtr);
      if (encoder && source && destination) encoder.object.copyBufferToTexture(source, destination, size);
    },

    wgpuCommandEncoderCopyTextureToBuffer: (handle, sourcePtr, destinationPtr, copySizePtr) => {
      const encoder = get(handle, 'CommandEncoder');
      const source = texelCopyTextureInfo(sourcePtr, 'source');
      const destination = texelCopyBufferInfo(destinationPtr, 'destination');
      const size = copySize(copySizePtr);
      if (encoder && source && destination) encoder.object.copyTextureToBuffer(source, destination, size);
    },

    wgpuCommandEncoderCopyTextureToTexture: (handle, sourcePtr, destinationPtr, copySizePtr) => {
      const encoder = get(handle, 'CommandEncoder');
      const source = texelCopyTextureInfo(sourcePtr, 'source');
      const destination = texelCopyTextureInfo(destinationPtr, 'destination');
      const size = copySize(copySizePtr);
      if (encoder && source && destination) encoder.object.copyTextureToTexture(source, destination, size);
    },

    wgpuCommandEncoderClearBuffer: (handle, bufferHandle, offset, size) => {
      const encoder = get(handle, 'CommandEncoder');
      const buffer = get(bufferHandle, 'Buffer');
      if (!encoder || !buffer) return;
      encoder.object.clearBuffer(buffer.object, Number(offset), size === WGPU_WHOLE_SIZE ? undefined : Number(size));
    },

    wgpuCommandEncoderResolveQuerySet: (handle, querySetHandle, firstQuery, queryCount, targetHandle, targetOffset) => {
      const encoder = get(handle, 'CommandEncoder');
      const querySet = get(querySetHandle, 'QuerySet');
      const target = get(targetHandle, 'Buffer');
      if (!encoder || !querySet || !target) return;
      encoder.object.resolveQuerySet(querySet.object, firstQuery, queryCount, target.object, Number(targetOffset));
    },

    // Not part of the JavaScript API any more: there, only passes write
    // timestamps. A browser may still have the method (Chrome does, for pages
    // allowed its unsafe APIs), and where it has, the call goes through, as it
    // does in Dawn's own Emscripten binding. Where it has not, the guest gets
    // a validation error instead of that binding's exception.
    wgpuCommandEncoderWriteTimestamp: (handle, querySetHandle, index) => {
      const encoder = get(handle, 'CommandEncoder');
      const querySet = get(querySetHandle, 'QuerySet');
      if (!encoder || !querySet) return;
      if (typeof encoder.object.writeTimestamp !== 'function') {
        throw new Invalid(
          'wgpuCommandEncoderWriteTimestamp: this browser cannot write timestamps from a command encoder.',
        );
      }
      encoder.object.writeTimestamp(querySet.object, index);
    },

    // -- compute pass ---------------------------------------------------------

    wgpuComputePassEncoderSetPipeline: (handle, pipelineHandle) => {
      const pass = get(handle, 'ComputePassEncoder');
      const pipeline = get(pipelineHandle, 'ComputePipeline');
      if (pass && pipeline) pass.object.setPipeline(pipeline.object);
    },

    wgpuComputePassEncoderDispatchWorkgroups: (handle, x, y, z) => {
      get(handle, 'ComputePassEncoder')?.object.dispatchWorkgroups(x, y, z);
    },

    wgpuComputePassEncoderDispatchWorkgroupsIndirect: (handle, bufferHandle, offset) => {
      const pass = get(handle, 'ComputePassEncoder');
      const buffer = get(bufferHandle, 'Buffer');
      if (pass && buffer) pass.object.dispatchWorkgroupsIndirect(buffer.object, Number(offset));
    },

    wgpuComputePassEncoderEnd: (handle) => {
      get(handle, 'ComputePassEncoder')?.object.end();
    },

    // -- pipelines ------------------------------------------------------------

    wgpuComputePipelineGetBindGroupLayout: (handle, index) => {
      const pipeline = get(handle, 'ComputePipeline');
      if (!pipeline) return 0;
      const layout = { object: pipeline.object.getBindGroupLayout(index), device: pipeline.device };
      return insert('BindGroupLayout', layout, handle);
    },

    wgpuRenderPipelineGetBindGroupLayout: (handle, index) => {
      const pipeline = get(handle, 'RenderPipeline');
      if (!pipeline) return 0;
      const layout = { object: pipeline.object.getBindGroupLayout(index), device: pipeline.device };
      return insert('BindGroupLayout', layout, handle);
    },

    // -- device ---------------------------------------------------------------

    wgpuDeviceCreateBindGroup: (handle, descriptorPtr) => {
      const D = S.WGPUBindGroupDescriptor;
      required(descriptorPtr, 'descriptor');
      chainOf(descriptorPtr, D);
      const device = get(handle, 'Device');
      if (!device) return 0;
      const layout = get(mem.u32(descriptorPtr + D.layout), 'BindGroupLayout');
      if (!layout) throw new Invalid('wgpuDeviceCreateBindGroup: invalid layout.');
      const group = device.object.createBindGroup({
        label: text(descriptorPtr + D.label),
        layout: layout.object,
        entries: array(
          mem.u32(descriptorPtr + D.entries), mem.u32(descriptorPtr + D.entryCount), S.WGPUBindGroupEntry.__size,
          bindGroupEntry,
        ),
      });
      return insert('BindGroup', { object: group, device }, handle);
    },

    wgpuDeviceCreateBindGroupLayout: (handle, descriptorPtr) => {
      const D = S.WGPUBindGroupLayoutDescriptor;
      required(descriptorPtr, 'descriptor');
      chainOf(descriptorPtr, D);
      const device = get(handle, 'Device');
      if (!device) return 0;
      const layout = device.object.createBindGroupLayout({
        label: text(descriptorPtr + D.label),
        entries: array(
          mem.u32(descriptorPtr + D.entries), mem.u32(descriptorPtr + D.entryCount),
          S.WGPUBindGroupLayoutEntry.__size, bindGroupLayoutEntry,
        ),
      });
      return insert('BindGroupLayout', { object: layout, device }, handle);
    },

    wgpuDeviceCreateBuffer: (handle, descriptorPtr) => {
      const D = S.WGPUBufferDescriptor;
      required(descriptorPtr, 'descriptor');
      chainOf(descriptorPtr, D);
      const label = text(descriptorPtr + D.label);
      const usage = mem.u64(descriptorPtr + D.usage);
      const size = mem.u64(descriptorPtr + D.size);
      const mappedAtCreation = bool(descriptorPtr + D.mappedAtCreation);
      const device = get(handle, 'Device');
      if (!device) return 0;
      const usageBits = flags(usage, BUFFER_USAGES, `${hex(usage)} is not a valid WGPUBufferUsage`);
      if (overBudget(Number(size))) {
        reportError(
          device, ErrorType.OutOfMemory,
          `creating a ${size} byte buffer would exceed this guest's GPU memory budget`,
        );
        return 0;
      }
      let object;
      try {
        object = device.object.createBuffer({ label, usage: usageBits, size: Number(size), mappedAtCreation });
      } catch (error) {
        // A RangeError for a size the mapping rules allow means the mapping
        // could not be allocated: webgpu.h returns null for that.
        if (!(error instanceof RangeError) || size % 4n !== 0n) throw error;
        reportError(device, ErrorType.OutOfMemory, `wgpuDeviceCreateBuffer: ${describe(error)}`);
        return 0;
      }
      const map = mappedAtCreation
        ? { state: MapState.Mapped, mode: MAP_WRITE, offset: 0, size: Number(size), request: null }
        : unmapped();
      return insertStorage('Buffer', { object, device, size, usage, map, mapped: null }, Number(size), handle);
    },

    wgpuDeviceCreateCommandEncoder: (handle, descriptorPtr) => {
      let descriptor;
      if (descriptorPtr !== 0) {
        chainOf(descriptorPtr, S.WGPUCommandEncoderDescriptor);
        descriptor = { label: text(descriptorPtr + S.WGPUCommandEncoderDescriptor.label) };
      }
      const device = get(handle, 'Device');
      if (!device) return 0;
      return insert('CommandEncoder', { object: device.object.createCommandEncoder(descriptor), device }, handle);
    },

    wgpuDeviceCreateComputePipeline: (handle, descriptorPtr) => {
      required(descriptorPtr, 'descriptor');
      const device = get(handle, 'Device');
      if (!device) return 0;
      const pipeline = device.object.createComputePipeline(computePipelineDescriptor(descriptorPtr));
      return insert('ComputePipeline', { object: pipeline, device }, handle);
    },

    wgpuDeviceCreatePipelineLayout: (handle, descriptorPtr) => {
      const D = S.WGPUPipelineLayoutDescriptor;
      const call = 'wgpuDeviceCreatePipelineLayout';
      required(descriptorPtr, 'descriptor');
      chainOf(descriptorPtr, D);
      const label = text(descriptorPtr + D.label);
      const handles = u32Array(
        mem.u32(descriptorPtr + D.bindGroupLayouts), mem.u32(descriptorPtr + D.bindGroupLayoutCount),
      );
      const immediateSize = mem.u32(descriptorPtr + D.immediateSize);
      const device = get(handle, 'Device');
      if (!device) return 0;
      // A null entry is a deliberate hole in the layout.
      const bindGroupLayouts = handles.map((layoutHandle) => {
        const layout = optional(layoutHandle, 'BindGroupLayout');
        if (layout === undefined) throw new Invalid(`${call}: invalid bind group layout.`);
        return layout?.object ?? null;
      });
      const descriptor = { label, bindGroupLayouts };
      if (immediateSize !== 0) {
        // A browser without immediate data would ignore the member.
        if (!('maxImmediateSize' in device.object.limits)) {
          throw new Invalid(`${call}: immediate data is not supported by this browser.`);
        }
        descriptor.immediateSize = immediateSize;
      }
      return insert('PipelineLayout', { object: device.object.createPipelineLayout(descriptor), device }, handle);
    },

    wgpuDeviceCreateQuerySet: (handle, descriptorPtr) => {
      const D = S.WGPUQuerySetDescriptor;
      required(descriptorPtr, 'descriptor');
      chainOf(descriptorPtr, D);
      const type = enumValue('QueryType', descriptorPtr + D.type);
      const count = mem.u32(descriptorPtr + D.count);
      const label = text(descriptorPtr + D.label);
      const device = get(handle, 'Device');
      if (!device) return 0;
      const object = device.object.createQuerySet({ label, type: WEBIDL.QueryType[type], count });
      return insert('QuerySet', { object, device, type, count }, handle);
    },

    wgpuDeviceCreateRenderBundleEncoder: (handle, descriptorPtr) => {
      const D = S.WGPURenderBundleEncoderDescriptor;
      required(descriptorPtr, 'descriptor');
      chainOf(descriptorPtr, D);
      const descriptor = {
        label: text(descriptorPtr + D.label),
        // An undefined color format is a hole in the attachment list.
        colorFormats: array(
          mem.u32(descriptorPtr + D.colorFormats), mem.u32(descriptorPtr + D.colorFormatCount), 4,
          (format) => idl('TextureFormat', format) ?? null,
        ),
        depthStencilFormat: idl('TextureFormat', descriptorPtr + D.depthStencilFormat),
        sampleCount: Math.max(mem.u32(descriptorPtr + D.sampleCount), 1),
        depthReadOnly: bool(descriptorPtr + D.depthReadOnly),
        stencilReadOnly: bool(descriptorPtr + D.stencilReadOnly),
      };
      const device = get(handle, 'Device');
      if (!device) return 0;
      const encoder = { object: device.object.createRenderBundleEncoder(descriptor), device };
      return insert('RenderBundleEncoder', encoder, handle);
    },

    wgpuDeviceCreateRenderPipeline: (handle, descriptorPtr) => {
      required(descriptorPtr, 'descriptor');
      const device = get(handle, 'Device');
      if (!device) return 0;
      const pipeline = device.object.createRenderPipeline(renderPipelineDescriptor(descriptorPtr));
      return insert('RenderPipeline', { object: pipeline, device }, handle);
    },

    wgpuDeviceCreateSampler: (handle, descriptorPtr) => {
      let descriptor;
      if (descriptorPtr !== 0) {
        const D = S.WGPUSamplerDescriptor;
        chainOf(descriptorPtr, D);
        descriptor = {
          label: text(descriptorPtr + D.label),
          addressModeU: idl('AddressMode', descriptorPtr + D.addressModeU),
          addressModeV: idl('AddressMode', descriptorPtr + D.addressModeV),
          addressModeW: idl('AddressMode', descriptorPtr + D.addressModeW),
          magFilter: idl('FilterMode', descriptorPtr + D.magFilter),
          minFilter: idl('FilterMode', descriptorPtr + D.minFilter),
          mipmapFilter: idl('MipmapFilterMode', descriptorPtr + D.mipmapFilter),
          lodMinClamp: mem.f32(descriptorPtr + D.lodMinClamp),
          lodMaxClamp: mem.f32(descriptorPtr + D.lodMaxClamp),
          compare: idl('CompareFunction', descriptorPtr + D.compare),
          // Zero is what a zero-initialized struct says; it means one.
          maxAnisotropy: Math.max(mem.u16(descriptorPtr + D.maxAnisotropy), 1),
        };
      }
      const device = get(handle, 'Device');
      if (!device) return 0;
      return insert('Sampler', { object: device.object.createSampler(descriptor), device }, handle);
    },

    wgpuDeviceCreateShaderModule: (handle, descriptorPtr) => {
      const D = S.WGPUShaderModuleDescriptor;
      required(descriptorPtr, 'descriptor');
      const links = chainOf(descriptorPtr, D, [STYPE.ShaderSourceSPIRV, STYPE.ShaderSourceWGSL]);
      const label = text(descriptorPtr + D.label);
      const wgsl = linked(links, STYPE.ShaderSourceWGSL);
      const code = wgsl === undefined ? undefined : text(wgsl + S.WGPUShaderSourceWGSL.code);
      const device = get(handle, 'Device');
      if (!device) return 0;
      if (code === undefined) {
        throw new Invalid(
          linked(links, STYPE.ShaderSourceSPIRV) === undefined
            ? 'A shader module descriptor must chain WGPUShaderSourceWGSL.'
            : 'SPIR-V shader sources are not supported ' +
              '(WGPUInstanceFeatureName_ShaderSourceSPIRV is not available).',
        );
      }
      const module = {
        object: device.object.createShaderModule({ label, code }),
        device,
        // Kept only when its compilation messages need their positions
        // converted from UTF-16 to UTF-8 code units.
        source: NON_ASCII.test(code) ? code : null,
      };
      return insert('ShaderModule', module, handle);
    },

    wgpuDeviceCreateTexture: (handle, descriptorPtr) => {
      const D = S.WGPUTextureDescriptor;
      const call = 'wgpuDeviceCreateTexture';
      required(descriptorPtr, 'descriptor');
      const links = chainOf(descriptorPtr, D, [STYPE.TextureBindingViewDimension]);
      const bindingViewDimension = linked(links, STYPE.TextureBindingViewDimension);
      const usage = mem.u64(descriptorPtr + D.usage);
      const descriptor = {
        label: text(descriptorPtr + D.label),
        size: extent3D(descriptorPtr + D.size),
        mipLevelCount: mem.u32(descriptorPtr + D.mipLevelCount),
        sampleCount: mem.u32(descriptorPtr + D.sampleCount),
        dimension: idl('TextureDimension', descriptorPtr + D.dimension),
        format: idl('TextureFormat', descriptorPtr + D.format),
        viewFormats: array(
          mem.u32(descriptorPtr + D.viewFormats), mem.u32(descriptorPtr + D.viewFormatCount), 4,
          (format) => idl('TextureFormat', format),
        ),
        textureBindingViewDimension: bindingViewDimension === undefined
          ? undefined
          : idl(
            'TextureViewDimension',
            bindingViewDimension + S.WGPUTextureBindingViewDimension.textureBindingViewDimension,
          ),
      };
      const device = get(handle, 'Device');
      if (!device) return 0;
      if (descriptor.format === undefined) throw new Invalid(`${call}: a texture needs a format.`);
      descriptor.usage = flags(usage, TEXTURE_USAGES, `${call}: invalid texture usage.`);
      if (descriptor.viewFormats.includes(undefined)) throw new Invalid(`${call}: invalid view format.`);
      const bytes = textureBytes(descriptor);
      if (overBudget(bytes)) {
        reportError(
          device, ErrorType.OutOfMemory, "creating this texture would exceed this guest's GPU memory budget",
        );
        return 0;
      }
      const texture = { object: device.object.createTexture(descriptor), device, usage };
      return insertStorage('Texture', texture, bytes, handle);
    },

    wgpuDeviceDestroy: (handle) => {
      const device = get(handle, 'Device');
      if (device) destroyDevice(device);
    },

    wgpuDeviceGetLimits: (handle, limitsPtr) => {
      const device = get(handle, 'Device');
      return device ? writeLimits(limitsPtr, device.object.limits) : Status.Error;
    },

    wgpuDeviceHasFeature: (handle, feature) => {
      const device = get(handle, 'Device');
      return Number(device !== null && hasFeature(device.features, feature));
    },

    wgpuDeviceGetQueue: (handle) => {
      const device = get(handle, 'Device');
      if (!device) return 0;
      // Hand back the handle the guest already has, if it still has it.
      if (device.queueHandle !== 0 && table.peek(device.queueHandle, 'Queue') === device) {
        table.addRef(device.queueHandle, 'Queue');
      } else {
        device.queueHandle = insert('Queue', device, handle);
      }
      return device.queueHandle;
    },

    wgpuDevicePushErrorScope: (handle, filter) => {
      checkEnum('ErrorFilter', filter);
      const device = get(handle, 'Device');
      if (!device) return;
      device.scopes.push({ filter, error: null });
      device.object.pushErrorScope(WEBIDL.ErrorFilter[filter]);
    },

    // -- instance -------------------------------------------------------------

    wgpuInstanceCreateSurface: suspending
      ? async (handle, descriptorPtr) => {
        const request = surfaceRequest(handle, descriptorPtr);
        return request === null ? 0 : openSurface(request, await getCanvas(request.selector));
      }
      : (handle, descriptorPtr) => {
        const request = surfaceRequest(handle, descriptorPtr);
        if (request === null) return 0;
        const found = getCanvas(request.selector);
        if (typeof found?.then === 'function') {
          log('wgpuInstanceCreateSurface: the target is not available without suspending the guest');
          // It arrives too late to be of use: hand it straight back.
          found.then((late) => releaseTarget(targetOf(late)), () => {});
          return 0;
        }
        return openSurface(request, found);
      },

    wgpuInstanceHasWGSLLanguageFeature: (handle, feature) =>
      Number(get(handle, 'Instance') !== null && hasWgslFeature(feature)),

    // -- query set ------------------------------------------------------------

    wgpuQuerySetGetType: (handle) => get(handle, 'QuerySet')?.type ?? E.QueryType.Occlusion,

    wgpuQuerySetGetCount: (handle) => get(handle, 'QuerySet')?.count ?? 0,

    wgpuQuerySetDestroy: (handle) => {
      get(handle, 'QuerySet')?.object.destroy();
    },

    // -- queue ----------------------------------------------------------------

    wgpuQueueSubmit: (handle, count, commandsPtr) => {
      const handles = u32Array(commandsPtr, count);
      const queue = get(handle, 'Queue');
      if (!queue) return;
      const buffers = handles.map((commandsHandle) => {
        const commands = get(commandsHandle, 'CommandBuffer');
        if (!commands) throw new Invalid('wgpuQueueSubmit: invalid command buffer.');
        // A command buffer can be submitted once.
        if (commands.object === null) throw new Invalid('wgpuQueueSubmit: command buffer was already submitted.');
        return commands;
      });
      queue.object.queue.submit(
        buffers.map((commands) => {
          const { object } = commands;
          commands.object = null;
          return object;
        }),
      );
      keepUntilWorkIsDone(queue.object.queue);
    },

    wgpuQueueWriteBuffer: (handle, bufferHandle, offset, data, size) => {
      const queue = get(handle, 'Queue');
      const buffer = get(bufferHandle, 'Buffer');
      if (!queue || !buffer) return;
      let bytes;
      try {
        bytes = mem.shared(data, size);
      } catch (error) {
        if (!(error instanceof GuestFault)) throw error;
        throw new Invalid('wgpuQueueWriteBuffer: data is outside guest memory.');
      }
      queue.object.queue.writeBuffer(buffer.object, Number(offset), bytes, 0, size);
    },

    wgpuQueueWriteTexture: (handle, destinationPtr, data, dataSize, dataLayoutPtr, writeSizePtr) => {
      const queue = get(handle, 'Queue');
      const destination = texelCopyTextureInfo(destinationPtr, 'destination');
      required(dataLayoutPtr, 'data_layout');
      const dataLayout = texelCopyBufferLayout(dataLayoutPtr);
      required(writeSizePtr, 'write_size');
      const writeSize = extent3D(writeSizePtr);
      if (!queue || !destination) return;
      let bytes;
      try {
        bytes = mem.shared(data, dataSize);
      } catch (error) {
        if (!(error instanceof GuestFault)) throw error;
        throw new Invalid('wgpuQueueWriteTexture: data is outside guest memory.');
      }
      queue.object.queue.writeTexture(destination, bytes, dataLayout, writeSize);
    },

    // -- render bundle encoder ------------------------------------------------

    wgpuRenderBundleEncoderFinish: (handle, descriptorPtr) => {
      let descriptor;
      if (descriptorPtr !== 0) {
        chainOf(descriptorPtr, S.WGPURenderBundleDescriptor);
        descriptor = { label: text(descriptorPtr + S.WGPURenderBundleDescriptor.label) };
      }
      const encoder = get(handle, 'RenderBundleEncoder');
      if (!encoder) return 0;
      const bundle = { object: encoder.object.finish(descriptor), device: encoder.device };
      return insert('RenderBundle', bundle, handle);
    },

    // -- render pass ----------------------------------------------------------

    wgpuRenderPassEncoderExecuteBundles: (handle, count, bundlesPtr) => {
      const handles = u32Array(bundlesPtr, count);
      const pass = get(handle, 'RenderPassEncoder');
      const bundles = handles.map((bundleHandle) => get(bundleHandle, 'RenderBundle'));
      if (pass && !bundles.includes(null)) pass.object.executeBundles(bundles.map((bundle) => bundle.object));
    },

    wgpuRenderPassEncoderSetStencilReference: (handle, reference) => {
      get(handle, 'RenderPassEncoder')?.object.setStencilReference(reference);
    },

    wgpuRenderPassEncoderSetBlendConstant: (handle, colorPtr) => {
      required(colorPtr, 'color');
      const value = color(colorPtr);
      get(handle, 'RenderPassEncoder')?.object.setBlendConstant(value);
    },

    wgpuRenderPassEncoderSetViewport: (handle, x, y, width, height, minDepth, maxDepth) => {
      get(handle, 'RenderPassEncoder')?.object.setViewport(x, y, width, height, minDepth, maxDepth);
    },

    wgpuRenderPassEncoderSetScissorRect: (handle, x, y, width, height) => {
      get(handle, 'RenderPassEncoder')?.object.setScissorRect(x, y, width, height);
    },

    wgpuRenderPassEncoderBeginOcclusionQuery: (handle, index) => {
      get(handle, 'RenderPassEncoder')?.object.beginOcclusionQuery(index);
    },

    wgpuRenderPassEncoderEndOcclusionQuery: (handle) => {
      get(handle, 'RenderPassEncoder')?.object.endOcclusionQuery();
    },

    wgpuRenderPassEncoderEnd: (handle) => {
      get(handle, 'RenderPassEncoder')?.object.end();
    },

    // -- surface --------------------------------------------------------------

    wgpuSurfaceConfigure: (handle, configPtr) => {
      const C = S.WGPUSurfaceConfiguration;
      required(configPtr, 'config');
      const links = chainOf(configPtr, C, [STYPE.SurfaceColorManagement]);
      const format = idl('TextureFormat', configPtr + C.format);
      const usage = mem.u64(configPtr + C.usage);
      const width = mem.u32(configPtr + C.width);
      const height = mem.u32(configPtr + C.height);
      const viewFormats = array(
        mem.u32(configPtr + C.viewFormats), mem.u32(configPtr + C.viewFormatCount), 4,
        (viewFormat) => idl('TextureFormat', viewFormat),
      );
      const alphaMode = enumValue('CompositeAlphaMode', configPtr + C.alphaMode);
      const presentMode = enumValue('PresentMode', configPtr + C.presentMode);
      const configuration = {};
      const management = linked(links, STYPE.SurfaceColorManagement);
      if (management !== undefined) {
        const M = S.WGPUSurfaceColorManagement;
        configuration.colorSpace = idl('PredefinedColorSpace', management + M.colorSpace);
        configuration.toneMapping = { mode: idl('ToneMappingMode', management + M.toneMappingMode) };
      }

      const surface = get(handle, 'Surface');
      const device = get(mem.u32(configPtr + C.device), 'Device');
      if (!surface || !device) return;
      const fail = (message) => {
        throw new Invalid(`wgpuSurfaceConfigure: ${message}`);
      };
      if (format === undefined) fail('a surface configuration needs a format.');
      const usageBits = flags(usage, TEXTURE_USAGES, 'wgpuSurfaceConfigure: invalid usage.');
      if (viewFormats.includes(undefined)) fail('invalid view format.');
      if (width === 0 || height === 0) fail('the size must not be zero.');
      // What `wasmer_wgpu_surface_get_capabilities` lists: a canvas is opaque
      // or premultiplied, and presents in step with the display.
      if (alphaMode === E.CompositeAlphaMode.Premultiplied) configuration.alphaMode = 'premultiplied';
      else if (alphaMode === E.CompositeAlphaMode.Opaque) configuration.alphaMode = 'opaque';
      else if (alphaMode !== E.CompositeAlphaMode.Auto) fail('this surface does not support that alpha mode.');
      if (presentMode !== E.PresentMode.Undefined && presentMode !== E.PresentMode.Fifo) {
        fail('this surface does not support that present mode.');
      }

      // Reconfiguring first unconfigures: the current frame is gone.
      surface.hasFrame = false;
      surface.configured = null;
      surface.canvas.width = width;
      surface.canvas.height = height;
      surface.context.configure({ ...configuration, device: device.object, format, usage: usageBits, viewFormats });
      surface.configured = { device, usage, width, height };
    },

    wgpuSurfaceGetCurrentTexture: (handle, surfaceTexturePtr) => {
      if (surfaceTexturePtr === 0) return;
      const T = S.WGPUSurfaceTexture;
      const FrameStatus = E.SurfaceGetCurrentTextureStatus;
      let texture = 0;
      let status = FrameStatus.Error;
      const surface = get(handle, 'Surface');
      // Asking again before presenting is an error, as is asking an
      // unconfigured surface.
      if (surface?.configured && !surface.hasFrame) {
        const { device, usage, width, height } = surface.configured;
        try {
          const frame = { object: surface.context.getCurrentTexture(), device, usage, accounted: 0 };
          texture = insert('Texture', frame, handle);
        } catch (error) {
          log(`wgpuSurfaceGetCurrentTexture: ${describe(error)}`);
        }
        if (texture !== 0) {
          surface.hasFrame = true;
          // The size the embedder wants may have changed since the guest
          // configured; tell it to reconfigure.
          const wanted = surfaceSize(surface);
          const resized = wanted !== null && (wanted[0] !== width || wanted[1] !== height);
          status = resized ? FrameStatus.SuccessSuboptimal : FrameStatus.SuccessOptimal;
        }
      }
      mem.setU32(surfaceTexturePtr + T.texture, texture);
      mem.setU32(surfaceTexturePtr + T.status, status);
    },

    // A canvas presents what was drawn to it when its thread returns to the
    // event loop. Waiting for the next animation frame does that, and paces a
    // guest's frame loop the way a native swap chain does. (In a page that is
    // not visible there are no frames, and the guest stays here.)
    wgpuSurfacePresent: suspending
      ? async (handle) => {
        if (!presentFrame(handle)) return Status.Error;
        await nextFrame();
        return Status.Success;
      }
      : (handle) => (presentFrame(handle) ? Status.Success : Status.Error),

    wgpuSurfaceUnconfigure: (handle) => {
      const surface = get(handle, 'Surface');
      if (!surface) return;
      surface.hasFrame = false;
      surface.configured = null;
      surface.context.unconfigure();
    },

    // A canvas context has no label to set.
    wgpuSurfaceSetLabel: (handle, labelPtr) => {
      text(labelPtr);
      get(handle, 'Surface');
    },

    wgpuSurfaceRelease: (handle) => {
      const surface = table.release(handle, 'Surface');
      if (surface !== null) closeSurface(surface);
    },

    wgpuWasixSurfaceGetSize: (handle, widthPtr, heightPtr) => {
      const surface = get(handle, 'Surface');
      const size = surface && surfaceSize(surface);
      if (!size) return Status.Error;
      if (widthPtr !== 0) mem.setU32(widthPtr, size[0]);
      if (heightPtr !== 0) mem.setU32(heightPtr, size[1]);
      return Status.Success;
    },

    // -- texture --------------------------------------------------------------

    wgpuTextureCreateView: (handle, descriptorPtr) => {
      const texture = get(handle, 'Texture');
      let descriptor;
      if (descriptorPtr !== 0) {
        const D = S.WGPUTextureViewDescriptor;
        const links = chainOf(descriptorPtr, D, [STYPE.TextureComponentSwizzleDescriptor]);
        const mipLevelCount = mem.u32(descriptorPtr + D.mipLevelCount);
        const arrayLayerCount = mem.u32(descriptorPtr + D.arrayLayerCount);
        const usage = mem.u64(descriptorPtr + D.usage);
        descriptor = {
          label: text(descriptorPtr + D.label),
          format: idl('TextureFormat', descriptorPtr + D.format),
          dimension: idl('TextureViewDimension', descriptorPtr + D.dimension),
          aspect: idl('TextureAspect', descriptorPtr + D.aspect),
          baseMipLevel: mem.u32(descriptorPtr + D.baseMipLevel),
          mipLevelCount: mipLevelCount === WGPU_MIP_LEVEL_COUNT_UNDEFINED ? undefined : mipLevelCount,
          baseArrayLayer: mem.u32(descriptorPtr + D.baseArrayLayer),
          arrayLayerCount: arrayLayerCount === WGPU_ARRAY_LAYER_COUNT_UNDEFINED ? undefined : arrayLayerCount,
        };
        const swizzlePtr = linked(links, STYPE.TextureComponentSwizzleDescriptor);
        if (swizzlePtr !== undefined) {
          const W = S.WGPUTextureComponentSwizzle;
          const at = swizzlePtr + S.WGPUTextureComponentSwizzleDescriptor.swizzle;
          const swizzle = ['r', 'g', 'b', 'a'].map((channel) => idl('ComponentSwizzle', at + W[channel]) ?? channel);
          // A browser without the feature would ignore the member.
          if (swizzle.join('') !== 'rgba') {
            if (!texture?.device.object.features.has('texture-component-swizzle')) {
              throw new Invalid('Texture component swizzle is not supported by this backend.');
            }
            descriptor.swizzle = swizzle.join('');
          }
        }
        // Zero inherits the texture's usage.
        if (usage !== 0n) descriptor.usage = flags(usage, TEXTURE_USAGES, 'wgpuTextureCreateView: invalid usage.');
      }
      if (!texture) return 0;
      return insert('TextureView', { object: texture.object.createView(descriptor), device: texture.device }, handle);
    },

    wgpuTextureGetWidth: (handle) => get(handle, 'Texture')?.object.width ?? 0,

    wgpuTextureGetHeight: (handle) => get(handle, 'Texture')?.object.height ?? 0,

    wgpuTextureGetDepthOrArrayLayers: (handle) => get(handle, 'Texture')?.object.depthOrArrayLayers ?? 0,

    wgpuTextureGetMipLevelCount: (handle) => get(handle, 'Texture')?.object.mipLevelCount ?? 0,

    wgpuTextureGetSampleCount: (handle) => get(handle, 'Texture')?.object.sampleCount ?? 0,

    wgpuTextureGetDimension: (handle) =>
      TEXTURE_DIMENSION_VALUES[get(handle, 'Texture')?.object.dimension] ?? E.TextureDimension.Undefined,

    // Only meaningful in compatibility mode.
    wgpuTextureGetTextureBindingViewDimension: (handle) =>
      TEXTURE_VIEW_DIMENSION_VALUES[get(handle, 'Texture')?.object.textureBindingViewDimension] ??
      E.TextureViewDimension.Undefined,

    wgpuTextureGetFormat: (handle) =>
      TEXTURE_FORMAT_VALUES[get(handle, 'Texture')?.object.format] ?? E.TextureFormat.Undefined,

    wgpuTextureGetUsage: (handle) => get(handle, 'Texture')?.usage ?? 0n,

    wgpuTextureDestroy: (handle) => {
      const texture = get(handle, 'Texture');
      if (!texture) return;
      texture.object.destroy();
      returnStorage(texture);
    },

    wgpuTextureRelease: (handle) => {
      const texture = table.release(handle, 'Texture');
      if (texture !== null) returnStorage(texture);
    },

    // -- for the shim ---------------------------------------------------------

    wasmer_wgpu_abi_version: () => ABI_VERSION,

    wasmer_wgpu_object_instance: (handle) => table.instanceOf(handle),

    wasmer_wgpu_instance_request_adapter: (handle, optionsPtr, future) => {
      const AdapterStatus = E.RequestAdapterStatus;
      settle(future, EVENT.RequestAdapter, AdapterStatus.Error, () => {
        const O = S.WGPURequestAdapterOptions;
        let options;
        let backend = E.BackendType.Undefined;
        let surface = 0;
        if (optionsPtr !== 0) {
          const links = chainOf(optionsPtr, O, [STYPE.RequestAdapterWebXROptions]);
          options = {
            featureLevel: idl('FeatureLevel', optionsPtr + O.featureLevel),
            powerPreference: idl('PowerPreference', optionsPtr + O.powerPreference),
            forceFallbackAdapter: bool(optionsPtr + O.forceFallbackAdapter),
          };
          const webxr = linked(links, STYPE.RequestAdapterWebXROptions);
          if (webxr !== undefined) {
            options.xrCompatible = bool(webxr + S.WGPURequestAdapterWebXROptions.xrCompatible);
          }
          backend = enumValue('BackendType', optionsPtr + O.backendType);
          surface = mem.u32(optionsPtr + O.compatibleSurface);
        }
        const unavailable = (message) => ({ status: AdapterStatus.Unavailable, message });
        checkTerminated();
        if (terminated) return unavailable('The WebGPU context was shut down.');
        if (!get(handle, 'Instance')) return unavailable('Invalid instance.');
        // Every adapter of a browser can present to every canvas.
        if (surface !== 0 && !get(surface, 'Surface')) return unavailable('Invalid compatible surface.');
        if (backend !== E.BackendType.Undefined && backend !== E.BackendType.WebGPU) {
          return unavailable('No adapter of the requested backend type is available.');
        }
        if (!gpu) return unavailable('WebGPU is not available here.');
        if (forceFallbackAdapter) options = { ...options, forceFallbackAdapter: true };
        return gpu.requestAdapter(options).then((object) => {
          // The guest may have been stopped while the browser was looking.
          if (terminated) return unavailable('The WebGPU context was shut down.');
          if (!object) return unavailable('No suitable graphics adapter is available.');
          const features = new Set(object.features);
          // A browser that predates the feature's name only has core
          // adapters. One that knows it says so itself.
          const assumesCore = !features.has(CORE_FEATURE) && options?.featureLevel !== 'compatibility';
          if (assumesCore) features.add(CORE_FEATURE);
          const adapter = { object, features, assumesCore, info: encodeAdapterInfo(object) };
          const adapterHandle = table.insert('Adapter', adapter, handle);
          return adapterHandle !== 0
            ? { status: AdapterStatus.Success, handle: adapterHandle }
            : { status: AdapterStatus.Error };
        });
      });
    },

    wasmer_wgpu_adapter_request_device: (handle, descriptorPtr, future, lostFuture) => {
      const fail = (error) => {
        const message = describe(error);
        events.push({ future, kind: EVENT.RequestDevice, status: E.RequestDeviceStatus.Error, message });
        // The device-lost future of a device that never existed.
        events.push({
          future: lostFuture, kind: EVENT.DeviceLost, status: E.DeviceLostReason.FailedCreation, message,
        });
      };
      let request;
      try {
        request = requestDevice(handle, descriptorPtr);
      } catch (error) {
        if (error instanceof GuestFault) throw error;
        fail(error);
        return;
      }
      request.promise
        .then((gpuDevice) => {
          // A device that arrives for a guest that was stopped in the meantime
          // is not one more thing for it to use.
          if (terminated) {
            gpuDevice.destroy();
            throw new Error('The WebGPU context was shut down.');
          }
          const device = adoptDevice(gpuDevice, request.adapter, lostFuture, request.instance);
          events.push({
            future,
            kind: EVENT.RequestDevice,
            status: device.handle !== 0 ? E.RequestDeviceStatus.Success : E.RequestDeviceStatus.Error,
            handle: device.handle,
          });
        })
        .catch(fail);
    },

    wasmer_wgpu_buffer_map_async: (handle, mode, offset, size, future) => {
      const MapStatus = E.MapAsyncStatus;
      const push = (status, message) => events.push({ future, kind: EVENT.BufferMap, status, message });
      const buffer = get(handle, 'Buffer');
      if (!buffer) {
        push(MapStatus.Error, 'Invalid buffer.');
        return;
      }
      const refuse = (message) => {
        reportError(buffer.device, ErrorType.Validation, message);
        push(MapStatus.Error, message);
      };
      if (mode !== MAP_READ && mode !== MAP_WRITE) {
        refuse('A map mode must be exactly Read or Write.');
        return;
      }
      // WGPU_WHOLE_MAP_SIZE: everything from the offset on.
      const length = size === WGPU_WHOLE_MAP_SIZE ? Math.max(Number(buffer.size) - offset, 0) : size;
      if (buffer.map.state !== MapState.Unmapped) {
        refuse('The buffer is already mapped or has a map pending.');
        return;
      }
      if (BigInt(offset) + BigInt(length) > buffer.size) {
        refuse('The mapped range is outside the buffer.');
        return;
      }
      // Identifies this map: the buffer may be unmapped and mapped again
      // before the browser reports how this one went.
      const request = { aborted: false };
      buffer.map = { state: MapState.Pending, mode, offset, size: length, request };
      const failed = (error) => {
        const aborted = request.aborted || error?.name === 'AbortError';
        if (buffer.map.request === request) buffer.map = unmapped();
        if (aborted) push(MapStatus.Aborted, 'The map was cancelled.');
        else push(MapStatus.Error, describe(error));
      };
      let mapping;
      try {
        mapping = buffer.object.mapAsync(Number(mode), offset, length);
      } catch (error) {
        failed(error);
        return;
      }
      mapping.then(() => {
        if (buffer.map.request === request) buffer.map.state = MapState.Mapped;
        push(MapStatus.Success);
      }, failed);
    },

    wasmer_wgpu_device_pop_error_scope: (handle, future) => {
      const PopStatus = E.PopErrorScopeStatus;
      settle(future, EVENT.PopErrorScope, PopStatus.Error, () => {
        const device = get(handle, 'Device');
        const scope = device?.scopes.pop();
        // Nothing to pop: the stack was empty (or the device is invalid).
        if (!scope) {
          return { status: PopStatus.Error, aux: ErrorType.NoError, message: 'There is no error scope to pop.' };
        }
        return device.object.popErrorScope().then((gpuError) => {
          // An error this host raised is told apart from the browser's only
          // here; it was found first, at the call, so it wins.
          const error = scope.error ?? (gpuError && { type: errorTypeOf(gpuError), message: gpuError.message });
          return { status: PopStatus.Success, aux: error?.type ?? ErrorType.NoError, message: error?.message };
        });
      });
    },

    wasmer_wgpu_device_create_compute_pipeline_async: createPipelineAsync(
      EVENT.CreateComputePipeline, 'ComputePipeline', 'createComputePipelineAsync', computePipelineDescriptor,
    ),

    wasmer_wgpu_device_create_render_pipeline_async: createPipelineAsync(
      EVENT.CreateRenderPipeline, 'RenderPipeline', 'createRenderPipelineAsync', renderPipelineDescriptor,
    ),

    wasmer_wgpu_queue_on_submitted_work_done: (handle, future) => {
      const WorkStatus = E.QueueWorkDoneStatus;
      settle(future, EVENT.QueueWorkDone, WorkStatus.Error, () => {
        const queue = get(handle, 'Queue');
        if (!queue) return { status: WorkStatus.Error, message: 'Invalid queue.' };
        return queue.object.queue.onSubmittedWorkDone().then(() => ({ status: WorkStatus.Success }));
      });
    },

    wasmer_wgpu_shader_module_get_compilation_info: (handle, future) => {
      // The request itself cannot fail: an invalid module has no messages.
      const success = (messages, source) => ({
        status: E.CompilationInfoRequestStatus.Success,
        extra: encodeCompilationMessages(messages, source),
      });
      settle(future, EVENT.CompilationInfo, E.CompilationInfoRequestStatus.Success, () => {
        const module = get(handle, 'ShaderModule');
        if (!module) return success([], null);
        return module.object.getCompilationInfo().then((info) => success(info.messages, module.source));
      });
    },

    // Nothing to drive: the browser makes progress whenever the guest is
    // suspended, and what completed is already queued.
    wasmer_wgpu_poll: (buf, cap) => events.drainInto(mem, buf, cap),

    wasmer_wgpu_epoch: () => events.epoch,

    wasmer_wgpu_wait: suspending ? wait : (_futuresPtr, _count, _timeoutNs, _epoch, _buf, _cap) => CANNOT_BLOCK,

    wasmer_wgpu_yield: suspending ? yieldToBrowser : () => {},

    wasmer_wgpu_instance_release: (handle) => Number(table.release(handle, 'Instance') !== null),

    wasmer_wgpu_buffer_release: (handle) => {
      const buffer = table.release(handle, 'Buffer');
      if (buffer === null) return 0;
      returnStorage(buffer);
      return 1;
    },

    wasmer_wgpu_device_release: (handle) => {
      const device = table.release(handle, 'Device');
      if (device === null) return 0;
      // Dropping the last reference implies destroying the device, and the
      // lost callback then receives a null device.
      device.handle = 0;
      destroyDevice(device);
      return 1;
    },

    /** Writes `{ u64 offset; u64 size; u32 mode; u32 state }` of the current mapping. */
    wasmer_wgpu_buffer_map_info: (handle, out) => {
      const buffer = get(handle, 'Buffer');
      if (!buffer) return 0;
      const { offset, size, mode, state } = buffer.map;
      mem.setU64(out, BigInt(offset));
      mem.setU64(out + 8, BigInt(size));
      mem.setU32(out + 16, Number(mode));
      mem.setU32(out + 20, state);
      return 1;
    },

    wasmer_wgpu_adapter_get_info: (handle, buf, cap) => {
      const adapter = get(handle, 'Adapter');
      return adapter ? writeBlob(buf, cap, adapter.info) : -1;
    },

    wasmer_wgpu_device_get_adapter_info: (handle, buf, cap) => {
      const device = get(handle, 'Device');
      return device ? writeBlob(buf, cap, device.info) : -1;
    },

    wasmer_wgpu_adapter_get_features: (handle, buf, cap) => {
      const adapter = get(handle, 'Adapter');
      return adapter ? writeBlob(buf, cap, u32Blob(featureList(adapter.features))) : -1;
    },

    wasmer_wgpu_device_get_features: (handle, buf, cap) => {
      const device = get(handle, 'Device');
      return device ? writeBlob(buf, cap, u32Blob(featureList(device.features))) : -1;
    },

    wasmer_wgpu_instance_get_wgsl_language_features: (handle, buf, cap) => {
      if (!get(handle, 'Instance')) return -1;
      return writeBlob(buf, cap, u32Blob(WGSL_FEATURE_VALUES.filter(hasWgslFeature)));
    },

    /**
     * Serialized `WGPUSurfaceCapabilities` (src/host/surface.rs):
     *
     *     u64 usages; u32 formatCount, presentModeCount, alphaModeCount, reserved;
     *     u32 formats[]; u32 presentModes[]; u32 alphaModes[];
     */
    wasmer_wgpu_surface_get_capabilities: (surfaceHandle, adapterHandle, buf, cap) => {
      if (!get(surfaceHandle, 'Surface') || !get(adapterHandle, 'Adapter')) return -1;
      const Format = E.TextureFormat;
      const Usage = F.TextureUsage;
      // What every canvas takes: the preferred format first.
      const formats = [...new Set([
        TEXTURE_FORMAT_VALUES[gpu.getPreferredCanvasFormat()], Format.BGRA8Unorm, Format.RGBA8Unorm,
        Format.RGBA16Float,
      ])];
      const presentModes = [E.PresentMode.Fifo];
      const alphaModes = [E.CompositeAlphaMode.Opaque, E.CompositeAlphaMode.Premultiplied];
      const usages = Usage.RenderAttachment | Usage.CopySrc | Usage.CopyDst | Usage.TextureBinding;
      const lists = u32Blob([
        formats.length, presentModes.length, alphaModes.length, 0, ...formats, ...presentModes, ...alphaModes,
      ]);
      const out = new Uint8Array(8 + lists.length);
      new DataView(out.buffer).setBigUint64(0, BigInt(usages), true);
      out.set(lists, 8);
      return writeBlob(buf, cap, out);
    },
  };

  // What several kinds of object share is defined once, for every import of
  // that name the ABI has.
  const declared = new Set(IMPORTS.map(([name]) => name));
  const define = (name, implementation) => {
    if (declared.has(name)) implementations[name] ??= implementation;
  };

  const KINDS = [
    'Adapter', 'BindGroup', 'BindGroupLayout', 'Buffer', 'CommandBuffer', 'CommandEncoder', 'ComputePassEncoder',
    'ComputePipeline', 'Device', 'ExternalTexture', 'Instance', 'PipelineLayout', 'QuerySet', 'Queue',
    'RenderBundle', 'RenderBundleEncoder', 'RenderPassEncoder', 'RenderPipeline', 'Sampler', 'ShaderModule',
    'Surface', 'Texture', 'TextureView',
  ];
  for (const kind of KINDS) {
    define(`wgpu${kind}AddRef`, (handle) => table.addRef(handle, kind));
    define(`wgpu${kind}Release`, (handle) => {
      table.release(handle, kind);
    });
    define(`wgpu${kind}SetLabel`, (handle, labelPtr) => {
      const label = text(labelPtr);
      const state = get(handle, kind);
      // A queue shares its device's state.
      const object = kind === 'Queue' ? state?.object.queue : state?.object;
      if (object) object.label = label;
    });
  }

  for (const kind of ['CommandEncoder', 'ComputePassEncoder', 'RenderPassEncoder', 'RenderBundleEncoder']) {
    define(`wgpu${kind}InsertDebugMarker`, (handle, labelPtr) => {
      const label = text(labelPtr);
      get(handle, kind)?.object.insertDebugMarker(label);
    });
    define(`wgpu${kind}PushDebugGroup`, (handle, labelPtr) => {
      const label = text(labelPtr);
      get(handle, kind)?.object.pushDebugGroup(label);
    });
    define(`wgpu${kind}PopDebugGroup`, (handle) => {
      get(handle, kind)?.object.popDebugGroup();
    });
  }

  for (const kind of ['ComputePassEncoder', 'RenderPassEncoder', 'RenderBundleEncoder']) {
    define(`wgpu${kind}SetBindGroup`, (handle, index, groupHandle, offsetCount, offsetsPtr) => {
      const offsets = u32Array(offsetsPtr, offsetCount);
      const encoder = get(handle, kind);
      // A null group unbinds the slot.
      const group = optional(groupHandle, 'BindGroup');
      if (encoder && group !== undefined) encoder.object.setBindGroup(index, group?.object ?? null, offsets);
    });
    define(`wgpu${kind}SetImmediates`, (handle, offset, data, size) => {
      const encoder = get(handle, kind);
      if (!encoder) return;
      if (typeof encoder.object.setImmediates !== 'function') {
        throw new Invalid(`wgpu${kind}SetImmediates: immediate data is not supported by this browser.`);
      }
      encoder.object.setImmediates(offset, mem.copy(data, size));
    });
  }

  for (const kind of ['RenderPassEncoder', 'RenderBundleEncoder']) {
    define(`wgpu${kind}SetPipeline`, (handle, pipelineHandle) => {
      const encoder = get(handle, kind);
      const pipeline = get(pipelineHandle, 'RenderPipeline');
      if (encoder && pipeline) encoder.object.setPipeline(pipeline.object);
    });
    define(`wgpu${kind}Draw`, (handle, vertexCount, instanceCount, firstVertex, firstInstance) => {
      get(handle, kind)?.object.draw(vertexCount, instanceCount, firstVertex, firstInstance);
    });
    define(`wgpu${kind}DrawIndexed`, (handle, indexCount, instanceCount, firstIndex, baseVertex, firstInstance) => {
      // The one signed argument.
      get(handle, kind)?.object.drawIndexed(indexCount, instanceCount, firstIndex, baseVertex | 0, firstInstance);
    });
    define(`wgpu${kind}DrawIndirect`, (handle, bufferHandle, offset) => {
      const encoder = get(handle, kind);
      const buffer = get(bufferHandle, 'Buffer');
      if (encoder && buffer) encoder.object.drawIndirect(buffer.object, Number(offset));
    });
    define(`wgpu${kind}DrawIndexedIndirect`, (handle, bufferHandle, offset) => {
      const encoder = get(handle, kind);
      const buffer = get(bufferHandle, 'Buffer');
      if (encoder && buffer) encoder.object.drawIndexedIndirect(buffer.object, Number(offset));
    });
    define(`wgpu${kind}SetVertexBuffer`, (handle, slot, bufferHandle, offset, size) => {
      const encoder = get(handle, kind);
      // A null buffer unbinds the slot.
      const buffer = optional(bufferHandle, 'Buffer');
      if (!encoder || buffer === undefined) return;
      if (buffer === null) encoder.object.setVertexBuffer(slot, null);
      else {
        encoder.object.setVertexBuffer(
          slot, buffer.object, Number(offset), size === WGPU_WHOLE_SIZE ? undefined : Number(size),
        );
      }
    });
    define(`wgpu${kind}SetIndexBuffer`, (handle, bufferHandle, formatValue, offset, size) => {
      const format = WEBIDL.IndexFormat[checkEnum('IndexFormat', formatValue)];
      const encoder = get(handle, kind);
      const buffer = get(bufferHandle, 'Buffer');
      if (!encoder || !buffer) return;
      if (format === undefined) throw new Invalid(`wgpu${kind}SetIndexBuffer: invalid range or index format.`);
      encoder.object.setIndexBuffer(
        buffer.object, format, Number(offset), size === WGPU_WHOLE_SIZE ? undefined : Number(size),
      );
    });
  }

  // Every import of the ABI, with exactly its signature: a host that falls
  // behind the generated table must fail here, not when a guest calls.
  const imports = {};
  for (const signature of IMPORTS) {
    const [name, params] = signature;
    const implementation = implementations[name];
    if (implementation === undefined) throw new Error(`webgpu_host.js does not implement ${name}`);
    if (implementation.length !== params.length) {
      throw new Error(`${name} takes ${params.length} arguments in the ABI, ${implementation.length} here`);
    }
    imports[name] = guard(signature, implementation);
  }
  const unknown = Object.keys(implementations).filter((name) => !declared.has(name));
  if (unknown.length !== 0) throw new Error(`webgpu_host.js implements imports the ABI does not have: ${unknown}`);

  return {
    imports,
    /** The guest's `WebAssembly.Memory`. Call it before the guest runs. */
    setMemory(memory) {
      mem.set(memory);
    },
    /**
     * Copies an image the embedder holds into the guest's texture `handle`,
     * on the GPU, with `GPUQueue.copyExternalImageToTexture`: how an embedder
     * hands its guest a frame it decoded, captured or drew itself. `source` is
     * any `GPUCopyExternalImageSource` (an `ImageBitmap`, a canvas, a
     * `VideoFrame`...). Options:
     * - `origin`: where the image goes in the texture, `[x, y]` (default
     *   `[0, 0]`);
     * - `size`: the `[width, height]` copied (default: the whole image);
     * - `flipY`, `premultipliedAlpha` and `colorSpace`, with WebGPU's
     *   defaults (`false`, `false` and `'srgb'`).
     *
     * The texture needs CopyDst and RenderAttachment usage. A failed copy is
     * the embedder's to hear, not the guest's: the copy runs under error
     * scopes of its own. Returns a promise of whether the copy was made; it is
     * false, after logging, for a texture that is gone, a device lost or a
     * host stopped before the copy completed, or a copy the browser refused.
     */
    copyExternalImage(handle, source, options = {}) {
      const {
        origin = [0, 0], size = imageSize(source), flipY = false, premultipliedAlpha = false, colorSpace = 'srgb',
      } = options;
      const fail = (message) => {
        log(`copyExternalImage: ${message}`);
        return false;
      };
      checkTerminated();
      if (terminated) return Promise.resolve(fail('the host was stopped'));
      const texture = table.peek(handle, 'Texture');
      if (texture === null) return Promise.resolve(fail(`${hex(handle)} is not a live texture`));
      if (texture.device.lost) return Promise.resolve(fail('the texture\'s device is lost'));
      const gpu = texture.device.object;
      // Pushed and popped before the guest runs again, so its own scopes
      // never see them.
      const filters = ['validation', 'out-of-memory', 'internal'];
      for (const filter of filters) gpu.pushErrorScope(filter);
      let thrown = null;
      try {
        gpu.queue.copyExternalImageToTexture(
          { source, flipY }, { texture: texture.object, origin, premultipliedAlpha, colorSpace }, size,
        );
      } catch (error) {
        // An image that is closed, detached or not same-origin.
        thrown = error;
      }
      const popped = filters.map(() => gpu.popErrorScope());
      return Promise.all(popped).then(
        (errors) => {
          const error = thrown ?? errors.find((found) => found !== null);
          if (error) return fail(describe(error));
          // A lost device reports no errors, and this host may learn of the
          // loss only now: no error is not proof of a copy.
          checkTerminated();
          if (terminated) return fail('the host was stopped');
          if (texture.device.lost) return fail('the texture\'s device is lost');
          return true;
        },
        (error) => fail(describe(error)),
      );
    },
    /**
     * Ends the host: stops the guest's GPU work (see `isTerminated`),
     * releases the target of every surface still open, and makes no surface
     * afterwards.
     */
    dispose() {
      if (disposed) return;
      disposed = true;
      terminate();
      for (const surface of [...surfaces]) closeSurface(surface);
      channel?.port1.close();
      channel?.port2.close();
    },
  };
}

// Checked when the module loads, with a host that touches nothing.
createWebGpuHost({ gpu: null });
