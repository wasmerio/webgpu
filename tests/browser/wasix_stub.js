// The WASI and WASIX imports the conformance programs need, and no more.
//
// This is not a WASIX implementation. The programs are single-threaded C that
// prints to stdout; what libc asks of the system for that is a clock, a few
// calls about its three standard descriptors, the futex behind its locks and
// a way to exit. Anything else a program imports is provided as a function
// that throws when called, so a program that starts to depend on more fails
// loudly instead of quietly misbehaving.

/** Thrown out of the guest by `proc_exit`: how a C program ends. */
export class ExitSignal extends Error {
  constructor(code) {
    super(`the program exited with code ${code}`);
    this.code = code;
  }
}

const ERRNO = { SUCCESS: 0, BADF: 8, SPIPE: 70 };
const FILETYPE_CHARACTER_DEVICE = 2;
const CLOCK_REALTIME = 0;

/**
 * Creates the stub for one program run.
 *
 * - `memory`: the program's (shared) `WebAssembly.Memory`.
 * - `args`: its command line, program name first.
 * - `write(fd, bytes)`: receives what the program writes to stdout (1) and
 *   stderr (2).
 *
 * `importsFor(module)` returns the `wasi_snapshot_preview1` and `wasix_32v1`
 * import objects for exactly what `module` imports.
 */
export function createWasixStub({ memory, args = [], write }) {
  const encoder = new TextEncoder();
  const argv = args.map((arg) => encoder.encode(`${arg}\0`));
  // For sleeping: nothing ever notifies it.
  const sleeper = new Int32Array(new SharedArrayBuffer(4));

  // The memory can grow, which gives it a new buffer.
  const view = () => new DataView(memory.buffer);
  const bytes = () => new Uint8Array(memory.buffer);
  const words = () => new Int32Array(memory.buffer);

  const monotonicNs = () => BigInt(Math.round(performance.now() * 1e6));
  const realtimeNs = () => BigInt(Math.round((performance.timeOrigin + performance.now()) * 1e6));

  const wasi = {
    args_sizes_get(countPtr, sizePtr) {
      view().setUint32(countPtr, argv.length, true);
      view().setUint32(sizePtr, argv.reduce((total, arg) => total + arg.length, 0), true);
      return ERRNO.SUCCESS;
    },
    args_get(argvPtr, bufPtr) {
      let at = bufPtr;
      argv.forEach((arg, i) => {
        view().setUint32(argvPtr + 4 * i, at, true);
        bytes().set(arg, at);
        at += arg.length;
      });
      return ERRNO.SUCCESS;
    },
    environ_sizes_get(countPtr, sizePtr) {
      view().setUint32(countPtr, 0, true);
      view().setUint32(sizePtr, 0, true);
      return ERRNO.SUCCESS;
    },
    environ_get() {
      return ERRNO.SUCCESS;
    },
    clock_time_get(clock, _precision, timePtr) {
      view().setBigUint64(timePtr, clock === CLOCK_REALTIME ? realtimeNs() : monotonicNs(), true);
      return ERRNO.SUCCESS;
    },
    clock_res_get(_clock, resolutionPtr) {
      view().setBigUint64(resolutionPtr, 1000n, true);
      return ERRNO.SUCCESS;
    },
    fd_write(fd, iovsPtr, iovsLength, writtenPtr) {
      if (fd !== 1 && fd !== 2) return ERRNO.BADF;
      let written = 0;
      for (let i = 0; i < iovsLength; i++) {
        const ptr = view().getUint32(iovsPtr + 8 * i, true);
        const length = view().getUint32(iovsPtr + 8 * i + 4, true);
        // A copy: the program's memory is shared and goes on changing.
        write(fd, bytes().slice(ptr, ptr + length));
        written += length;
      }
      view().setUint32(writtenPtr, written, true);
      return ERRNO.SUCCESS;
    },
    fd_read(fd, _iovsPtr, _iovsLength, readPtr) {
      if (fd !== 0) return ERRNO.BADF;
      // Standard input is empty.
      view().setUint32(readPtr, 0, true);
      return ERRNO.SUCCESS;
    },
    fd_close(fd) {
      return fd <= 2 ? ERRNO.SUCCESS : ERRNO.BADF;
    },
    fd_seek(fd) {
      return fd <= 2 ? ERRNO.SPIPE : ERRNO.BADF;
    },
    fd_fdstat_get(fd, statPtr) {
      if (fd > 2) return ERRNO.BADF;
      // { u8 filetype; u16 flags; u64 rights_base; u64 rights_inheriting }
      bytes().fill(0, statPtr, statPtr + 24);
      view().setUint8(statPtr, FILETYPE_CHARACTER_DEVICE);
      return ERRNO.SUCCESS;
    },
    fd_fdstat_set_flags(fd) {
      return fd <= 2 ? ERRNO.SUCCESS : ERRNO.BADF;
    },
    // No preopened directories: the first descriptor libc asks about is bad.
    fd_prestat_get() {
      return ERRNO.BADF;
    },
    fd_prestat_dir_name() {
      return ERRNO.BADF;
    },
    sched_yield() {
      return ERRNO.SUCCESS;
    },
    random_get(bufPtr, length) {
      // getRandomValues does not fill shared memory.
      const random = new Uint8Array(length);
      for (let at = 0; at < length; at += 65536) crypto.getRandomValues(random.subarray(at, at + 65536));
      bytes().set(random, bufPtr);
      return ERRNO.SUCCESS;
    },
    proc_exit(code) {
      throw new ExitSignal(code);
    },
  };

  const wasix = {
    // Names the export to call for a signal. Nothing here sends signals.
    callback_signal() {},
    thread_signal() {
      return ERRNO.SUCCESS;
    },
    proc_signals_sizes_get(countPtr) {
      view().setUint32(countPtr, 0, true);
      return ERRNO.SUCCESS;
    },
    proc_signals_get() {
      return ERRNO.SUCCESS;
    },
    proc_exit2(code) {
      throw new ExitSignal(code);
    },
    thread_id(idPtr) {
      view().setUint32(idPtr, 1, true);
      return ERRNO.SUCCESS;
    },
    thread_sleep(durationNs) {
      Atomics.wait(sleeper, 0, 0, Number(durationNs) / 1e6);
      return ERRNO.SUCCESS;
    },
    // The memory is shared, so the futex calls are the real thing.
    futex_wait(futexPtr, expected, timeoutPtr, wokenPtr) {
      // { u8 tag; u64 nanoseconds }: an optional timeout.
      const timeout = view().getUint8(timeoutPtr) === 0 ? undefined : view().getBigUint64(timeoutPtr + 8, true);
      let woken = false;
      if (Atomics.load(words(), futexPtr >>> 2) === (expected | 0)) {
        if (timeout === undefined) {
          throw new Error('futex_wait would block forever: this harness runs single-threaded programs');
        }
        woken = Atomics.wait(words(), futexPtr >>> 2, expected | 0, Number(timeout) / 1e6) === 'ok';
      }
      view().setUint8(wokenPtr, Number(woken));
      return ERRNO.SUCCESS;
    },
    futex_wake(futexPtr, wokenPtr) {
      view().setUint8(wokenPtr, Number(Atomics.notify(words(), futexPtr >>> 2, 1) > 0));
      return ERRNO.SUCCESS;
    },
    futex_wake_all(futexPtr, wokenPtr) {
      view().setUint8(wokenPtr, Number(Atomics.notify(words(), futexPtr >>> 2) > 0));
      return ERRNO.SUCCESS;
    },
  };

  const namespaces = { wasi_snapshot_preview1: wasi, wasix_32v1: wasix };

  // wasm passes an i32 as a signed number; every one of them here is a
  // pointer, a length or a small id.
  const unsigned = (implementation) => (...values) =>
    implementation(...values.map((value) => (typeof value === 'number' ? value >>> 0 : value)));

  return {
    importsFor(module) {
      const imports = { wasi_snapshot_preview1: {}, wasix_32v1: {} };
      for (const { module: namespace, name, kind } of WebAssembly.Module.imports(module)) {
        if (!(namespace in namespaces) || kind !== 'function') continue;
        const implementation = namespaces[namespace][name];
        imports[namespace][name] = implementation
          ? unsigned(implementation)
          : () => {
            throw new Error(`${namespace}.${name} is not implemented by the browser test harness`);
          };
      }
      return imports;
    },
  };
}
