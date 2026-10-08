// Runs one conformance program: the dedicated worker a guest lives in.
//
// The page sends `{ wasmUrl, name, args, canvas, host, stop, then }`. The
// program is instantiated with a shared memory, the WASIX stub and the WebGPU
// host under test, and started through JavaScript Promise Integration, so
// that the host's suspending imports can hold it while the browser works.
//
// `host` configures the WebGPU host the way an embedder would:
//   { limits, forceFallbackAdapter, suspending, capture: [width, height] }
// and `stop` is a shared flag the page raises to stop the guest. `then`
// (`{ wasmUrl, name, args }`) is a program to start on this thread once the
// first has exited, with a host of its own, as when the Wasmer SDK reuses a
// worker; what is reported at the end is then how that one went.
//
// Messages back to the page:
//   { type: 'output', fd, text }   what the program wrote, as it writes it
//   { type: 'log', message }       the WebGPU host's diagnostics
//   { type: 'done', exitCode, error, surfaces, adapterRequests }

import { IMPORTS } from '../../js/webgpu_abi.gen.js';
import { IMPORT_MODULE, createWebGpuHost } from '../../js/webgpu_host.js';
import { ExitSignal, createWasixStub } from './wasix_stub.js';

/** Decodes an unsigned LEB128 number at `at`. Returns `[value, next]`. */
function leb(bytes, at) {
  let value = 0;
  let shift = 0;
  for (;;) {
    const byte = bytes[at++];
    value += (byte & 0x7f) * 2 ** shift;
    shift += 7;
    if ((byte & 0x80) === 0) return [value, at];
  }
}

/**
 * The limits of the memory a module imports. `WebAssembly.Module.imports`
 * does not tell them, and the memory handed to the module has to match what
 * it declares, so they are read out of the import section.
 */
function importedMemoryLimits(bytes) {
  const limits = (at) => {
    const flags = bytes[at++];
    let initial;
    let maximum;
    [initial, at] = leb(bytes, at);
    if (flags & 1) [maximum, at] = leb(bytes, at);
    return [{ initial, maximum, shared: (flags & 2) !== 0 }, at];
  };
  let at = 8; // Past the magic number and the version.
  while (at < bytes.length) {
    const id = bytes[at++];
    let size;
    [size, at] = leb(bytes, at);
    if (id !== 2) {
      at += size;
      continue;
    }
    let count;
    [count, at] = leb(bytes, at);
    for (let i = 0; i < count; i++) {
      // Module name, field name, then what kind of thing is imported.
      for (let name = 0; name < 2; name++) {
        let length;
        [length, at] = leb(bytes, at);
        at += length;
      }
      const kind = bytes[at++];
      if (kind === 0) [, at] = leb(bytes, at); // function: type index
      else if (kind === 1) [, at] = limits(at + 1); // table: element type, limits
      else if (kind === 2) return limits(at)[0]; // memory
      else if (kind === 3) at += 2; // global: value type, mutability
      else if (kind === 4) [, at] = leb(bytes, at + 1); // tag: attribute, type index
      else throw new Error(`import of unknown kind ${kind}`);
    }
  }
  throw new Error('the module does not import a memory');
}

/**
 * The target every surface of the program presents to, as an embedder would
 * bind it, and a record of what the host did with it.
 *
 * By default that is the page's canvas, so what the program draws can be
 * seen. With `capture`, it is a canvas of that size whose presented frames
 * are taken the way the Wasmer SDK takes them, to check what was drawn.
 */
function surfaceTarget(pageCanvas, capture) {
  const canvas = capture ? new OffscreenCanvas(...capture) : pageCanvas;
  const size = [canvas.width, canvas.height];
  const use = { opened: 0, presented: 0, released: 0, lastFrame: null };
  return {
    use,
    getCanvas: () => {
      use.opened += 1;
      return {
        canvas,
        getSize: () => size,
        present: () => {
          use.presented += 1;
          if (capture) use.lastFrame = describeFrame(canvas.transferToImageBitmap());
        },
        release: () => {
          use.released += 1;
        },
      };
    },
  };
}

/** The size of a presented frame, and its color if it is a single one. */
function describeFrame(bitmap) {
  const { width, height } = bitmap;
  const context = new OffscreenCanvas(width, height).getContext('2d');
  context.drawImage(bitmap, 0, 0);
  const pixels = new Uint32Array(context.getImageData(0, 0, width, height).data.buffer);
  const solid = pixels.every((pixel) => pixel === pixels[0]);
  return { width, height, color: solid ? [...new Uint8Array(pixels.buffer, 0, 4)] : null };
}

/** `navigator.gpu`, noting the options of each adapter request. */
function recordingGpu(requests) {
  const gpu = navigator.gpu;
  return new Proxy(gpu, {
    get(_, key) {
      if (key === 'requestAdapter') {
        return (options) => {
          requests.push({ ...options });
          return gpu.requestAdapter(options);
        };
      }
      const value = gpu[key];
      return typeof value === 'function' ? value.bind(gpu) : value;
    },
  });
}

/** Fetches and compiles a program, and reads which memory it asks for. */
async function load(wasmUrl) {
  const response = await fetch(wasmUrl);
  if (!response.ok) throw new Error(`${wasmUrl}: ${response.status} ${response.statusText}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  return { module: await WebAssembly.compile(bytes), limits: importedMemoryLimits(bytes) };
}

/**
 * Collects what a finished guest left behind, where the browser lets a page
 * ask for that: Chrome with `--js-flags=--expose-gc`, which run.mjs passes.
 */
async function collectGarbage() {
  for (let turn = 0; turn < 2; turn++) {
    // In a task of its own, for the collector's tasks to get their turn too.
    await new Promise((resolve) => setTimeout(resolve, turn * 2));
    globalThis.gc?.();
    globalThis.gc?.();
  }
}

async function run({ module, limits }, { name, args, canvas, host: configuration = {}, stop }) {
  const memory = new WebAssembly.Memory(limits);

  const decoders = { 1: new TextDecoder(), 2: new TextDecoder() };
  const wasix = createWasixStub({
    memory,
    args: [name, ...args],
    write: (fd, chunk) => postMessage({ type: 'output', fd, text: decoders[fd].decode(chunk, { stream: true }) }),
  });

  const { capture, suspending = true, ...options } = configuration;
  const target = surfaceTarget(canvas, capture);
  const adapterRequests = [];
  const host = createWebGpuHost({
    ...options,
    suspending,
    gpu: recordingGpu(adapterRequests),
    getCanvas: target.getCanvas,
    isTerminated: () => Atomics.load(stop, 0) !== 0,
    log: (message) => postMessage({ type: 'log', message }),
  });
  host.setMemory(memory);
  const webgpu = { ...host.imports };
  if (suspending) {
    for (const [importName, , , blocks] of IMPORTS) {
      if (blocks) webgpu[importName] = new WebAssembly.Suspending(host.imports[importName]);
    }
  }

  const instance = await WebAssembly.instantiate(module, {
    env: { memory },
    [IMPORT_MODULE]: webgpu,
    ...wasix.importsFor(module),
  });

  let exitCode = 0;
  try {
    // Without suspension the program runs to its end in one go.
    if (suspending) await WebAssembly.promising(instance.exports._start)();
    else instance.exports._start();
  } catch (error) {
    if (!(error instanceof ExitSignal)) throw error;
    exitCode = error.code;
  } finally {
    host.dispose();
  }
  return { exitCode, surfaces: target.use, adapterRequests };
}

onmessage = async ({ data }) => {
  try {
    const program = await load(data.wasmUrl);
    // Loaded up front, so that it starts right behind the first one.
    const next = data.then ? await load(data.then.wasmUrl) : null;
    let outcome = await run(program, data);
    if (next !== null && outcome.exitCode === 0) {
      await collectGarbage();
      outcome = await run(next, { ...data, ...data.then, host: {} });
    }
    postMessage({ type: 'done', ...outcome });
  } catch (error) {
    postMessage({ type: 'done', error: error?.stack ?? String(error) });
  }
};
