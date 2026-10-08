// The page side of the browser tests: starts a conformance program in a
// dedicated worker (where the Wasmer SDK runs guests too) and collects what
// it printed.
//
//     const result = await runProgram('compute_double');
//     // { name, exitCode, stdout, stderr, log, error, surfaces, ... }
//
// `run.mjs` drives this from Node; opening index.html does the same by hand.

const CANVAS_WIDTH = 640;
const CANVAS_HEIGHT = 480;

/**
 * Runs `tests/programs/out/<name>.wasm`. Resolves with what happened; it
 * never rejects, so that a crash or a hang is a result like any other.
 *
 * - `args`: the program's arguments.
 * - `host`: options for the WebGPU host, as an embedder would set them (see
 *   worker.js).
 * - `stop`: `{ afterOutput, delayMs }` raises the embedder's kill switch
 *   that long after the program printed that text; `true` raises it before
 *   the program starts.
 * - `then`: `{ name, args }` of a program to start on the same thread once
 *   this one has exited. The exit code reported is that program's.
 */
export function runProgram(name, { args = [], host = {}, stop = null, then = null, timeoutMs = 60_000 } = {}) {
  return new Promise((resolve) => {
    // What the program's surface presents to. It is the size of the native
    // runner's default target, so both hosts are asked for the same frames.
    const canvas = document.createElement('canvas');
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;
    canvas.title = name;
    document.body.append(canvas);
    const offscreen = canvas.transferControlToOffscreen();

    // The embedder's kill switch: memory the guest's worker can read while
    // the guest keeps it busy.
    const stopFlag = new Int32Array(new SharedArrayBuffer(4));
    if (stop === true) Atomics.store(stopFlag, 0, 1);

    const started = performance.now();
    const result = {
      name, exitCode: null, stdout: '', stderr: '', log: [], error: null, surfaces: null, adapterRequests: [],
      stoppedAfterMs: null,
    };
    let stopping = null;
    let stoppedAt = null;
    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    const finish = (outcome) => {
      clearTimeout(timer);
      clearTimeout(stopping);
      worker.terminate();
      if (stoppedAt !== null) result.stoppedAfterMs = performance.now() - stoppedAt;
      result.elapsedMs = performance.now() - started;
      resolve(Object.assign(result, outcome));
    };
    const timer = setTimeout(() => finish({ error: `timed out after ${timeoutMs} ms` }), timeoutMs);

    worker.onmessage = ({ data }) => {
      if (data.type === 'output') {
        result[data.fd === 1 ? 'stdout' : 'stderr'] += data.text;
        if (stop?.afterOutput && stopping === null && result.stdout.includes(stop.afterOutput)) {
          stopping = setTimeout(() => {
            stoppedAt = performance.now();
            Atomics.store(stopFlag, 0, 1);
          }, stop.delayMs);
        }
      } else if (data.type === 'log') {
        result.log.push(data.message);
      } else {
        const { type: _, ...outcome } = data;
        finish(outcome);
      }
    };
    worker.onerror = (event) => finish({ error: `the worker failed: ${event.message}` });
    const wasmUrl = (program) => new URL(`../programs/out/${program}.wasm`, import.meta.url).href;
    worker.postMessage(
      {
        wasmUrl: wasmUrl(name),
        name, args, host, canvas: offscreen, stop: stopFlag,
        then: then && { name: then.name, args: then.args ?? [], wasmUrl: wasmUrl(then.name) },
      },
      [offscreen],
    );
  });
}
