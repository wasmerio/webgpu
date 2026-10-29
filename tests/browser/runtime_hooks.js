// Runs the tests of runtime_hooks.rs, the crate's browser backend built with
// wasm-bindgen (build-runtime-hooks.sh), in a dedicated worker: where the
// Wasmer SDK runs guests.

/** How each test went: `[{ name, failure, milliseconds }]`. */
export function runRuntimeHookTests() {
  return new Promise((resolve) => {
    const worker = new Worker(new URL('./runtime_hooks_worker.js', import.meta.url), { type: 'module' });
    const finish = (results) => {
      worker.terminate();
      resolve(results);
    };
    worker.onmessage = ({ data }) => finish(data);
    worker.onerror = (event) =>
      finish([{ name: 'the worker', failure: event.message || 'the worker failed to start', milliseconds: 0 }]);
  });
}
