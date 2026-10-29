// The worker of runtime_hooks.js: loads the fixture and runs its tests.

const failed = (name, error) => [{ name, failure: String(error?.stack ?? error), milliseconds: 0 }];

let fixture = null;
try {
  const loaded = await import('./out/runtime_hooks/runtime_hooks.js');
  await loaded.default();
  fixture = loaded;
} catch (error) {
  postMessage(failed('loading the fixture (tests/browser/build-runtime-hooks.sh builds it)', error));
}
if (fixture !== null) {
  // A panic traps, which ends every test at once.
  postMessage(await fixture.run().catch((error) => failed('running the fixture', error)));
}
