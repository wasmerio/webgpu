#!/usr/bin/env node
// Tests the JavaScript WebGPU host (js/webgpu_host.js) in the system's Chrome:
//
//  - every browser program of tests/manifest.json must print what the
//    manifest says (the browser twin of tests/manifest_tests.rs);
//  - the same programs under the options an embedder sets: limits, a kill
//    switch, its own surface target (tests/embedder_tests.rs);
//  - every import called with null and garbage arguments (tests/abi_sweep.rs);
//  - the host's imports called by hand, to see what the browser is asked for.
//
//     node tests/browser/run.mjs [name...] [--headed] [--no-build] [--verbose] [--repeat <n>]
//                                [--timeout <seconds>] [--serve]
//
// Names select tests of any of the four groups. `WEBGPU_TEST_FILTER=<substring>`
// does too, as for the native tests. `WEBGPU_CHROME_ARGS` adds command-line
// switches to Chrome. See README.md for what this needs.

import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('../..', import.meta.url)));
/** Where the programs are built: the build script's own override applies. */
const PROGRAMS = resolve(process.env.WEBGPU_TEST_OUT_DIR ?? join(ROOT, 'tests/programs/out'));
const PROGRAMS_URL = '/tests/programs/out/';

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
};

function parseArguments(argv) {
  const options = {
    names: [], headed: false, build: true, serve: false, verbose: false, repeat: 1, timeoutMs: 60_000,
  };
  for (let i = 0; i < argv.length; i++) {
    const argument = argv[i];
    if (argument === '--headed') options.headed = true;
    else if (argument === '--no-build') options.build = false;
    else if (argument === '--serve') options.serve = true;
    else if (argument === '--verbose') options.verbose = true;
    else if (argument === '--timeout') options.timeoutMs = Number(argv[++i]) * 1000;
    else if (argument === '--repeat') options.repeat = Number(argv[++i]);
    else if (argument.startsWith('--')) throw new Error(`unknown option ${argument}`);
    else options.names.push(argument);
  }
  if (!(options.timeoutMs > 0)) throw new Error('--timeout takes a number of seconds');
  if (!(options.repeat >= 1)) throw new Error('--repeat takes a count');
  return options;
}

/**
 * Serves the repository on a free localhost port. Shared memory, which the
 * guests are built for, is only available to cross-origin isolated pages:
 * that is what the two headers ask for.
 */
function serveRepository() {
  const server = createServer(async (request, response) => {
    const path = normalize(decodeURIComponent(new URL(request.url, 'http://localhost').pathname));
    const [directory, relative] = path.startsWith(PROGRAMS_URL)
      ? [PROGRAMS, path.slice(PROGRAMS_URL.length)]
      : [ROOT, path];
    const file = join(directory, relative);
    try {
      if (!file.startsWith(directory + sep)) throw new Error('outside the served directory');
      const body = await readFile(file);
      response.writeHead(200, {
        'Content-Type': CONTENT_TYPES[extname(file)] ?? 'application/octet-stream',
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cache-Control': 'no-store',
      });
      response.end(body);
    } catch {
      response.writeHead(404, { 'Content-Type': 'text/plain' });
      response.end('not found');
    }
  });
  return new Promise((ready, failed) => {
    server.once('error', failed);
    server.listen(0, '127.0.0.1', () => ready(server));
  });
}

function buildPrograms() {
  const build = spawnSync(join(ROOT, 'tests/build-test-wasix.sh'), { stdio: 'inherit' });
  if (build.status !== 0) throw new Error('tests/build-test-wasix.sh failed (is wasixcc installed?)');
}

async function launchChrome(headed) {
  let chromium;
  try {
    ({ chromium } = await import('playwright-core'));
  } catch {
    throw new Error('playwright-core is missing: run `npm install` in tests/browser');
  }
  return chromium.launch({
    // The Chrome installed on this machine; Playwright downloads nothing.
    channel: 'chrome',
    headless: !headed,
    args: [
      // A window that is covered or in the background gets no animation
      // frames, which is what a presenting guest waits for.
      '--disable-backgrounding-occluded-windows',
      '--disable-renderer-backgrounding',
      '--disable-background-timer-throttling',
      // Lets a test collect garbage when it matters (embedder_thread_reuse).
      '--js-flags=--expose-gc',
      ...(process.env.WEBGPU_CHROME_ARGS ?? '').split(' ').filter(Boolean),
    ],
  });
}

/** Why a run is not what the manifest says, or `null` when it is. */
function manifestFailure(test, outcome) {
  if (outcome.error !== null) return outcome.error;
  if (outcome.exitCode !== 0) return `exit code ${outcome.exitCode}`;
  if (outcome.stdout !== test.stdout) return 'stdout differs from the manifest';
  // Whatever target a program opened, the host gave back.
  const { opened, released } = outcome.surfaces;
  if (released !== opened) return `${opened} surface target(s) opened, ${released} released`;
  return null;
}

/** The first of `checks` (pairs of a condition and what it means) that fails. */
const firstFailure = (outcome, checks) =>
  outcome.error ?? checks.find(([holds]) => !holds)?.[1] ?? null;

/**
 * The conformance programs under the options an embedder sets, as
 * tests/embedder_tests.rs runs them on the native host. `expect` returns why
 * the outcome is wrong, or `null`.
 */
const EMBEDDER_TESTS = [
  {
    name: 'embedder_surface_target',
    about: 'presented frames reach the embedder\'s target, at the size it asked for',
    program: 'surface',
    options: { host: { capture: [32, 24] } },
    expect: (outcome) => {
      const { opened, presented, released, lastFrame } = outcome.surfaces ?? {};
      return firstFailure(outcome, [
        [outcome.exitCode === 0, `exit code ${outcome.exitCode}`],
        // The program makes a surface, releases it, and makes another.
        [opened === 2 && released === 2, `target opened ${opened} time(s), released ${released}`],
        [presented === 3, `${presented} frames presented, not 3`],
        // The program presents red, green, then blue.
        [
          lastFrame?.width === 32 && lastFrame?.height === 24 && String(lastFrame?.color) === '0,0,255,255',
          `the last frame is ${JSON.stringify(lastFrame)}, not 32x24 of solid blue`,
        ],
      ]);
    },
  },
  {
    name: 'embedder_object_limit',
    about: 'the object limit fails creation instead of the host',
    program: 'compute_double',
    // Instance, adapter and device fit; the queue and everything after do not.
    options: { host: { limits: { maxObjects: 3 } } },
    expect: (outcome) => firstFailure(outcome, [
      [outcome.exitCode === 1, `exit code ${outcome.exitCode}, not 1`],
      [outcome.stdout.includes('FAILED'), 'the program did not notice'],
    ]),
  },
  {
    name: 'embedder_device_limit',
    about: 'the device limit fails the second request',
    program: 'instance_info',
    options: { host: { limits: { maxDevices: 1 } } },
    expect: (outcome) => firstFailure(outcome, [
      [outcome.exitCode === 1, `exit code ${outcome.exitCode}, not 1`],
      [outcome.stdout.includes('device: ok'), 'the first device did not work'],
      [outcome.stdout.includes('This guest may have at most 1 devices.'), 'the second request did not fail'],
    ]),
  },
  {
    name: 'embedder_gpu_memory_budget',
    about: 'exceeding the GPU memory budget is an out-of-memory error',
    program: 'render_triangle',
    // A 64x64 RGBA8 render target is 16 KiB.
    options: { host: { limits: { maxGpuMemoryBytes: 1024 } } },
    expect: (outcome) => firstFailure(outcome, [
      [outcome.exitCode === 1, `exit code ${outcome.exitCode}, not 1`],
      [outcome.stdout.includes('texture != NULL'), 'the texture was not refused'],
    ]),
  },
  {
    name: 'embedder_gpu_memory_budget_roomy',
    about: 'a budget with room changes nothing',
    program: 'render_triangle',
    options: { host: { limits: { maxGpuMemoryBytes: 64 << 20 } } },
    expect: (outcome, manifest) => manifestFailure(manifest, outcome),
  },
  {
    name: 'embedder_surface_colors',
    about: 'a canvas takes the colour space and tone mapping a guest configures',
    program: 'surface_colors',
    expect: (outcome) => firstFailure(outcome, [
      [outcome.exitCode === 0, `exit code ${outcome.exitCode}`],
      [outcome.stdout === 'presented in display-p3\nok\n', `unexpected transcript:\n${outcome.stdout}`],
    ]),
  },
  {
    name: 'embedder_gpu_memory_returned',
    about: 'destroyed and released storage returns to the budget',
    program: 'budget',
    options: { host: { limits: { maxGpuMemoryBytes: 1 << 20 } } },
    expect: (outcome) => firstFailure(outcome, [
      [outcome.exitCode === 0, `exit code ${outcome.exitCode}`],
      [
        outcome.stdout ===
          '768 KiB: fits\n768 KiB more: out of memory\nafter destroying the first: fits\n' +
            'after releasing it: fits\ntextures: counted, and given back\nok\n',
        `unexpected transcript:\n${outcome.stdout}`,
      ],
    ]),
  },
  {
    name: 'embedder_thread_reuse',
    about: 'a guest that exits with work in flight leaves its thread to the next one',
    program: 'submit_and_exit',
    // What the Wasmer SDK does with a worker. Chrome 154 deadlocks the thread
    // when the first guest's queue is collected before its submit is answered.
    options: { then: { name: 'compute_double' } },
    expect: (outcome, _, manifest) => firstFailure(outcome, [
      [outcome.exitCode === 0, `exit code ${outcome.exitCode}`],
      [
        outcome.stdout === `submitted\n${manifest.tests.find(({ name }) => name === 'compute_double').stdout}`,
        `unexpected transcript:\n${outcome.stdout}`,
      ],
    ]),
  },
  {
    name: 'embedder_terminate',
    about: 'the kill switch gets a guest out of a wait',
    program: 'wait_forever',
    options: { stop: { afterOutput: 'waiting\n', delayMs: 500 } },
    expect: (outcome) => firstFailure(outcome, [
      [outcome.stoppedAfterMs !== null, 'the program ended before it was stopped'],
      [outcome.stoppedAfterMs < 5000, `the wait took ${Math.round(outcome.stoppedAfterMs)} ms to end`],
      [outcome.exitCode === 0, `exit code ${outcome.exitCode}`],
      // Destroying its device is how the guest learns: reason 2 is Destroyed.
      [outcome.stdout.includes('device lost: reason 2'), 'the lost callback did not run'],
    ]),
  },
  {
    name: 'embedder_terminate_sticky',
    about: 'a stopped guest gets no adapter',
    program: 'compute_double',
    options: { stop: true },
    expect: (outcome) => firstFailure(outcome, [
      [outcome.exitCode === 1, `exit code ${outcome.exitCode}, not 1`],
      [outcome.stdout.includes('The WebGPU context was shut down.'), 'the adapter request did not fail'],
    ]),
  },
  {
    name: 'embedder_fallback_adapter',
    about: 'forceFallbackAdapter overrides what the guest asks for',
    program: 'compute_double',
    options: { host: { forceFallbackAdapter: true } },
    // Whether this Chrome has a software adapter is not the point.
    expect: (outcome) => {
      const requests = outcome.adapterRequests;
      return firstFailure(outcome, [
        [requests.length !== 0, 'no adapter was requested'],
        [requests.every((request) => request.forceFallbackAdapter === true), 'a request did not force it'],
        [outcome.exitCode === 0 || outcome.stdout.includes('request adapter failed'), `exit code ${outcome.exitCode}`],
      ]);
    },
  },
  {
    name: 'embedder_no_suspension',
    about: 'without JSPI the host does not offer timed waits',
    program: 'compute_double',
    options: { host: { suspending: false } },
    // The programs require WGPUInstanceFeatureName_TimedWaitAny.
    expect: (outcome) => firstFailure(outcome, [
      [outcome.exitCode === 1, `exit code ${outcome.exitCode}, not 1`],
      [outcome.stdout.includes('instance != NULL'), 'the instance was not refused'],
    ]),
  },
];

const SWEEP_TEST = 'abi_sweep';
const HOST_TESTS = 'host_tests';

function report(name, expected, outcome, failure) {
  const block = (title, text) => `--- ${title} ---\n${text}${text.endsWith('\n') ? '' : '\n'}`;
  let details = `${name}: ${failure}\n`;
  if (expected !== null) details += block('expected stdout', expected);
  details += block('actual stdout', outcome.stdout);
  if (outcome.stderr) details += block('stderr', outcome.stderr);
  if (outcome.log.length !== 0) details += block('host log', outcome.log.join('\n'));
  return details;
}

async function main() {
  const options = parseArguments(process.argv.slice(2));
  const server = await serveRepository();
  const origin = `http://127.0.0.1:${server.address().port}`;
  if (options.serve) {
    console.log(`serving ${ROOT}\nopen ${origin}/tests/browser/index.html in a browser with WebGPU`);
    return;
  }

  let browser;
  try {
    if (options.build) buildPrograms();
    const manifest = JSON.parse(await readFile(join(ROOT, 'tests/manifest.json'), 'utf8'));
    const filter = process.env.WEBGPU_TEST_FILTER;
    const selected = ({ name }) =>
      (options.names.length === 0 || options.names.includes(name)) && (!filter || name.includes(filter));
    // Results that depend on timing only show when a test is run many times.
    const repeated = (tests) => tests.flatMap((test) => Array(options.repeat).fill(test));
    const programs = repeated(manifest.tests.filter((test) => test.browser !== false && selected(test)));
    const embedderTests = repeated(EMBEDDER_TESTS.filter(selected));
    const sweep = selected({ name: SWEEP_TEST });
    const hostTests = selected({ name: HOST_TESTS });
    if (programs.length + embedderTests.length + Number(sweep) + Number(hostTests) === 0) {
      throw new Error('no test matched');
    }

    browser = await launchChrome(options.headed);
    const page = await browser.newPage();
    page.on('pageerror', (error) => console.error(`page error: ${error.message}`));
    await page.goto(`${origin}/tests/browser/index.html`);
    const environment = await page.evaluate(async () => ({
      isolated: crossOriginIsolated,
      jspi: typeof WebAssembly.Suspending === 'function',
      adapter: (await navigator.gpu?.requestAdapter())?.info.vendor ?? null,
      userAgent: navigator.userAgent,
    }));
    if (!environment.isolated) throw new Error('the page is not cross-origin isolated');
    if (!environment.jspi) throw new Error('this Chrome has no JavaScript Promise Integration');
    if (environment.adapter === null) {
      throw new Error(`this Chrome has no WebGPU adapter${options.headed ? '' : ' in headless mode; try --headed'}`);
    }
    console.log(`${environment.userAgent}\nWebGPU adapter: ${environment.adapter || 'unnamed'}\n`);

    const failures = [];
    let total = 0;
    const runProgram = (name, runOptions) =>
      page.evaluate(([program, programOptions]) => window.runProgram(program, programOptions), [
        name,
        { timeoutMs: options.timeoutMs, ...runOptions },
      ]);
    const result = (name, about, milliseconds, failure, { stdout = '', log = [] } = {}) => {
      total += 1;
      const status = failure === null ? 'ok      ' : 'FAILED  ';
      console.log(`${status} ${name} (${Math.round(milliseconds)} ms)${about ? `: ${about}` : ''}`);
      if (!options.verbose) return;
      for (const line of stdout.split('\n').filter(Boolean)) console.log(`         | ${line}`);
      // What the host had to say, each distinct line once.
      const counts = new Map();
      for (const line of log) counts.set(line, (counts.get(line) ?? 0) + 1);
      for (const [line, count] of counts) console.log(`           host: ${line}${count > 1 ? ` (x${count})` : ''}`);
    };

    for (const test of programs) {
      const outcome = await runProgram(test.name, { args: test.args ?? [] });
      const failure = manifestFailure(test, outcome);
      result(test.name, '', outcome.elapsedMs, failure, outcome);
      if (failure !== null) failures.push(report(test.name, test.stdout, outcome, failure));
    }

    for (const test of embedderTests) {
      const program = manifest.tests.find(({ name }) => name === test.program);
      const outcome = await runProgram(test.program, { args: program?.args ?? [], ...test.options });
      const failure = test.expect(outcome, program, manifest);
      result(test.name, test.about, outcome.elapsedMs, failure, outcome);
      if (failure !== null) failures.push(report(`${test.name} (${test.program})`, null, outcome, failure));
    }

    if (sweep) {
      const started = Date.now();
      const outcome = await page.evaluate(() => window.runAbiSweep());
      const failed = outcome.failures.length !== 0;
      const about = `${outcome.calls} calls, ${outcome.traps} traps`;
      result(SWEEP_TEST, about, Date.now() - started, failed ? about : null, outcome);
      if (failed) failures.push(`${SWEEP_TEST}:\n${outcome.failures.join('\n')}\n`);
    }

    if (hostTests) {
      for (const { name, failure, milliseconds } of await page.evaluate(() => window.runHostTests())) {
        result(HOST_TESTS, name, milliseconds, failure);
        if (failure !== null) failures.push(`${HOST_TESTS}: ${name}\n${failure}\n`);
      }
    }

    console.log(`\n${total - failures.length} of ${total} tests passed`);
    if (failures.length !== 0) {
      console.log(`\n${failures.join('\n')}`);
      process.exitCode = 1;
    }
  } finally {
    await browser?.close();
    server.close();
  }
}

main().catch((error) => {
  console.error(`run.mjs: ${error.message}`);
  process.exitCode = 1;
});
