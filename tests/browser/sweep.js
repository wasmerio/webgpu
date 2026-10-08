// The import surface is the trust boundary: whatever a guest passes, the
// host must answer with a value or a trap, never with a JavaScript exception
// of its own. This is the browser twin of tests/abi_sweep.rs.
//
// Every import is called directly, with no guest and no shim in between:
//
//  1. on a fresh host, with all-zero and with all-ones arguments (null and
//     garbage handles, null and wild pointers, absurd counts and sizes);
//  2. on a host with one live object of every kind this file can make, with
//     every argument set to each of those handles in turn. That reaches the
//     browser's own WebGPU with the right object and nonsense around it.
//
// A trap (a `WebAssembly.RuntimeError` naming the import) is a fine answer to
// a wild pointer. Anything else that is thrown, or a promise left rejected,
// fails the sweep. So does a complaint that reads like a mistake of the
// host's own rather than the guest's: the host reports what the browser
// throws at it as a WebGPU error, and a bug in the host would hide there.

import { E, EVENT, F, IMPORTS, S, STYPE } from '../../js/webgpu_abi.gen.js';
import { createWebGpuHost } from '../../js/webgpu_host.js';

/** Where this file writes the structs it passes, and reads records back. */
const SCRATCH = 0x4000;
const RECORDS = 0x8000;
const RECORDS_CAPACITY = 0x4000;
const CALL_TIMEOUT_MS = 5000;
/** What a JavaScript engine says about broken code, not about bad arguments. */
const HOST_BUG = new RegExp([
  'is not a function', 'is not defined', 'is not iterable', 'is not a constructor', 'Cannot (read|set) propert',
  'Cannot (convert|mix) ',
].join('|'));

const argument = (type, value) => {
  if (type === 'j') return BigInt(value);
  // What wasm passes for an f32 of all ones.
  if (type === 'f' && value === -1) return NaN;
  return value;
};

/** A host with its own small memory, and the means to drive it by hand. */
function createSubject(failures) {
  const memory = new WebAssembly.Memory({ initial: 1, maximum: 1, shared: true });
  const view = new DataView(memory.buffer);
  const log = [];
  const host = createWebGpuHost({ getCanvas: () => new OffscreenCanvas(16, 16), log: (message) => log.push(message) });
  host.setMemory(memory);
  const subject = { host, view, log, calls: 0, traps: 0 };

  /** Calls one import with every argument set to `value`. */
  subject.call = async (signature, value) => {
    const [name, params, , suspending] = signature;
    const args = Array.from(params, (type) => argument(type, value));
    subject.calls += 1;
    try {
      const result = host.imports[name](...args);
      if (!suspending) return;
      const timedOut = Symbol('timed out');
      const outcome = await Promise.race([
        result, new Promise((resolve) => setTimeout(resolve, CALL_TIMEOUT_MS, timedOut)),
      ]);
      if (outcome === timedOut) failures.push(`${name}(${value}) did not return within ${CALL_TIMEOUT_MS} ms`);
    } catch (error) {
      if (error instanceof WebAssembly.RuntimeError && error.message.includes(name)) subject.traps += 1;
      else failures.push(`${name}(${value}) threw ${error?.stack ?? error}`);
    }
  };

  /**
   * Collects the event records the host has ready. Uncaptured errors go to
   * the log; the rest are returned as `{ future, handle }`.
   */
  subject.records = () => {
    const records = [];
    for (;;) {
      const written = host.imports.wasmer_wgpu_poll(RECORDS, RECORDS_CAPACITY);
      if (written <= 0) return records;
      // u64 future; u32 kind, status, handle, aux, messageLength, extraLength; message; extra
      for (let at = RECORDS; at < RECORDS + written;) {
        const messageLength = view.getUint32(at + 24, true);
        if (view.getUint32(at + 8, true) === EVENT.UncapturedError) {
          // A copy: TextDecoder does not read shared memory.
          const message = new Uint8Array(memory.buffer).slice(at + 32, at + 32 + messageLength);
          log.push(`uncaptured error: ${new TextDecoder().decode(message)}`);
        } else {
          records.push({ future: view.getBigUint64(at, true), handle: view.getUint32(at + 16, true) });
        }
        at += 32 + ((messageLength + 7) & ~7) + ((view.getUint32(at + 28, true) + 7) & ~7);
      }
    }
  };

  /** Waits for the record of `future` and returns the handle it carries. */
  subject.result = async (future) => {
    for (let turn = 0; turn < 10_000; turn++) {
      const record = subject.records().find((candidate) => candidate.future === future);
      if (record) return record.handle;
      await host.imports.wasmer_wgpu_yield();
    }
    throw new Error(`future ${future} never completed`);
  };

  /** Ends the subject: lets what the calls started come back, then judges what was said. */
  subject.finish = async () => {
    await host.imports.wasmer_wgpu_yield();
    subject.records();
    host.dispose();
    for (const line of log) {
      if (HOST_BUG.test(line)) failures.push(`the host seems to have a bug: ${line}`);
    }
  };
  return subject;
}

/** Makes one live object of every kind that can be made from here. */
async function populate({ host: { imports }, view, result }) {
  const zero = (ptr, size) => new Uint8Array(view.buffer, ptr, size).fill(0);
  const handles = {};
  handles.instance = imports.wgpuCreateInstance(0);
  imports.wasmer_wgpu_instance_request_adapter(handles.instance, 0, 1n);
  handles.adapter = await result(1n);
  imports.wasmer_wgpu_adapter_request_device(handles.adapter, 0, 2n, 3n);
  handles.device = await result(2n);
  const { device } = handles;
  handles.queue = imports.wgpuDeviceGetQueue(device);

  // An extension struct naming the default canvas, chained to a descriptor.
  const source = SCRATCH + 256;
  zero(SCRATCH, 512);
  view.setUint32(source + S.WGPUChainedStruct.sType, STYPE.WasixSurfaceSourceCanvas, true);
  view.setUint32(SCRATCH + S.WGPUSurfaceDescriptor.nextInChain, source, true);
  handles.surface = await imports.wgpuInstanceCreateSurface(handles.instance, SCRATCH);

  zero(SCRATCH, 512);
  const bufferUsage = BigInt(F.BufferUsage.Uniform | F.BufferUsage.CopyDst);
  view.setBigUint64(SCRATCH + S.WGPUBufferDescriptor.usage, bufferUsage, true);
  view.setBigUint64(SCRATCH + S.WGPUBufferDescriptor.size, 64n, true);
  handles.buffer = imports.wgpuDeviceCreateBuffer(device, SCRATCH);

  zero(SCRATCH, 512);
  const T = S.WGPUTextureDescriptor;
  view.setBigUint64(SCRATCH + T.usage, BigInt(F.TextureUsage.RenderAttachment | F.TextureUsage.TextureBinding), true);
  view.setUint32(SCRATCH + T.size + S.WGPUExtent3D.width, 4, true);
  view.setUint32(SCRATCH + T.size + S.WGPUExtent3D.height, 4, true);
  view.setUint32(SCRATCH + T.size + S.WGPUExtent3D.depthOrArrayLayers, 1, true);
  view.setUint32(SCRATCH + T.format, E.TextureFormat.RGBA8Unorm, true);
  view.setUint32(SCRATCH + T.mipLevelCount, 1, true);
  view.setUint32(SCRATCH + T.sampleCount, 1, true);
  handles.texture = imports.wgpuDeviceCreateTexture(device, SCRATCH);
  handles.textureView = imports.wgpuTextureCreateView(handles.texture, 0);

  zero(SCRATCH, 512);
  view.setUint32(SCRATCH + S.WGPUQuerySetDescriptor.type, E.QueryType.Occlusion, true);
  view.setUint32(SCRATCH + S.WGPUQuerySetDescriptor.count, 1, true);
  handles.querySet = imports.wgpuDeviceCreateQuerySet(device, SCRATCH);

  // Zeroed descriptors: an empty layout, and a bundle encoder with no targets.
  zero(SCRATCH, 512);
  handles.bindGroupLayout = imports.wgpuDeviceCreateBindGroupLayout(device, SCRATCH);
  handles.pipelineLayout = imports.wgpuDeviceCreatePipelineLayout(device, SCRATCH);
  handles.renderBundleEncoder = imports.wgpuDeviceCreateRenderBundleEncoder(device, SCRATCH);
  view.setUint32(SCRATCH + S.WGPUBindGroupDescriptor.layout, handles.bindGroupLayout, true);
  handles.bindGroup = imports.wgpuDeviceCreateBindGroup(device, SCRATCH);

  handles.sampler = imports.wgpuDeviceCreateSampler(device, 0);
  handles.commandEncoder = imports.wgpuDeviceCreateCommandEncoder(device, 0);
  handles.computePass = imports.wgpuCommandEncoderBeginComputePass(handles.commandEncoder, 0);
  zero(SCRATCH, 512);
  return handles;
}

/**
 * Runs the sweep. Returns `{ calls, traps, failures, log }`, where `log` is
 * what the hosts had to say about the calls, for a reader to judge.
 */
export async function runAbiSweep() {
  const failures = [];
  const log = [];
  const rejected = (event) => {
    event.preventDefault();
    failures.push(`a promise was left rejected: ${event.reason?.stack ?? event.reason}`);
  };
  addEventListener('unhandledrejection', rejected);
  let calls = 0;
  let traps = 0;
  try {
    // 1. Null and garbage, on a host that has nothing.
    const empty = createSubject(failures);
    for (let round = 0; round < 2; round++) {
      for (const value of [0, -1]) {
        for (const signature of IMPORTS) await empty.call(signature, value);
      }
    }
    await empty.finish();
    if (empty.traps === 0) failures.push('no import trapped on a wild pointer');
    calls += empty.calls;
    traps += empty.traps;
    log.push(...empty.log);

    // 2. Every live handle, in every position. The objects are made anew for
    // each one: the sweep destroys and releases what it is given.
    for (let swept = 0, kinds = 1; swept < kinds; swept++) {
      const subject = createSubject(failures);
      const handles = await populate(subject);
      const [kind, handle] = Object.entries(handles)[swept];
      kinds = Object.keys(handles).length;
      if (handle === 0) failures.push(`could not create a ${kind} to sweep with`);
      for (const signature of IMPORTS) await subject.call(signature, handle);
      await subject.finish();
      calls += subject.calls;
      traps += subject.traps;
      log.push(...subject.log);
    }
    // Rejections are reported a turn after they happen.
    await new Promise((resolve) => setTimeout(resolve, 50));
  } catch (error) {
    failures.push(`the sweep itself failed: ${error?.stack ?? error}`);
  } finally {
    removeEventListener('unhandledrejection', rejected);
  }
  return { calls, traps, failures, log };
}
