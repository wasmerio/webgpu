// Tests of the JavaScript host that need no guest program: the import
// functions are called the way wasm would call them, on a memory this file
// fills in by hand.
//
// They cover what the conformance programs cannot observe from inside a
// guest: exactly which descriptor the host builds from a C struct (through a
// spy on the browser's WebGPU objects), how event records are handed over,
// and how the host follows a memory that grows. They run on the page's main
// thread, where the host has to work as well as in a worker.

import { E, EVENT, F, S, STYPE, WEBIDL } from '../../js/webgpu_abi.gen.js';
import { createWebGpuHost } from '../../js/webgpu_host.js';

const PAGE = 65536;
const U32_MAX = 0xffff_ffff;
const U64_MAX = 0xffff_ffff_ffff_ffffn;
const RECORDS_CAPACITY = 4096;

/**
 * `navigator.gpu` with every call recorded. Objects the browser returns are
 * wrapped as well, and unwrapped again where the host passes them back, so
 * the browser only ever sees its own objects.
 */
function spyOn(gpu) {
  const calls = [];
  const targets = new WeakMap();
  const wrap = (value) => {
    if (value instanceof Promise) return value.then(wrap);
    if (value === null || typeof value !== 'object' || !/^GPU/.test(value.constructor?.name)) return value;
    const proxy = new Proxy(value, {
      get(target, key) {
        const member = target[key];
        if (typeof member !== 'function') return wrap(member);
        return (...args) => {
          const call = { method: key, args };
          calls.push(call);
          call.result = wrap(member.apply(target, unwrap(args)));
          return call.result;
        };
      },
      set(target, key, member) {
        target[key] = member;
        return true;
      },
    });
    targets.set(proxy, value);
    return proxy;
  };
  const unwrap = (value) => {
    if (targets.has(value)) return targets.get(value);
    if (Array.isArray(value)) return value.map(unwrap);
    if (value !== null && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) {
      return Object.fromEntries(Object.entries(value).map(([key, member]) => [key, unwrap(member)]));
    }
    return value;
  };
  return {
    gpu: wrap(gpu),
    /** The arguments of the last call of `method`. */
    last: (method) => calls.findLast((call) => call.method === method)?.args,
    /** What the last call of `method` returned. */
    lastResult: (method) => calls.findLast((call) => call.method === method)?.result,
    count: (method) => calls.filter((call) => call.method === method).length,
  };
}

/** A host, the memory it reads, and what a guest shim would do with both. */
class Guest {
  constructor(options = {}) {
    this.memory = new WebAssembly.Memory({ initial: 2, maximum: 16, shared: true });
    this.log = [];
    this.spy = spyOn(navigator.gpu);
    this.host = createWebGpuHost({ gpu: this.spy.gpu, log: (message) => this.log.push(message), ...options });
    this.host.setMemory(this.memory);
    this.imports = this.host.imports;
    this.top = 64;
    this.future = 0n;
    /** Where event records are collected. */
    this.recordBuffer = this.alloc(RECORDS_CAPACITY);
  }

  get view() {
    return new DataView(this.memory.buffer);
  }

  /** Zeroed, 8-aligned guest memory. */
  alloc(size) {
    const ptr = (this.top + 7) & ~7;
    this.top = ptr + size;
    new Uint8Array(this.memory.buffer, ptr, size).fill(0);
    return ptr;
  }

  /** Writes a WGPUStringView at `ptr`: `text` with its length, or as a C string. */
  stringView(ptr, text, { terminated = false } = {}) {
    const bytes = new TextEncoder().encode(text);
    const data = this.alloc(bytes.length + 1);
    new Uint8Array(this.memory.buffer, data, bytes.length).set(bytes);
    this.view.setUint32(ptr + S.WGPUStringView.data, data, true);
    this.view.setUint32(ptr + S.WGPUStringView.length, terminated ? U32_MAX : bytes.length, true);
  }

  /** Collects the event records the host has ready. */
  records() {
    const out = [];
    const buffer = this.recordBuffer;
    for (;;) {
      const written = this.imports.wasmer_wgpu_poll(buffer, RECORDS_CAPACITY);
      if (written <= 0) return out;
      for (let at = buffer; at < buffer + written;) {
        const messageLength = this.view.getUint32(at + 24, true);
        const extraLength = this.view.getUint32(at + 28, true);
        out.push({
          future: this.view.getBigUint64(at, true),
          kind: this.view.getUint32(at + 8, true),
          status: this.view.getUint32(at + 12, true),
          handle: this.view.getUint32(at + 16, true),
          aux: this.view.getUint32(at + 20, true),
          // A copy: TextDecoder does not read shared memory.
          message: new TextDecoder().decode(
            new Uint8Array(this.memory.buffer).slice(at + 32, at + 32 + messageLength),
          ),
        });
        at += 32 + ((messageLength + 7) & ~7) + ((extraLength + 7) & ~7);
      }
    }
  }

  /** Starts an operation that takes a future, and waits for its record. */
  async complete(start) {
    const future = ++this.future;
    start(future);
    for (let turn = 0; turn < 10_000; turn++) {
      const record = this.records().find((candidate) => candidate.future === future);
      if (record) return record;
      await this.imports.wasmer_wgpu_yield();
    }
    throw new Error('the future never completed');
  }

  /** An instance, an adapter and a device, as a program would get them. */
  async device() {
    const { imports } = this;
    const instance = imports.wgpuCreateInstance(0);
    const requestedAdapter = await this.complete(
      (future) => imports.wasmer_wgpu_instance_request_adapter(instance, 0, future),
    );
    const adapter = requestedAdapter.handle;
    const lostFuture = ++this.future;
    this.lostFuture = lostFuture;
    const requestedDevice = await this.complete(
      (future) => imports.wasmer_wgpu_adapter_request_device(adapter, 0, future, lostFuture),
    );
    const device = requestedDevice.handle;
    if (device === 0) throw new Error(`no device: ${requestedDevice.message}`);
    return { instance, adapter, device };
  }

  /** A 4x4 RGBA8 render target and a view of it. */
  renderTarget(device, format = E.TextureFormat.RGBA8Unorm) {
    const T = S.WGPUTextureDescriptor;
    const descriptor = this.alloc(T.__size);
    this.view.setBigUint64(descriptor + T.usage, BigInt(F.TextureUsage.RenderAttachment), true);
    this.view.setUint32(descriptor + T.size + S.WGPUExtent3D.width, 4, true);
    this.view.setUint32(descriptor + T.size + S.WGPUExtent3D.height, 4, true);
    this.view.setUint32(descriptor + T.size + S.WGPUExtent3D.depthOrArrayLayers, 1, true);
    this.view.setUint32(descriptor + T.format, format, true);
    this.view.setUint32(descriptor + T.mipLevelCount, 1, true);
    this.view.setUint32(descriptor + T.sampleCount, 1, true);
    const texture = this.imports.wgpuDeviceCreateTexture(device, descriptor);
    return { texture, view: this.imports.wgpuTextureCreateView(texture, 0) };
  }
}

function check(condition, message) {
  if (!condition) throw new Error(message);
}

/** Compares by value; 64-bit numbers included. */
function equal(actual, expected, what) {
  const show = (value) => JSON.stringify(value, (_, member) => (typeof member === 'bigint' ? `${member}n` : member));
  check(show(actual) === show(expected), `${what}: got ${show(actual)}, expected ${show(expected)}`);
}

/** A 4x4 RGBA8 guest texture an image can be copied into, and its browser object. */
function imageTarget(guest, device) {
  const T = S.WGPUTextureDescriptor;
  const descriptor = guest.alloc(T.__size);
  const usage = F.TextureUsage.CopyDst | F.TextureUsage.CopySrc | F.TextureUsage.RenderAttachment;
  guest.view.setBigUint64(descriptor + T.usage, BigInt(usage), true);
  guest.view.setUint32(descriptor + T.size + S.WGPUExtent3D.width, 4, true);
  guest.view.setUint32(descriptor + T.size + S.WGPUExtent3D.height, 4, true);
  guest.view.setUint32(descriptor + T.size + S.WGPUExtent3D.depthOrArrayLayers, 1, true);
  guest.view.setUint32(descriptor + T.format, E.TextureFormat.RGBA8Unorm, true);
  guest.view.setUint32(descriptor + T.mipLevelCount, 1, true);
  guest.view.setUint32(descriptor + T.sampleCount, 1, true);
  const texture = guest.imports.wgpuDeviceCreateTexture(device, descriptor);
  return { texture, gpuTexture: guest.spy.lastResult('createTexture') };
}

/** The texels of a 4x4 RGBA8 texture, rows 256 bytes apart. */
async function readTexels(gpuDevice, gpuTexture) {
  const readback = gpuDevice.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = gpuDevice.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture: gpuTexture }, { buffer: readback, bytesPerRow: 256 }, [4, 4]);
  gpuDevice.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const texels = new Uint8Array(readback.getMappedRange()).slice();
  readback.unmap();
  return texels;
}

const TESTS = {
  async 'a string view with a length is taken whole, a null one is empty'(guest) {
    const { device } = await guest.device();
    const B = S.WGPUBufferDescriptor;
    const descriptor = guest.alloc(B.__size);
    guest.view.setBigUint64(descriptor + B.usage, BigInt(F.BufferUsage.Uniform), true);
    guest.view.setBigUint64(descriptor + B.size, 16n, true);
    // An explicit length is the whole label, NUL included.
    guest.stringView(descriptor + B.label, 'a\0b');
    check(guest.imports.wgpuDeviceCreateBuffer(device, descriptor) !== 0, 'the buffer was not created');
    equal(guest.spy.last('createBuffer')[0].label, 'a\0b', 'label with an explicit length');
    // WGPU_STRLEN stops at the terminator.
    guest.stringView(descriptor + B.label, 'terminated', { terminated: true });
    guest.imports.wgpuDeviceCreateBuffer(device, descriptor);
    equal(guest.spy.last('createBuffer')[0].label, 'terminated', 'NUL-terminated label');
    // { NULL, WGPU_STRLEN }: no label.
    guest.view.setUint32(descriptor + B.label + S.WGPUStringView.data, 0, true);
    guest.imports.wgpuDeviceCreateBuffer(device, descriptor);
    equal(guest.spy.last('createBuffer')[0].label, '', 'null label');
  },

  async 'the undefined values of webgpu.h leave members out'(guest) {
    const { device } = await guest.device();
    const { imports, view } = guest;
    const color = guest.renderTarget(device);
    const depth = guest.renderTarget(device, E.TextureFormat.Depth24PlusStencil8);

    const Q = S.WGPUQuerySetDescriptor;
    const queryDescriptor = guest.alloc(Q.__size);
    view.setUint32(queryDescriptor + Q.type, E.QueryType.Occlusion, true);
    view.setUint32(queryDescriptor + Q.count, 2, true);
    const querySet = imports.wgpuDeviceCreateQuerySet(device, queryDescriptor);

    const C = S.WGPURenderPassColorAttachment;
    const colorAttachment = guest.alloc(C.__size);
    view.setUint32(colorAttachment + C.view, color.view, true);
    view.setUint32(colorAttachment + C.depthSlice, U32_MAX, true); // WGPU_DEPTH_SLICE_UNDEFINED
    view.setUint32(colorAttachment + C.loadOp, E.LoadOp.Clear, true);
    view.setUint32(colorAttachment + C.storeOp, E.StoreOp.Undefined, true);

    const D = S.WGPURenderPassDepthStencilAttachment;
    const depthAttachment = guest.alloc(D.__size);
    view.setUint32(depthAttachment + D.view, depth.view, true);
    view.setUint32(depthAttachment + D.depthLoadOp, E.LoadOp.Load, true);
    view.setUint32(depthAttachment + D.depthStoreOp, E.StoreOp.Store, true);
    view.setFloat32(depthAttachment + D.depthClearValue, NaN, true); // WGPU_DEPTH_CLEAR_VALUE_UNDEFINED
    view.setUint32(depthAttachment + D.stencilReadOnly, 1, true);
    // Ops on a read-only aspect are not the pass's to give.
    view.setUint32(depthAttachment + D.stencilLoadOp, E.LoadOp.Clear, true);

    const W = S.WGPUPassTimestampWrites;
    const writes = guest.alloc(W.__size);
    view.setUint32(writes + W.querySet, querySet, true);
    view.setUint32(writes + W.beginningOfPassWriteIndex, U32_MAX, true); // WGPU_QUERY_SET_INDEX_UNDEFINED
    view.setUint32(writes + W.endOfPassWriteIndex, 1, true);

    const P = S.WGPURenderPassDescriptor;
    const pass = guest.alloc(P.__size);
    view.setUint32(pass + P.colorAttachmentCount, 1, true);
    view.setUint32(pass + P.colorAttachments, colorAttachment, true);
    view.setUint32(pass + P.depthStencilAttachment, depthAttachment, true);
    view.setUint32(pass + P.timestampWrites, writes, true);

    const encoder = imports.wgpuDeviceCreateCommandEncoder(device, 0);
    check(imports.wgpuCommandEncoderBeginRenderPass(encoder, pass) !== 0, 'the pass was not begun');
    const [descriptor] = guest.spy.last('beginRenderPass');
    const [attachment] = descriptor.colorAttachments;
    equal(attachment.depthSlice, undefined, 'depthSlice');
    equal(attachment.storeOp, 'store', 'an undefined storeOp');
    equal(descriptor.depthStencilAttachment.depthClearValue, undefined, 'a NaN depthClearValue');
    equal(descriptor.depthStencilAttachment.depthLoadOp, 'load', 'depthLoadOp');
    equal(descriptor.depthStencilAttachment.stencilLoadOp, undefined, 'the load op of a read-only aspect');
    equal(descriptor.timestampWrites.beginningOfPassWriteIndex, undefined, 'beginningOfPassWriteIndex');
    equal(descriptor.timestampWrites.endOfPassWriteIndex, 1, 'endOfPassWriteIndex');
    equal(descriptor.occlusionQuerySet, undefined, 'a null occlusionQuerySet');

    // A depth-stencil state that leaves depthWriteEnabled undefined.
    const code = '@vertex fn vs() -> @builtin(position) vec4f { return vec4f(); }';
    const source = guest.alloc(S.WGPUShaderSourceWGSL.__size);
    view.setUint32(source + S.WGPUChainedStruct.sType, STYPE.ShaderSourceWGSL, true);
    guest.stringView(source + S.WGPUShaderSourceWGSL.code, code);
    const moduleDescriptor = guest.alloc(S.WGPUShaderModuleDescriptor.__size);
    view.setUint32(moduleDescriptor + S.WGPUShaderModuleDescriptor.nextInChain, source, true);
    const module = imports.wgpuDeviceCreateShaderModule(device, moduleDescriptor);

    const DS = S.WGPUDepthStencilState;
    const depthStencil = guest.alloc(DS.__size);
    view.setUint32(depthStencil + DS.format, E.TextureFormat.Depth24PlusStencil8, true);
    view.setUint32(depthStencil + DS.depthWriteEnabled, E.OptionalBool.Undefined, true);
    const R = S.WGPURenderPipelineDescriptor;
    const pipeline = guest.alloc(R.__size);
    view.setUint32(pipeline + R.vertex + S.WGPUVertexState.module, module, true);
    // { NULL, WGPU_STRLEN }: the null string, which lets the shader choose.
    view.setUint32(pipeline + R.vertex + S.WGPUVertexState.entryPoint + S.WGPUStringView.length, U32_MAX, true);
    view.setUint32(pipeline + R.depthStencil, depthStencil, true);
    imports.wgpuDeviceCreateRenderPipeline(device, pipeline);
    const [pipelineDescriptor] = guest.spy.last('createRenderPipeline');
    equal(pipelineDescriptor.depthStencil.depthWriteEnabled, undefined, 'depthWriteEnabled');
    equal(pipelineDescriptor.depthStencil.depthCompare, undefined, 'depthCompare');
    equal(pipelineDescriptor.vertex.entryPoint, undefined, 'a null entry point');
    equal(pipelineDescriptor.layout, 'auto', 'a null layout');
    equal(pipelineDescriptor.multisample.count, 1, 'a zero sample count');
    equal(pipelineDescriptor.fragment, undefined, 'no fragment stage');
  },

  async 'a struct chained where it does not belong is refused, not ignored'(guest) {
    const { device } = await guest.device();
    const B = S.WGPUBufferDescriptor;
    const stray = guest.alloc(S.WGPUShaderSourceWGSL.__size);
    guest.view.setUint32(stray + S.WGPUChainedStruct.sType, STYPE.ShaderSourceWGSL, true);
    const descriptor = guest.alloc(B.__size);
    guest.view.setUint32(descriptor + B.nextInChain, stray, true);
    guest.view.setBigUint64(descriptor + B.usage, BigInt(F.BufferUsage.Uniform), true);
    guest.view.setBigUint64(descriptor + B.size, 16n, true);
    equal(guest.imports.wgpuDeviceCreateBuffer(device, descriptor), 0, 'the handle');
    equal(guest.spy.count('createBuffer'), 0, 'calls of createBuffer');
    check(guest.log.some((line) => line.includes('is not allowed in this chain')), `not logged: ${guest.log}`);
    // An enum value from the future is the same kind of mistake.
    const sampler = guest.alloc(S.WGPUSamplerDescriptor.__size);
    guest.view.setUint32(sampler + S.WGPUSamplerDescriptor.magFilter, 77, true);
    equal(guest.imports.wgpuDeviceCreateSampler(device, sampler), 0, 'a sampler with an unknown filter');
    equal(guest.spy.count('createSampler'), 0, 'calls of createSampler');
  },

  async 'records are handed over whole, and say how much room they need'(guest) {
    const { imports } = guest;
    // An invalid instance: the request fails at once, with a message.
    imports.wasmer_wgpu_instance_request_adapter(0, 0, 7n);
    imports.wasmer_wgpu_instance_request_adapter(0, 0, 8n);
    equal(imports.wasmer_wgpu_epoch(), 2n, 'epoch after two records');
    const buffer = guest.alloc(256);
    const needed = -imports.wasmer_wgpu_poll(buffer, 16);
    // 32 bytes of header and "Invalid instance." padded to 8.
    equal(needed, 32 + 24, 'bytes the first record needs');
    equal(imports.wasmer_wgpu_poll(buffer, needed), needed, 'bytes written when one record fits');
    equal(guest.view.getBigUint64(buffer, true), 7n, 'the first record');
    equal(guest.records(), [{
      future: 8n, kind: EVENT.RequestAdapter, status: E.RequestAdapterStatus.Unavailable, handle: 0, aux: 0,
      message: 'Invalid instance.',
    }], 'the second record');
    equal(imports.wasmer_wgpu_poll(buffer, 256), 0, 'bytes written when nothing is ready');
    equal(imports.wasmer_wgpu_epoch(), 2n, 'epoch after draining');
  },

  async 'a wait ends at its timeout, at once for a past epoch, and when an event arrives'(guest) {
    const { imports } = guest;
    const futures = guest.alloc(8);
    guest.view.setBigUint64(futures, 99n, true);
    const buffer = guest.alloc(256);
    const wait = (timeoutNs, epoch) => imports.wasmer_wgpu_wait(futures, 1, timeoutNs, epoch, buffer, 256);
    let started = performance.now();
    equal(await wait(30_000_000n, imports.wasmer_wgpu_epoch()), 0, 'bytes from a wait that times out');
    check(performance.now() - started >= 29, `a 30 ms wait returned after ${performance.now() - started} ms`);
    // The shim read the epoch before an event was queued: no sleep.
    const epoch = imports.wasmer_wgpu_epoch();
    imports.wasmer_wgpu_instance_request_adapter(0, 0, 5n);
    started = performance.now();
    check((await wait(U64_MAX, epoch)) > 0, 'the queued record was not returned');
    check(performance.now() - started < 1000, 'a wait on a past epoch slept');
    // Asleep until something is queued, whatever future it belongs to.
    setTimeout(() => imports.wasmer_wgpu_instance_request_adapter(0, 0, 6n), 20);
    check((await wait(U64_MAX, imports.wasmer_wgpu_epoch())) > 0, 'the wait did not wake');
    equal(guest.view.getBigUint64(buffer, true), 6n, 'the future that woke the wait');
  },

  async 'the host follows a memory that grows'(guest) {
    const limits = guest.memory.buffer.byteLength + 64;
    let trapped = false;
    try {
      guest.imports.wgpuGetInstanceLimits(limits);
    } catch (error) {
      trapped = error instanceof WebAssembly.RuntimeError && error.message.includes('wgpuGetInstanceLimits');
    }
    check(trapped, 'an address past the end of memory did not trap');
    guest.memory.grow(1);
    equal(guest.imports.wgpuGetInstanceLimits(limits), E.Status.Success, 'the call after growing');
    equal(guest.view.getUint32(limits + S.WGPUInstanceLimits.timedWaitAnyMaxCount, true), 64, 'timedWaitAnyMaxCount');
    check(guest.memory.buffer.byteLength === 3 * PAGE, 'the memory did not grow');
  },

  async 'an error the host finds itself reaches the scope a browser error would'(guest) {
    const { imports } = guest;
    const { device } = await guest.device();
    // A bind group whose layout handle is not a layout.
    const descriptor = guest.alloc(S.WGPUBindGroupDescriptor.__size);
    guest.view.setUint32(descriptor + S.WGPUBindGroupDescriptor.layout, device, true);
    const pop = () => guest.complete((future) => imports.wasmer_wgpu_device_pop_error_scope(device, future));

    imports.wgpuDevicePushErrorScope(device, E.ErrorFilter.Validation);
    imports.wgpuDevicePushErrorScope(device, E.ErrorFilter.OutOfMemory);
    equal(imports.wgpuDeviceCreateBindGroup(device, descriptor), 0, 'the handle of an invalid bind group');
    const inner = await pop();
    equal([inner.status, inner.aux], [E.PopErrorScopeStatus.Success, E.ErrorType.NoError], 'the out-of-memory scope');
    const outer = await pop();
    equal([outer.status, outer.aux, outer.message],
      [E.PopErrorScopeStatus.Success, E.ErrorType.Validation, 'wgpuDeviceCreateBindGroup: invalid layout.'],
      'the validation scope');
    const empty = await pop();
    equal(
      [empty.status, empty.message], [E.PopErrorScopeStatus.Error, 'There is no error scope to pop.'],
      'an empty stack',
    );

    // With no scope it is an uncaptured error of that device.
    imports.wgpuDeviceCreateBindGroup(device, descriptor);
    const [uncaptured] = guest.records();
    equal([uncaptured.kind, uncaptured.handle, uncaptured.aux, uncaptured.future],
      [EVENT.UncapturedError, device, E.ErrorType.Validation, 0n], 'the uncaptured error');
  },

  async 'an embedder copies an image into a guest texture, and alone hears what fails'(guest) {
    const { imports } = guest;
    const { device } = await guest.device();
    const gpuDevice = await guest.spy.lastResult('requestDevice');
    const { texture, gpuTexture } = imageTarget(guest, device);
    // Opaque texels (x * 60, y * 60, 200).
    const pixels = new Uint8ClampedArray(4 * 4 * 4);
    for (let i = 0; i < 16; i++) pixels.set([(i % 4) * 60, Math.floor(i / 4) * 60, 200, 255], i * 4);
    const image = await createImageBitmap(new ImageData(pixels, 4, 4));

    // The top-left 2x2 of the image, at (2, 1).
    equal(await guest.host.copyExternalImage(texture, image, { origin: [2, 1], size: [2, 2] }), true, 'a copy');
    const texels = await readTexels(gpuDevice, gpuTexture);
    for (let y = 0; y < 4; y++) {
      for (let x = 0; x < 4; x++) {
        const inside = x >= 2 && x < 4 && y >= 1 && y < 3;
        const expected = inside ? [(x - 2) * 60, (y - 1) * 60, 200, 255] : [0, 0, 0, 0];
        equal([...texels.subarray(y * 256 + x * 4, y * 256 + x * 4 + 4)], expected, `texel (${x}, ${y})`);
      }
    }

    // Failures reach the embedder's log, never the guest's scopes or callback.
    const pop = () => guest.complete((future) => imports.wasmer_wgpu_device_pop_error_scope(device, future));
    imports.wgpuDevicePushErrorScope(device, E.ErrorFilter.Validation);
    const noCopyDst = guest.renderTarget(device).texture;
    equal(await guest.host.copyExternalImage(noCopyDst, image), false, 'a copy into a texture without CopyDst');
    check(guest.log.at(-1).startsWith('copyExternalImage: '), `not logged: ${guest.log.at(-1)}`);
    equal((await pop()).aux, E.ErrorType.NoError, 'the guest\'s scope');
    equal(await guest.host.copyExternalImage(device, image), false, 'a copy into a device handle');
    check(guest.log.at(-1).includes('is not a live texture'), `not logged: ${guest.log.at(-1)}`);
    image.close();
    equal(await guest.host.copyExternalImage(texture, image), false, 'a copy of a closed image');
    equal(guest.records().filter((record) => record.kind === EVENT.UncapturedError), [], 'uncaptured errors');
    guest.host.dispose();
    equal(await guest.host.copyExternalImage(texture, image), false, 'a copy after dispose');
    check(guest.log.at(-1).includes('the host was stopped'), `not logged: ${guest.log.at(-1)}`);
  },

  async 'a copied image keeps straight alpha unless premultiplied alpha is asked for'(guest) {
    const { device } = await guest.device();
    const gpuDevice = await guest.spy.lastResult('requestDevice');
    const { texture, gpuTexture } = imageTarget(guest, device);
    const image = await createImageBitmap(new ImageData(new Uint8ClampedArray([200, 100, 50, 128]), 1, 1),
      { premultiplyAlpha: 'none' });
    const near = (actual, expected, what) => check(actual.every((value, i) => Math.abs(value - expected[i]) <= 2),
      `${what}: got ${[...actual]}, expected ${expected}`);
    // WebGPU's default: the texel is what the image holds.
    equal(await guest.host.copyExternalImage(texture, image), true, 'a copy with the default');
    near((await readTexels(gpuDevice, gpuTexture)).subarray(0, 4), [200, 100, 50, 128], 'straight alpha');
    equal(await guest.host.copyExternalImage(texture, image, { premultipliedAlpha: true }), true, 'a premultiplied copy');
    near((await readTexels(gpuDevice, gpuTexture)).subarray(0, 4), [100, 50, 25, 128], 'premultiplied alpha');
  },

  async 'a copy is not reported as made once its device or host is gone'(guest) {
    const image = await createImageBitmap(new ImageData(4, 4));
    // The browser destroys the device under the host, which hears of it later.
    const { device } = await guest.device();
    const { texture } = imageTarget(guest, device);
    (await guest.spy.lastResult('requestDevice')).destroy();
    equal(await guest.host.copyExternalImage(texture, image), false, 'a copy onto a destroyed device');
    check(guest.log.at(-1).includes('device is lost'), `not logged: ${guest.log.at(-1)}`);
    // The host is disposed while the copy is under way.
    const other = new Guest();
    try {
      const target = imageTarget(other, (await other.device()).device).texture;
      const copied = other.host.copyExternalImage(target, image);
      other.host.dispose();
      equal(await copied, false, 'a copy whose host was disposed meanwhile');
      check(other.log.at(-1).includes('the host was stopped'), `not logged: ${other.log.at(-1)}`);
    } finally {
      other.host.dispose();
    }
  },

  async 'a released handle does not come back as another object'(guest) {
    const { imports } = guest;
    const { instance, device } = await guest.device();
    const first = imports.wgpuDeviceCreateSampler(device, 0);
    imports.wgpuSamplerAddRef(first);
    imports.wgpuSamplerRelease(first);
    imports.wgpuSamplerRelease(first);
    const second = imports.wgpuDeviceCreateSampler(device, 0);
    check(first !== 0 && second !== 0 && first !== second, `handles ${first} and ${second}`);
    // Same slot, next generation.
    equal(second & 0xffffff, first & 0xffffff, 'the slot of the second sampler');
    imports.wgpuSamplerRelease(first);
    check(guest.log.at(-1).includes('Release on an invalid Sampler handle'), `not refused: ${guest.log.at(-1)}`);
    // The queue is one object: asking again returns the same handle.
    const queue = imports.wgpuDeviceGetQueue(device);
    equal(imports.wgpuDeviceGetQueue(device), queue, 'the queue handle');
    imports.wgpuQueueRelease(queue);
    imports.wgpuQueueRelease(queue);
    check(imports.wgpuDeviceGetQueue(device) !== queue, 'a released queue handle was handed out again');
    // Events are delivered per instance: every object knows the one it descends from.
    equal(imports.wasmer_wgpu_object_instance(second), instance, 'the instance of a sampler');
    equal(imports.wasmer_wgpu_object_instance(instance), instance, 'the instance of an instance');
    equal(imports.wasmer_wgpu_object_instance(first), 0, 'the instance of a stale handle');
  },

  async 'mapped ranges are copied, whatever their alignment'(guest) {
    const { imports, view } = guest;
    const { device } = await guest.device();
    const B = S.WGPUBufferDescriptor;
    const descriptor = guest.alloc(B.__size);
    view.setBigUint64(descriptor + B.usage, BigInt(F.BufferUsage.MapRead | F.BufferUsage.CopyDst), true);
    view.setBigUint64(descriptor + B.size, 32n, true);
    view.setUint32(descriptor + B.mappedAtCreation, 1, true);
    const buffer = imports.wgpuDeviceCreateBuffer(device, descriptor);
    const data = guest.alloc(32);
    for (let i = 0; i < 32; i++) view.setUint8(data + i, 100 + i);
    equal(imports.wgpuBufferWriteMappedRange(buffer, 3, data, 7), E.Status.Success, 'an unaligned write');
    const back = guest.alloc(32);
    equal(imports.wgpuBufferReadMappedRange(buffer, 1, back, 11), E.Status.Success, 'an unaligned read');
    equal(
      [...new Uint8Array(guest.memory.buffer).slice(back, back + 11)], [0, 0, 100, 101, 102, 103, 104, 105, 106, 0, 0],
      'bytes read back',
    );
    equal(imports.wgpuBufferReadMappedRange(buffer, 30, back, 4), E.Status.Error, 'a read past the mapping');
    const info = guest.alloc(24);
    equal(imports.wasmer_wgpu_buffer_map_info(buffer, info), 1, 'map info of a buffer');
    // { u64 offset; u64 size; u32 mode; u32 state }
    equal(
      [view.getBigUint64(info, true), view.getBigUint64(info + 8, true), view.getUint32(info + 16, true),
        view.getUint32(info + 20, true)],
      [0n, 32n, F.MapMode.Write, E.BufferMapState.Mapped], 'map info',
    );
    imports.wgpuBufferUnmap(buffer);
    equal(imports.wgpuBufferGetMapState(buffer), E.BufferMapState.Unmapped, 'state after unmap');
    equal(imports.wgpuBufferReadMappedRange(buffer, 0, back, 4), E.Status.Error, 'a read of an unmapped buffer');
    // Mapping a part of it: only that part can be read.
    const mapped = await guest.complete(
      (future) => imports.wasmer_wgpu_buffer_map_async(buffer, BigInt(F.MapMode.Read), 8, 16, future),
    );
    equal(mapped.status, E.MapAsyncStatus.Success, 'the map');
    equal(imports.wgpuBufferReadMappedRange(buffer, 9, back, 3), E.Status.Success, 'a read inside the mapping');
    equal(imports.wgpuBufferReadMappedRange(buffer, 4, back, 8), E.Status.Error, 'a read before the mapping');
    equal(imports.wgpuBufferWriteMappedRange(buffer, 8, data, 4), E.Status.Error, 'a write through a read mapping');
    equal(imports.wgpuBufferGetSize(buffer), 32n, 'the size, a 64-bit result');
  },

  async 'what the browser answers after the guest was stopped is refused'() {
    let stopped = false;
    const guest = new Guest({ isTerminated: () => stopped });
    try {
      const { imports } = guest;
      const { instance } = await guest.device();
      // A browser makes one device per adapter: a second adapter for a second.
      const { handle: adapter } = await guest.complete(
        (future) => imports.wasmer_wgpu_instance_request_adapter(instance, 0, future),
      );
      // Both requests are with the browser when the embedder stops the guest.
      imports.wasmer_wgpu_instance_request_adapter(instance, 0, 20n);
      imports.wasmer_wgpu_adapter_request_device(adapter, 0, 21n, 22n);
      stopped = true;
      const records = [];
      for (let turn = 0; turn < 10_000 && records.length < 4; turn++) {
        // Yielding is what notices the switch: the first device is destroyed.
        await imports.wasmer_wgpu_yield();
        records.push(...guest.records());
      }
      const byFuture = (future) => records.find((record) => record.future === future);
      const refusal = 'The WebGPU context was shut down.';
      const lost = byFuture(guest.lostFuture);
      equal([lost?.kind, lost?.status], [EVENT.DeviceLost, E.DeviceLostReason.Destroyed], 'the first device');
      equal([byFuture(20n)?.status, byFuture(20n)?.handle, byFuture(20n)?.message],
        [E.RequestAdapterStatus.Unavailable, 0, refusal], 'the adapter request');
      equal([byFuture(21n)?.status, byFuture(21n)?.handle, byFuture(21n)?.message],
        [E.RequestDeviceStatus.Error, 0, refusal], 'the device request');
      equal([byFuture(22n)?.kind, byFuture(22n)?.status],
        [EVENT.DeviceLost, E.DeviceLostReason.FailedCreation], 'the lost future of the device that never was');
      // The handles stay good for what a dying guest still does with them.
      equal(imports.wasmer_wgpu_object_instance(adapter), instance, 'the instance of the adapter');
      imports.wgpuAdapterRelease(adapter);
      equal(imports.wasmer_wgpu_instance_release(instance), 1, 'releasing the instance');
    } finally {
      guest.host.dispose();
    }
  },

  async 'a surface is sized, presented and released as the embedder binds it'() {
    let size = [40, 30];
    let failSizing = false;
    let failPresenting = false;
    const use = { presented: 0, released: 0 };
    const canvas = new OffscreenCanvas(8, 8);
    // The browser's own objects: a canvas is configured with a real device.
    const guest = new Guest({
      gpu: navigator.gpu,
      getCanvas: async (selector) => selector === '#bound' && {
        canvas,
        getSize: () => {
          if (failSizing) throw new Error('no layout yet');
          return size;
        },
        present: () => {
          use.presented += 1;
          if (failPresenting) throw new Error('the page is gone');
        },
        release: () => {
          use.released += 1;
        },
      },
    });
    try {
      const { imports, view } = guest;
      const { instance, device } = await guest.device();
      const createSurface = (selector) => {
        const source = guest.alloc(S.WGPUWasixSurfaceSourceCanvas.__size);
        view.setUint32(source + S.WGPUChainedStruct.sType, STYPE.WasixSurfaceSourceCanvas, true);
        guest.stringView(source + S.WGPUWasixSurfaceSourceCanvas.selector, selector);
        const descriptor = guest.alloc(S.WGPUSurfaceDescriptor.__size);
        view.setUint32(descriptor + S.WGPUSurfaceDescriptor.nextInChain, source, true);
        return imports.wgpuInstanceCreateSurface(instance, descriptor);
      };
      equal(await createSurface('#unknown'), 0, 'a surface for a selector the embedder does not know');
      const surface = await createSurface('#bound');
      check(surface !== 0, 'the surface was not created');

      const sizePtr = guest.alloc(8);
      const wanted = () => {
        equal(imports.wgpuWasixSurfaceGetSize(surface, sizePtr, sizePtr + 4), E.Status.Success, 'GetSize');
        return [view.getUint32(sizePtr, true), view.getUint32(sizePtr + 4, true)];
      };
      equal(wanted(), [40, 30], 'the size the embedder asks for');

      const C = S.WGPUSurfaceConfiguration;
      const format = Number(Object.entries(WEBIDL.TextureFormat)
        .find(([, name]) => name === navigator.gpu.getPreferredCanvasFormat())[0]);
      const configure = ([width, height]) => {
        const configuration = guest.alloc(C.__size);
        view.setUint32(configuration + C.device, device, true);
        view.setUint32(configuration + C.format, format, true);
        view.setBigUint64(configuration + C.usage, BigInt(F.TextureUsage.RenderAttachment), true);
        view.setUint32(configuration + C.width, width, true);
        view.setUint32(configuration + C.height, height, true);
        imports.wgpuSurfaceConfigure(surface, configuration);
      };
      const frame = guest.alloc(S.WGPUSurfaceTexture.__size);
      const acquire = () => {
        // Cleared first, so that an answer left over from the last call
        // cannot pass for this one.
        view.setUint32(frame + S.WGPUSurfaceTexture.status, 0, true);
        view.setUint32(frame + S.WGPUSurfaceTexture.texture, 0, true);
        imports.wgpuSurfaceGetCurrentTexture(surface, frame);
        return [
          view.getUint32(frame + S.WGPUSurfaceTexture.status, true),
          view.getUint32(frame + S.WGPUSurfaceTexture.texture, true) !== 0,
        ];
      };
      const FrameStatus = E.SurfaceGetCurrentTextureStatus;

      equal(acquire(), [FrameStatus.Error, false], 'a frame of an unconfigured surface');
      configure(wanted());
      equal([canvas.width, canvas.height], [40, 30], 'the canvas after configuring');
      equal(acquire(), [FrameStatus.SuccessOptimal, true], 'the first frame');
      equal(acquire(), [FrameStatus.Error, false], 'a second frame before presenting');
      equal(await imports.wgpuSurfacePresent(surface), E.Status.Success, 'presenting');
      equal(use.presented, 1, 'calls of present()');
      equal(await imports.wgpuSurfacePresent(surface), E.Status.Error, 'presenting with no frame');

      // The embedder wants another size: the guest is told to reconfigure.
      size = [20, 10];
      equal(acquire(), [FrameStatus.SuccessSuboptimal, true], 'a frame after the target was resized');
      equal(wanted(), [20, 10], 'the new size');
      configure(wanted());
      equal(acquire(), [FrameStatus.SuccessOptimal, true], 'a frame after reconfiguring');
      equal(await imports.wgpuSurfacePresent(surface), E.Status.Success, 'presenting the reconfigured frame');

      // The embedder's getSize() failing costs the guest that answer, not its frame.
      failSizing = true;
      equal(
        imports.wgpuWasixSurfaceGetSize(surface, sizePtr, sizePtr + 4), E.Status.Error,
        'GetSize when the embedder cannot say',
      );
      equal(acquire(), [FrameStatus.SuccessOptimal, true], 'a frame when the embedder cannot say its size');
      check(guest.log.some((line) => line.includes('no layout yet')), `not logged: ${guest.log}`);
      failSizing = false;

      // The embedder's present() failing is the guest's error status, nothing more.
      failPresenting = true;
      equal(await imports.wgpuSurfacePresent(surface), E.Status.Error, 'presenting when the embedder cannot');
      check(guest.log.some((line) => line.includes('the page is gone')), `not logged: ${guest.log}`);

      equal(use.released, 0, 'releases while the surface is open');
      guest.host.dispose();
      equal(use.released, 1, 'releases after dispose');
      imports.wgpuSurfaceRelease(surface);
      equal(use.released, 1, 'releases after the handle went too');
      // A target the embedder only finds now is not kept: nothing would release it.
      equal(await createSurface('#bound'), 0, 'a surface made after dispose');
      equal(use.released, 2, 'releases after a target arrived too late');
      equal(guest.records().filter((record) => record.kind === EVENT.UncapturedError), [], 'uncaptured errors');
    } finally {
      guest.host.dispose();
    }
  },
};

/** Runs the tests. Returns `[{ name, failure, milliseconds }]`, `failure` null for a pass. */
export async function runHostTests() {
  const results = [];
  for (const [name, test] of Object.entries(TESTS)) {
    const started = performance.now();
    const guest = new Guest();
    let failure = null;
    try {
      await test(guest);
    } catch (error) {
      failure = error?.stack ?? String(error);
    } finally {
      guest.host.dispose();
    }
    results.push({ name, failure, milliseconds: performance.now() - started });
  }
  return results;
}
