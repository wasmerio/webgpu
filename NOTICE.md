# Third-party material

This repository is MIT licensed (see `LICENSE`). It includes, or is derived
from, the following work of others.

## webgpu-headers

`spec/webgpu.h`, `spec/webgpu.json`, `spec/webgpu.yml` and
`include/webgpu/webgpu.h` are unmodified copies from
<https://github.com/webgpu-native/webgpu-headers> at commit
`a1dc77409bb2b8bd80540cbf4b8db219e123e909`.

Copyright 2019-2023 WebGPU-Native developers. BSD-3-Clause.

The files generated from them by `codegen/` (`*.gen.*`, and
`include/webgpu/webgpu_wasix.h`) restate their declarations.

## Dawn

`spec/webidl_enums.json` is derived from `src/dawn/dawn.json` of
<https://dawn.googlesource.com/dawn> by `codegen/derive_webidl_enums.py`.

Copyright 2017 The Dawn & Tint Authors. BSD-3-Clause.

## Dawn's Emscripten binding

The descriptor decoding in `js/webgpu_host.js` follows
`third_party/emdawnwebgpu/pkg/webgpu/src/library_webgpu.js` of the same
repository, adapted to this project's ABI.

Copyright 2019 The Emscripten Authors. MIT.

## BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
