#!/usr/bin/env bash
# Builds lib/wasm32-wasix/libwebgpu.a, the guest half of wasmer-webgpu.
#
# Guests compile against include/webgpu/webgpu.h (the unmodified upstream
# header) and link this library:
#
#     wasixcc app.c -I<repo>/include -L<repo>/lib/wasm32-wasix -lwebgpu -o app.wasm
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${WEBGPU_GUEST_OUT_DIR:-$ROOT/lib/wasm32-wasix}"
OBJ="$OUT/obj"

if ! command -v wasixcc >/dev/null 2>&1 && [[ -x "${HOME}/.wasixcc/bin/wasixcc" ]]; then
  export PATH="${HOME}/.wasixcc/bin:${PATH}"
fi

# Nothing to do while the library is newer than everything it is made from.
if [[ -f "$OUT/libwebgpu.a" && -z "$(find "$ROOT/guest" "$ROOT/include" -newer "$OUT/libwebgpu.a" -type f -print -quit)" ]]; then
  exit 0
fi

mkdir -p "$OBJ"
objects=()
for source in webgpu_shim.c wgpu_forwarders.gen.c wgpu_proc_table.gen.c; do
  object="$OBJ/${source%.c}.o"
  wasixcc --target=wasm32-wasix -O2 -Wall -Wextra -Werror \
    -I"$ROOT/include" -c "$ROOT/guest/src/$source" -o "$object"
  objects+=("$object")
done
rm -f "$OUT/libwebgpu.a"
wasixar rcs "$OUT/libwebgpu.a" "${objects[@]}"
echo "Built $OUT/libwebgpu.a"
