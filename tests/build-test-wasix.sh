#!/usr/bin/env bash
# Builds tests/programs/<name>.c (or every program) to tests/programs/out/<name>.wasm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${WEBGPU_TEST_OUT_DIR:-$ROOT/tests/programs/out}"

if ! command -v wasixcc >/dev/null 2>&1 && [[ -x "${HOME}/.wasixcc/bin/wasixcc" ]]; then
  export PATH="${HOME}/.wasixcc/bin:${PATH}"
fi

"$ROOT/guest/build.sh" >/dev/null
mkdir -p "$OUT"

build() {
  local source="$1"
  local name
  name="$(basename "${source%.*}")"
  local driver=wasixcc
  case "$source" in *.cc | *.cpp) driver=wasixcc++ ;; esac
  local output="$OUT/$name.wasm"
  if [[ -f "$output" && "$output" -nt "$source" && "$output" -nt "$ROOT/lib/wasm32-wasix/libwebgpu.a" \
        && "$output" -nt "$ROOT/tests/programs/wgpu_test.h" ]]; then
    return
  fi
  "$driver" --target=wasm32-wasix -O2 -Wall -Wextra -Werror -Wno-unused-function \
    -I"$ROOT/include" "$source" -L"$ROOT/lib/wasm32-wasix" -lwebgpu -o "$output"
  echo "built $name.wasm"
}

if [[ $# -eq 0 ]]; then
  for source in "$ROOT"/tests/programs/*.c "$ROOT"/tests/programs/*.cc; do
    if [[ -e "$source" ]]; then build "$source"; fi
  done
else
  for name in "$@"; do
    for ext in c cc cpp; do
      if [[ -f "$ROOT/tests/programs/$name.$ext" ]]; then build "$ROOT/tests/programs/$name.$ext"; fi
    done
  done
fi
