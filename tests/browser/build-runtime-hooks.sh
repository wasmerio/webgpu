#!/usr/bin/env bash
# Builds runtime_hooks.rs, the crate's browser backend with tests of its own, to
# tests/browser/out/runtime_hooks/: the wasm-bindgen module runtime_hooks_worker.js
# imports. Needs the wasm32-unknown-unknown target and the wasm-bindgen CLI of the
# version in Cargo.lock (`WASM_BINDGEN` names another binary than `wasm-bindgen`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"

"${CARGO:-cargo}" build --manifest-path "$ROOT/Cargo.toml" --target wasm32-unknown-unknown \
  --features js --example runtime_hooks
"${WASM_BINDGEN:-wasm-bindgen}" --target web --no-typescript --out-dir "$ROOT/tests/browser/out/runtime_hooks" \
  "$TARGET/wasm32-unknown-unknown/debug/examples/runtime_hooks.wasm"
