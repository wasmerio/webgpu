#!/usr/bin/env python3
"""Regenerates every ABI artifact from spec/webgpu.json and spec/abi.json.

    python3 codegen/gen.py           # rewrite generated files
    python3 codegen/gen.py --check   # fail if any generated file is stale

Generated files are checked in, so building the crate or the guest shim never
needs Python.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import emit_c  # noqa: E402
import emit_js  # noqa: E402
import emit_rust  # noqa: E402
from spec import ROOT, Spec  # noqa: E402


def outputs(spec):
    return {
        "include/webgpu/webgpu_wasix.h": emit_c.emit_extension_header(spec),
        "guest/src/wgpu_imports.gen.h": emit_c.emit_imports_header(spec),
        "guest/src/wgpu_forwarders.gen.c": emit_c.emit_forwarders(spec),
        "guest/src/wgpu_proc_table.gen.c": emit_c.emit_proc_table(spec),
        "guest/tests/abi_check.gen.c": emit_c.emit_abi_check(spec),
        "guest/tests/link_all.gen.c": emit_c.emit_link_all(spec),
        "src/abi/enums.gen.rs": emit_rust.emit_enums(spec),
        "src/abi/structs.gen.rs": emit_rust.emit_structs(spec),
        "src/abi/api.gen.rs": emit_rust.emit_api(spec),
        "src/abi/imports.gen.rs": emit_rust.emit_imports(spec),
        "js/webgpu_abi.gen.js": emit_js.emit_abi(spec),
    }


def main():
    check = "--check" in sys.argv[1:]
    spec = Spec()
    stale = []
    for rel, text in sorted(outputs(spec).items()):
        path = os.path.join(ROOT, rel)
        old = None
        if os.path.exists(path):
            with open(path) as f:
                old = f.read()
        if old == text:
            continue
        if check:
            stale.append(rel)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        print("wrote " + rel)
    if stale:
        print("stale generated files (run python3 codegen/gen.py):\n  " + "\n  ".join(stale))
        return 1
    print("%d functions, %d structs, %d enums; %d raw imports + %d special imports" % (
        len(spec.functions), len(spec.structs), len(spec.enums),
        len(spec.raw_import_functions()), len(spec.special_imports)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
