#!/usr/bin/env python3
"""Records what the test programs print on the native host into tests/manifest.json.

    python3 tests/record-manifest.py name [name...]

For a new or changed program, after reading its output and agreeing with it:
the manifest is the expectation every host is held to, not a snapshot to
refresh when a test fails.
"""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tests", "manifest.json")
RUNNER = os.path.join(ROOT, "target", "debug", "webgpu_wasmer")


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    by_name = dict((t["name"], t) for t in manifest["tests"])
    for name in sys.argv[1:]:
        wasm = os.path.join(ROOT, "tests", "programs", "out", name + ".wasm")
        run = subprocess.run([RUNNER, wasm], capture_output=True, text=True)
        if run.returncode != 0:
            sys.exit("%s exited with %d:\n%s%s" % (name, run.returncode, run.stdout, run.stderr))
        entry = by_name.get(name)
        if entry is None:
            entry = {"name": name}
            manifest["tests"].append(entry)
            by_name[name] = entry
        entry["stdout"] = run.stdout
        print("recorded %s (%d lines)" % (name, run.stdout.count("\n")))
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
