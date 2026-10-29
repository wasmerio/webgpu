# Everyday tasks. Nothing here is needed to use the crate from Rust: cargo
# builds it. The guest library and the test programs need wasixcc
# (https://github.com/wasix-org/wasixcc); the browser fixture needs the
# wasm32-unknown-unknown target and the wasm-bindgen CLI of Cargo.lock's version.

CARGO ?= cargo
PYTHON ?= python3
WASIXCC ?= wasixcc
ifeq ($(shell command -v $(WASIXCC) 2>/dev/null),)
WASIXCC := $(HOME)/.wasixcc/bin/wasixcc
endif

EXAMPLES := $(patsubst examples/%.c,examples/out/%.wasm,$(wildcard examples/*.c))

VERSION := $(shell sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
DIST := dist/wasmer-webgpu-guest-$(VERSION)

.PHONY: all guest programs browser-fixture examples dist gen check fmt lint lint-dawn check-web test test-native test-dawn test-browser dawn clean

all: guest examples

## What guests link: lib/wasm32-wasix/libwebgpu.a
guest:
	guest/build.sh

## The conformance programs: tests/programs/out/*.wasm
programs: guest
	tests/build-test-wasix.sh

## The runtime hooks as a browser embedder uses them: tests/browser/out/runtime_hooks
browser-fixture:
	CARGO="$(CARGO)" tests/browser/build-runtime-hooks.sh

examples: $(EXAMPLES)

examples/out/%.wasm: examples/%.c guest
	@mkdir -p examples/out
	$(WASIXCC) --target=wasm32-wasix -O2 -Wall -Wextra -Werror $< -Iinclude -Llib/wasm32-wasix -lwebgpu -o $@

## What a guest toolchain needs, as one archive: the headers and the library.
dist: guest
	rm -rf $(DIST) && mkdir -p $(DIST)/include/webgpu $(DIST)/lib/wasm32-wasix
	cp include/webgpu/webgpu.h include/webgpu/webgpu_wasix.h $(DIST)/include/webgpu/
	cp lib/wasm32-wasix/libwebgpu.a $(DIST)/lib/wasm32-wasix/
	cp LICENSE NOTICE.md $(DIST)/
	tar -czf $(DIST).tar.gz -C dist $(notdir $(DIST))
	@echo "$(DIST).tar.gz"

## Regenerate everything derived from spec/ (the results are checked in).
gen:
	$(PYTHON) codegen/gen.py

## What CI holds a change to, short of running it on a GPU.
check: lint check-web
	$(PYTHON) codegen/gen.py --check
	$(CARGO) fmt --check

fmt:
	$(CARGO) fmt

lint:
	$(CARGO) clippy --features cli,window --all-targets -- -D warnings
	$(CARGO) clippy --target wasm32-unknown-unknown --features js --example runtime_hooks -- -D warnings

## The same with the native host on Dawn (needs `make dawn` first).
lint-dawn:
	$(CARGO) clippy --features cli,window,dawn --all-targets -- -D warnings

## The browser half of the crate, as the Wasmer SDK compiles it.
check-web:
	$(CARGO) check --target wasm32-unknown-unknown --no-default-features --features js,wasix

test: test-native

## Unit tests, the ABI sweep, the conformance programs and the embedder tests.
## WEBGPU_TEST_FALLBACK_ADAPTER=1 runs them on a software adapter (empty or 0: off).
test-native: programs
	$(CARGO) test --features cli

## The same suite on the Dawn host.
test-dawn: programs dawn
	$(CARGO) test --features cli,dawn

## Dawn for the `dawn` feature, at the pinned revision: third_party/dawn/install.
## Quick when that revision is installed already (the script checks).
dawn:
	scripts/build-dawn.sh

## The conformance programs against js/webgpu_host.js, and the runtime hooks
## built for a browser, in the installed Chrome.
test-browser: programs browser-fixture
	cd tests/browser && npm install --no-audit --no-fund && node run.mjs --no-build

clean:
	$(CARGO) clean
	rm -rf lib dist tests/programs/out tests/browser/out examples/out
	@echo "(third_party/dawn is kept: it takes a while to build)"
