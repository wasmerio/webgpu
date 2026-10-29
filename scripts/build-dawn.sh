#!/bin/sh
# Builds the Dawn revision the `dawn` feature is written against, and installs
# its headers and monolithic static library where build.rs looks for them:
#
#   scripts/build-dawn.sh [prefix]    # default: third_party/dawn/install
#
# Then: cargo build --features dawn (or set WASMER_WEBGPU_DAWN_DIR=prefix).
#
# Needs git, cmake, ninja and a C++20 compiler; Dawn fetches the rest of its
# dependencies itself. DAWN_SOURCE and DAWN_BUILD override where the checkout
# and the build tree go.
set -eu

# The revision whose webgpu.h is spec/webgpu.h: Dawn's own pin of
# webgpu-headers must be the commit spec/ was taken from (checked below).
DAWN_REVISION=432d5a4f703606c6a48b14fbae581bc04a089099

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=${DAWN_SOURCE:-$ROOT/third_party/dawn/src}
OUT=${DAWN_BUILD:-$ROOT/third_party/dawn/out}
PREFIX=${1:-$ROOT/third_party/dawn/install}
STAMP=$PREFIX/.dawn-revision

# An install of this revision is kept (a cache CI restores, for one).
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$DAWN_REVISION" ] && [ -f "$PREFIX/include/webgpu/webgpu.h" ]; then
    echo "Dawn $DAWN_REVISION is already installed in $PREFIX"
    exit 0
fi

if [ ! -d "$SRC/.git" ]; then
    mkdir -p "$(dirname "$SRC")"
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://dawn.googlesource.com/dawn
fi
if [ "$(git -C "$SRC" rev-parse -q --verify HEAD 2>/dev/null || true)" != "$DAWN_REVISION" ]; then
    git -C "$SRC" fetch -q --depth 1 origin "$DAWN_REVISION"
    git -C "$SRC" checkout -q --detach "$DAWN_REVISION"
fi

# A backend per platform, and only the shading-language writers it needs.
case "$(uname -s)" in
    Darwin)
        PLATFORM="-DDAWN_ENABLE_METAL=ON -DDAWN_ENABLE_VULKAN=OFF
                  -DTINT_BUILD_SPV_WRITER=OFF -DTINT_BUILD_HLSL_WRITER=OFF"
        ;;
    Linux)
        PLATFORM="-DDAWN_ENABLE_VULKAN=ON -DDAWN_USE_X11=ON -DDAWN_USE_WAYLAND=ON
                  -DTINT_BUILD_HLSL_WRITER=OFF"
        ;;
    MINGW* | MSYS* | CYGWIN*)
        PLATFORM="-DDAWN_ENABLE_D3D12=ON -DDAWN_ENABLE_VULKAN=OFF -DTINT_BUILD_SPV_WRITER=OFF"
        ;;
    *)
        echo "build-dawn.sh: no Dawn backend configured for $(uname -s)" >&2
        exit 1
        ;;
esac

# DAWN_SUPPORTS_CXX_MODULES=OFF: the bindings need no C++ module, and GCC 13
# cannot scan one (CMake refuses to generate the build).
# shellcheck disable=SC2086 # PLATFORM is a list of arguments
cmake -S "$SRC" -B "$OUT" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DDAWN_FETCH_DEPENDENCIES=ON \
    -DDAWN_ENABLE_INSTALL=ON \
    -DDAWN_BUILD_MONOLITHIC_LIBRARY=STATIC \
    -DDAWN_ENABLE_OPENGLES=OFF \
    -DDAWN_ENABLE_DESKTOP_GL=OFF \
    -DDAWN_ENABLE_NULL=OFF \
    -DDAWN_BUILD_SAMPLES=OFF \
    -DDAWN_BUILD_TESTS=OFF \
    -DDAWN_BUILD_BENCHMARKS=OFF \
    -DDAWN_BUILD_NODE_BINDINGS=OFF \
    -DDAWN_BUILD_PROTOBUF=OFF \
    -DDAWN_USE_GLFW=OFF \
    -DDAWN_SUPPORTS_CXX_MODULES=OFF \
    -DTINT_BUILD_TESTS=OFF \
    -DTINT_BUILD_CMD_TOOLS=OFF \
    -DTINT_BUILD_BENCHMARKS=OFF \
    -DTINT_BUILD_FUZZERS=OFF \
    -DTINT_BUILD_IR_BINARY=OFF \
    -DTINT_BUILD_SPV_READER=OFF \
    -DTINT_BUILD_GLSL_WRITER=OFF \
    -DTINT_BUILD_GLSL_VALIDATOR=OFF \
    $PLATFORM

# Dawn implements the header it pins; the guest is built against spec/'s.
if ! cmp -s "$SRC/third_party/webgpu-headers/src/webgpu.h" "$ROOT/spec/webgpu.h"; then
    echo "build-dawn.sh: Dawn $DAWN_REVISION implements a webgpu.h other than spec/webgpu.h" >&2
    exit 1
fi

cmake --build "$OUT"
cmake --install "$OUT"
echo "$DAWN_REVISION" > "$STAMP"
echo "Dawn $DAWN_REVISION installed in $PREFIX"
