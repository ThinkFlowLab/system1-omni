#!/usr/bin/env bash
#
# Build liblaya_cuda.so through one interface.
#
#   src/backends/cuda/build.sh <output dir> [compute capability]
#
# This is a wrapper, not a replacement: `tools/export.py` generates the CUDA
# source from TileLang and `tools/build.py` compiles it, exactly as
# recipe/laya/native/README.md documents. Keeping a single entry point means CI
# can build this backend the same way it builds any other, without knowing that
# this one happens to be generated rather than hand-written.
#
# It does NOT remove TileLang from the build: export.py imports it, so a machine
# building this bundle needs TileLang installed. Checking in the generated
# `generated.cu` and compiling only that with nvcc is a separate change, and the
# model owner's call.
#
#   BUILD_STAGE=<dir>   stage the export there and retain it (default: mktemp -d)
#   KEEP_STAGE=1        keep the staging directory and print its path
#   PYTHON=<python>     interpreter for the tools (default: python3, else python)
#   CUDA_HOME=<dir>     CUDA toolkit root (default: /usr/local/cuda)
#
# A checkpoint is not needed here. `tools/export_tables.py` needs one; it is a
# separate step in the recipe and is not required to produce the library.

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: build.sh <output dir> [compute capability]}
arch=${2:-${CUDA_COMPUTE_CAP:-90}}

case "$arch" in
    90) ;;
    *)
        echo "build.sh: these kernels are pinned to sm_90a (Hopper); got '$arch'." >&2
        echo "          The TileLang export embeds compute_90a and the kernels use" >&2
        echo "          Hopper-only instructions, so another target cannot be built." >&2
        exit 2
        ;;
esac

python=${PYTHON:-}
if [ -z "$python" ]; then
    if command -v python3 >/dev/null 2>&1; then python=python3; else python=python; fi
fi
if ! command -v "$python" >/dev/null 2>&1; then
    echo "build.sh: no Python interpreter found; set PYTHON" >&2
    exit 2
fi
if ! "$python" -c 'import tilelang' >/dev/null 2>&1; then
    echo "build.sh: TileLang is not importable by '$python', and tools/export.py needs it." >&2
    echo "          Install it (see recipe/laya/native/README.md) or set PYTHON." >&2
    exit 2
fi

stage=${BUILD_STAGE:-}
if [ -z "$stage" ]; then
    stage=$(mktemp -d "${TMPDIR:-/tmp}/laya-cuda.XXXXXX")
    if [ -z "${KEEP_STAGE:-}" ]; then
        trap 'rm -rf "$stage"' EXIT
    fi
else
    mkdir -p "$stage"
fi

echo "build.sh: staging in $stage"
"$python" "$here/tools/export.py" "$stage"
"$python" "$here/tools/build.py" "$stage"

library=$stage/liblaya_cuda.so
if [ ! -f "$library" ]; then
    echo "build.sh: tools/build.py did not produce $library" >&2
    exit 1
fi

mkdir -p "$out"
if ! [ "$stage" -ef "$out" ]; then
    cp "$library" "$out/liblaya_cuda.so"
    if [ -f "$stage/build-manifest.json" ]; then
        cp "$stage/build-manifest.json" "$out/build-manifest.json"
    fi
fi

echo "build.sh: built $out/liblaya_cuda.so for sm_${arch}a"
