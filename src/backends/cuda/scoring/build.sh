#!/usr/bin/env bash
#
# Build libscoring.so.
#
#   src/backends/cuda/scoring/build.sh <output dir> [compute capability]
#
# Needs nvcc only: this kernel has no Python, TileLang or cuBLAS dependency. The
# reference implementation and the tests are Python, but they are never part of
# the built artifact.
#
#   NVCC / CUDA_HOME   as tools elsewhere in this repository read them
#   CUDA_ARCH_LIST     extra targets, e.g. CUDA_ARCH_LIST="80 89 90"

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: build.sh <output dir> [compute capability]}
arch=${2:-${CUDA_COMPUTE_CAP:-80}}

nvcc=${NVCC:-}
if [ -z "$nvcc" ]; then
    if [ -n "${CUDA_HOME:-}" ]; then nvcc=$CUDA_HOME/bin/nvcc; else nvcc=$(command -v nvcc); fi
fi
if [ -z "$nvcc" ] || ! command -v "$nvcc" >/dev/null 2>&1; then
    echo "build.sh: nvcc not found; set NVCC or CUDA_HOME" >&2
    exit 2
fi

targets=$arch
for extra in ${CUDA_ARCH_LIST:-}; do
    targets="$targets $extra"
done
gencode=()
for target in $targets; do
    gencode+=(-gencode "arch=compute_${target},code=sm_${target}")
done

mkdir -p "$out"
"$nvcc" -O3 -std=c++17 "${gencode[@]}" \
    -shared -Xcompiler -fPIC -Xcompiler -Wall,-Wextra \
    -I"$here" "$here/candidate_scoring.cu" \
    -lcudart -o "$out/libscoring.so"

echo "build.sh: built $out/libscoring.so for sm_$arch (targets: $targets)"
