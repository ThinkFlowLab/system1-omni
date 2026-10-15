#!/usr/bin/env bash
# Build libqwen3_5_cuda.so from the kernels in this directory.
#
#   src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, e.g. 89]
#
# Needs nvcc (from NVCC, CUDA_HOME/bin or PATH) and cuBLASLt; the compute capability
# defaults to CUDA_COMPUTE_CAP, else 89. The library holds machine code for that
# compute capability and PTX that newer GPUs can compile at load time. The CUDA
# runtime is linked statically and cuBLASLt dynamically, with an rpath to the
# toolkit's library directory when there is one next to nvcc.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: build.sh <output dir> [compute capability]}
arch=${2:-${CUDA_COMPUTE_CAP:-89}}
nvcc=${NVCC:-}
if [ -z "$nvcc" ]; then
    if [ -n "${CUDA_HOME:-}" ]; then nvcc=$CUDA_HOME/bin/nvcc; else nvcc=$(command -v nvcc); fi
fi
link=(-lcublasLt)
if lib=$(cd "$(dirname "$nvcc")/../lib64" 2>/dev/null && pwd); then
    link=(-L"$lib" -lcublasLt -Xlinker -rpath -Xlinker "$lib")
fi
extra=()
# Optional JEMM framework-compatible Conv3D. Legacy builds do not require cuDNN.
if [ -n "${CUDNN_INCLUDE_DIR:-}" ] || [ -n "${CUDNN_LIB_DIR:-}" ]; then
    : "${CUDNN_INCLUDE_DIR:?set CUDNN_INCLUDE_DIR and CUDNN_LIB_DIR together}"
    : "${CUDNN_LIB_DIR:?set CUDNN_INCLUDE_DIR and CUDNN_LIB_DIR together}"
    extra=(-DCS1_REFERENCE_CUDNN -I"$CUDNN_INCLUDE_DIR")
    link+=(-L"$CUDNN_LIB_DIR" -l:libcudnn.so.9 -Xlinker -rpath -Xlinker "$CUDNN_LIB_DIR")
fi
mkdir -p "$out"
"$nvcc" -O3 -std=c++17 -gencode "arch=compute_${arch},code=[sm_${arch},compute_${arch}]" \
    -shared -Xcompiler -fPIC -Xcompiler -Wall,-Wextra -I"$here" "$here"/*.cu \
    "${extra[@]}" "${link[@]}" -o "$out/libqwen3_5_cuda.so"
echo "built $out/libqwen3_5_cuda.so for sm_${arch}"
