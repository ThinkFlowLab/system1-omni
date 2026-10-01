# Qwen3.5 prefill operations

CUDA kernels for a prefill-only Qwen3.5 forward pass, built into `libqwen3_5_cuda.so` with a C interface ([`ops.h`](ops.h)), so that a Rust model engine loads it at run time and builds without a CUDA toolkit. The Cua-S1 native worker ([`src/models/cua_s1/native/`](../../../models/cua_s1/native/)) uses it and keeps the layer loop and buffers.

```sh
src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, default 89]
```

The norm, elementwise and q/k preparation kernels round to bfloat16 where Transformers (`modeling_qwen3_5.py`) does. Attention (FlashAttention-2 style, on tensor cores) and the chunked gated delta rule keep some intermediate results in bfloat16, as FlashAttention and flash-linear-attention do. GEMMs go through cuBLASLt with its first heuristic choice. Tensor-core kernels need sm_80 or newer; only sm_89 has been run.
