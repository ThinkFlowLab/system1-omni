# Qwen3.5/3.8 prefill operations

CUDA kernels for a prefill-only Qwen3.5/3.8 forward pass, built into `libqwen3_5_cuda.so` with a C interface ([`ops.h`](ops.h)), so that a Rust model engine loads it at run time and builds without a CUDA toolkit. The Cua-S1 and Open-Jev native workers share the layer loop and buffers in [`src/models/qwen3_5/native/`](../../../models/qwen3_5/native/).

```sh
src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, default 89]
```

The norm, elementwise and q/k preparation kernels round to bfloat16 where Transformers (`modeling_qwen3_5.py`) does. Attention (FlashAttention-2 style, on tensor cores) and the chunked gated delta rule keep some intermediate results in bfloat16, as FlashAttention and flash-linear-attention do. GEMMs go through cuBLASLt with its first heuristic choice. Tensor-core kernels need sm_80 or newer; PR #19 validated the original kernels on sm_89. The current reference tests, including fused gating, cached residual RMSNorm and packed SiLU, passed on H200 (sm_90). Compilation passed for sm_80, sm_89 and sm_90; execution of the modified GDN kernel on sm_80/sm_89 remains unverified.

`cs1_attention_gated` fuses the sigmoid gate into the attention epilogue, preserving
the BF16 rounding of both attention and sigmoid before multiplication. The native
workers use this entry point; the separate operations remain available for kernel
comparisons. Rebuild the library and workers together for ABI version 6, which
includes vision, CUDA Graph, gated attention and prefix-continuation entry points.

Gated DeltaNet preparation stores converted TF32 operands in three-byte component planes, preserves the original four-term TF32 accumulation, and writes U/W fragments directly as bfloat16. Dynamic shared memory is 72 KiB per block. The [H200 comparison](../../../../benchmarks/gdn/README.md) records complete GDN call latency, numerical checks, and the small end-to-end change measured with the Open-Jev worker from PR #55.

The shared Rust model can pack independent sequences for input and gate/up GEMMs.
Output/down GEMMs retain each prompt's original shape and reduction order;
attention, convolution and GDN calls remain sequence-local. Packing adds no CUDA
entry points. Open-Jev uses this path within requests; Cua-S1 keeps single-prompt calls.

Prefix continuation retains full-attention KV, three convolution input rows,
and FP32 Gated DeltaNet state at 64-token chunk boundaries. It is used by the
experimental JEV-VL worker; existing full-prefill callers keep their own paths.
ABI 6 adds mandatory copy and prefix entry points after ABI 5 native vision.
Rebuild the library and **all** Qwen worker binaries together; older libraries
are rejected. Current-branch GPU regression is required before release.

## Optional JEMM reference arithmetic

JEMM enables the additive `cs1_reference_*` API group after model load and before
any forward. It requires cuDNN 9; build with `CUDNN_INCLUDE_DIR` and
`CUDNN_LIB_DIR`. Builds without those variables retain the legacy ABI-6 functions;
JEMM reports a missing-capability error rather than silently selecting older numerics.

The reference path uses a per-handle GEMM reduction policy, separate normalization,
attention and GDN functions, and shape-owned cuDNN Conv3D plans. Attention and
convolution workspaces allocate before capture and retire with their model/shape.
The rotary entry point consumes FP32 angles and computes trigonometry on device;
legacy rotary functions still consume cos/sin tables. Prefix caching is unsupported
on this opt-in path. The shared workers' default arithmetic remains unchanged.

The [JEMM numerical record](../../../../docs/benchmarks/jemm-reference-20261011/README.md)
contains a frozen 1–4-image corpus and reproducible operator/HTTP checks.
