# Qwen3.5/3.8 prefill operations

CUDA kernels for a prefill-only Qwen3.5/3.8 forward pass, built into `libqwen3_5_cuda.so` with a C interface ([`ops.h`](ops.h)), so that a Rust model engine loads it at run time and builds without a CUDA toolkit. The Cua-S1 and Open-Jev native workers share the layer loop and buffers in [`src/models/qwen3_5/native/`](../../../models/qwen3_5/native/).

```sh
src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, default 89]
```

The norm, elementwise and q/k preparation kernels round to bfloat16 where Transformers (`modeling_qwen3_5.py`) does. Attention (FlashAttention-2 style, on tensor cores) and the chunked gated delta rule keep some intermediate results in bfloat16, as FlashAttention and flash-linear-attention do. GEMMs go through cuBLASLt with its first heuristic choice. Tensor-core kernels need sm_80 or newer; PR #19 validated the original kernels on sm_89. The current reference tests, including fused gating, cached residual RMSNorm and packed SiLU, passed on H200 (sm_90). Compilation passed for sm_80, sm_89 and sm_90; execution of the modified GDN kernel on sm_80/sm_89 remains unverified.

`cs1_attention_gated` fuses the sigmoid gate into the attention epilogue, preserving
the BF16 rounding of both attention and sigmoid before multiplication. The native
workers use this entry point; the separate operations remain available for kernel
comparisons. Rebuild the library and workers together for ABI version 7, which
includes vision, CUDA Graph, gated attention, prefix-continuation entry points and
the fixed-algorithm GEMM handle below.

cuBLASLt's heuristic picks a GEMM algorithm per M, so a row's result can change with
the number of rows in the call: a prefix and its branch, run separately, round
differently from the same tokens in one pass. `cs1_gemm_create_fixed` returns a handle
that keeps one algorithm per weight shape for every M, the heuristic's first choice at a
reference M among algorithms without split-K, so each row's result is the same whatever
M is and wherever the row sits. cuBLASLt doesn't document this property;
`fixed_gemm_rows_do_not_depend_on_m` checks it, so run it on a new GPU before relying
on it (it passed on sm_89). It is slower for some shapes and faster for others: on an
RTX 6000 Ada, the down and output projections take up to about 4 times as long at small
M without split-K. `cs1_gemm_create` remains the default.

Gated DeltaNet preparation stores converted TF32 operands in three-byte component planes, preserves the original four-term TF32 accumulation, and writes U/W fragments directly as bfloat16. Dynamic shared memory is 72 KiB per block. The [H200 comparison](../../../../benchmarks/gdn/README.md) records complete GDN call latency, numerical checks, and the small end-to-end change measured with the Open-Jev worker from PR #55.

The shared Rust model can pack independent sequences for input and gate/up GEMMs.
Output/down GEMMs retain each prompt's original shape and reduction order;
attention, convolution and GDN calls remain sequence-local. Packing adds no CUDA
entry points. Open-Jev uses this path within requests; Cua-S1 keeps single-prompt calls.

Prefix continuation retains full-attention KV, three convolution input rows,
and FP32 Gated DeltaNet state at 64-token chunk boundaries. It is used by the
experimental JEV-VL worker and by the shared executor's `forward_shared`, which
runs prompts that share prefixes once per prefix with the fixed-algorithm GEMM
handle ([#85](https://github.com/ThinkFlowLab/system1-omni/issues/85)); existing
full-prefill callers keep their own paths.
ABI 6 adds mandatory copy and prefix entry points after ABI 5 native vision,
and ABI 7 adds the fixed-algorithm GEMM handle.
Rebuild the library and **all** Qwen worker binaries together; older libraries
are rejected. Current-branch GPU regression is required before release.
