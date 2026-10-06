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
includes the CUDA Graph entry points, gated attention, the vision operations and
the continuation operations below.

Three operations continue a sequence after a shared prefix, for prefix reuse
([#85](https://github.com/ThinkFlowLab/system1-omni/issues/85)); the native
workers do not call them yet:

- `cs1_gdn_conv_history` reads the conv inputs of the three positions before its
  first token and can write those of its last three. Every output equals the
  unsplit conv's.
- `cs1_gdn_prefill_state` starts the chunked gated delta rule from a float32 state
  `[H, 128, 128]` and can write the final state. When every split falls on a multiple
  of 64 tokens, it reproduces the unsplit prefill bit for bit; elsewhere the chunks
  fall differently, within the float64-reference tolerance of the unsplit kernel.
- `cs1_attention_gated_cached` runs the queries of the last positions against keys
  and values that also cover the positions before them. Key tiles start at position
  0 either way, so each output row matches the unsplit call.

`cs1_copy_rows` queues a pitched device-to-device copy, for keeping cached values
in their own rows. The existing `cs1_gdn_conv`, `cs1_gdn_prefill` and
`cs1_attention_gated` are these operations without history, state or cached
positions. `tests/qwen3_5/kernels.rs` checks each against the unsplit call at
prefix lengths around and inside 64-token chunks.

Gated DeltaNet preparation stores converted TF32 operands in three-byte component planes, preserves the original four-term TF32 accumulation, and writes U/W fragments directly as bfloat16. Dynamic shared memory is 72 KiB per block. The [H200 comparison](../../../../benchmarks/gdn/README.md) records complete GDN call latency, numerical checks, and the small end-to-end change measured with the Open-Jev worker from PR #55.

The shared Rust model can pack independent sequences for input and gate/up GEMMs.
Output/down GEMMs retain each prompt's original shape and reduction order;
attention, convolution and GDN calls remain sequence-local. Packing adds no CUDA
entry points. Open-Jev uses this path within requests; Cua-S1 keeps single-prompt calls.
