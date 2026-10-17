# Qwen3.5/3.8 prefill operations

CUDA kernels for a prefill-only Qwen3.5/3.8 forward pass, built into `libqwen3_5_cuda.so` with a C interface ([`ops.h`](ops.h)), so that a Rust model engine loads it at run time and builds without a CUDA toolkit. The Cua-S1, Open-Jev and experimental JEV-VL native workers share the layer loop and buffers in [`src/models/qwen3_5/native/`](../../../models/qwen3_5/native/).

The backend manifest remains `experimental`. Maintained CI checks source/build
metadata and CPU tests; it does not compile this backend with NVCC or establish
GPU numerical parity. Current-head CUDA compilation and GPU regressions remain
separate release checks.

```sh
src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, default 89]
```

The norm, elementwise and q/k preparation kernels round to bfloat16 where Transformers (`modeling_qwen3_5.py`) does. Attention (FlashAttention-2 style, on tensor cores) and the chunked gated delta rule keep some intermediate results in bfloat16, as FlashAttention and flash-linear-attention do. GEMMs go through cuBLASLt with its first heuristic choice. Tensor-core kernels need sm_80 or newer; PR #19 validated the original kernels on sm_89. The current reference tests, including fused gating, cached residual RMSNorm and packed SiLU, passed on H200 (sm_90). Compilation passed for sm_80, sm_89 and sm_90; execution of the modified GDN kernel on sm_80/sm_89 remains unverified.

`cs1_attention_gated` fuses the sigmoid gate into the attention epilogue, preserving
the BF16 rounding of both attention and sigmoid before multiplication. The native
workers use this entry point; the separate operations remain available for kernel
comparisons. Rebuild the library and all native Rust consumers together for
**ABI version 8**, even with resident features disabled. ABI 8 retains the vision,
CUDA Graph, gated attention and prefix-continuation entry points and adds
`cs1_get_device` and `cs1_bf16_finite` for device feature validation. Older
libraries are rejected at load. See the
[Cua-S1 resident-feature recipe](../../../../recipe/cua_s1/native_vision_graph.md#gpu-resident-vision-features)
for the opt-in path, lifetime contract and regression commands.

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

## Shared vision cache and operator checks

The generic shared vision constructor keeps a four-entry exact-grid FIFO for
compatibility. Its cap counts entries, not device bytes; four large grids may
retain substantially more memory than four small grids. A borrowed resident
output prevents mutation or eviction until its consumer is finished.

[Cua-S1's bounded-cache integration in PR #127](https://github.com/ThinkFlowLab/system1-omni/pull/127)
uses a separate explicit policy: count/byte LRU limits, default one entry and
256 MiB, with zero retention available. The Cua wrapper owns environment parsing;
generic Qwen/JEMM users retain their existing policy. Retention-disabled or
oversized Cua output stays model-owned through both producer and consumer stream
drains. Graph enablement and scratch retention are separate policies: turning
Graph off alone does not release retained scratch.

The weight-free 27B operator regressions directly check 1152-wide positional
interpolation, head-72 rotary pairing and noncausal attention, including strided
V, non-tile-aligned row counts, changed inputs and poisoned padding workspace:

```sh
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-qwen3-5-native --test kernels \
  vision_v2:: -- --ignored --test-threads=1
```

Position/rotary assert exact BF16 readouts on dyadic inputs. Attention compares
against a host float64 softmax reference with a fixed 0.004 absolute gate. These
are explicit device tests; ordinary CPU CI compiles and skips them. Their addition
does not establish a fresh device pass or whole-model 27B qualification.

Shared preprocessing and vision adaptation notices accompany the Qwen crate in
[THIRD_PARTY_NOTICES.md](../../../models/qwen3_5/native/THIRD_PARTY_NOTICES.md).
