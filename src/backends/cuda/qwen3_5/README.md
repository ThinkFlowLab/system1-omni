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

## Optional JEMM reference arithmetic

JEMM enables the additive `cs1_reference_*` API group after model load and before
any forward. It requires cuDNN 9; build with `CUDNN_INCLUDE_DIR` and
`CUDNN_LIB_DIR`. Builds without those variables retain the current ABI-8 functions;
JEMM reports a missing-capability error rather than silently selecting older numerics.

The reference path uses a per-handle GEMM reduction policy, separate normalization,
attention and GDN functions, and shape-owned cuDNN Conv3D plans. Attention and
convolution workspaces allocate before capture and retire with their model/shape.
The rotary entry point consumes FP32 angles and computes trigonometry on device;
legacy rotary functions still consume cos/sin tables. Prefix caching is unsupported
on this opt-in path. The shared workers' default arithmetic remains unchanged.

The [JEMM numerical record](../../../../docs/benchmarks/jemm-reference-20261011/README.md)
contains a frozen 1–4-image corpus and reproducible operator/HTTP checks.

## Optional Cua-S1 no-FLA text arithmetic

The unmerged Cua-S1 text export reuses the reference norm, attention and GEMM
operations without requiring cuDNN. Norm additionally supports hidden size 2560.
The additive `cs1_gdn_fp32_*` group owns a cuBLAS handle and 32 MiB internal
workspace; the caller supplies shape-dependent FP32 buffers. It implements
64-token chunked GDN, FP32 normalization, triangular solves and recurrent state,
Torch 2.14 cumulative-decay association, and a final BF16 output cast. Each call
starts with zero state; prefix continuation is rejected by the model adapter.
The context drains its last stream before destruction, and is retired before
that stream. Rebuild the library and all consuming workers together for the
current ABI 8, including this optional group. The independent A800 qualification
at `27f77e49868d84b091aa804f02ccb4d3474a9a00` used ABI 6; the earlier cumulative
campaign used ABI 8. Neither campaign qualifies the current integrated source.
Builds now link both cuBLAS and cuBLASLt. Every consumer of the rebuilt shared
library must provide both runtime libraries (`libcublas.so` and
`libcublasLt.so`), even when it does not enable the optional FP32 GDN group.
An existing cuBLASLt-only deployment must add cuBLAS before upgrading.

`tests/qwen3_5/test_gdn_fp32.py` compares against the pinned Transformers 5.17
no-FLA implementation. On the cumulative A800 sm_80/CUDA 13.0/Torch 2.14 build, all 14 cases passed
with identical BF16 outputs, covering chunk boundaries, grouped heads,
changed-input replay/state reset, invalid dimensions/scales and context scratch
release. The fixed relative-L2 gate is 1e-4. This does not qualify other GPUs or
Torch versions; the original approximate GDN entry points remain available.

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
