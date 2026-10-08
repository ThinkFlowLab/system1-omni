# Supported models and hardware

This page covers what runs from `main`. Start with the
[CPU decision walkthrough](getting-started.md). The
[rolling model tracker (#83)](https://github.com/ThinkFlowLab/system1-omni/issues/83)
records available paths separately from proposed integrations and hardware evidence.
Models that are being added are also tracked in issues labeled [new model](https://github.com/ThinkFlowLab/system1-omni/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22new%20model%22).

| Model | Worker | CPU | NVIDIA CUDA | Apple GPU | Requirements |
| --- | --- | --- | --- | --- | --- |
| LAYA, English checkpoint | [External worker](../recipe/laya/README.md) running the upstream Laya runtime, for text requests | Validated ([#2](https://github.com/ThinkFlowLab/system1-omni/pull/2)); [CPU demo](../recipe/laya/validation.md) | Unverified ([#39](https://github.com/ThinkFlowLab/system1-omni/issues/39)) | Use the dedicated MPS worker below | Python 3.12, [pinned CPU dependencies](../recipe/laya/requirements-cpu.txt) |
| LAYA, English checkpoint | [Python MPS/CPU worker](../recipe/laya/apple-silicon.md) in this repository | Contract checks documented ([#67](https://github.com/ThinkFlowLab/system1-omni/pull/67)) | No validation recorded for this path | **PyTorch MPS validated** on M1 Pro; M4/M5 checks documented ([#30](https://github.com/ThinkFlowLab/system1-omni/pull/30), [#67](https://github.com/ThinkFlowLab/system1-omni/pull/67)) | Python 3.12, [MPS dependency versions](../recipe/laya/requirements-mps.txt); native Metal execution remains planned |
| LAYA, English checkpoint | [Native Rust worker](../recipe/laya/native/README.md) | Processing/checkpoint checks; no CPU inference | Hopper `sm_90a`; [validation scope](../recipe/laya/native/VALIDATION.md) | Not supported | CUDA toolkit, TileLang for AOT generation, pinned checkpoint and rotary tables |
| Cua-S1 4B 0.2, `text` adapter | [Reference worker](../recipe/cua_s1/text.md) on Transformers and PEFT | Unverified | Validated ([#13](https://github.com/ThinkFlowLab/system1-omni/pull/13)) | Unverified | Python 3.12, the versions in `requirements-text.txt` |
| Cua-S1 4B 0.2, `text` adapter | [Native Rust worker](../recipe/cua_s1/native.md) on the [Qwen3.5 CUDA kernels](../src/backends/cuda/qwen3_5/README.md) | Not supported | Validated on compute capability 8.9 ([#19](https://github.com/ThinkFlowLab/system1-omni/pull/19), [#52](https://github.com/ThinkFlowLab/system1-omni/pull/52)) | Not supported | Compute capability 8.0 or newer, the CUDA toolkit to build, weights merged with `export_text_merged.py` |
| Cua-S1 4B 0.2, `multimodal` adapter | Reference worker on Transformers and PEFT, [`src/frontend/cua_s1.py`](../src/frontend/cua_s1.py); no recipe yet | Not supported | Validated ([#17](https://github.com/ThinkFlowLab/system1-omni/pull/17), [#18](https://github.com/ThinkFlowLab/system1-omni/pull/18)) | Not supported | The state is one PNG or JPEG image; upstream's `weights.lock.json` next to the base weights |
| Valen-Preview-0923 | [Python reference worker](../recipe/valen/reference.md); text state or one inline PNG/JPEG image; `choice` questions | Not supported | Validated on RTX 5060 Laptop GPU (sm_120) for text and image state, with recorded reference parity for both | Not supported | Python 3.12, pinned Valen source/checkpoint, Qwen3.5-2B base, BF16 CUDA |
| Open-Jev-27B-v1.1 | [Native Rust/CUDA worker](../recipe/open_jev/native.md) on the shared Qwen3.5/3.8 executor | Not supported | Validated on H200 (sm_90) for the [74 single-candidate workload](../recipe/open_jev/validation.md) | Not supported | Compute capability 8.0 or newer, CUDA toolkit to build, exported merged weights and trained head |
| Open-Jev-9B | The same [native Rust/CUDA worker](../recipe/open_jev/native.md) | Not supported | Validated on compute capability 8.9 against the reference for [253 requests](../recipe/open_jev/validation-9b.md) | Not supported | Compute capability 8.0 or newer, CUDA toolkit to build, exported merged weights and trained head |
| CLM-v0.1-8B | [External `clm-serve` recipe](../recipe/clm/README.md) with a CPU stub embeddings server | **Stub-encoder contract checks only** ([#23](https://github.com/ThinkFlowLab/system1-omni/pull/23)); not real Qwen3-8B decisions | Real encoder unverified by the merged recipe | Unverified | Python, upstream CLM and head checkpoint; a real encoder requires a separate embeddings server |

- **Validated:** covered by the recipe on `main` or by the checks in the linked merged pull request.
- **Unverified:** the worker accepts this device, but no recipe or merged pull request covers it.
- **Planned:** not implemented yet; the linked issue tracks it.

The Cua-S1 workers answer `choice` questions only.
LAYA's English worker and Open-Jev support `choice`, `score`, and `noul` text questions.
CLM's merged recipe exercises these answer shapes with stub embeddings; it does
not validate decision quality. MPS validation above is for a Python/PyTorch
worker, not a native Metal backend.

The [architecture contracts](architecture.md) describe the native target.
Shared processing orchestration and dynamic batching remain planned. Native
workers reuse serial admission. Qwen workers run independent single-prompt
prefills with CPU heads; Laya packs questions within one request and runs its
scorer/action head on CUDA. These target layers do not expand
the validated model or hardware coverage above.
