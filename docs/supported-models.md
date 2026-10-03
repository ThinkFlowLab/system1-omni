# Supported models and hardware

This page covers what runs from `main`. Models that are being added are tracked in issues labeled [new model](https://github.com/ThinkFlowLab/system1-omni/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22new%20model%22).

| Model | Worker | CPU | NVIDIA CUDA | Apple Metal | Requirements |
| --- | --- | --- | --- | --- | --- |
| LAYA, English checkpoint | [External worker](../recipe/laya/README.md) running the upstream Laya runtime, for text requests | Validated ([#2](https://github.com/ThinkFlowLab/system1-omni/pull/2)) | Unverified ([#39](https://github.com/ThinkFlowLab/system1-omni/issues/39)) | Unverified ([#3](https://github.com/ThinkFlowLab/system1-omni/issues/3)) | Python 3.12, `laya[serve]==0.3.20` |
| LAYA | In-repository model engine | Planned ([#14](https://github.com/ThinkFlowLab/system1-omni/issues/14)) | Planned ([#14](https://github.com/ThinkFlowLab/system1-omni/issues/14)) | Planned ([#3](https://github.com/ThinkFlowLab/system1-omni/issues/3)) | |
| Cua-S1 4B 0.2, `text` adapter | [Reference worker](../recipe/cua_s1/text.md) on Transformers and PEFT | Unverified | Validated ([#13](https://github.com/ThinkFlowLab/system1-omni/pull/13)) | Unverified | Python 3.12, the versions in `requirements-text.txt` |
| Cua-S1 4B 0.2, `text` adapter | [Native Rust worker](../recipe/cua_s1/native.md) on the [Qwen3.5 CUDA kernels](../src/backends/cuda/qwen3_5/README.md) | Not supported | Validated on compute capability 8.9 ([#19](https://github.com/ThinkFlowLab/system1-omni/pull/19), [#52](https://github.com/ThinkFlowLab/system1-omni/pull/52)) | Not supported | Compute capability 8.0 or newer, the CUDA toolkit to build, weights merged with `export_text_merged.py` |
| Cua-S1 4B 0.2, `multimodal` adapter | Reference worker on Transformers and PEFT, [`src/frontend/cua_s1.py`](../src/frontend/cua_s1.py); no recipe yet | Not supported | Validated ([#17](https://github.com/ThinkFlowLab/system1-omni/pull/17), [#18](https://github.com/ThinkFlowLab/system1-omni/pull/18)) | Not supported | The state is one PNG or JPEG image; upstream's `weights.lock.json` next to the base weights |

- **Validated:** covered by the recipe on `main` or by the checks in the linked merged pull request.
- **Unverified:** the worker accepts this device, but no recipe or merged pull request covers it.
- **Planned:** not implemented yet; the linked issue tracks it.

The Cua-S1 workers answer `choice` questions only.
