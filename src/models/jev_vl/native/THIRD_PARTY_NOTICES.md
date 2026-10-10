# JEV-27B-VL adaptation notices

This worker and its preparation scripts use the following pinned references.
The applicable license text is retained as
[Apache License, Version 2.0](licenses/APACHE-2.0).

- **autotrust/JEV-27B-VL**, revision
  `f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc`:
  [`serve_decide.py`](https://huggingface.co/autotrust/JEV-27B-VL/blob/f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc/serve_decide.py).
  `src/contract.rs` adapts its state rendering, raw System-1 prompt and decision
  response semantics to Rust. `recipe/jev_vl/export_merged.py` reproduces its
  single-token option-label scan. The Rust worker supports a restricted
  single-question, System-1-only interface and does not include the upstream
  server, adaptive thinking or multi-pass strategies. The pinned
  [model card](https://huggingface.co/autotrust/JEV-27B-VL/blob/f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc/README.md)
  declares Apache-2.0 for the release; the included license text comes from its
  `LICENSE` file. No weights are redistributed with this integration.
- **Transformers 5.17.0**:
  [`modeling_qwen3_5.py`](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py),
  Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
  `src/images.rs` adapts the multimodal position-index calculation to Rust,
  preencoded image rows and bounded grid dimensions.
  [`vision_utils.py`](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/vision_utils.py),
  Copyright 2026 The HuggingFace Inc. team. All rights reserved, is the related
  vision-position reference. These sources are licensed under Apache-2.0.

The offline preencoder invokes Transformers, PyTorch and Pillow as installed
dependencies. It does not vendor their vision model or image decoder code.
Existing shared Cua-S1 native vision adaptations retain their separate
[notices](../../cua_s1/native/THIRD_PARTY_NOTICES.md).
