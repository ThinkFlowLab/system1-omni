# Image preprocessing attributions

`src/image_preprocess.rs` is a Rust adaptation of the following algorithms.
It is modified for decoded interleaved RGB8 input, fixed Cua-S1 4B settings,
bounded dimensions, standard-library buffers, and a standalone CPU API.

- PyTorch 2.14.0, `aten/src/ATen/native/cpu/UpSampleKernel.cpp`
  (`_compute_indices_min_size_weights_aa`, `_compute_index_ranges_int16_weights`,
  and the separable uint8 horizontal/vertical loops), plus the cubic polynomial
  helpers in `aten/src/ATen/native/UpSample.h`. See [PyTorch license](licenses/PYTORCH-LICENSE)
  for the retained copyright notices, redistribution conditions, and disclaimer.
- PyTorch's bicubic filter credits Pillow's `src/libImaging/Resample.c`.
  The retained PIL/Pillow notice is in [Pillow license](licenses/PILLOW-LICENSE).
- Transformers 5.17.0,
  `src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py` (smart resize
  and patch ordering) and `src/transformers/image_processing_backends.py`
  (fused normalization). Copyright 2024 The Qwen team, Alibaba Group and the
  HuggingFace Inc. team. All rights reserved. The backend file is
  Copyright 2025 The HuggingFace Inc. team. Licensed under the
  [Apache License, Version 2.0](licenses/APACHE-2.0).

No upstream runtime or image decoder is linked by this module. The above
notices and license texts must accompany redistributed adaptations as required
by their respective licenses.
