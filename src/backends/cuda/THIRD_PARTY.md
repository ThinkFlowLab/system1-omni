# Sources

- `kernels/laya_tilelang.py`: Laya 0.3.20 `tl_kernels.py`, Apache-2.0; see `licenses/Laya.txt`. CUDA export preserves its arithmetic; the emitted Attention adds an unconditional wait before shared-memory reuse on empty key ranges.
- `kernels/model_ops.cu`: Welford reduction and normalization order adapted from [PyTorch 2.11 CUDA LayerNorm](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu), BSD-3-Clause; see `licenses/PyTorch.txt`. Compiled without fast math, matching that reference.
- AOT host-stub inspection follows the approach in [PegaInfer](https://github.com/pegainfer-project/pegainfer), commit b2efe52726cda0ae9c460e56f398fbe7c1b2b584. No runtime dependency on PegaInfer or TileLang.
