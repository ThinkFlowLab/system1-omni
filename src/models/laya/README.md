# LAYA model engine

LAYA is the first planned System1-Omni model. This directory owns its complete request-to-result path: preprocessing, postprocessing, batching policy, state, execution, and backend-specific kernel selection.

GPU operations and kernel implementations belong in [`backends/cuda/`](../../backends/cuda/) and [`backends/metal/`](../../backends/metal/). Setup and usage examples belong in the top-level [`recipe/`](../../../recipe/) directory.

The `omni-laya` crate currently reads and checks the English Laya 0.3.20 checkpoint. `Config::load` validates the architecture and temperatures; `Weights` checks tensor names and shapes and converts FP32, FP16 and BF16 values. `checkpoint_tensors()` lists the 206 expected tensors. Each backend chooses its own storage precision.

Keep checkpoint files unchanged while `Weights` holds a read-only memory mapping. This crate does not yet execute inference.

## CPU checks

The normal workspace tests cover configuration errors, malformed tensors, inventory mismatches and conversion boundaries without downloading weights.

To check the complete checkpoint, use `convaiinnovations/laya` revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` and a Python environment with PyTorch, safetensors and NumPy:

```sh
export LAYA_CHECKPOINT=/path/to/laya/snapshot
export LAYA_WEIGHT_ORACLE=/tmp/laya-weight-oracle.json
python recipe/laya/native/export_weights.py "$LAYA_CHECKPOINT" "$LAYA_WEIGHT_ORACLE"
cargo test --release --locked -p omni-laya --test weights -- --ignored
```

These two CPU tests check all 206 tensor names and shapes, 618 conversion hashes, and the legacy temperature buffer. The normal CI job skips them because it does not download the full checkpoint.

## GPU weight residency

`ResidentWeights::upload(&cuda, &weights)` validates the checkpoint inventory and
uploads the weights once. Embeddings use FP16; encoder norms, head norms and biases,
and the scorer input norm use FP32; other weights use BF16. Layouts stay unchanged.
The legacy `temperature` buffer is validated but not uploaded.

`get(name)` returns the resident buffer. `bytes()` reports weight allocations only,
excluding CUDA context and allocator overhead. Buffers keep their CUDA context alive
after the caller drops the source mapping or `Cuda`. Failed loads release partial
allocations. Workspace, rotary tables and inference are separate modules.

The ignored `real_checkpoint_residency_matches_torch` test requires
`LAYA_CUDA_LIBRARY`, `LAYA_CUDA_DEVICE`, `LAYA_CHECKPOINT` and `LAYA_WEIGHT_ORACLE`.
It uploads all 205 used tensors and compares readback hashes with the existing Torch
oracle; it does not test model outputs or latency.
