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

## Inference workspace

`Workspace::new(&cuda, batch, sequence)` allocates fixed scratch buffers for one
shape. Batch must be 1, 2, 4, 8 or 16; sequence must be a multiple of 16 in 16..=512.
Invalid shapes fail before any allocation; a failed allocation releases the partial
workspace. Contents are uninitialized and must be written before use.

`buffers()` borrows the named buffers without allowing allocations to be replaced.
The workspace outlives the caller's `Cuda` handle. `bytes()` reports scratch
allocations only, excluding resident weights and CUDA overhead.

Let `B` be batch, `L` sequence, `D=1024`, and `M=MAX_MARKERS=2048`. Layouts are:

| Buffers | Shape and dtype |
| --- | --- |
| ids / lengths / types | `[B,L]` int64 / `[B]` int32 / `[B]` int64 |
| residual / hidden / attention | `[B,L,D]` FP32 / BF16 / BF16 |
| qkv / gated / feed_forward | `[B,L,3D]` / `[B,L,2624]` / `[B,L,4096]`, BF16 |
| indices / offsets | `[M]` / `[B+1]`, int32 |
| markers / scored / logits | `[M,D]` / `[M,D]` / `[M]`, BF16 |
| features / action_hidden / actions | `[B,1028]` / `[B,256]` / `[B,2]`, BF16 |

At `(B,L)=(1,512)`, allocations total 22,628,896 bytes; at `(16,512)`,
236,048,836 bytes. The caller must enforce at most `MAX_MARKERS` scored positions.
No Graph cache, kernel launch, cuBLAS workspace or inference is included.

The ignored `real_gpu_workspace_capacity_and_reuse` test requires
`LAYA_CUDA_LIBRARY` and `LAYA_CUDA_DEVICE`. It writes and reads all 17 buffers twice
at three shapes, including both capacity bounds, using deterministic byte patterns.
This tests allocation and transfer, not model numerics or latency.

## Python worker

The Python worker serves LAYA through laya-serve on CPU and Apple Silicon (PyTorch MPS,
validated on an M1 Pro and, by another contributor, an M5). No native CUDA or Metal backend yet.

- [`src/frontend/laya_mps.py`](../../frontend/laya_mps.py): the HTTP worker. laya-serve (`laya[serve]==0.3.20`)
  with its request handling unchanged, started as `PYTHONPATH=src python -m frontend.laya_mps --device mps`.
- `engine.py`: what the worker runs before readiness (a warmup of every loaded model over short, long and
  multi-question requests) and what `/health` reports about a loaded model, read on every call: device,
  weight and autocast dtypes, checkpoint and the revision the weights were loaded from, `device_mismatch`.
- `optimize.py`: the two GPU options. `--compile` compiles one-question requests end to end and, for several
  questions, only the encoder (Laya's decision head is slower compiled on MPS). `--weights fp16` keeps the
  checkpoint's fp16 weights instead of Laya's fp32 upcast (`act_head` stays fp32). Both apply on the GPU
  only: on the CPU, including after Laya falls back to it on a GPU out-of-memory error, the worker runs
  Laya's fp32 model uncompiled.
- Tests: [`tests/laya/`](../../../tests/laya/). The unit tests use a fake router; `LAYA_CONTRACT=1` adds
  contract tests against a real worker on the CPU.

What the worker changes against laya-serve, with measurements, is in the
[Apple Silicon recipe](../../../recipe/laya/apple-silicon.md): it binds only after the warmup (laya-serve
answers `/health` before any forward pass, so its first request took 0.7–1.1 s against 70–81 ms), and
`/health` tells the device the model is actually on (laya-serve reports the configured one; Laya falls
back to the CPU with only a printed warning). `--require-device` exits at startup if a model is not on
the requested device.

```sh
PYTHONPATH=src python -m frontend.laya_mps --device mps --model english
PYTHONPATH=src python -m pytest tests/laya                     # unit tests, no model
LAYA_CONTRACT=1 PYTHONPATH=src python -m pytest tests/laya     # plus contract tests on CPU, loads the checkpoint
```
