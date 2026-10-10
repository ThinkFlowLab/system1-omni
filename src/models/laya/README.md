# Laya English native executor

The `omni-laya` crate supports English Laya 0.3.20 text requests with Rust host
processing and a Hopper CUDA executor. The Python worker below remains available.
The pinned checkpoint is `convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.

`processing::Processor` validates requests and retains response context.
`preprocess` preserves token/option order; `packing` pads questions to a
power-of-two row count and sequence lengths to multiples of 16. The executor
accepts up to 16 questions, 512 tokens per sequence and 2048 option markers.
Input-token usage excludes padding. `choice`, `score` and `noul` answers use the
checkpoint temperatures, first-option ties and four-decimal output rounding.
Only `model=english` and English language selectors are accepted; there is no
automatic language routing or image/audio/video path.

The model owns all 206 checkpoint tensors, 28 ModernBERT encoder layers, two
decision transformer layers, the scorer and action head. Residuals/norm weights
are FP32; token embeddings are FP16, and projection weights and GPU activations are BF16. Graphs capture only
Encoder/Decision and are cached by padded shape (four entries, 512 MiB workspace
budget). Scorer/action head and copyback run outside Graph. Complete-Graph and
continuous batching are not implemented.

The native worker reuses `omni-runtime::SerialScheduler` for one complete request
per admission. A dedicated thread owns CUDA's non-Send state; a rendezvous channel
transports admitted work without another pending queue. Cancellation after
dispatch retains admission and model resources until device work and readback
finish. CUDA failure stops this executor; `/health` returns 503. A real warmup
finishes before the HTTP listener binds.

`Config::load` checks architecture and temperatures; `Weights` preserves the
main-branch inventory and conversion checks. Keep checkpoint files immutable
while mapped. Bundle/table/checkpoint hashes are checked before CUDA loading.

See the [CUDA setup and worker recipe](../../../recipe/laya/native/README.md)
and [measured scope](../../../recipe/laya/native/VALIDATION.md).

## CPU checks

The normal workspace tests cover configuration errors, malformed tensors, inventory mismatches and conversion boundaries without downloading weights.
Feature-enabled CI also runs [CPU host/ABI regressions](../../../tests/cuda/laya/README.md)
for dispatch, staging, graph lifetimes and cache admission. They compile a small
native fixture with `cc` on Linux/macOS and require no CUDA toolkit or weights;
they do not establish GPU numerical parity.

To check the complete checkpoint, use `convaiinnovations/laya` revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` and a Python environment with PyTorch, safetensors and NumPy:

```sh
export LAYA_CHECKPOINT=/path/to/laya/snapshot
export LAYA_WEIGHT_ORACLE=/tmp/laya-weight-oracle.json
python recipe/laya/native/export_weights.py "$LAYA_CHECKPOINT" "$LAYA_WEIGHT_ORACLE"
cargo test --release --locked -p omni-laya --test weights -- --ignored
```

These two CPU tests check all 206 tensor names and shapes, 618 conversion hashes, and the legacy temperature buffer. The normal CI job skips them because it does not download the full checkpoint.

## Python worker

The Python worker serves LAYA through laya-serve on CPU and Apple Silicon (PyTorch MPS,
validated on an M1 Pro and, by another contributor, an M5). The native CUDA worker is separate; native Metal remains unimplemented.

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
