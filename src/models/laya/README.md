# LAYA model engine

LAYA is the first planned System1-Omni model. This directory owns its complete request-to-result path: preprocessing, postprocessing, batching policy, state, execution, and backend-specific kernel selection.

GPU operations and kernel implementations belong in [`backends/cuda/`](../../backends/cuda/) and [`backends/metal/`](../../backends/metal/). Setup and usage examples belong in the top-level [`recipe/`](../../../recipe/) directory.

The `omni-laya` crate currently reads and checks the English Laya 0.3.20 checkpoint. `Config::load` validates the architecture and temperatures; `Weights` checks tensor names and shapes and converts FP32, FP16 and BF16 values. `checkpoint_tensors()` lists the 206 expected tensors. Each backend chooses its own storage precision.

Keep checkpoint files unchanged while `Weights` holds a read-only memory mapping. This crate does not yet execute inference.

`Preprocessor::load` reads a tokenizer JSON file. `prepare` packs English `choice`, `score` and `noul` questions into ordered token rows, option-marker positions and type IDs. Rows follow Laya 0.3.20's 512-token limit and 192-token head budget. Conversation lists keep the newest state tokens; other state values keep the beginning. The result includes normalized criteria for later decoding and the total input-token usage. Backends own padding, batching and resource limits.

`decision::decode` converts raw option and action logits into answers, probabilities and confidence. Rows follow `Prepared.questions`; option logits follow marker order, and action rows contain two logits with "act" first. It applies the configured temperatures and preserves question and option order. It rejects inconsistent rows, invalid metadata, non-finite values and temperature-scaling overflow. Callers retain `Prepared.usage` and add routing and HTTP response fields.

## CPU checks

The normal workspace tests cover configuration errors, malformed tensors, inventory mismatches and conversion boundaries without downloading weights.

They also compare decoding against a checked-in 16-case reference from Laya 0.3.20, PyTorch 2.14.0 and NumPy 2.5.3. Those rounded answers must match exactly. The cases cover all three question types, temperature buckets and clamps, ties, single options, ordering and rounding boundaries. A separate 16-option probe covers FP32 reduction differences: the first probability is 0.0100 in Rust and 0.0101 in NumPy. That probe allows one displayed decimal unit for probabilities and requires all other fields to match. This is not a universal error bound; scorer integration must check real model outputs. No Python environment is needed to run the Rust tests. To regenerate the reference with those Python package versions:

```sh
python recipe/laya/native/export_decisions.py /path/to/laya/snapshot /tmp/laya-decisions.json
```

The generator reads only `rl_agent_config.json` and calls Laya's official CPU decoding functions. It records source hashes and package versions in the output. These fixed logits test decoding, not model quality or GPU execution.

To check the complete checkpoint, use `convaiinnovations/laya` revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` and a Python environment with PyTorch, safetensors and NumPy:

```sh
export LAYA_CHECKPOINT=/path/to/laya/snapshot
export LAYA_WEIGHT_ORACLE=/tmp/laya-weight-oracle.json
python recipe/laya/native/export_weights.py "$LAYA_CHECKPOINT" "$LAYA_WEIGHT_ORACLE"
cargo test --release --locked -p omni-laya --test weights -- --ignored
```

These two CPU tests check all 206 tensor names and shapes, 618 conversion hashes, and the legacy temperature buffer. The normal CI job skips them because it does not download the full checkpoint.

The normal tests also check input validation, question and option order, truncation and JSON rendering with a small test tokenizer. CPU CI separately downloads the [official tokenizer](https://huggingface.co/convaiinnovations/laya/blob/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/tokenizer/tokenizer.json) and a [frozen 17-case packing reference](https://github.com/linear3735/system1-omni/blob/5e4dd4215c925ebd93bb9ce4097b27bd6375f7c0/recipe/laya/native/packing-golden.json). Both files are checked by SHA-256 before comparison. No weights or GPU are needed.

To run that check locally with the same files:

```sh
export LAYA_TOKENIZER=/path/to/laya/tokenizer/tokenizer.json
export LAYA_PACKING_ORACLE=/path/to/packing-golden.json
cargo test --locked -p omni-laya --test packing -- --ignored
```

The comparison covers every token, marker, question type, row length, question order and usage count. It excludes the reference's backend padding and bucket dimensions. The [reference generator and inputs](https://github.com/linear3735/system1-omni/tree/5e4dd4215c925ebd93bb9ce4097b27bd6375f7c0/recipe/laya/native) use `laya==0.3.20`; packing parity does not measure model quality.

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
