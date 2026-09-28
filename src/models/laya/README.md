# LAYA model engine

LAYA is the first planned System1-Omni model. This directory owns its complete request-to-result path: preprocessing, postprocessing, batching policy, state, execution, and backend-specific kernel selection.

GPU operations and kernel implementations belong in [`backends/cuda/`](../../backends/cuda/) and [`backends/metal/`](../../backends/metal/). Setup and usage examples belong in the top-level [`recipe/`](../../../recipe/) directory.

The `omni-laya` crate currently reads and checks the English Laya 0.3.20 checkpoint. `Config::load` validates the architecture and temperatures; `Weights` checks tensor names and shapes and converts FP32, FP16 and BF16 values. `checkpoint_tensors()` lists the 206 expected tensors. Each backend chooses its own storage precision.

Keep checkpoint files unchanged while `Weights` holds a read-only memory mapping. This crate does not yet execute inference.

`Preprocessor::load` reads a tokenizer JSON file. `prepare` packs English `choice`, `score` and `noul` questions into ordered token rows, option-marker positions and type IDs. Rows follow Laya 0.3.20's 512-token limit and 192-token head budget. Conversation lists keep the newest state tokens; other state values keep the beginning. The result includes normalized criteria for later decoding and the total input-token usage. Backends own padding, batching and resource limits.

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

The normal tests also check input validation, question and option order, truncation and JSON rendering with a small test tokenizer. CPU CI separately downloads the [official tokenizer](https://huggingface.co/convaiinnovations/laya/blob/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/tokenizer/tokenizer.json) and a [frozen 17-case packing reference](https://github.com/linear3735/system1-omni/blob/5e4dd4215c925ebd93bb9ce4097b27bd6375f7c0/recipe/laya/native/packing-golden.json). Both files are checked by SHA-256 before comparison. No weights or GPU are needed.

To run that check locally with the same files:

```sh
export LAYA_TOKENIZER=/path/to/laya/tokenizer/tokenizer.json
export LAYA_PACKING_ORACLE=/path/to/packing-golden.json
cargo test --locked -p omni-laya --test packing -- --ignored
```

The comparison covers every token, marker, question type, row length, question order and usage count. It excludes the reference's backend padding and bucket dimensions. The [reference generator and inputs](https://github.com/linear3735/system1-omni/tree/5e4dd4215c925ebd93bb9ce4097b27bd6375f7c0/recipe/laya/native) use `laya==0.3.20`; packing parity does not measure model quality.
