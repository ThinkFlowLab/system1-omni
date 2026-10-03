# Laya text worker

This recipe runs the external Laya Python package behind the Rust frontend.
It validates text decisions; image, audio and video inference are not covered.

Run all commands from the repository root. To serve on the GPU of an Apple Silicon Mac, see
[Laya on Apple Silicon](apple-silicon.md).

## Start the worker

Use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install 'laya[serve]==0.3.20'
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=cpu \
LAYA_MODELS=english LAYA_PRELOAD=1 LAYA_THREADS=4 \
  .venv/bin/laya-serve
```

First startup downloads the English checkpoint. Wait for the worker to become ready.

## Start the frontend

In another terminal:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

## Send a request

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}'
```

## Compare responses

With both services running:

```sh
python3 recipe/laya/compare_with_backend.py --model english \
  --backend http://127.0.0.1:8000 --frontend http://127.0.0.1:8080
```

The script checks health and all three decision types, separately and together.
Each request must return `200`, with identical status, content type and body bytes
through both paths. Use a deterministic worker response. Set `OMNI_JEV_TEST_TOKEN`
if the worker requires a bearer token.

See the [frontend documentation](../../src/frontend/README.md) for configuration
and transport behavior.

## Native CPU packing check

The `omni-laya` preprocessor packs English Laya 0.3.20 requests without weights
or a GPU.

```sh
cargo test --locked -p omni-laya --test preprocess
```

Native callers use `Request::from_json(&str)` for a single top-level JSON request,
or `Request::from_value(Value)` for an existing structured value. `Request`
retains its public fields and `Serialize`; it does not implement generic
`Deserialize`. The JSON entry checks the complete request against serde_json's
default nesting limit. The value entry preserves existing nested values without
reparsing. Both preserve literal private Number/RawValue object keys and reject
unknown request fields.

The normal tests cover validation, JSON rendering, question and option order,
and truncation with a small tokenizer:
Pass raw JSON directly to `from_json`.

For the official 17-case comparison, use the same pinned inputs as CPU CI.
The test checks both files by SHA-256 before comparing:

```sh
LAYA_PACKING_DIR=$(mktemp -d)
export LAYA_TOKENIZER="$LAYA_PACKING_DIR/tokenizer.json"
export LAYA_PACKING_ORACLE="$LAYA_PACKING_DIR/packing-golden.json"
curl --fail --location --retry 3 \
  https://huggingface.co/convaiinnovations/laya/resolve/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851/tokenizer/tokenizer.json \
  --output "$LAYA_TOKENIZER"
curl --fail --location --retry 3 \
  https://raw.githubusercontent.com/linear3735/system1-omni/5e4dd4215c925ebd93bb9ce4097b27bd6375f7c0/recipe/laya/native/packing-golden.json \
  --output "$LAYA_PACKING_ORACLE"
cargo test --locked -p omni-laya --test packing -- --ignored
```

Existing copies of these pinned files can be supplied through `LAYA_TOKENIZER`
and `LAYA_PACKING_ORACLE` instead. The comparison covers every token, marker,
question type, row length, question order and usage count; it excludes backend
padding and bucket dimensions. The [reference generator and inputs](https://github.com/linear3735/system1-omni/tree/5e4dd4215c925ebd93bb9ce4097b27bd6375f7c0/recipe/laya/native)
use `laya==0.3.20`. Packing parity does not measure model quality or execute
native model inference.

## Native decoder validation

Run the CPU decoding checks without Python, weights or a GPU:

```sh
cargo test --locked -p omni-laya --test decision
```

The checked-in reference covers all three question types, temperatures, ordering,
ties and rounding boundaries. Its 16 cases require exact rounded answers; a
separate FP32 reduction probe allows one displayed decimal unit for probabilities.
These fixed logits check decoding, not model quality or GPU execution.

To regenerate the reference, use `laya==0.3.20`, `torch==2.14.0` and
`numpy==2.5.3`, matching the recorded fixture. Supply the English checkpoint
`convaiinnovations/laya` at revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`:

```sh
export LAYA_CHECKPOINT=/path/to/laya/snapshot
python recipe/laya/native/export_decisions.py "$LAYA_CHECKPOINT" /path/to/new-decisions.json
```

The generator reads only `rl_agent_config.json` and calls Laya's official CPU
decoding functions. The output records source hashes and package versions.
