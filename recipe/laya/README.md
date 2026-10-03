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

## Native encoder validation

The Rust eager encoder runs on Hopper with the English Laya 0.3.20 checkpoint
(`convaiinnovations/laya`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`).
It covers embedding, 28 encoder layers and two decision transformer layers.
Scorer, decoding, HTTP and CUDA Graphs are separate steps.

Use a CUDA-enabled PyTorch environment with `laya==0.3.20` and its TileLang fast-path
dependencies for the reference. Build the operator bundle with the
[existing Hopper build entry](https://github.com/linear3735/system1-omni/blob/5ff41a5/src/backends/cuda/build.sh)
and export rotary tables with that version's `tools/export_tables.py`. Keep
`liblaya_cuda.so`, `build-manifest.json`, `tables.json` and the four rotary table
files in one directory. The operator bundle is separate from the resource library
built below. Load only trusted native libraries on a compatible Hopper GPU.

Run GPU checks on an allocated device. Device 0 below means the first visible GPU.
`requests.json` is a list of `{ "name": "...", "request": { "state": ..., "questions": ... } }`
cases, each with 1–16 questions. The exporter constructs `mixed_16` from the
longest row and the first 15 other distinct token/type rows. That selection must
cover all three question types and include both a 512-token row and a shorter row.
Use a new output directory for each validation run.

```sh
export LAYA_CHECKPOINT=/path/to/laya/snapshot
export LAYA_KERNEL_BUNDLE=/path/to/hopper-bundle
export LAYA_CUDA_DEVICE=0
export LAYA_CUDA_LIBRARY=/tmp/liblaya_resources.so
export LAYA_ENCODER_ORACLE=/path/to/new-encoder-oracle
nvcc -shared -Xcompiler=-fPIC -O2 src/backends/cuda/kernels/runtime.cu -o "$LAYA_CUDA_LIBRARY"
python recipe/laya/native/export_encoder.py "$LAYA_CHECKPOINT" requests.json "$LAYA_ENCODER_ORACLE"
cargo test --release --locked -p omni-laya --lib real_encoder_matches_official_hidden_states -- --ignored --nocapture
```

The test compares selected intermediates and final hidden states, reverses request
order and repeats each workspace. It checks valid tokens and requires finite
candidate padding. This validates implementation parity, not model quality or latency.

The [frozen H800 benchmark](https://gist.github.com/linear3735/c777b0dc449676ebf1c26f0990a90b16)
contains the five-case inputs, runner scripts, host/CUDA-event timing boundaries,
software versions and commands for the separate latency comparison.
