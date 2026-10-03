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

## Native residency and workspace validation

These opt-in Rust checks test CUDA allocations and transfers. They do not need the
Python worker or frontend and do not test inference, model outputs or latency.
The normal CPU tests skip them.

Use a Linux host with an approved CUDA GPU, a working NVIDIA driver, the CUDA
toolkit (`nvcc`) and Rust. Build the trusted resource library from this checkout;
it uses ABI version 1 and needs neither TileLang nor cuBLAS. `LAYA_CUDA_DEVICE` is
the approved device ordinal after `CUDA_VISIBLE_DEVICES` filtering.

```sh
export LAYA_CUDA_LIBRARY=/tmp/liblaya-resources.so
export LAYA_CUDA_DEVICE=0
nvcc -shared -Xcompiler=-fPIC -O2 src/backends/cuda/kernels/runtime.cu \
  -o "$LAYA_CUDA_LIBRARY"
```

The workspace check needs no checkpoint. It writes and reads all 17 buffers twice
at `(batch, sequence) = (1, 16), (1, 512), (16, 512)`, using deterministic byte
patterns:

```sh
cargo test --release --locked -p omni-laya --lib \
  workspace::tests::real_gpu_workspace_capacity_and_reuse \
  -- --ignored --exact --nocapture
```

For the residency check, use an unchanged local snapshot of
`convaiinnovations/laya` revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`,
including `model.safetensors`. Generate the oracle from that same snapshot in a
Python environment with PyTorch, safetensors and NumPy:

```sh
export LAYA_CHECKPOINT=/path/to/laya/snapshot
export LAYA_WEIGHT_ORACLE=/tmp/laya-weight-oracle.json
python recipe/laya/native/export_weights.py "$LAYA_CHECKPOINT" "$LAYA_WEIGHT_ORACLE"
cargo test --release --locked -p omni-laya --lib \
  resident::tests::real_checkpoint_residency_matches_torch \
  -- --ignored --exact --nocapture
```

The residency check validates all 206 checkpoint tensors, uploads the 205 used
tensors and compares readback hashes with the Torch conversion oracle. The legacy
`temperature` buffer is validated but not uploaded. Reported allocation bytes
exclude CUDA context and library overhead. See the
[model contracts](https://github.com/linear3735/system1-omni/blob/codex/laya-workspace/src/models/laya/README.md) for storage precision,
workspace layouts and ownership.
