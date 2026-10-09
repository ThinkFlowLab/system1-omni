# Laya text worker

This recipe runs the external Laya Python package behind the Rust frontend.
For the Rust/CUDA worker on Hopper, see [native CUDA setup](native/README.md).
It validates text decisions; image, audio and video inference are not covered.

Run all commands from the repository root. To serve on the GPU of an Apple Silicon Mac, see
[Laya on Apple Silicon](apple-silicon.md).

New users should start with the [complete CPU walkthrough](../../docs/getting-started.md)
([中文](../../docs/getting-started.zh.md)): it lists hardware/storage allowances,
toolchains, downloads, readiness and expected output before the commands.
The [recorded reproduction and demo](validation.md) documents the tested environment.

## Start the worker

Use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r recipe/laya/requirements-cpu.txt
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 LAYA_DEVICE=cpu \
LAYA_MODELS=english LAYA_PRELOAD=1 LAYA_THREADS=4 \
  .venv/bin/laya-serve
```

First startup downloads about 846 MB for the English checkpoint. Wait for
`Uvicorn running` and check direct worker health before starting the frontend.
Health confirms loading; plain `laya-serve` does not perform an inference warmup.

## Start the frontend

In another terminal:

```sh
cargo build -p omni-jev --release --locked
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
python3 recipe/compare_with_backend.py --model english \
  --backend http://127.0.0.1:8000 --frontend http://127.0.0.1:8080
```

The script checks health and all three decision types, separately and together.
Each request must return `200`, with identical status and content type. It first
compares body bytes, then accepts equal parsed `answers` if serialization or usage
differs; see the [CLM contract discussion](../clm/README.md#what-the-comparison-had-to-learn).
Use a deterministic worker response. Set `OMNI_JEV_TEST_TOKEN`
if the worker requires a bearer token.

See the [frontend documentation](../../src/frontend/README.md) for configuration
and transport behavior.
