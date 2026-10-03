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
