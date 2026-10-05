# Rust frontend

An Axum/Tokio server that forwards requests to a separately running model worker
using Reqwest. The worker handles validation, media loading, preprocessing and
inference.

In the [target architecture](../../docs/architecture.md), a shared Rust worker
runtime coordinates independent processors, a scheduler/batcher, and model
executors. Rust handles host orchestration; CUDA/Metal handle device operations
through backend bindings. Native workers share bounded serial admission and
dispatch; further orchestration and GPU batching remain planned. The frontend
continues to forward requests to existing model-specific worker pipelines.

## Run

Run these commands from the repository root:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

Both variables are optional; the values above are their defaults. The bind address
must be an IP address and port. The backend URL accepts a path prefix, such as
`http://localhost:8000/worker`, but no credentials, query or fragment. Backend
connections bypass system HTTP proxies.

## HTTP interface

- `POST /v1/systemone` forwards the `model`, `state` and `questions` envelope
  unchanged. Workers return `choice`, `score` or `noul` decisions; see the
  [Jev API reference](https://docs.typesafe.ai/api).
- `GET /health` returns the worker's health response, including unhealthy status codes.
- Authorization and other end-to-end headers are forwarded. Response status,
  content type and body are preserved. Redirects are returned without following them.
- One shared client reuses connections with a 60-second total timeout and no retries.
  Connection failures return `502`; timeouts, including response-body timeouts, return `504`.
- Uploads are streamed. Responses are buffered so a body timeout can still return `504`.
  Set request size and concurrency limits at the ingress or worker.

Native Cua-S1 and Open-Jev workers configure pending execution-unit capacity with
`OMNI_NATIVE_MAX_PENDING` (default 64). Their HTTP `503` overload responses pass
through unchanged, with no frontend retry. See the
[native runtime](../runtime/README.md) for each worker's unit and cancellation rules.

Text, image, audio, video and mixed payloads pass through as bytes. Actual inference
support depends on the worker. The [Laya recipe](../../recipe/laya/README.md) verifies
text decisions against a real backend.

## Checks

From the repository root:

```sh
# Focused CPU-only API integration suite.
cargo test -p omni-jev --test frontend --locked

# Full workspace checks.
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
```

Tests use local mock workers; no model weights or GPU are needed. They cover
multimodal byte preservation, authorization, connection reuse, large uploads,
backend errors, timeouts, health and binary startup/shutdown.
