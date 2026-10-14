# Rust frontend

The default `omni-jev` binary is an Axum/Tokio server that forwards requests to a
separately running model worker using Reqwest. The library also exposes an
in-process transport described below. In both paths the worker handles model
validation, media loading, preprocessing and inference.

In the [target architecture](../../docs/architecture.md), a shared Rust worker
runtime coordinates independent processors, a scheduler/batcher, and model
executors. Rust handles host orchestration; CUDA/Metal handle device operations
through backend bindings. Serial admission and an in-process `Engine` interface
exist; shared processing orchestration and dynamic batching remain planned. The
default binary continues to forward to existing model-specific worker pipelines.

## Run

Run these commands from the repository root:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

All environment variables are optional. The bind address
must be an IP address and port. The backend URL accepts a path prefix, such as
`http://localhost:8000/worker`, but no credentials, query or fragment. Backend
connections bypass system HTTP proxies.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OMNI_JEV_BIND` | `127.0.0.1:8080` | Listen address |
| `OMNI_JEV_BACKEND_URL` | `http://127.0.0.1:8000` | Worker URL |
| `OMNI_JEV_HEALTH_TIMEOUT_MS` | `2000` | Total health timeout in milliseconds |
| `OMNI_JEV_MAX_RESPONSE_BYTES` | `16777216` (16 MiB) | Maximum buffered worker response on either route |

The health timeout and response limit must be positive integers.

## Forwarding HTTP interface

- `POST /v1/systemone` forwards the `model`, `state` and `questions` envelope
  unchanged. Workers return `choice`, `score` or `noul` decisions; see the
  [Jev API reference](https://docs.typesafe.ai/api).
- `GET /health` returns the worker's health response, including unhealthy status codes.
- Authorization and other end-to-end headers are forwarded. Response status,
  content type and body are preserved. Redirects are returned without following them.
- One shared client reuses connections with no retries. Inference has a 60-second
  total timeout; health has a separate 2-second timeout, capped by the client timeout.
  Connection failures return `502`; timeouts, including response-body timeouts, return `504`.
- Uploads are streamed. Responses are buffered up to the configured limit so a body
  timeout can still return `504`. Oversized responses return `502`, including chunked
  responses without `Content-Length`. Raise the limit for workers returning larger bodies.
  Set request size and concurrency limits at the ingress or worker.

Text, image, audio, video and mixed payloads pass through as bytes. Actual inference
support depends on the worker. The [Laya recipe](../../recipe/laya/README.md) verifies
text decisions against a real backend.

## In-process library transport

[`omni_jev::engine::app`](src/engine.rs) accepts an `Arc<dyn Engine>` and
`ServiceConfig`; [`Engine`](../runtime/src/engine.rs) lives in `omni-runtime`.
It exposes the same two route paths, but it is a separate library integration,
not a binary configuration switch. `src/main.rs` selects the forwarding router.

The transport passes request/response bytes without parsing the decision envelope.
It owns body limits, the total request deadline and HTTP status mapping; the
worker owns admission, execution, readiness and optional queue reports. Defaults
are a 30-second total budget and 1 MiB request body limit. Invalid requests map to
400, oversized bodies to 413, busy/unavailable engines to 503, inference failures
to 500, and upload/queue/inference timeouts to 504. Health reads engine readiness
without waiting for inference; successful replies expose the reported queue depth.
`engine::serve` drains HTTP connections; stopping admission and joining worker
threads remain the caller's responsibility.

## Checks

From the repository root:

```sh
# Focused CPU-only API integration suite.
cargo test -p omni-jev --test frontend --locked

# Other transport paths and the in-process interface.
cargo test -p omni-jev --test forwarding_limits --test in_process --locked
```

Tests use local mock workers; no model weights or GPU are needed. They cover
multimodal byte preservation, authorization, connection reuse, large uploads,
backend errors, timeouts, health and binary startup/shutdown.

For full workspace checks, use the current [Rust CI commands](../../CONTRIBUTING.md#rust-ci-checks).
