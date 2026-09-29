# Rust frontend

An Axum/Tokio server for the `/v1/systemone` decision API. It serves the same HTTP
surface two ways:

- **Forwarding** (default). Requests are passed to a separately running model worker
  with Reqwest. The worker handles validation, media loading, preprocessing and
  inference, and this process owns no model state.
- **In-process.** A model linked into this binary answers the requests through the
  [`Engine`](src/engine.rs) trait. Admission, the request budget and readiness belong to
  the transport; the model owns the bytes.

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

### Forwarding mode

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

Text, image, audio, video and mixed payloads pass through as bytes. Actual inference
support depends on the worker. The [Laya recipe](../../recipe/laya/README.md) verifies
text decisions against a real backend.

### In-process mode

The same two routes, answered by [`engine::app`](src/engine.rs):

- `POST /v1/systemone` passes the body to the engine and returns the body it produced.
  Nothing is parsed or re-serialized, so a field the frontend has never heard of
  survives, and a model's error text cannot produce invalid JSON.
- `GET /health` is the readiness probe. `200` carries `{"status":"ok"}`; before the
  engine can take work it is `503` with `{"status":"starting"}`, and an engine that
  failed answers `503` with `{"status":"failed","reason":…}`. When the engine reports
  its queue, `depth` and `rejected` are included; an engine that reports nothing gets
  neither field rather than a zero it did not measure.
- Successful responses carry `x-queue-depth`, measured when the request was dequeued.
- Status codes: `400` invalid request, `413` body over the configured limit, `500` the
  engine failed to run an accepted request, `503` not ready / no queue capacity /
  request budget expired / the engine stopped, `504` the engine did not answer inside
  the budget.

```rust
let app = omni_jev::engine::app(engine, omni_jev::engine::ServiceConfig::default());
```

An `Engine` reports readiness, accepts one request and hands back a reply carrying the
response body and the queue depth to publish with it. A model implements that and
nothing else; the thread that owns it, the bounded queue and the shutdown sequence
around it arrive in a following change.

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

Tests use local mock workers and a fake engine; no model weights and no GPU are needed.
They cover multimodal byte preservation, authorization, connection reuse, large uploads,
backend errors, timeouts, health, and — for the in-process path — readiness, byte
preservation, the body limit, queue-full refusal, the deadline, error mapping and the
queue depth a reply publishes.
