# Rust frontend

An Axum/Tokio server for the `/v1/systemone` decision API. It runs in one of two
modes, and the HTTP surface is the same in both:

- **Forwarding** (default). Requests are passed to a separately running model worker
  with Reqwest. The worker handles validation, media loading, preprocessing and
  inference, and this process owns no model state.
- **Native.** A model linked into this binary answers the requests in-process. The
  transport owns the request lifecycle — budget, queue admission, readiness,
  shutdown — and the model owns the bytes.

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

A model is linked in by selecting an engine. `passthrough` is a stand-in that echoes
every valid request:

```sh
OMNI_SYSTEMONE_ENGINE=passthrough \
OMNI_JEV_BIND=127.0.0.1:8080 \
  ./target/release/omni-jev
```

It exists to check a deployment's wiring — bind address, readiness probe, queue size,
signal handling — before the model is available, and to exercise the lifecycle without
hardware. It is not a model and is not a production path.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OMNI_JEV_BIND` | `127.0.0.1:8080` | Listen address, in both modes. |
| `OMNI_JEV_BACKEND_URL` | `http://127.0.0.1:8000` | Forwarding mode only. |
| `OMNI_SYSTEMONE_ENGINE` | unset | Unset forwards; `passthrough` serves the stand-in. An unknown value is a startup error. |
| `OMNI_SYSTEMONE_CHECKPOINT` | `/checkpoints` | Where the engine's weights are. |
| `OMNI_SYSTEMONE_QUEUE` | `32` | Requests accepted and not yet completed. |
| `OMNI_SYSTEMONE_TIMEOUT_MS` | `30000` | Request budget, covering upload, queueing and inference. |
| `OMNI_SYSTEMONE_MAX_BODY` | `1048576` | Largest request body accepted. |
| `OMNI_SYSTEMONE_DRAIN_MS` | `10000` | How long shutdown waits for accepted work. |

A malformed value is a startup error rather than a silent fallback to the default.

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

### Native mode

- `POST /v1/systemone` passes the body to the engine and returns the body it produced.
  Nothing is parsed or re-serialized, so a field the frontend has never heard of
  survives, and a model's error text cannot produce invalid JSON.
- `GET /health` is the readiness probe. It answers `200` with `{"status":"ok"}` only
  after the engine has loaded **and** warmed up; before that it is `503` with
  `{"status":"starting"}`. An engine that failed after startup answers `503` with
  `{"status":"failed","reason":…}` and the process keeps running, so the failure is
  readable by an operator instead of vanishing with the process. When the engine
  reports one, `depth`, `capacity` and `rejected` are included; an engine that reports
  nothing gets none of them rather than a zero it did not measure. `rejected` counts work
  the engine did not complete: refused because there was no room, or accepted and then
  dropped.
- Successful responses carry `x-queue-depth`: how many requests were outstanding when
  this one was dequeued, so it includes this one. Backpressure is visible before it
  turns into a refusal.
- Status codes: `400` invalid request, `413` body over `OMNI_SYSTEMONE_MAX_BODY`,
  `500` the engine failed to run an accepted request, `503` not ready / queue full /
  request budget expired / the engine stopped, `504` the engine did not answer inside
  the budget.

Lifecycle, which the acceptance criteria for a persistent server depend on:

- One worker thread owns the engine, so a runtime that binds to its creating thread is
  created by the thread that keeps it. Requests run one at a time, in arrival order,
  with no cross-request batching.
- The queue is bounded and refuses rather than grows: once `OMNI_SYSTEMONE_QUEUE`
  requests are outstanding — running **or** queued, not only waiting in the channel —
  further requests get `503` immediately. The slot is taken, and the limit checked, under
  the admission lock and before the request is handed over, so a request that is running
  still counts against it and the depth cannot release a slot that was never reserved.
- The budget starts when the request headers arrive. A body that arrives too slowly, a
  request that waits too long, or one whose client has already gone is dropped without
  reaching the engine, so no GPU work is spent on an answer nobody will read. Work
  already submitted is never cancelled. `Request::deadline_expired` reports the same
  condition to a model that wants to check before starting something expensive.
- `SIGTERM` or `SIGINT` closes admission immediately and gives the rest of the shutdown
  the same budget: responses in flight get up to `OMNI_SYSTEMONE_DRAIN_MS` to finish,
  accepted work is drained within it, and the process exits when the budget is spent. A
  request that is still uploading at that point is not going to finish, and holding the
  process open for its whole request budget would mean the drain budget bounded nothing.
  A drain that could not finish is reported rather than hidden.
- An engine that fails one request is retired: the caller gets `500`, `/health` turns
  `503` with the reason, and no further request is offered to an engine whose state can
  no longer be vouched for. A malformed request does not retire anything.

A model supplies loading, warmup and one function from a request body to a response
body; the queue, readiness and shutdown behaviour above come from the transport:

```rust
let running = omni_jev::worker::spawn(
    checkpoint,
    omni_jev::worker::Options::default(),
    Model::load,
    |model| model.warmup(),
    |request| match model_decide(request.engine, request.body) {
        Ok(body) => Ok(omni_jev::engine::Answer {
            body,
            ..request.answer()
        }),
        Err(detail) => Err(omni_jev::worker::Failure::InvalidRequest(detail)),
    },
)?;
```

Shutting down is `running.stop_accepting()` followed by `running.run_until_drained(deadline)`,
where the deadline comes from `running.drain_budget()` measured when the signal arrived —
one deadline for the whole sequence rather than one per step.

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
backend errors, timeouts, health, and — for the native path — readiness, queue
admission, expired and cancelled requests, error mapping, panic isolation, drain, and
binary startup and shutdown.
