//! In-process serving: the `/v1/systemone` surface answered by a worker in this binary.
//!
//! The contract this serves is [`omni_runtime::engine::Engine`] — admission and the queue
//! belong to the worker-side runtime, and this module is the transport above it. It owns what
//! the transport owns and nothing else: the two routes, the request budget measured from the
//! headers, the body limit, the status mapping, and the queue depth a reply publishes.
//!
//! Nothing here parses a decision envelope. Request bytes go to the engine and response bytes
//! come back, so a field the frontend has never heard of survives and a model's error text
//! cannot produce invalid JSON.
//!
//! **One timeout semantic.** A request that cannot be answered inside its budget is `504`,
//! whatever part of the budget ran out — the upload, the queue, or the inference. `503` is
//! reserved for a worker that cannot take work at all: still loading, failed, out of capacity,
//! or gone. The forwarding path in [`crate`] answers the same way, so a caller does not have to
//! learn which mode it is talking to in order to read a timeout.

use std::{
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    Router,
    body::{Body, to_bytes},
    extract::{Request, State},
    http::{HeaderName, HeaderValue, StatusCode, header},
    response::Response,
    routing::{get, post},
};
use omni_runtime::engine::{Answer, Engine, EngineError, Readiness};
use tokio::net::TcpListener;

use crate::BoxError;

/// Work accepted and not yet completed, as reported when a response was produced. Only sent
/// when the worker reports one, so a missing header means "not observed", not "empty".
pub const QUEUE_DEPTH: HeaderName = HeaderName::from_static("x-queue-depth");

/// The transport's own limits: what it will read, and how long it will wait.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// Total request budget, covering upload, queueing and inference.
    pub timeout: Duration,
    pub max_body: usize,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_body: 1024 * 1024,
        }
    }
}

#[derive(Clone)]
struct Service {
    engine: Arc<dyn Engine>,
    config: ServiceConfig,
}

/// The in-process router: `POST /v1/systemone` and `GET /health`.
pub fn app(engine: Arc<dyn Engine>, config: ServiceConfig) -> Router {
    Router::new()
        .route("/v1/systemone", post(infer))
        .route("/health", get(health))
        .with_state(Service { engine, config })
}

/// Serves until `shutdown` resolves, then drains in-flight connections.
///
/// Draining here covers the transport only. Stopping admission and joining the worker's own
/// thread belong to the caller, and happen after this returns.
pub async fn serve(
    listener: TcpListener,
    router: Router,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), BoxError> {
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

async fn health(State(service): State<Service>) -> Response {
    let readiness = service.engine.readiness();
    let report = match readiness {
        Readiness::Ready => service.engine.report(),
        // A starting worker knows nothing yet, and a failed one has already said why.
        _ => None,
    };
    let (status, value) = match readiness {
        Readiness::Ready => (StatusCode::OK, "ok"),
        Readiness::Starting => (StatusCode::SERVICE_UNAVAILABLE, "starting"),
        Readiness::Failed => (StatusCode::SERVICE_UNAVAILABLE, "failed"),
    };
    let mut fields = vec![format!("\"status\":\"{value}\"")];
    if let (Readiness::Failed, Some(reason)) = (readiness, service.engine.failure()) {
        fields.push(format!("\"reason\":{}", quote(&reason)));
    }
    if let Some(report) = report {
        fields.push(format!("\"depth\":{}", report.depth));
        fields.push(format!("\"capacity\":{}", report.capacity));
        fields.push(format!("\"rejected\":{}", report.rejected));
    }
    json(status, &format!("{{{}}}", fields.join(",")))
}

async fn infer(State(service): State<Service>, request: Request) -> Response {
    // The budget starts here, so a slow upload is charged to the caller rather than to the
    // worker, and work that cannot finish in time is refused before it is admitted.
    let deadline = Instant::now() + service.config.timeout;
    if service.engine.readiness() != Readiness::Ready {
        return fail(StatusCode::SERVICE_UNAVAILABLE, "model unavailable");
    }

    // Reading the body is charged to the same budget. Too large and too slow are separate
    // answers so a caller can tell a big request from a late one.
    let body = match tokio::time::timeout_at(
        deadline.into(),
        to_bytes(request.into_body(), service.config.max_body),
    )
    .await
    {
        Ok(Ok(bytes)) => bytes.to_vec(),
        Ok(Err(_)) => return fail(StatusCode::PAYLOAD_TOO_LARGE, "request body too large"),
        Err(_) => {
            return fail(
                StatusCode::GATEWAY_TIMEOUT,
                "request timed out while uploading",
            );
        }
    };
    if Instant::now() >= deadline {
        return fail(StatusCode::GATEWAY_TIMEOUT, "request timed out");
    }

    let reply = service.engine.submit(body, deadline);
    match tokio::time::timeout_at(deadline.into(), reply).await {
        Ok(Ok(Ok(answer))) => with_depth(answer),
        Ok(Ok(Err(error))) => error_response(error),
        // The worker dropped a request it had accepted. Not bad input and not a failed
        // inference, so it reads as lost capacity.
        Ok(Err(_)) => fail(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
        Err(_) => fail(StatusCode::GATEWAY_TIMEOUT, "request timed out"),
    }
}

fn error_response(error: EngineError) -> Response {
    let (status, message) = match error {
        EngineError::InvalidRequest(detail) => (StatusCode::BAD_REQUEST, detail),
        EngineError::Busy => (
            StatusCode::SERVICE_UNAVAILABLE,
            "inference queue full".into(),
        ),
        EngineError::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "model unavailable".into()),
        EngineError::InferenceFailed => {
            (StatusCode::INTERNAL_SERVER_ERROR, "inference failed".into())
        }
    };
    fail(status, &message)
}

fn fail(status: StatusCode, message: &str) -> Response {
    json(status, &format!("{{\"error\":{}}}", quote(message)))
}

/// Answers with the worker's body, publishing the queue depth it reported.
fn with_depth(answer: Answer) -> Response {
    let mut response = Response::new(Body::from(answer.body));
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    if let Ok(depth) = HeaderValue::from_str(&answer.depth.to_string()) {
        headers.insert(QUEUE_DEPTH, depth);
    }
    response
}

fn json(status: StatusCode, body: &str) -> Response {
    let mut response = Response::new(Body::from(body.to_owned()));
    *response.status_mut() = status;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// Serializes a string as a JSON string literal. Hand-rolled so this crate keeps parsing
/// nothing: a worker's error text must not be able to produce invalid JSON.
fn quote(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
