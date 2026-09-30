//! In-process serving: the same `/v1/systemone` surface, answered by a linked engine.
//!
//! The forwarding path in [`crate`] stays available for workers that already speak HTTP.
//! This module is for a model in the same binary: the transport owns admission, the
//! request budget and readiness, and the model owns the bytes. Nothing here parses the
//! decision envelope, so a field this crate has never heard of survives.

use std::{
    fmt,
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
use tokio::{net::TcpListener, sync::oneshot};

use crate::BoxError;

/// Requests accepted and not yet completed, as reported when a response was produced. Only
/// sent when the engine reports one, so a missing header means "not observed", not "empty".
pub const QUEUE_DEPTH: HeaderName = HeaderName::from_static("x-queue-depth");

/// How far the engine has got. `GET /health` is the readiness probe a load balancer or a
/// benchmark script polls before sending traffic, and `Starting` is what it sees while
/// weights load and the engine warms up.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Readiness {
    Starting,
    Ready,
    /// Startup failed. The process keeps running and `/health` carries the reason, so the
    /// failure is readable by an operator instead of vanishing with the process.
    Failed,
}

/// What an engine tells `/health` about itself. Advisory: readiness never waits on it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    /// Requests accepted and not yet completed, including the one running.
    pub depth: usize,
    /// Requests the engine refused or dropped since it started. Monotonic.
    pub rejected: u64,
}

/// The reply the transport sends. `depth` is how many requests were outstanding when this
/// one was dequeued, so it includes this one; it becomes the `x-queue-depth` header, which
/// makes backpressure visible before it turns into a refusal.
pub struct Answer {
    pub body: Vec<u8>,
    pub depth: usize,
}

/// A closed receiver means the client is gone: a queued request must then be skipped
/// rather than executed, and an engine that has already started work must still finish it
/// before reusing GPU buffers.
pub type Reply = oneshot::Receiver<Result<Answer, EngineError>>;

/// Why an engine could not answer. The transport maps each variant to one status code and
/// one message, so the mapping is a property of this crate rather than of a model.
#[derive(Debug)]
pub enum EngineError {
    /// The caller's body is not a request this engine accepts.
    InvalidRequest(String),
    /// Out of capacity right now. The caller may retry.
    Busy,
    /// Not able to accept work at all: still loading, drained, or dead.
    Unavailable,
    /// The request was well formed and the engine failed to execute it.
    InferenceFailed,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest(detail) => write!(f, "invalid request: {detail}"),
            Self::Busy => f.write_str("no queue capacity"),
            Self::Unavailable => f.write_str("engine unavailable"),
            Self::InferenceFailed => f.write_str("inference failed"),
        }
    }
}

impl std::error::Error for EngineError {}

/// The model side of the transport. Implementations are usually a [`crate::worker::Handle`],
/// which forwards to the thread that owns the engine.
pub trait Engine: Send + Sync + 'static {
    /// Non-blocking. Polled on every request, so it must not lock behind queued work.
    fn readiness(&self) -> Readiness;

    /// Accepts one request. `Ok` means the engine owns it and the transport waits for the
    /// reply until `deadline`; [`EngineError::Busy`] means nothing was accepted.
    fn submit(&self, body: Vec<u8>, deadline: Instant) -> Result<Reply, EngineError>;

    /// What `/health` publishes while ready. `None` leaves those fields out.
    fn report(&self) -> Option<Report> {
        None
    }

    /// Set only when [`Engine::readiness`] is [`Readiness::Failed`], and surfaced verbatim
    /// so an operator can read the cause without the server log.
    fn failure(&self) -> Option<String> {
        None
    }
}

/// Admission policy: a bounded queue, and a budget that starts when the headers arrive.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// Total request budget: upload, queueing and inference together.
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

/// The native router: `POST /v1/systemone` and `GET /health`.
pub fn app(engine: Arc<dyn Engine>, config: ServiceConfig) -> Router {
    Router::new()
        .route("/v1/systemone", post(infer))
        .route("/health", get(health))
        .with_state(Service { engine, config })
}

/// Serves until `shutdown` resolves, then drains in-flight connections.
///
/// Draining here covers the transport only. Stopping admission and joining the engine's
/// own worker happen after this returns, and belong to the caller.
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
        // A starting engine knows nothing yet, and a failed one has already said why.
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
        fields.push(format!("\"rejected\":{}", report.rejected));
    }
    json(status, &format!("{{{}}}", fields.join(",")))
}

async fn infer(State(service): State<Service>, request: Request) -> Response {
    // The budget starts here, so a slow upload is charged to the caller rather than to the
    // engine, and work that cannot finish in time is refused before it queues.
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
                StatusCode::SERVICE_UNAVAILABLE,
                "request deadline expired while uploading",
            );
        }
    };
    if Instant::now() >= deadline {
        return fail(StatusCode::SERVICE_UNAVAILABLE, "request deadline expired");
    }

    let reply = match service.engine.submit(body, deadline) {
        Ok(reply) => reply,
        Err(error) => return error_response(error),
    };
    match tokio::time::timeout_at(deadline.into(), reply).await {
        Ok(Ok(Ok(answer))) => with_depth(answer),
        Ok(Ok(Err(error))) => error_response(error),
        // The engine dropped a request it had already accepted. Not bad input and not a
        // failed inference, so it reads as lost capacity.
        Ok(Err(_)) => fail(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
        Err(_) => fail(StatusCode::GATEWAY_TIMEOUT, "inference timed out"),
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

/// Answers with the engine's body, publishing the queue depth it reported.
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
/// nothing: a model's error text must not be able to produce invalid JSON.
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
