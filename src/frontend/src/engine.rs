//! Native engines accept owned request bytes. Cancellation drops the reply receiver;
//! a worker must retain its GPU allocations until submitted work has finished.
use axum::{
    Router,
    body::{Body, to_bytes},
    extract::{Request, State},
    http::{StatusCode, header},
    response::Response,
    routing::{get, post},
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

#[derive(Debug)]
pub enum EngineError {
    InvalidRequest(String),
    Busy,
    Unavailable,
    InferenceFailed,
}
pub type Reply = oneshot::Receiver<Result<Vec<u8>, EngineError>>;
pub trait Engine: Send + Sync + 'static {
    fn submit(&self, body: Vec<u8>, deadline: Instant) -> Result<Reply, EngineError>;
    fn ready(&self) -> bool;
}
#[derive(Clone)]
struct Service {
    engine: Arc<dyn Engine>,
    timeout: Duration,
    max_body: usize,
}
pub fn app(engine: Arc<dyn Engine>, timeout: Duration, max_body: usize) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/systemone", post(infer))
        .with_state(Service {
            engine,
            timeout,
            max_body,
        })
}
async fn health(State(s): State<Service>) -> Response {
    if s.engine.ready() {
        response(StatusCode::OK, br#"{"status":"ok"}"#.to_vec())
    } else {
        error(EngineError::Unavailable)
    }
}
async fn infer(State(s): State<Service>, request: Request) -> Response {
    let deadline = Instant::now() + s.timeout;
    let work = async {
        if !s.engine.ready() {
            return error(EngineError::Unavailable);
        }
        let bytes = match to_bytes(request.into_body(), s.max_body).await {
            Ok(b) => b,
            Err(_) => {
                return response(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    br#"{"error":"request body too large or unreadable"}"#.to_vec(),
                );
            }
        };
        let reply = match s.engine.submit(bytes.to_vec(), deadline) {
            Ok(r) => r,
            Err(e) => return error(e),
        };
        match reply.await {
            Ok(Ok(body)) => response(StatusCode::OK, body),
            Ok(Err(e)) => error(e),
            Err(_) => error(EngineError::Unavailable),
        }
    };
    match tokio::time::timeout(s.timeout, work).await {
        Ok(r) => r,
        Err(_) => response(
            StatusCode::GATEWAY_TIMEOUT,
            br#"{"error":"inference timed out"}"#.to_vec(),
        ),
    }
}
fn response(status: StatusCode, body: Vec<u8>) -> Response {
    let mut r = Response::new(Body::from(body));
    *r.status_mut() = status;
    r.headers_mut().insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
    r
}
fn error(e: EngineError) -> Response {
    let (status, message) = match e {
        EngineError::InvalidRequest(s) => (StatusCode::BAD_REQUEST, s),
        EngineError::Busy => (
            StatusCode::SERVICE_UNAVAILABLE,
            "inference queue full".into(),
        ),
        EngineError::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "model unavailable".into()),
        EngineError::InferenceFailed => {
            (StatusCode::INTERNAL_SERVER_ERROR, "inference failed".into())
        }
    };
    response(
        status,
        serde_json::to_vec(&serde_json::json!({"error":message})).expect("serialize error string"),
    )
}
