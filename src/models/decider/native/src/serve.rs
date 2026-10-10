//! Model worker HTTP transport; Decider interpretation belongs to Processor.
use crate::{Limits, engine::Engine};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde_json::json;
use std::sync::Arc;

fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"detail":message.to_string()}))).into_response()
}
async fn health(State(engine): State<Arc<Engine>>) -> Response {
    let status = if engine.executor.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(engine.health())).into_response()
}
async fn decide(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let content_type = headers
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim();
    if !content_type.eq_ignore_ascii_case("application/json") {
        return error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
        );
    }
    let raw = match body {
        Ok(raw) => raw,
        Err(e) => return error(e.status(), e.body_text()),
    };
    if !engine.executor.is_ready() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "model unavailable");
    }
    // Tokenization/row construction runs outside admission and the model mutex.
    let processor = engine.processor.clone();
    let prepared = match tokio::task::spawn_blocking(move || processor.prepare(&raw)).await {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(e)) => return error(StatusCode::UNPROCESSABLE_ENTITY, e),
        Err(e) => {
            eprintln!("Decider preparation failed: {e}");
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "request preparation failed",
            );
        }
    };
    let result = engine
        .executor
        .execute(&engine.scheduler, prepared.rows)
        .await
        .and_then(|rows| prepared.context.finish(rows));
    match result {
        Ok(body) => Json(body).into_response(),
        Err(e) => {
            eprintln!("Decider inference failed: {e:#}");
            error(
                if engine.executor.is_ready() {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::SERVICE_UNAVAILABLE
                },
                "model inference failed",
            )
        }
    }
}
pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/systemone", post(decide))
        .layer(DefaultBodyLimit::max(Limits::default().max_request_bytes))
        .with_state(engine)
}
