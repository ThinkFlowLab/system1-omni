//! Model worker HTTP transport; Decider interpretation belongs to Processor.
use crate::{
    Limits,
    engine::Engine,
    runtime::{Overloaded, Runtime},
};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, FromRequest, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_runtime::control::{ExecutionContext, ExecutionStopped};
use serde_json::json;
use std::sync::Arc;
use std::{future::Future, time::Instant};

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
async fn decide(State(engine): State<Arc<Engine>>, request: Request) -> Response {
    let content_type = request
        .headers()
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
    if !engine.executor.is_ready() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "model unavailable");
    }
    within_budget(&engine.runtime, request, |raw, context| {
        infer(&engine, raw, context)
    })
    .await
}

async fn within_budget<F, Fut>(runtime: &Runtime, request: Request, infer: F) -> Response
where
    F: FnOnce(Bytes, ExecutionContext) -> Fut,
    Fut: Future<Output = Response>,
{
    let context = match runtime.admit(Instant::now()) {
        Ok(context) => context,
        Err(error) => return runtime_error(error, StatusCode::INTERNAL_SERVER_ERROR),
    };
    let _cancel = context.cancel_on_drop();
    let deadline = context.deadline();
    let work = async move {
        if let Err(error) = context.check() {
            return runtime_error(error.into(), StatusCode::INTERNAL_SERVER_ERROR);
        }
        let raw = match Bytes::from_request(request, &()).await {
            Ok(raw) => raw,
            Err(error) => return self::error(error.status(), error.body_text()),
        };
        if let Err(error) = context.check() {
            return runtime_error(error.into(), StatusCode::INTERNAL_SERVER_ERROR);
        }
        infer(raw, context).await
    };
    match deadline {
        Some(deadline) => match tokio::time::timeout_at(deadline.into(), work).await {
            Ok(response) => response,
            Err(_) => error(StatusCode::GATEWAY_TIMEOUT, "request timed out"),
        },
        None => work.await,
    }
}

fn runtime_error(error: anyhow::Error, fallback: StatusCode) -> Response {
    if error.is::<Overloaded>() {
        let mut response = self::error(
            StatusCode::SERVICE_UNAVAILABLE,
            "inference capacity exhausted",
        );
        response
            .headers_mut()
            .insert("retry-after", axum::http::HeaderValue::from_static("1"));
        response
    } else if error.is::<ExecutionStopped>() {
        self::error(StatusCode::GATEWAY_TIMEOUT, "request timed out")
    } else if fallback == StatusCode::UNPROCESSABLE_ENTITY {
        self::error(fallback, error)
    } else {
        self::error(fallback, "model inference failed")
    }
}

async fn infer(engine: &Engine, raw: Bytes, context: ExecutionContext) -> Response {
    // Request capacity covers preparation; processed-token admission follows it.
    // Tokenization and row construction stay outside the model mutex.
    let processor = engine.processor.clone();
    let preparing = context.clone();
    let prepared = match tokio::task::spawn_blocking(move || {
        preparing.check()?;
        let prepared = processor.prepare(&raw)?;
        preparing.check()?;
        Ok::<_, anyhow::Error>(prepared)
    })
    .await
    {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(e)) => return runtime_error(e, StatusCode::UNPROCESSABLE_ENTITY),
        Err(e) => {
            eprintln!("Decider preparation failed: {e}");
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "request preparation failed",
            );
        }
    };
    let context = match engine
        .runtime
        .reserve_tokens(context, prepared.processed_tokens())
    {
        Ok(context) => context,
        Err(error) => return runtime_error(error, StatusCode::INTERNAL_SERVER_ERROR),
    };
    let result = engine
        .executor
        .execute_with_context(&engine.scheduler, prepared.rows, context.clone())
        .await
        .and_then(|rows| prepared.context.finish(rows));
    match result {
        Ok(body) => match context.check() {
            Ok(()) => Json(body).into_response(),
            Err(error) => runtime_error(error.into(), StatusCode::INTERNAL_SERVER_ERROR),
        },
        Err(e) => {
            eprintln!("Decider inference failed: {e:#}");
            runtime_error(
                e,
                if engine.executor.is_ready() {
                    StatusCode::INTERNAL_SERVER_ERROR
                } else {
                    StatusCode::SERVICE_UNAVAILABLE
                },
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

#[cfg(test)]
#[path = "../../../../../tests/decider/serve_runtime.rs"]
mod runtime_tests;
