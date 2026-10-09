//! Native worker assembly and HTTP adapter; the public frontend remains a proxy.
use crate::{executor::Executor, processing::Processor};
use anyhow::Result;
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_runtime::SerialScheduler;
use serde_json::json;
use std::{path::Path, sync::Arc};

pub struct Engine {
    pub processor: Processor,
    pub scheduler: SerialScheduler,
    pub executor: Executor,
}

impl Engine {
    pub async fn load(checkpoint: &Path, bundle: &Path) -> Result<Self> {
        Ok(Self {
            processor: Processor::load(checkpoint)?,
            scheduler: SerialScheduler::default(),
            executor: Executor::load(checkpoint, bundle).await?,
        })
    }
    pub async fn decide(&self, raw: &[u8]) -> Response {
        let prepared = match self.processor.prepare(raw) {
            Ok(prepared) => prepared,
            Err(e) => return error(StatusCode::UNPROCESSABLE_ENTITY, e),
        };
        let result = async {
            let (logits, actions) = self
                .executor
                .execute(&self.scheduler, prepared.inputs)
                .await?;
            prepared.context.finish(logits, actions)
        }
        .await;
        match result {
            Ok(body) => Json(body).into_response(),
            Err(e) => {
                eprintln!("inference failed: {e:#}");
                error(StatusCode::INTERNAL_SERVER_ERROR, "model inference failed")
            }
        }
    }
}

fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"error": message.to_string()}))).into_response()
}

async fn systemone(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
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
    match body {
        Ok(raw) => engine.decide(&raw).await,
        Err(e) => error(e.status(), e.body_text()),
    }
}

async fn health(State(engine): State<Arc<Engine>>) -> Response {
    if engine.executor.ready() {
        Json(json!({"status":"ready", "model":"laya-rl-agent"})).into_response()
    } else {
        error(StatusCode::SERVICE_UNAVAILABLE, "CUDA worker unavailable")
    }
}

pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/systemone", post(systemone))
        .layer(DefaultBodyLimit::max(4 << 20))
        .with_state(engine)
}
