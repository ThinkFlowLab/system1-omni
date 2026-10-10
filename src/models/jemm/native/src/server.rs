//! JEMM HTTP transport and model readiness.
use crate::{executor::Executor, processing::Processor};
use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_qwen3_5_native::cuda;
use omni_runtime::SerialScheduler;
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};
const TEXT_WARMUP:&[u8]=br#"{"state":"Update installed.","questions":{"q":{"instructions":"Choose the action.","criteria":{"close":"Close dialog","wait":"Wait"}}}}"#;
const WARMUP: &[u8] = br#"{"state":"Update installed.","questions":{"q":{"instructions":"Choose the action.","criteria":{"close":"Close dialog","wait":"Wait"}}},"images":["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKklEQVR4nGPgUbKgKWIYtWDUglELRi0YtWDUglELRi0YtWDUglELhooFAG0HmBCThFO/AAAAAElFTkSuQmCC"]}"#;
pub struct Engine {
    pub processor: Processor,
    pub executor: Executor,
    pub scheduler: SerialScheduler,
}
fn manifest(dir: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(
        &std::fs::read(dir.join("jemm_export.json"))
            .context("export JEMM before launching the native worker")?,
    )?)
}
impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        let export = manifest(dir)?;
        let processor = Processor::load(dir, &export)?;
        let executor = Executor::load(dir, library, &export).await?;
        let engine = Self {
            processor,
            executor,
            scheduler: SerialScheduler::default(),
        };
        // Validate both device paths before binding the listener or reporting ready.
        for raw in [TEXT_WARMUP, WARMUP] {
            let prepared = engine.processor.prepare(raw)?;
            let rows = engine
                .executor
                .execute(&engine.scheduler, prepared.inputs)
                .await?;
            prepared.context.finish(rows)?;
        }
        ensure!(engine.executor.available(), "warmup retired the model");
        Ok(engine)
    }
}
fn error(status: StatusCode, message: impl ToString) -> Response {
    let kind = match status {
        StatusCode::UNPROCESSABLE_ENTITY | StatusCode::PAYLOAD_TOO_LARGE => "ValueError",
        StatusCode::UNSUPPORTED_MEDIA_TYPE => "TypeError",
        _ => "RuntimeError",
    };
    let detail: String = message.to_string().chars().take(200).collect();
    (status, Json(json!({"error":kind,"detail":detail}))).into_response()
}
pub async fn decide(engine: &Engine, raw: &[u8]) -> Response {
    if !engine.executor.available() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "model is unavailable");
    }
    let prepared = match engine.processor.prepare(raw) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::UNPROCESSABLE_ENTITY, e),
    };
    match engine
        .executor
        .execute(&engine.scheduler, prepared.inputs)
        .await
        .and_then(|rows| prepared.context.finish(rows))
    {
        Ok(body) => Json(body).into_response(),
        Err(e) => {
            eprintln!("inference failed: {e:#}");
            error(StatusCode::SERVICE_UNAVAILABLE, "model inference failed")
        }
    }
}
async fn systemone(
    State(engine): State<Arc<Engine>>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    match body {
        Ok(raw) => decide(&engine, &raw).await,
        Err(e) => error(e.status(), e.body_text()),
    }
}
pub fn router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route(
            "/health",
            get(|State(engine): State<Arc<Engine>>| async move {
                if engine.executor.available() {
                    Json(json!({"status":"READY","model":"JEMM"})).into_response()
                } else {
                    error(StatusCode::SERVICE_UNAVAILABLE, "model is unavailable")
                }
            }),
        )
        .route("/v1/systemone", post(systemone))
        .layer(DefaultBodyLimit::max(crate::contract::MAX_BODY_BYTES))
        .with_state(engine)
}
pub async fn run() -> Result<()> {
    let dir = std::env::var_os("JEMM_MODEL").context("set JEMM_MODEL to the native export")?;
    let dir = Path::new(&dir);
    ensure!(
        std::env::args_os().len() == 1,
        "usage: omni-jemm-native (configure using JEMM_MODEL and JEMM_CUDA_LIB)"
    );
    let library = std::env::var_os("JEMM_CUDA_LIB")
        .map(Into::into)
        .map_or_else(cuda::default_library, Ok)?;
    let engine = Arc::new(Engine::load(dir, &library).await?);
    let host = std::env::var("JEMM_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("JEMM_PORT")
        .map_or(Ok(8000), |v| v.parse())
        .context("JEMM_PORT")?;
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    eprintln!("JEMM ready on {host}:{port}");
    axum::serve(listener, router(engine)).await?;
    Ok(())
}

#[cfg(test)]
#[path = "../../../../../tests/jemm/server.rs"]
mod tests;
