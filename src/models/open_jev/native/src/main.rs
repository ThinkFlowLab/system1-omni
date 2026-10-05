//! OPEN_JEV_MODEL=<merged export> omni-open-jev-native

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_open_jev_native::{contract::MODEL_ID, engine::Engine};
use omni_qwen3_5_native::cuda;
use serde_json::json;

const WARMUP: &[u8] = br#"{"state":"Dialog: Update installed.","questions":{"q":{"type":"choice","instructions":"Close it.","criteria":{"ok":"OK","wait":"Wait"}}}}"#;

fn error(status: StatusCode, message: impl ToString) -> Response {
    (status, Json(json!({"error": message.to_string()}))).into_response()
}

async fn decide(engine: &Engine, raw: &[u8]) -> Response {
    let prepared = match engine.processor.prepare(raw) {
        Ok(prepared) => prepared,
        Err(e) => return error(StatusCode::UNPROCESSABLE_ENTITY, e),
    };
    let result = async {
        let rows = engine
            .executor
            .execute(&engine.scheduler, prepared.inputs)
            .await?;
        prepared.context.finish(rows)
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
        Ok(raw) => decide(&engine, &raw).await,
        Err(e) => error(e.status(), e.body_text()),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var_os("OPEN_JEV_MODEL").context("set OPEN_JEV_MODEL")?;
    let library = std::env::var_os("OPEN_JEV_CUDA_LIB")
        .map(Into::into)
        .map_or_else(cuda::default_library, Ok)?;
    let engine = Arc::new(Engine::load(model.as_ref(), &library).await?);
    ensure!(
        decide(&engine, WARMUP).await.status() == StatusCode::OK,
        "warmup failed"
    );
    let host = std::env::var("OPEN_JEV_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("OPEN_JEV_PORT")
        .map_or(Ok(8000), |v| v.parse())
        .context("OPEN_JEV_PORT")?;
    let app = Router::new()
        .route(
            "/health",
            get(|| async { Json(json!({"status": "ready", "model": MODEL_ID})) }),
        )
        .route("/v1/systemone", post(systemone))
        .layer(DefaultBodyLimit::max(4 << 20))
        .with_state(engine);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    println!("listening on {host}:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
