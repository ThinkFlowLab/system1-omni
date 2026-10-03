//! Screenshot `/v1/systemone` worker using native RGB, vision and language stages.
use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_cua_s1_native::{
    contract, cuda,
    image_request::{self, MODEL_ID},
    vision_engine::VisionEngine,
};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

type Shared = Arc<Mutex<VisionEngine>>;
fn reply(status: StatusCode, value: Value) -> Response {
    (status, Json(value)).into_response()
}

async fn decide(State(engine): State<Shared>, body: Result<Bytes, BytesRejection>) -> Response {
    let raw = match body {
        Ok(b) => b,
        Err(e) => return reply(e.status(), json!({"detail": e.body_text()})),
    };
    let body = match contract::parse_body(&raw) {
        Ok(b) => b,
        Err(e) => {
            return reply(
                StatusCode::from_u16(e.status).unwrap(),
                json!({"detail":e.message}),
            );
        }
    };
    // Decode and inference both run off the async executor. The mutex serializes
    // the CUDA models; prepared image features live only for this request.
    match tokio::task::spawn_blocking(move || -> Result<(StatusCode, Value)> {
        let request = match image_request::parse_image_body(&body) {
            Ok(r) => r,
            Err(e) => {
                return Ok((
                    StatusCode::UNPROCESSABLE_ENTITY,
                    json!({"detail":e.to_string()}),
                ));
            }
        };
        let mut engine = engine
            .lock()
            .map_err(|_| anyhow::anyhow!("poisoned engine"))?;
        let prepared = match engine.prepare(
            request.width,
            request.height,
            &request.rgb,
            &request.questions,
        ) {
            Ok(p) => p,
            Err(e) => {
                return Ok((
                    StatusCode::UNPROCESSABLE_ENTITY,
                    json!({"detail": e.to_string()}),
                ));
            }
        };
        Ok((StatusCode::OK, engine.predict_prepared(&prepared)?))
    })
    .await
    {
        Ok(Ok((status, body))) => reply(status, body),
        err => {
            eprintln!("native image inference failed: {err:?}");
            reply(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail":"inference failed"}),
            )
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let path = |name| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .with_context(|| format!("set {name}"))
    };
    let library = std::env::var_os("CUA_S1_CUDA_LIB")
        .map(PathBuf::from)
        .map_or_else(cuda::default_library, Ok)?;
    let engine = VisionEngine::load(
        &path("CUA_S1_BASE")?,
        &path("CUA_S1_VISION_ADAPTER")?,
        &path("CUA_S1_MODEL")?,
        &library,
    )?;
    let host = std::env::var("CUA_S1_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("CUA_S1_PORT").map_or(Ok(8000), |p| p.parse())?;
    let app = Router::new()
        .route(
            "/health",
            get(|| async { Json(json!({"status":"ready", "model": MODEL_ID})) }),
        )
        .route("/v1/systemone", post(decide))
        .layer(DefaultBodyLimit::max(image_request::MAX_BODY))
        .with_state(Arc::new(Mutex::new(engine)));
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    println!("native vision worker listening on {host}:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
