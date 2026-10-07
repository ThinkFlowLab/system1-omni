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
    vision_processing::VisionProcessor,
};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
struct Shared {
    processor: Arc<VisionProcessor>,
    executor: Arc<Mutex<VisionEngine>>,
    scheduler: omni_runtime::SerialScheduler,
}
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
    // CPU decode/preparation does not hold the model mutex or admission permit.
    let processor = engine.processor.clone();
    let prepared = match tokio::task::spawn_blocking(move || {
        let request = image_request::parse_image_body(&body)?;
        processor.prepare(
            request.width,
            request.height,
            &request.rgb,
            &request.questions,
        )
    })
    .await
    {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(e)) => {
            return reply(
                StatusCode::UNPROCESSABLE_ENTITY,
                json!({"detail": e.to_string()}),
            );
        }
        Err(e) => {
            eprintln!("native image preparation failed: {e}");
            return reply(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail":"inference failed"}),
            );
        }
    };
    let executor = engine.executor.clone();
    let result = engine
        .scheduler
        .run(move || {
            let rows = executor
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned engine"))?
                .execute_prepared(&prepared)?;
            Ok((prepared.context, rows))
        })
        .await
        .and_then(|(context, rows)| context.finish(rows));
    match result {
        Ok(body) => reply(StatusCode::OK, body),
        Err(e) => {
            eprintln!("native image inference failed: {e:#}");
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
    let (base, adapter, language) = (
        path("CUA_S1_BASE")?,
        path("CUA_S1_VISION_ADAPTER")?,
        path("CUA_S1_MODEL")?,
    );
    let mut executor = tokio::task::spawn_blocking(move || {
        VisionEngine::load(&base, &adapter, &language, &library)
    })
    .await??;
    let warmup = executor.prepare(
        1,
        1,
        &[0, 0, 0],
        &[contract::Question {
            name: "warmup".into(),
            goal: String::new(),
            keys: vec!["continue".into()],
            labels: vec!["Continue".into()],
        }],
    )?;
    let engine = tokio::task::spawn_blocking(move || -> Result<VisionEngine> {
        executor.predict_prepared(&warmup)?;
        Ok(executor)
    })
    .await?
    .map(|executor| Shared {
        processor: executor.processor.clone(),
        executor: Arc::new(Mutex::new(executor)),
        scheduler: omni_runtime::SerialScheduler::default(),
    })?;
    let host = std::env::var("CUA_S1_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("CUA_S1_PORT").map_or(Ok(8000), |p| p.parse())?;
    let app = Router::new()
        .route(
            "/health",
            get(|| async { Json(json!({"status":"ready", "model": MODEL_ID})) }),
        )
        .route("/v1/systemone", post(decide))
        .layer(DefaultBodyLimit::max(image_request::MAX_BODY))
        .with_state(engine);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    println!("native vision worker listening on {host}:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
