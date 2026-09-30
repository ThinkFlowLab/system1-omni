//! Cua-S1 4B 0.2 (`text` adapter) `/v1/systemone` worker on the native CUDA kernels.
//!
//!     CUA_S1_MODEL=<merged text checkpoint> omni-cua-s1-native
//!
//! `CUA_S1_CUDA_LIB` (default: next to this executable), `CUA_S1_HOST` and `CUA_S1_PORT`
//! are optional; see recipe/cua_s1/native.md.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use omni_cua_s1_native::contract::{self, MODEL_ID};
use omni_cua_s1_native::cuda;
use omni_cua_s1_native::engine::Engine;
use omni_cua_s1_native::json::quote;

const MAX_BODY_BYTES: usize = 4 << 20;
const MAX_PROMPT_TOKENS: usize = 16384;
const WARMUP: &[u8] = br#"{"model": "cua-s1-4b-0.2", "state": "Dialog: Update installed.", "questions": {"q": {"type": "choice", "instructions": "Close it.", "criteria": {"ok": "OK", "wait": "Wait"}}}}"#;

fn reply(status: u16, body: Value) -> Response {
    (StatusCode::from_u16(status).unwrap(), Json(body)).into_response()
}

/// Every prompt is tokenized and checked against the limit before any forward pass.
async fn decide(engine: &Engine, raw: &[u8]) -> Response {
    let (state, questions) = match contract::parse_body(raw).and_then(|b| contract::map_request(&b))
    {
        Ok(request) => request,
        Err(e) => return reply(e.status, json!({"detail": e.message})),
    };
    let failed = |e: anyhow::Error| {
        eprintln!("inference failed: {e:#}");
        reply(500, json!({"detail": "inference failed"}))
    };
    let mut prompts = Vec::with_capacity(questions.len());
    for q in &questions {
        let ids = match engine.encode(&state, q) {
            Ok(ids) => ids,
            Err(e) => return failed(e),
        };
        if ids.len() > MAX_PROMPT_TOKENS {
            let message = format!(
                "question {}: {} prompt tokens, over {MAX_PROMPT_TOKENS}",
                quote(&q.name),
                ids.len()
            );
            return reply(413, json!({"detail": message}));
        }
        prompts.push(ids);
    }
    let tokens: usize = prompts.iter().map(Vec::len).sum();
    let mut answers = serde_json::Map::new();
    for (q, ids) in questions.iter().zip(prompts) {
        match engine.score(ids, q.keys.len()).await {
            Ok(probs) => answers.insert(q.name.clone(), contract::answer(q, &probs)),
            Err(e) => return failed(e),
        };
    }
    reply(
        200,
        json!({"model": MODEL_ID, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0}}),
    )
}

async fn systemone(
    State(engine): State<Arc<Engine>>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    match body {
        Ok(raw) => decide(&engine, &raw).await,
        Err(e) => reply(e.status().as_u16(), json!({"detail": e.body_text()})),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var_os("CUA_S1_MODEL").context("set CUA_S1_MODEL")?;
    let library = match std::env::var_os("CUA_S1_CUDA_LIB") {
        Some(path) => path.into(),
        None => cuda::default_library()?,
    };
    let engine = Arc::new(Engine::load(model.as_ref(), &library).await?);
    // one decision before listening, so the first request does not pay for first-call setup
    ensure!(
        decide(&engine, WARMUP).await.status() == StatusCode::OK,
        "warmup failed"
    );
    let host = std::env::var("CUA_S1_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("CUA_S1_PORT")
        .map_or(Ok(8000), |p| p.parse())
        .context("CUA_S1_PORT")?;
    let app = Router::new()
        .route(
            "/health",
            get(|| async { Json(json!({"status": "ready", "model": MODEL_ID})) }),
        )
        .route("/v1/systemone", post(systemone))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(engine);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    println!("listening on {host}:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
