//! JEV_VL_MODEL=<merged export> omni-jev-vl-native

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_jev_vl_native::contract::{MODEL_ID, Reject};
use omni_jev_vl_native::engine::Engine;
use omni_qwen3_5_native::cuda;
use serde_json::{Value, json};

const WARMUP: &[u8] = br#"{"kind":"choice","state":"Dialog: Update installed.","question":"Close it?","options":["OK","Wait"]}"#;

fn reject(r: Reject) -> Response {
    (
        StatusCode::from_u16(r.status).unwrap_or(StatusCode::BAD_REQUEST),
        Json(r.body),
    )
        .into_response()
}

async fn decide(engine: &Engine, raw: &[u8]) -> Response {
    let prepared = match engine
        .processor
        .prepare(&engine.head, &engine.manifest, raw)
    {
        Ok(prepared) => prepared,
        Err(r) => return reject(r),
    };
    let cache_note = prepared.cache_note.clone();
    let result = async {
        let probs = engine
            .executor
            .execute(&engine.scheduler, prepared.plan, prepared.readout)
            .await?;
        anyhow::Ok(prepared.context.finish(probs))
    }
    .await;
    match result {
        Ok(body) => {
            let mut resp = Json(body).into_response();
            if !cache_note.is_empty() {
                // Expose cache decisions without changing the response schema.
                resp.headers_mut().insert(
                    "x-jev-cache",
                    axum::http::HeaderValue::from_str(&cache_note)
                        .unwrap_or(axum::http::HeaderValue::from_static("l1=?,l3=?,p=?")),
                );
            }
            resp
        }
        Err(e) => {
            eprintln!("inference failed: {e:#}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "model inference failed"})),
            )
                .into_response()
        }
    }
}

async fn systemone(State(engine): State<Arc<Engine>>, headers: HeaderMap, body: Bytes) -> Response {
    // Upstream pydantic validates the raw body even for a non-JSON content type;
    // mirroring it keeps the wrong-content-type probe's 400 (not 415) semantics.
    let ctype = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim();
    if !ctype.eq_ignore_ascii_case("application/json") {
        return reject(Reject::validation(
            format!(
                "1 validation error:\n  {{'type': 'model_attributes_type', 'loc': 'body', 'msg': 'Input should be a valid dictionary or object to extract fields from', 'input': '<bytes of {} bytes>'}}",
                body.len()
            ),
            "body",
        ));
    }
    decide(&engine, &body).await
}

async fn fallback(uri: Uri, body: Bytes) -> Response {
    let model = serde_json::from_slice::<Value>(&body)
        .ok()
        .and_then(|v| v.get("model").and_then(Value::as_str).map(str::to_owned));
    let _ = uri;
    reject(Reject::unknown_model(
        model.as_deref().unwrap_or("no-such-model"),
    ))
}

/// Cumulative cache counters and resident sizes, separate from decision responses.
async fn cache_stats(State(engine): State<Arc<Engine>>) -> Response {
    let s = engine.caches.snapshot();
    let cfg = &engine.caches.cfg;
    Json(json!({
        "enabled": cfg.enabled,
        "l1": cfg.l1, "l2": cfg.l2, "l3": cfg.l3,
        "counters": {
            "l1_hit": s.l1_hit, "l1_miss": s.l1_miss,
            "l2_hit": s.l2_hit, "l2_miss": s.l2_miss,
            "l3_hit": s.l3_hit, "l3_miss": s.l3_miss,
            "l3_populate": s.l3_populate, "l3_fallback_full": s.l3_fallback_full,
        },
        "records": {
            "l1_records": s.l1_records, "l1_bytes": s.l1_bytes,
            "l2_records": s.l2_records, "l2_bytes": s.l2_bytes,
            "l3_records": s.l3_records, "l3_bytes": s.l3_bytes,
        },
        "budgets": {
            "l1_max": cfg.l1_max, "l2_bytes": cfg.l2_bytes, "l3_bytes": cfg.l3_bytes,
        },
    }))
    .into_response()
}

async fn cache_reset(State(engine): State<Arc<Engine>>) -> Response {
    engine.caches.reset_stats();
    Json(json!({"reset": true})).into_response()
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var_os("JEV_VL_MODEL").context("set JEV_VL_MODEL")?;
    let library = std::env::var_os("JEV_VL_CUDA_LIB")
        .map(Into::into)
        .map_or_else(cuda::default_library, Ok)?;
    let engine = Arc::new(Engine::load(model.as_ref(), &library).await?);
    ensure!(
        decide(&engine, WARMUP).await.status() == StatusCode::OK,
        "warmup failed"
    );
    let host = std::env::var("JEV_VL_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("JEV_VL_PORT")
        .map_or(Ok(8001), |v| v.parse())
        .context("JEV_VL_PORT")?;
    let app = Router::new()
        .route(
            "/health",
            get(|| async { Json(json!({"status": "ready", "model": MODEL_ID})) }),
        )
        .route("/v1/systemone", post(systemone))
        .route("/v1/cache/stats", get(cache_stats))
        .route("/v1/cache/reset", post(cache_reset))
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(4 << 20))
        .with_state(engine);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    println!("listening on {host}:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
#[path = "../../../../../tests/jev_vl/http.rs"]
mod http_tests;
