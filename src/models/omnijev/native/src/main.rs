//! OmniJev-4B v1.1 `/v1/systemone` worker: CPU preparation, then native vision and
//! language passes behind serial admission.
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use omni_omnijev_native::{
    contract::{MAX_BODY, MODEL_ID},
    executor::{self, Executor},
    processing::Processor,
};
use omni_qwen3_5_native::{cuda, json};
use serde_json::{Value, json};
use tokio::sync::Semaphore;

/// Requests prepared at once; each decodes an image of up to 4096 × 4096.
const PREPARING: usize = 2;
/// Requests in the worker at once, preparing, waiting or running; more get 503.
const ADMITTED: usize = 16;

#[derive(Clone)]
struct Shared {
    processor: Arc<Processor>,
    executor: Arc<Mutex<Executor>>,
    scheduler: omni_runtime::SerialScheduler,
    preparing: Arc<Semaphore>,
    admitted: Arc<AtomicUsize>,
}

/// One admitted request, released on drop.
struct Admission(Arc<AtomicUsize>);

impl Drop for Admission {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn reply(status: StatusCode, value: Value) -> Response {
    (status, Json(value)).into_response()
}

async fn decide(State(shared): State<Shared>, body: Result<Bytes, BytesRejection>) -> Response {
    let raw = match body {
        Ok(b) => b,
        Err(e) => return reply(e.status(), json!({"detail": e.body_text()})),
    };
    // The strict parser the contract uses: duplicate keys and out-of-range numbers are
    // malformed too.
    if let Err(e) = json::parse(&raw) {
        return reply(StatusCode::BAD_REQUEST, json!({"detail": e}));
    }
    if shared.admitted.fetch_add(1, Ordering::SeqCst) >= ADMITTED {
        shared.admitted.fetch_sub(1, Ordering::SeqCst);
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            [(header::RETRY_AFTER, "1")],
            Json(json!({"detail": "the worker is busy"})),
        )
            .into_response();
    }
    let _admission = Admission(shared.admitted.clone());
    // Decoding and preparation run before the scheduler and without the model lock.
    let permit = shared.preparing.clone().acquire_owned().await;
    let processor = shared.processor.clone();
    let prepared = match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        processor.prepare(&raw)
    })
    .await
    {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(e)) => {
            return reply(
                StatusCode::UNPROCESSABLE_ENTITY,
                json!({"detail": format!("{e:#}")}),
            );
        }
        Err(e) => {
            eprintln!("preparation failed: {e}");
            return reply(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": "inference failed"}),
            );
        }
    };
    let executor = shared.executor.clone();
    let result = shared
        .scheduler
        .run(move || {
            let answers = executor
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned executor"))?
                .execute(&prepared)?;
            Ok(executor::response(&prepared, answers))
        })
        .await;
    match result {
        Ok(body) => reply(StatusCode::OK, body),
        Err(e) => {
            eprintln!("inference failed: {e:#}");
            reply(
                StatusCode::INTERNAL_SERVER_ERROR,
                json!({"detail": "inference failed"}),
            )
        }
    }
}

/// A request with every question type, an option text for the vocabulary pass and a
/// region, run once before the worker reports ready.
fn warmup_request() -> Vec<u8> {
    let image = image::RgbImage::from_fn(96, 64, |x, y| image::Rgb([x as u8, y as u8, 128]));
    let mut png = Vec::new();
    image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encoding the warmup image");
    let url = format!(
        "data:image/png;base64,{}",
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, png)
    );
    serde_json::to_vec(&json!({
        "model": MODEL_ID,
        "state": {"images": [url]},
        "questions": {
            "noul": {"type": "noul", "instructions": "The image is mostly blue.",
                     "region": {"box": [0, 0, 48, 32]}},
            "choice": {"type": "choice", "instructions": "Which colour dominates?",
                       "criteria": {"red": null, "green": "Mostly green", "blue": null}},
            "score": {"type": "score", "instructions": "How bright is the image?",
                      "levels": ["dark", "medium", "bright"]},
        },
    }))
    .expect("serializing the warmup request")
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::var_os("OMNIJEV_MODEL")
        .map(PathBuf::from)
        .context("set OMNIJEV_MODEL to the export directory")?;
    let library = std::env::var_os("OMNIJEV_CUDA_LIB")
        .map(PathBuf::from)
        .map_or_else(cuda::default_library, Ok)?;
    let shared = tokio::task::spawn_blocking(move || -> Result<Shared> {
        let (mut executor, processor) = Executor::load(&dir, &library)?;
        let warmup = processor.prepare(&warmup_request())?;
        executor.execute(&warmup).context("warmup request")?;
        Ok(Shared {
            processor: Arc::new(processor),
            executor: Arc::new(Mutex::new(executor)),
            scheduler: omni_runtime::SerialScheduler::default(),
            preparing: Arc::new(Semaphore::new(PREPARING)),
            admitted: Arc::new(AtomicUsize::new(0)),
        })
    })
    .await??;
    let host = std::env::var("OMNIJEV_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("OMNIJEV_PORT").map_or(Ok(8000), |p| p.parse())?;
    let app = Router::new()
        .route(
            "/health",
            get(|| async { Json(json!({"status": "ready", "model": MODEL_ID})) }),
        )
        .route("/v1/systemone", post(decide))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(shared);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    println!("OmniJev worker listening on {host}:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
