use anyhow::{Context, Result};
use omni_decider_native::{engine::Engine, serve};
use omni_qwen3_5_native::cuda;
use std::{path::PathBuf, sync::Arc};

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var_os("DECIDER_MODEL")
        .map(PathBuf::from)
        .context("set DECIDER_MODEL to pinned Decider-2B v11 directory")?;
    let library = std::env::var_os("DECIDER_CUDA_LIB")
        .map(PathBuf::from)
        .map_or_else(cuda::default_library, Ok)?;
    let host = std::env::var("DECIDER_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("DECIDER_PORT")
        .map_or(Ok(8000), |v| v.parse())
        .context("DECIDER_PORT must be a valid port")?;
    let engine = Arc::new(Engine::load(&model, &library).await?);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    println!("native Decider worker listening on {host}:{port}");
    axum::serve(listener, serve::router(engine))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
