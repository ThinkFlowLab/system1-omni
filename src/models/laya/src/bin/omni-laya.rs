#[cfg(feature = "serve")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use omni_laya::serve::{Engine, router};
    use std::{path::PathBuf, sync::Arc};
    let checkpoint =
        PathBuf::from(std::env::var_os("LAYA_CHECKPOINT").context("set LAYA_CHECKPOINT")?);
    let bundle =
        PathBuf::from(std::env::var_os("LAYA_CUDA_BUNDLE").context("set LAYA_CUDA_BUNDLE")?);
    let engine = Arc::new(Engine::load(&checkpoint, &bundle).await?);
    let warmup = br#"{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}"#;
    ensure!(
        engine.decide(warmup).await.status().is_success(),
        "warmup failed"
    );
    let host = std::env::var("LAYA_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("LAYA_PORT")
        .map_or(Ok(8000), |s| s.parse())
        .context("LAYA_PORT")?;
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
    println!("listening on {host}:{port}");
    axum::serve(listener, router(engine))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
#[cfg(not(feature = "serve"))]
fn main() {
    eprintln!("omni-laya requires --features serve");
    std::process::exit(2);
}
