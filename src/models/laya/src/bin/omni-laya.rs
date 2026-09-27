#[cfg(feature = "serve")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use std::{path::PathBuf, time::Duration};
    let args: Vec<_> = std::env::args().collect();
    let checkpoint = PathBuf::from(
        args.get(1)
            .context("usage: omni-laya CHECKPOINT CUDA_BUNDLE [BIND]")?,
    );
    let bundle = PathBuf::from(args.get(2).context("missing CUDA bundle")?);
    let bind = args.get(3).map(String::as_str).unwrap_or("127.0.0.1:8080");
    let listener = tokio::net::TcpListener::bind(bind).await?;
    let (engine, worker) = omni_laya::serve::start(checkpoint, bundle, 32).await?;
    let app = omni_jev::engine::app(engine.clone(), Duration::from_secs(30), 1024 * 1024);
    eprintln!("READY native Laya HTTP {}", listener.local_addr()?);
    let stop = engine.clone();
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            #[cfg(unix)]
            {
                let mut term =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("install SIGTERM handler");
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {},
                    _ = term.recv() => {},
                }
            }
            #[cfg(not(unix))]
            let _ = tokio::signal::ctrl_c().await;
            stop.stop_accepting();
        })
        .await;
    engine.stop_accepting();
    drop(engine);
    tokio::task::spawn_blocking(move || worker.join())
        .await?
        .map_err(|_| anyhow::anyhow!("GPU worker panicked"))?;
    result?;
    Ok(())
}
#[cfg(not(feature = "serve"))]
fn main() {
    eprintln!("omni-laya requires --features serve");
    std::process::exit(2);
}
