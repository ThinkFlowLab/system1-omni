//! The `omni-jev` binary. Two modes, selected by `OMNI_SYSTEMONE_ENGINE`.
//!
//! Without it, the process forwards to a worker over HTTP and owns no model state. With
//! `OMNI_SYSTEMONE_ENGINE=passthrough` it serves a linked engine in-process, which is the
//! shape a real model takes: the engine is built on a worker thread, `/health` follows its
//! warmup, and SIGTERM stops admission and drains accepted work before the process exits.

use std::{
    env,
    path::PathBuf,
    process::ExitCode,
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};

use omni_jev::{
    BoxError, Config,
    engine::{self, Answer, Engine, ServiceConfig},
    passthrough, worker,
};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("omni-jev: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), BoxError> {
    let bind = setting("OMNI_JEV_BIND", Config::DEFAULT_BIND.to_owned())?;
    // An unrecognized engine name is a configuration error, not a fallback: silently
    // forwarding would send inference somewhere the operator did not ask for.
    match env::var("OMNI_SYSTEMONE_ENGINE") {
        Err(env::VarError::NotPresent) => return forward(&bind).await,
        Ok(name) if name.is_empty() => return forward(&bind).await,
        Ok(name) if name == "passthrough" => {}
        Ok(name) => return Err(format!("unknown OMNI_SYSTEMONE_ENGINE {name:?}").into()),
        Err(error) => return Err(format!("OMNI_SYSTEMONE_ENGINE: {error}").into()),
    }

    let queue = setting("OMNI_SYSTEMONE_QUEUE", 32)?;
    let service = ServiceConfig {
        timeout: Duration::from_millis(setting("OMNI_SYSTEMONE_TIMEOUT_MS", 30_000)?),
        max_body: setting("OMNI_SYSTEMONE_MAX_BODY", 1024 * 1024)?,
    };
    let running = worker::spawn(
        PathBuf::from(setting(
            "OMNI_SYSTEMONE_CHECKPOINT",
            "/checkpoints".to_owned(),
        )?),
        worker::Options {
            queue_capacity: queue,
            drain: Duration::from_millis(setting("OMNI_SYSTEMONE_DRAIN_MS", 10_000)?),
        },
        // The engine is built here, on the worker thread, which is where a CUDA context
        // would be created and where it has to stay.
        passthrough::Passthrough::load,
        |_engine| Ok(()),
        |request| match request.body {
            [] => Err(worker::Failure::InvalidRequest("empty request body".into())),
            body => Ok(Answer {
                body: body.to_vec(),
                ..request.answer()
            }),
        },
    )?;
    let app = engine::app(Arc::clone(&running.handle) as Arc<dyn Engine>, service);
    let listener = TcpListener::bind(&bind).await?;
    // The `spawn` above returned only after loading and warmup, so discovery by this line
    // means `/health` answers 200 rather than a race with startup.
    eprintln!(
        "omni-jev listening on {} (engine passthrough, queue {queue})",
        listener.local_addr()?
    );

    // No timer runs while serving: the budget belongs to shutdown, and arming it at startup
    // would retire a healthy process once it elapsed.
    let drain = running.drain_budget();
    if !serve_until_stopped(listener, app, running, drain).await? {
        return Err("engine did not drain before shutdown".into());
    }
    Ok(())
}

/// How long responses already in flight get to be written out once the engine has stopped.
const DRAIN_GRACE: Duration = Duration::from_millis(250);

/// Serves until the process is asked to stop, then shuts down. Returns false if the shutdown
/// could not finish within the drain budget.
///
/// One `serve` call runs for the whole life of the process, and everything that has to happen
/// in order once a signal arrives happens in its shutdown future: close admission, start the
/// budget there rather than at startup, drain the worker, and only then let the transport
/// close. Draining the worker and draining the transport run concurrently, so an idle engine
/// exits immediately instead of waiting the budget out.
async fn serve_until_stopped(
    listener: TcpListener,
    app: axum::Router,
    running: worker::Running,
    drain: Duration,
) -> Result<bool, BoxError> {
    // Signals that the engine has stopped, so the transport can close as soon as the
    // responses it is still holding have been written.
    let (outcome, drained) = tokio::sync::oneshot::channel::<bool>();
    let admission = Arc::clone(&running.handle);
    let shutdown = async move {
        stop_signal().await;
        // Admission closes first: a request still uploading when shutdown begins could
        // otherwise submit work that keeps the engine busy for its whole request budget,
        // long after the budget below has passed.
        admission.stop_accepting();
        let deadline = Instant::now() + drain;
        // Off the runtime thread: this waits on the engine thread, and blocking a worker here
        // would stall the transport draining alongside it.
        let done = tokio::task::spawn_blocking(move || running.run_until_drained(deadline))
            .await
            .unwrap_or(false);
        let _ = outcome.send(done);
        // Responses in flight get a short grace period to be written out, never more than
        // what is left of the budget: an engine that used the whole budget leaves none, and a
        // handler still stuck on a slow upload could not have finished either way.
        let grace = (Instant::now() + DRAIN_GRACE).min(deadline);
        tokio::time::sleep_until(grace.into()).await;
    };
    engine::serve(listener, app, shutdown).await?;
    // `serve` returns only once shutdown has finished, so the outcome is always published.
    Ok(drained.await.unwrap_or(false))
}

/// Forwards to an HTTP worker, owning no model state.
async fn forward(bind: &str) -> Result<(), BoxError> {
    let backend = setting(
        "OMNI_JEV_BACKEND_URL",
        Config::DEFAULT_BACKEND_URL.to_owned(),
    )?;
    let config = Config::new(bind, &backend)?;
    let app = omni_jev::app(&config)?;
    let listener = TcpListener::bind(&config.bind).await?;
    // This line is the documented way to find a port bound with `:0`, and the way a
    // supervisor knows the process is up. It stays first and stays unchanged.
    eprintln!("omni-jev listening on {}", listener.local_addr()?);
    eprintln!("forwarding to {}", config.backend_url);
    engine::serve(listener, app, shutdown_signal()).await?;
    Ok(())
}

/// Reads a setting, failing on a malformed value rather than quietly using the default: a
/// typo in `OMNI_SYSTEMONE_TIMEOUT_MS` must not become a timeout nobody asked for.
fn setting<T: FromStr>(name: &str, default: T) -> Result<T, BoxError>
where
    T::Err: std::fmt::Display,
{
    match env::var(name) {
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("{name}: {error}").into()),
        Ok(value) if value.is_empty() => Ok(default),
        Ok(value) => value
            .parse()
            .map_err(|error| format!("{name}: {error}").into()),
    }
}

/// Resolves when the process is asked to stop. The forwarding path owns no worker, so it
/// has nothing to admit or drain.
async fn shutdown_signal() {
    stop_signal().await
}

async fn stop_signal() {
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut sigterm) => {
                sigterm.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        () = terminate => {}
    }
}
