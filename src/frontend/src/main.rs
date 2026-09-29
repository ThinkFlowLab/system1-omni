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

/// How long the transport gets to write out what it is still holding once the engine has
/// stopped. An idle keep-alive connection is still an open connection and graceful shutdown
/// waits for it, so this is what stops that wait from running to the whole drain budget.
const CLOSE_GRACE: Duration = Duration::from_millis(250);

/// Serves until the process is asked to stop, then shuts down. Returns false if the engine
/// could not finish within the drain budget.
///
/// One `serve` call runs for the life of the process, and everything that has to happen in
/// order once a signal arrives happens in its shutdown future: close admission, take the one
/// deadline everything shares, drain the worker, then let the transport write out what it is
/// still holding.
///
/// **One absolute deadline, taken once at the signal, and no period added on top of it.** The
/// worker drains until that instant; the transport then gets whatever remains of it, up to a
/// bounded grace. A grace measured from the end of the drain instead would make the process
/// live `drain + grace` however little work was left. And the transport cannot be asked to
/// stop — it waits for open connections for as long as they are open, including idle
/// keep-alive ones that would never close on their own — so the sequence is bounded by no
/// longer waiting on it.
async fn serve_until_stopped(
    listener: TcpListener,
    app: axum::Router,
    running: worker::Running,
    drain: Duration,
) -> Result<bool, BoxError> {
    let (outcome, drained) = tokio::sync::oneshot::channel::<bool>();
    // Tells the outer wait that the engine is done, and when the transport's own grace period
    // ends. An oneshot rather than a `Notify`, which stores a permit and would let the wait
    // finish before the window had actually been served.
    let (closed, finish) = tokio::sync::oneshot::channel::<Instant>();
    let admission = Arc::clone(&running.handle);
    let shutdown = async move {
        stop_signal().await;
        // Admission closes first: a request still uploading when shutdown begins could
        // otherwise submit work that keeps the engine busy for its whole request budget,
        // long after the deadline below has passed.
        admission.stop_accepting();
        let deadline = Instant::now() + drain;
        // Off the runtime thread: this waits on the engine thread, and blocking a worker here
        // would stall the transport draining alongside it.
        let done = tokio::task::spawn_blocking(move || running.run_until_drained(deadline))
            .await
            .unwrap_or(false);
        let _ = outcome.send(done);
        // The transport's window is a bounded grace, clamped to the deadline itself. Both
        // halves matter: without the bound an idle keep-alive connection holds the process to
        // the whole drain budget, and without the clamp the process lives `drain + grace`
        // however little work was left.
        let _ = closed.send(deadline.min(Instant::now() + CLOSE_GRACE));
        // Nothing left to do but let the transport finish. The caller stops waiting at the
        // instant above whether or not this resolves, so this needs no bound of its own.
        std::future::pending::<()>().await;
    };

    let mut serving = std::pin::pin!(engine::serve(listener, app, shutdown));
    tokio::select! {
        // Both branches run only once a signal is in: `serve` cannot resolve before its
        // shutdown future does, and that future waits for the signal.
        waited = &mut serving => waited?,
        ends = finish => {
            let now = Instant::now();
            let ends = ends.unwrap_or_else(|_| (now + CLOSE_GRACE).min(now + drain));
            // Polled rather than awaited once: the transport writes out what it is holding and
            // is done, so this returns as soon as it does. Waiting a fixed period here instead
            // would make every shutdown last that period, however little was left.
            let mut serving = &mut serving;
            let settled = std::future::poll_fn(|cx| {
                if Instant::now() >= ends {
                    return std::task::Poll::Ready(false);
                }
                match std::future::Future::poll(std::pin::Pin::new(&mut serving), cx) {
                    std::task::Poll::Ready(Ok(())) => std::task::Poll::Ready(true),
                    std::task::Poll::Ready(Err(error)) => {
                        panic!("engine::serve failed during shutdown: {error}")
                    }
                    std::task::Poll::Pending => {
                        // Wake when the window closes, so a transport that never finishes
                        // cannot outlive it.
                        cx.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                }
            })
            .await;
            if !settled {
                eprintln!("omni-jev: shutdown did not finish within {drain:?}, exiting");
            }
        }
    }
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
