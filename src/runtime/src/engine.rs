//! The worker-side contract for serving one decision at a time.
//!
//! A worker in this repository is three things: model code that turns request bytes into
//! response bytes, admission that decides what may run, and HTTP that carries the result. This
//! module defines the middle seam, so a model supplies only the first and inherits the rest.
//!
//! It lives in the runtime rather than in the frontend because admission is the runtime's:
//! `docs/architecture.md` gives the frontend transport and response delivery, and gives queues,
//! back-pressure and request bookkeeping to the worker-side runtime. A worker that implements
//! [`Engine`] can be dispatched through any admission policy here — including
//! [`crate::SerialScheduler`] — without the transport knowing which one is in use.
//!
//! Nothing here parses a decision envelope. Request bytes go in, response bytes come out, so a
//! field the runtime has never heard of survives and a model's error text cannot produce
//! invalid JSON.

use std::time::Instant;

use tokio::sync::oneshot;

/// How far the worker has got. An HTTP transport publishes this as readiness; anything that
/// only needs to know whether work may be sent can compare against [`Readiness::Ready`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Readiness {
    /// Loading, or warming up. Work must not be sent yet.
    Starting,
    Ready,
    /// Loading or execution failed. [`Engine::failure`] explains why, and the worker stays
    /// alive so an operator can read the reason rather than guess from an exit code.
    Failed,
}

/// What the worker reports about the work it is holding. Advisory: readiness never waits on it,
/// and a field the worker does not measure is left out rather than defaulted.
#[derive(Clone, Debug, Default)]
pub struct Report {
    /// Work accepted and not yet completed, running or queued.
    pub depth: usize,
    /// The depth at which admission starts refusing, so a reader can see how close the queue is
    /// to back-pressure without knowing the configuration.
    pub capacity: usize,
    /// Work the worker did not complete since it started: refused because there was no room, or
    /// accepted and then dropped because its caller had gone. Monotonic.
    pub rejected: u64,
}

/// The answer to one request: the bytes to return, and the queue depth at the moment the work
/// was admitted, so a transport can publish back-pressure before it becomes a refusal.
#[derive(Clone, Debug)]
pub struct Answer {
    pub body: Vec<u8>,
    pub depth: usize,
}

/// Why a worker could not answer. Each variant maps to one status code and one message; that
/// mapping belongs to the transport above, which is why the distinction is made here rather
/// than left to a string.
#[derive(Debug)]
pub enum EngineError {
    /// The caller's body is not a request this worker accepts.
    InvalidRequest(String),
    /// Out of capacity right now. The caller may retry.
    Busy,
    /// Not able to accept work at all: still loading, drained, or dead.
    Unavailable,
    /// The request was well formed and the worker failed to execute it.
    InferenceFailed,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(detail) => write!(f, "invalid request: {detail}"),
            Self::Busy => f.write_str("no queue capacity"),
            Self::Unavailable => f.write_str("engine unavailable"),
            Self::InferenceFailed => f.write_str("inference failed"),
        }
    }
}

impl std::error::Error for EngineError {}

/// The outcome of one request, delivered exactly once. A refusal travels through the same
/// channel as an answer, so a caller has one thing to await and one place to read the reason:
///
/// ```no_run
/// # use std::time::Instant;
/// # use omni_runtime::engine::Engine;
/// # async fn f(engine: &dyn Engine) {
/// match engine.submit(b"{}".to_vec(), Instant::now()).await {
///     Ok(answer) => { /* the response body is answer.body */ }
///     Err(refused) => { /* an EngineError the transport maps to a status */ }
/// }
/// # }
/// ```
pub type Reply = oneshot::Receiver<Result<Answer, EngineError>>;

/// What a model implements to be served.
///
/// Two methods are the whole contract: say whether work may be sent, and accept it. An
/// implementation that answers inline does both directly; one that owns a queue admits in
/// [`Engine::submit`] and reports its state through [`Engine::report`].
pub trait Engine: Send + Sync + 'static {
    /// Non-blocking. Polled on every request, so it must not lock behind queued work.
    fn readiness(&self) -> Readiness;

    /// Accepts one request if there is room, and returns what will carry its outcome.
    ///
    /// This does not fail: a refusal is an outcome, so it is sent through the reply. Keeping
    /// admission and the answer on one path is what lets a caller tell "there was no room" from
    /// "the runtime dropped my receiver", which two channels would make easy to confuse.
    ///
    /// `deadline` is the caller's budget, covering everything from the request headers to the
    /// response. Work that cannot finish inside it, or whose caller has already gone, is better
    /// refused than started.
    fn submit(&self, body: Vec<u8>, deadline: Instant) -> Reply;

    /// What to publish about the work being held. `None` leaves those fields out.
    fn report(&self) -> Option<Report> {
        None
    }

    /// Set only when [`Engine::readiness`] is [`Readiness::Failed`]; surfaced verbatim so an
    /// operator can read the cause without the server log.
    fn failure(&self) -> Option<String> {
        None
    }
}

/// Sends `outcome` if anyone is still waiting for it.
///
/// A worker whose caller has gone should not treat that as an error: a dropped receiver is how
/// cancellation is observed, and work the worker already captured still has to finish before
/// any device buffer it used is reused.
pub fn answer(
    reply: oneshot::Sender<Result<Answer, EngineError>>,
    outcome: Result<Answer, EngineError>,
) {
    let _ = reply.send(outcome);
}
