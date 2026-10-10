//! The in-process path over real sockets: client -> omni-jev -> a worker implementing the
//! runtime's engine contract.
//!
//! The worker here is a stand-in, so none of this needs a model or a GPU. What it checks is the
//! transport's half of the contract: which status a given worker state produces, that bytes and
//! the budget are handled as documented, and that a worker which never answers is cut off.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use omni_jev::engine::{QUEUE_DEPTH, ServiceConfig, app};
use omni_runtime::engine::{Answer, Engine, EngineError, Readiness, Reply, Report};
use tokio::sync::oneshot;
mod common;

use common::{client, listen};

const REQUEST: &str = r#"{"model":"english","state":"refund please","questions":{"q":{"type":"noul","instructions":"Ask?"}}}"#;

/// How the stand-in answers. Each mode is one of the states the transport has to tell apart, so
/// the status codes below are the contract rather than a convention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Ready,
    Starting,
    Failed,
    /// Accepts the request and never answers it.
    Silent,
    /// Refuses because there is no room.
    Busy,
    /// Refuses because the body is not a request it accepts.
    Rejects,
    /// Accepts the request and fails to run it.
    Fails,
}

struct Fake {
    mode: Mode,
    depth: usize,
    capacity: usize,
    rejected: u64,
    /// Requests that reached the engine at all.
    seen: Arc<AtomicUsize>,
}

impl Fake {
    /// A ready worker that reports the given queue depth.
    fn reporting_depth(depth: usize) -> Arc<Self> {
        Arc::new(Self {
            mode: Mode::Ready,
            depth,
            ..Self::quiet()
        })
    }

    fn quiet() -> Self {
        Self {
            mode: Mode::Ready,
            depth: 0,
            capacity: 4,
            rejected: 0,
            seen: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn new(mode: Mode) -> Arc<Self> {
        Arc::new(Self {
            mode,
            ..Self::quiet()
        })
    }
}

impl Engine for Fake {
    fn readiness(&self) -> Readiness {
        match self.mode {
            Mode::Starting => Readiness::Starting,
            Mode::Failed => Readiness::Failed,
            _ => Readiness::Ready,
        }
    }

    fn submit(&self, body: Vec<u8>, _deadline: Instant) -> Reply {
        self.seen.fetch_add(1, Ordering::AcqRel);
        let (reply, receiver) = oneshot::channel();
        match self.mode {
            Mode::Busy => {
                let _ = reply.send(Err(EngineError::Busy));
            }
            Mode::Rejects => {
                let _ = reply.send(Err(EngineError::InvalidRequest(
                    "bad \"envelope\"\n".into(),
                )));
            }
            Mode::Fails => {
                let _ = reply.send(Err(EngineError::InferenceFailed));
            }
            // Keeping the sender alive means the request neither completes nor fails: only the
            // caller's budget can end it.
            Mode::Silent => std::mem::forget(reply),
            _ => {
                let depth = self.depth;
                tokio::spawn(async move {
                    let _ = reply.send(Ok(Answer { body, depth }));
                });
            }
        }
        receiver
    }

    fn report(&self) -> Option<Report> {
        Some(Report {
            depth: self.depth,
            capacity: self.capacity,
            rejected: self.rejected,
        })
    }

    fn failure(&self) -> Option<String> {
        matches!(self.mode, Mode::Failed).then(|| "no such checkpoint".to_owned())
    }
}

/// Starts the in-process service and returns its base URL, with the worker's own counters.
async fn start(mode: Mode, config: ServiceConfig) -> (String, Arc<AtomicUsize>) {
    let fake = Fake::new(mode);
    let seen = Arc::clone(&fake.seen);
    (listen(app(fake, config)).await, seen)
}

async fn post(url: &str, body: &str) -> reqwest::Response {
    client()
        .post(format!("{url}/v1/systemone"))
        .body(body.to_owned())
        .send()
        .await
        .unwrap()
}

async fn health(url: &str) -> (reqwest::StatusCode, String) {
    let response = client().get(format!("{url}/health")).send().await.unwrap();
    let status = response.status();
    (status, response.text().await.unwrap())
}

#[tokio::test]
async fn readiness_decides_whether_work_is_accepted() {
    // Ready answers; the other two refuse without the worker ever seeing the request, and the
    // health body says which of the three it is.
    for (mode, status, expected) in [
        (
            Mode::Ready,
            200,
            r#"{"status":"ok","depth":0,"capacity":4,"rejected":0}"#,
        ),
        (Mode::Starting, 503, r#"{"status":"starting"}"#),
        (
            Mode::Failed,
            503,
            r#"{"status":"failed","reason":"no such checkpoint"}"#,
        ),
    ] {
        let (url, seen) = start(mode, ServiceConfig::default()).await;
        let (code, body) = health(&url).await;
        assert_eq!(
            (code.as_u16(), body.as_str()),
            (status, expected),
            "{mode:?}"
        );

        let response = post(&url, REQUEST).await;
        assert_eq!(response.status().as_u16(), status, "{mode:?}");
        if mode != Mode::Ready {
            assert_eq!(
                seen.load(Ordering::Acquire),
                0,
                "{mode:?}: an unready worker must not see the request"
            );
        }
    }
}

#[tokio::test]
async fn a_decision_survives_the_transport_byte_for_byte() {
    let (url, _) = start(Mode::Ready, ServiceConfig::default()).await;
    let response = post(&url, REQUEST).await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()["content-type"],
        "application/json",
        "the transport labels its own responses"
    );
    assert_eq!(response.text().await.unwrap(), REQUEST);
}

#[tokio::test]
async fn a_body_over_the_limit_is_refused_before_the_worker_sees_it() {
    let (url, seen) = start(
        Mode::Ready,
        ServiceConfig {
            max_body: 32,
            ..Default::default()
        },
    )
    .await;
    assert_eq!(post(&url, &"x".repeat(64)).await.status(), 413);
    assert_eq!(seen.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn a_busy_worker_produces_503_and_is_not_retried() {
    let (url, seen) = start(Mode::Busy, ServiceConfig::default()).await;
    let response = post(&url, REQUEST).await;
    assert_eq!(response.status(), 503);
    assert_eq!(
        response.text().await.unwrap(),
        r#"{"error":"inference queue full"}"#
    );
    assert_eq!(
        seen.load(Ordering::Acquire),
        1,
        "a refusal is an answer, not a reason to ask again"
    );
}

#[tokio::test]
async fn worker_errors_map_to_one_status_each() {
    for (mode, status, expected) in [
        (
            Mode::Rejects,
            400,
            "{\"error\":\"bad \\\"envelope\\\"\\n\"}",
        ),
        (Mode::Fails, 500, r#"{"error":"inference failed"}"#),
    ] {
        let (url, _) = start(mode, ServiceConfig::default()).await;
        let response = post(&url, REQUEST).await;
        assert_eq!(response.status().as_u16(), status, "{mode:?}");
        // The worker's text is passed through as JSON, so it has to survive being quoted; the
        // fake's message contains a quote and a newline on purpose.
        assert_eq!(response.text().await.unwrap(), expected, "{mode:?}");
    }
}

#[tokio::test]
async fn the_reply_publishes_the_depth_the_worker_reported() {
    // The worker, not the transport, measures its own queue, so the header carries whatever the
    // reply said. A depth of zero is included rather than treated as absent.
    let url = listen(app(Fake::reporting_depth(3), ServiceConfig::default())).await;

    let response = post(&url, REQUEST).await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()[QUEUE_DEPTH],
        "3",
        "back-pressure has to be visible before it becomes a refusal"
    );
}

#[tokio::test]
async fn a_worker_that_never_answers_is_cut_off_by_the_budget() {
    let (url, _) = start(
        Mode::Silent,
        ServiceConfig {
            timeout: Duration::from_millis(150),
            ..Default::default()
        },
    )
    .await;

    let started = Instant::now();
    let response = post(&url, REQUEST).await;
    assert_eq!(
        response.status(),
        504,
        "a request that cannot be answered inside its budget is a gateway timeout"
    );
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the budget, not the client, has to end the wait: took {:?}",
        started.elapsed()
    );
    assert_eq!(
        response.text().await.unwrap(),
        r#"{"error":"request timed out"}"#
    );
}

/// A 503 is for a worker that cannot take work; a 504 is for work that ran out of budget. The
/// two must not blur, or a caller cannot tell "retry later" from "this took too long".
#[tokio::test]
async fn timeout_and_unavailability_are_different_answers() {
    let (slow, _) = start(
        Mode::Silent,
        ServiceConfig {
            timeout: Duration::from_millis(100),
            ..Default::default()
        },
    )
    .await;
    assert_eq!(post(&slow, REQUEST).await.status(), 504);

    let (loading, _) = start(Mode::Starting, ServiceConfig::default()).await;
    assert_eq!(post(&loading, REQUEST).await.status(), 503);
}
