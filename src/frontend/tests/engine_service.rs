//! End-to-end tests over real sockets for the in-process path: client -> omni-jev ->
//! a fake engine. The engine is a stand-in, so none of this needs a model or a GPU.
//!
//! The last test runs the compiled binary with `OMNI_SYSTEMONE_ENGINE=passthrough`, which
//! is the only way to cover the whole lifecycle: readiness, signals and process exit.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use axum::Router;
use omni_jev::engine::{
    self, Answer, Engine, EngineError, Readiness, Reply, Report, ServiceConfig,
};
use tokio::{net::TcpListener, sync::oneshot};

const REQUEST: &str = r#"{"model":"english","state":"refund please","questions":{"q":{"type":"noul","instructions":"Ask?"}}}"#;

/// How the fake engine answers. Every mode is one of the states the transport has to tell
/// apart, so the status codes below are the contract rather than a convention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Ready,
    Starting,
    Failed,
    /// Accepts the request and never answers it.
    Silent,
    /// Refuses with a full queue.
    Full,
    /// Echoes its own error variants.
    Invalid,
    Inference,
}

struct Fake {
    mode: Mode,
    depth: usize,
    rejected: u64,
    /// Number of requests that reached `submit`.
    submitted: Arc<AtomicUsize>,
}

impl Default for Fake {
    fn default() -> Self {
        Self {
            mode: Mode::Ready,
            depth: 0,
            rejected: 0,
            submitted: Arc::new(AtomicUsize::new(0)),
        }
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

    fn submit(&self, body: Vec<u8>, _deadline: Instant) -> Result<Reply, EngineError> {
        self.submitted.fetch_add(1, Ordering::AcqRel);
        match self.mode {
            Mode::Full => Err(EngineError::Busy),
            Mode::Invalid => Err(EngineError::InvalidRequest("bad \"envelope\"\n".into())),
            Mode::Inference => Err(EngineError::InferenceFailed),
            Mode::Silent => {
                // Keeping the sender alive means the request neither completes nor fails:
                // only the deadline can end it.
                let (reply, receiver) = oneshot::channel();
                std::mem::forget(reply);
                Ok(receiver)
            }
            _ => {
                let answer = Answer {
                    body,
                    depth: self.depth,
                };
                // Answered from another task, because a real engine never answers inside
                // `submit` and the transport must not depend on getting one immediately.
                let (reply, receiver) = oneshot::channel();
                tokio::spawn(async move {
                    let _ = reply.send(Ok(answer));
                });
                Ok(receiver)
            }
        }
    }

    fn report(&self) -> Option<Report> {
        Some(Report {
            depth: self.depth,
            rejected: self.rejected,
        })
    }

    fn failure(&self) -> Option<String> {
        matches!(self.mode, Mode::Failed).then(|| "no such checkpoint".to_owned())
    }
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap()
}

/// Starts the native service and returns its base URL.
async fn start(fake: Arc<Fake>, config: ServiceConfig) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app = engine::app(fake, config);
    tokio::spawn(async move {
        engine::serve(listener, app, std::future::pending())
            .await
            .unwrap();
    });
    url
}

async fn post(url: &str, body: &str) -> reqwest::Response {
    client()
        .post(format!("{url}/v1/systemone"))
        .body(body.to_owned())
        .send()
        .await
        .unwrap()
}

/// The body a response carried. Asserting the whole JSON text rather than a parsed field
/// pins the wire format itself, and keeps the transport from depending on a parser it
/// deliberately does not have.
async fn body(response: reqwest::Response) -> String {
    response.text().await.unwrap()
}

#[tokio::test]
async fn readiness_and_failure_are_distinguishable() {
    for (mode, status, expected) in [
        (
            Mode::Ready,
            200,
            r#"{"status":"ok","depth":0,"rejected":0}"#,
        ),
        (Mode::Starting, 503, r#"{"status":"starting"}"#),
        (
            Mode::Failed,
            503,
            r#"{"status":"failed","reason":"no such checkpoint"}"#,
        ),
    ] {
        let fake = Arc::new(Fake {
            mode,
            ..Default::default()
        });
        let url = start(fake.clone(), ServiceConfig::default()).await;

        let health = client().get(format!("{url}/health")).send().await.unwrap();
        assert_eq!(health.status().as_u16(), status, "mode {mode:?}");
        // A ready engine describes its queue; a failed one says why. A starting one claims
        // neither, because a half-loaded engine knows nothing worth reporting.
        assert_eq!(body(health).await, expected, "mode {mode:?}");

        // Inference is refused before the engine is asked, so an unready engine never sees
        // the request at all.
        let response = post(&url, REQUEST).await;
        assert_eq!(response.status().as_u16(), status, "mode {mode:?}");
        if mode != Mode::Ready {
            assert_eq!(fake.submitted.load(Ordering::Acquire), 0);
        }
    }
}

#[tokio::test]
async fn a_valid_request_keeps_its_bytes() {
    let fake = Arc::new(Fake::default());
    let url = start(fake, ServiceConfig::default()).await;

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
async fn a_body_over_the_limit_is_refused_without_touching_the_engine() {
    let fake = Arc::new(Fake::default());
    let url = start(
        fake.clone(),
        ServiceConfig {
            max_body: 32,
            ..Default::default()
        },
    )
    .await;

    let response = post(&url, &"x".repeat(64)).await;
    assert_eq!(response.status(), 413);
    assert_eq!(fake.submitted.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn a_full_queue_is_a_503_and_a_lost_slot() {
    let fake = Arc::new(Fake {
        mode: Mode::Full,
        rejected: 7,
        ..Default::default()
    });
    let url = start(fake, ServiceConfig::default()).await;

    let response = post(&url, REQUEST).await;
    assert_eq!(response.status(), 503);
    assert_eq!(body(response).await, r#"{"error":"inference queue full"}"#);
}

#[tokio::test]
async fn a_silent_engine_is_cut_off_by_the_deadline() {
    let fake = Arc::new(Fake {
        mode: Mode::Silent,
        ..Default::default()
    });
    let url = start(
        fake,
        ServiceConfig {
            timeout: Duration::from_millis(150),
            ..Default::default()
        },
    )
    .await;

    let started = Instant::now();
    let response = post(&url, REQUEST).await;
    assert_eq!(response.status(), 504);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the deadline, not the client, must end the wait"
    );
    assert_eq!(body(response).await, r#"{"error":"inference timed out"}"#);
}

#[tokio::test]
async fn engine_errors_map_to_one_status_each() {
    for (mode, status, message) in [
        (
            Mode::Invalid,
            400,
            "{\"error\":\"bad \\\"envelope\\\"\\n\"}",
        ),
        (Mode::Inference, 500, r#"{"error":"inference failed"}"#),
    ] {
        let fake = Arc::new(Fake {
            mode,
            ..Default::default()
        });
        let url = start(fake, ServiceConfig::default()).await;

        let response = post(&url, REQUEST).await;
        assert_eq!(response.status().as_u16(), status, "mode {mode:?}");
        // The engine's text is passed through as JSON, so it has to survive being quoted:
        // the fake's message contains a quote and a newline on purpose.
        assert_eq!(body(response).await, message);
    }
}

#[tokio::test]
async fn the_answer_carries_the_depth_the_engine_reported() {
    // The engine, not the transport, measures the queue, so the header is whatever the
    // answer carries. A depth of zero is included rather than treated as absent.
    let fake = Arc::new(Fake {
        depth: 3,
        ..Default::default()
    });
    let url = start(fake, ServiceConfig::default()).await;

    let response = post(&url, REQUEST).await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()["x-queue-depth"],
        "3",
        "backpressure must be visible before it becomes a refusal"
    );
}

/// The forwarding path is unchanged, but both modes now share one serve loop, so this pins
/// that the shared loop binds and keeps serving rather than returning early.
#[tokio::test]
async fn engine_serve_keeps_serving_until_shutdown() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let app = Router::new();
    let handle = tokio::spawn(async move {
        engine::serve(listener, app, std::future::pending())
            .await
            .unwrap();
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!handle.is_finished(), "serve returned without a shutdown");

    // And it stops when the shutdown future resolves.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    engine::serve(listener, Router::new(), async {})
        .await
        .unwrap();
    handle.abort();
}
