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
    capacity: usize,
    rejected: u64,
    /// Number of requests that reached `submit`.
    submitted: Arc<AtomicUsize>,
}

impl Default for Fake {
    fn default() -> Self {
        Self {
            mode: Mode::Ready,
            depth: 0,
            capacity: 4,
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
            capacity: self.capacity,
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
            r#"{"status":"ok","depth":0,"capacity":4,"rejected":0}"#,
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

/// Serving is not on a timer. A budget measured at startup would retire a healthy process
/// once it elapsed even though nothing asked it to stop — the drain budget belongs to
/// shutdown, and there is no shutdown until there is a signal.
#[cfg(unix)]
#[tokio::test]
async fn the_service_stays_up_beyond_the_drain_budget_without_a_signal() {
    use tokio::io::AsyncBufReadExt;

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_omni-jev"))
        .env("OMNI_SYSTEMONE_ENGINE", "passthrough")
        .env("OMNI_JEV_BIND", "127.0.0.1:0")
        // The whole point: far shorter than the test waits below.
        .env("OMNI_SYSTEMONE_DRAIN_MS", "200")
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();

    let mut stderr = tokio::io::BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(10), stderr.read_line(&mut line))
        .await
        .expect("binary did not start")
        .unwrap();
    let address = line
        .trim()
        .strip_prefix("omni-jev listening on ")
        .unwrap_or_else(|| panic!("unexpected startup line {line:?}"))
        .split(' ')
        .next()
        .unwrap()
        .to_owned();

    // No signal is sent at any point in this test.
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert!(
        child.try_wait().unwrap().is_none(),
        "the process exited on its own {DRAIN_MS} after starting",
        DRAIN_MS = 200
    );
    let response = post(&format!("http://{address}"), REQUEST).await;
    assert_eq!(
        response.status(),
        200,
        "the service stopped serving with no signal sent"
    );
}

/// An unknown engine name is a configuration error. Falling through to forwarding would send
/// inference somewhere the operator did not ask for, and the README says as much.
#[cfg(unix)]
#[tokio::test]
async fn an_unknown_engine_name_is_a_startup_error() {
    // Bounded, because the failure this guards against is the process *starting*: an
    // unbounded wait would hang the suite instead of reporting a regression.
    let output = tokio::time::timeout(
        Duration::from_secs(10),
        tokio::process::Command::new(env!("CARGO_BIN_EXE_omni-jev"))
            .env("OMNI_SYSTEMONE_ENGINE", "typo-engine")
            .env("OMNI_JEV_BIND", "127.0.0.1:0")
            .kill_on_drop(true)
            .output(),
    )
    .await
    .expect("an unknown engine name started a server instead of failing")
    .unwrap();
    assert!(
        !output.status.success(),
        "an unknown engine name started the process anyway"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("typo-engine"),
        "the error does not name the offending value: {stderr}"
    );
}

/// SIGTERM bounds the *whole* shutdown, not just the part after the listener stopped. A
/// request whose body never arrives holds a handler open for its entire request budget, and
/// the process must still exit within the drain budget rather than waiting that budget out.
#[cfg(unix)]
#[tokio::test]
async fn shutdown_is_bounded_by_the_drain_budget_not_the_request_budget() {
    use tokio::{io::AsyncBufReadExt, io::AsyncWriteExt};

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_omni-jev"))
        .env("OMNI_SYSTEMONE_ENGINE", "passthrough")
        .env("OMNI_JEV_BIND", "127.0.0.1:0")
        .env("OMNI_SYSTEMONE_DRAIN_MS", "200")
        // Far longer than the drain budget: without one budget for the whole shutdown, this
        // is how long the process would stay alive after the signal.
        .env("OMNI_SYSTEMONE_TIMEOUT_MS", "30000")
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();

    let mut stderr = tokio::io::BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(10), stderr.read_line(&mut line))
        .await
        .expect("binary did not start")
        .unwrap();
    let address = line
        .trim()
        .strip_prefix("omni-jev listening on ")
        .unwrap_or_else(|| panic!("unexpected startup line {line:?}"))
        .split(' ')
        .next()
        .unwrap()
        .to_owned();

    // A request whose body never arrives holds a handler for its whole request budget, which
    // is what the drain budget has to bound instead.
    let mut stalled = tokio::net::TcpStream::connect(&address).await.unwrap();
    stalled
        .write_all(
            format!(
                "POST /v1/systemone HTTP/1.1\r\nHost: {address}\r\nContent-Length: 100000\r\n\r\n"
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    stalled.flush().await.unwrap();
    // And then it stays open, deliberately. The transport waits for open connections, so this
    // is the case where the budget has to be enforced rather than awaited: an earlier version
    // of this test hung up first and so could not see the difference.
    let pid = child.id().unwrap().to_string();
    let started = Instant::now();
    assert!(
        std::process::Command::new("kill")
            .args(["-TERM", &pid])
            .status()
            .unwrap()
            .success()
    );
    let exit = tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .expect("binary did not exit after SIGTERM")
        .unwrap();
    let elapsed = started.elapsed();
    assert!(exit.success(), "exit status {exit}");
    // The budget has to bound the whole sequence, and the transport cannot be asked to stop:
    // it waits on the connection this test deliberately keeps open. Without a bound here, an
    // earlier version waited out the handler's entire request budget — 2.6 s of a 200 ms
    // budget, measured. With the close window clamped to the deadline rather than added to it,
    // the exit sits at ~240 ms; the threshold is tight enough that a version which adds its own
    // period instead stays red.
    assert!(
        elapsed < Duration::from_millis(450),
        "took {elapsed:?} to exit with a 200ms drain budget and a 30000ms request budget: the \
         whole sequence has to be bounded by the drain budget"
    );
    drop(stalled);
}

/// Runs the compiled binary in native mode: the full lifecycle in one process, including
/// the parts no in-process test can cover — a real worker thread, signal handling and a
/// clean exit.
/// A drain budget much larger than the shutdown needs must not become the shutdown's length.
/// Exit is bounded by the deadline taken at the signal, not by a period measured after the
/// drain: an earlier version drained instantly and then added its own grace to the budget, so
/// the process lived `budget + grace` however little work was left. With a budget this size,
/// that additive version takes ~5.4 s where the bounded one takes under 300 ms.
#[cfg(unix)]
#[tokio::test]
async fn shutdown_does_not_wait_out_a_budget_it_does_not_need() {
    use tokio::{io::AsyncBufReadExt, io::AsyncWriteExt};

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_omni-jev"))
        .env("OMNI_SYSTEMONE_ENGINE", "passthrough")
        .env("OMNI_JEV_BIND", "127.0.0.1:0")
        .env("OMNI_SYSTEMONE_DRAIN_MS", "5000")
        .env("OMNI_SYSTEMONE_TIMEOUT_MS", "30000")
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();

    let mut stderr = tokio::io::BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(10), stderr.read_line(&mut line))
        .await
        .expect("binary did not start")
        .unwrap();
    let address = line
        .trim()
        .strip_prefix("omni-jev listening on ")
        .unwrap_or_else(|| panic!("unexpected startup line {line:?}"))
        .split(' ')
        .next()
        .unwrap()
        .to_owned();

    // One answered request, and the client stays connected: an idle keep-alive connection is
    // still an open one, so the transport has something to wait for.
    let mut held = tokio::net::TcpStream::connect(&address).await.unwrap();
    held.write_all(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n").as_bytes())
        .await
        .unwrap();
    held.flush().await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    let pid = child.id().unwrap().to_string();
    let started = Instant::now();
    assert!(
        std::process::Command::new("kill")
            .args(["-TERM", &pid])
            .status()
            .unwrap()
            .success()
    );
    let exit = tokio::time::timeout(Duration::from_secs(20), child.wait())
        .await
        .expect("binary did not exit after SIGTERM")
        .unwrap();
    let elapsed = started.elapsed();
    assert!(exit.success(), "exit status {exit}");
    assert!(
        elapsed < Duration::from_millis(1500),
        "took {elapsed:?} to exit with a 5000ms drain budget: the budget is an upper bound, \
         not a duration to wait out"
    );
    drop(held);
}

#[cfg(unix)]
#[tokio::test]
async fn binary_serves_a_linked_engine_and_drains_on_sigterm() {
    use tokio::io::AsyncBufReadExt;

    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_omni-jev"))
        .env("OMNI_SYSTEMONE_ENGINE", "passthrough")
        .env("OMNI_JEV_BIND", "127.0.0.1:0")
        .env("OMNI_SYSTEMONE_QUEUE", "4")
        .env("OMNI_SYSTEMONE_MAX_BODY", "256")
        .env("OMNI_SYSTEMONE_TIMEOUT_MS", "5000")
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();

    let mut stderr = tokio::io::BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(10), stderr.read_line(&mut line))
        .await
        .expect("binary did not start")
        .unwrap();
    let address = line
        .trim()
        .strip_prefix("omni-jev listening on ")
        .unwrap_or_else(|| panic!("unexpected startup line {line:?}"))
        .split(' ')
        .next()
        .unwrap()
        .to_owned();
    let url = format!("http://{address}");
    let client = client();

    // The line above is printed only after loading and warmup, so readiness is already
    // true here: a supervisor that reads the port from stderr never races startup.
    let health = client.get(format!("{url}/health")).send().await.unwrap();
    assert_eq!(health.status(), 200);
    assert_eq!(
        body(health).await,
        r#"{"status":"ok","depth":0,"capacity":4,"rejected":0}"#
    );

    let response = post(&url, REQUEST).await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.headers()["x-queue-depth"],
        "1",
        "the request being answered is still outstanding when it is dequeued"
    );
    assert_eq!(body(response).await, REQUEST);

    // The configured body limit reaches the transport, not just the config.
    assert_eq!(post(&url, &"x".repeat(1024)).await.status(), 413);
    // And a refused request leaves the engine serving.
    assert_eq!(post(&url, REQUEST).await.status(), 200);

    let pid = child.id().unwrap().to_string();
    assert!(
        std::process::Command::new("kill")
            .args(["-TERM", &pid])
            .status()
            .unwrap()
            .success()
    );
    let exit = match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(exit) => exit.unwrap(),
        Err(_) => {
            let alive = std::process::Command::new("kill")
                .args(["-0", &pid])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            let mut rest = String::new();
            let _ =
                tokio::time::timeout(Duration::from_millis(200), stderr.read_line(&mut rest)).await;
            panic!("binary did not exit after SIGTERM (alive={alive}, stderr={rest:?})");
        }
    };
    assert!(exit.success(), "exit status {exit}");
}
