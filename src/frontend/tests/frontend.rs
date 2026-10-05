//! End-to-end tests over real sockets: client -> omni-jev -> mock worker.

use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    extract::{ConnectInfo, Request, State},
    http::{HeaderMap, StatusCode},
    response::Response,
};
use omni_jev::Config;
use tokio::net::TcpListener;

#[path = "../../../tests/frontend/common/mod.rs"]
mod common;

use common::{client, listen};

/// What the mock worker received.
#[derive(Clone, Debug)]
struct Seen {
    method: String,
    uri: String,
    headers: HeaderMap,
    body: Bytes,
    peer: SocketAddr,
}

/// How the mock worker replies. By default it echoes the request body with 200.
#[derive(Clone, Default)]
struct Reply {
    status: StatusCode,
    headers: Vec<(&'static str, &'static str)>,
    body: Option<Bytes>,
    header_delay: Duration,
    body_delay: Duration,
}

#[derive(Clone)]
struct Worker {
    reply: Reply,
    seen: Arc<Mutex<Vec<Seen>>>,
}

async fn start_worker(reply: Reply) -> (String, Arc<Mutex<Vec<Seen>>>) {
    let worker = Worker {
        reply,
        seen: Arc::default(),
    };
    let seen = worker.seen.clone();
    let url = listen(Router::new().fallback(respond).with_state(worker)).await;
    (url, seen)
}

async fn respond(
    State(worker): State<Worker>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
) -> Response {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, usize::MAX).await.unwrap();
    worker.seen.lock().unwrap().push(Seen {
        method: parts.method.to_string(),
        uri: parts.uri.to_string(),
        headers: parts.headers,
        body: body.clone(),
        peer,
    });

    let reply = worker.reply;
    tokio::time::sleep(reply.header_delay).await;
    let bytes = reply.body.unwrap_or(body);
    let body = Body::from_stream(futures_util::stream::once(async move {
        tokio::time::sleep(reply.body_delay).await;
        Ok::<_, Infallible>(bytes)
    }));
    let mut response = Response::new(body);
    *response.status_mut() = reply.status;
    for (name, value) in reply.headers {
        response.headers_mut().append(name, value.parse().unwrap());
    }
    response
}

async fn start_frontend(backend_url: &str, timeout: Duration) -> String {
    let mut config = Config::new("127.0.0.1:0", backend_url).unwrap();
    config.timeout = timeout;
    listen(omni_jev::app(&config).unwrap()).await
}

fn decision_request(state: &str) -> String {
    // Formatting and the unknown field are deliberate: the frontend must not re-serialize.
    format!(
        "{{\n  \"model\": \"any-worker\",  \"state\": {state},\n  \"questions\": {{\"safe\": {{\"type\": \"noul\", \"instructions\": \"Safe?\"}}}},\n  \"extension\": 1.00\n}}"
    )
}

#[tokio::test]
async fn forwards_every_modality_byte_for_byte() {
    let (worker, seen) = start_worker(Reply {
        headers: vec![("content-type", "application/json")],
        ..Default::default()
    })
    .await;
    let frontend = start_frontend(&worker, Config::DEFAULT_TIMEOUT).await;
    // Transport fixtures only; the frontend does not interpret media.
    let states = [
        r#""I was charged twice. 请退款。""#,
        r#"{"image_url": "https://example.invalid/a.png", "image": "data:image/png;base64,iVBORw0KGgo="}"#,
        r#"{"audio": {"data": "UklGRg==", "format": "wav"}}"#,
        r#"{"video_url": "https://example.invalid/a.mp4", "frames": [{"t": 0, "data": "AAAA"}]}"#,
        r#"[{"text": "hi"}, {"image": "iVBO"}, {"audio": "UklG"}, {"video": "https://example.invalid/v"}]"#,
    ];

    for state in states {
        let body = decision_request(state);
        let response = client()
            .post(format!("{frontend}/v1/systemone"))
            .header("content-type", "application/json")
            .body(body.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "application/json");
        assert_eq!(response.bytes().await.unwrap(), body);
    }

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), states.len());
    for (request, state) in seen.iter().zip(states) {
        assert_eq!(request.method, "POST");
        assert_eq!(request.uri, "/v1/systemone");
        assert_eq!(request.body, decision_request(state));
        assert_eq!(request.headers["content-type"], "application/json");
        assert!(!request.headers.contains_key("authorization"));
    }
    assert!(
        seen.iter().all(|request| request.peer == seen[0].peer),
        "backend connection was not reused"
    );
}

#[tokio::test]
async fn round_trips_decision_fixtures() {
    let fixtures = [
        (
            r#"{"model":"any-worker","state":"Please route this request.","questions":{"department":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges and refunds","technical":"Software problems"}}}}"#,
            r#"{"answers":{"department":{"type":"choice","choice":"billing"}}}"#,
        ),
        (
            r#"{"model":"any-worker","state":"Please route this request.","questions":{"urgency":{"type":"score","instructions":"How urgent is this?","criteria":["Not urgent","Needs attention soon","Immediate attention"]}}}"#,
            r#"{"answers":{"urgency":{"type":"score","score":0.8}}}"#,
        ),
        (
            r#"{"model":"any-worker","state":"Please route this request.","questions":{"refund":{"type":"noul","instructions":"Does this ask for a refund?"}}}"#,
            r#"{"answers":{"refund":{"type":"noul","noul":0.9}}}"#,
        ),
        (
            r#"{"model":"any-worker","state":"Please route this request.","questions":{"department":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges and refunds","technical":"Software problems"}},"urgency":{"type":"score","instructions":"How urgent is this?","criteria":["Not urgent","Needs attention soon","Immediate attention"]},"refund":{"type":"noul","instructions":"Does this ask for a refund?"}}}"#,
            r#"{"answers":{"department":{"type":"choice","choice":"billing"},"urgency":{"type":"score","score":0.8},"refund":{"type":"noul","noul":0.9}}}"#,
        ),
    ];

    for (request_body, response_body) in fixtures {
        let (worker, seen) = start_worker(Reply {
            headers: vec![("content-type", "application/json")],
            body: Some(Bytes::from_static(response_body.as_bytes())),
            ..Default::default()
        })
        .await;
        let frontend = start_frontend(&worker, Config::DEFAULT_TIMEOUT).await;
        let response = client()
            .post(format!("{frontend}/v1/systemone"))
            .header("content-type", "application/json")
            .body(request_body)
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["content-type"], "application/json");
        assert_eq!(response.bytes().await.unwrap(), response_body.as_bytes());
        assert_eq!(seen.lock().unwrap()[0].body, request_body.as_bytes());
    }
}

#[tokio::test]
async fn rejects_invalid_routes_before_forwarding() {
    let (worker, seen) = start_worker(Reply::default()).await;
    let frontend = start_frontend(&worker, Config::DEFAULT_TIMEOUT).await;

    let get_decision = client()
        .get(format!("{frontend}/v1/systemone"))
        .send()
        .await
        .unwrap();
    assert_eq!(get_decision.status(), StatusCode::METHOD_NOT_ALLOWED);

    let post_health = client()
        .post(format!("{frontend}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(post_health.status(), StatusCode::METHOD_NOT_ALLOWED);

    let unknown_path = client()
        .get(format!("{frontend}/v1/unknown"))
        .send()
        .await
        .unwrap();
    assert_eq!(unknown_path.status(), StatusCode::NOT_FOUND);

    assert!(
        seen.lock().unwrap().is_empty(),
        "rejected frontend requests must not reach the worker"
    );
}

#[tokio::test]
async fn forwards_authorization_when_present() {
    let (worker, seen) = start_worker(Reply::default()).await;
    let frontend = start_frontend(&worker, Config::DEFAULT_TIMEOUT).await;
    let response = client()
        .post(format!("{frontend}/v1/systemone"))
        .bearer_auth("test-token")
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        seen.lock().unwrap()[0].headers["authorization"],
        "Bearer test-token"
    );
}

#[tokio::test]
async fn streams_uploads_beyond_axum_default_body_limit() {
    let (worker, seen) = start_worker(Reply::default()).await;
    let frontend = start_frontend(&worker, Config::DEFAULT_TIMEOUT).await;
    let body = decision_request(&format!("\"{}\"", "A".repeat(3 << 20)));
    let chunks: Vec<Result<Bytes, Infallible>> = body
        .as_bytes()
        .chunks(64 << 10)
        .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
        .collect();

    // A streamed body has no Content-Length, so it arrives chunked.
    let response = client()
        .post(format!("{frontend}/v1/systemone"))
        .body(reqwest::Body::wrap_stream(futures_util::stream::iter(
            chunks,
        )))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.bytes().await.unwrap(), body);
    assert_eq!(seen.lock().unwrap()[0].body, body);
}

#[tokio::test]
async fn returns_worker_responses_unchanged_without_retrying() {
    for status in [201, 307, 400, 401, 422, 429, 500, 503] {
        let reply = Reply {
            status: StatusCode::from_u16(status).unwrap(),
            headers: vec![
                ("content-type", "application/octet-stream"),
                ("content-encoding", "gzip"),
                ("location", "/elsewhere"),
                ("retry-after", "0"),
            ],
            body: Some(Bytes::from_static(b"opaque\x00\xff bytes")),
            ..Default::default()
        };
        let (worker, seen) = start_worker(reply).await;
        let frontend = start_frontend(&worker, Config::DEFAULT_TIMEOUT).await;
        let response = client()
            .post(format!("{frontend}/v1/systemone"))
            .body("not even json")
            .send()
            .await
            .unwrap();

        assert_eq!(response.status().as_u16(), status);
        let headers = response.headers();
        assert_eq!(headers["content-type"], "application/octet-stream");
        assert_eq!(headers["content-encoding"], "gzip");
        assert_eq!(headers["location"], "/elsewhere");
        assert_eq!(headers["retry-after"], "0");
        assert_eq!(response.bytes().await.unwrap(), b"opaque\x00\xff bytes"[..]);
        assert_eq!(
            seen.lock().unwrap().len(),
            1,
            "status {status} was retried or followed"
        );
    }
}

#[tokio::test]
async fn strips_connection_specific_headers_in_both_directions() {
    let (worker, seen) = start_worker(Reply {
        headers: vec![("connection", "x-worker-hop"), ("x-worker-hop", "1")],
        ..Default::default()
    })
    .await;
    let frontend = start_frontend(&worker, Config::DEFAULT_TIMEOUT).await;
    let response = client()
        .post(format!("{frontend}/v1/systemone"))
        .header("connection", "x-client-hop")
        .header("x-client-hop", "1")
        .header("x-end-to-end", "1")
        .body("{}")
        .send()
        .await
        .unwrap();

    assert!(!response.headers().contains_key("x-worker-hop"));
    let seen = seen.lock().unwrap();
    assert!(!seen[0].headers.contains_key("x-client-hop"));
    assert_eq!(seen[0].headers["x-end-to-end"], "1");
}

#[tokio::test]
async fn health_reflects_worker_health_under_base_path() {
    for status in [200, 503] {
        let (worker, seen) = start_worker(Reply {
            status: StatusCode::from_u16(status).unwrap(),
            body: Some(Bytes::from_static(b"worker says hi")),
            ..Default::default()
        })
        .await;
        let frontend = start_frontend(&format!("{worker}/worker"), Config::DEFAULT_TIMEOUT).await;
        let response = client()
            .get(format!("{frontend}/health?verbose=1"))
            .bearer_auth("health-token")
            .send()
            .await
            .unwrap();

        assert_eq!(response.status().as_u16(), status);
        assert_eq!(response.bytes().await.unwrap(), "worker says hi");
        let seen = seen.lock().unwrap();
        assert_eq!(seen[0].method, "GET");
        assert_eq!(seen[0].uri, "/worker/health?verbose=1");
        assert_eq!(seen[0].headers["authorization"], "Bearer health-token");
    }
}

#[tokio::test]
async fn forwards_prefixed_decision_with_query() {
    let response_body = r#"{"answers":{"refund":{"type":"noul","noul":0.9}}}"#;
    let (worker, seen) = start_worker(Reply {
        headers: vec![("content-type", "application/json")],
        body: Some(Bytes::from_static(response_body.as_bytes())),
        ..Default::default()
    })
    .await;
    let frontend = start_frontend(&format!("{worker}/worker"), Config::DEFAULT_TIMEOUT).await;
    let request_body = decision_request(r#""A request with a backend prefix.""#);
    let response = client()
        .post(format!(
            "{frontend}/v1/systemone?request_id=param-test&verbose=1"
        ))
        .header("content-type", "application/json")
        .body(request_body.clone())
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "application/json");
    assert_eq!(response.bytes().await.unwrap(), response_body.as_bytes());

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].method, "POST");
    assert_eq!(
        seen[0].uri,
        "/worker/v1/systemone?request_id=param-test&verbose=1"
    );
    assert_eq!(seen[0].body, request_body.as_bytes());
}

#[tokio::test]
async fn unreachable_worker_returns_502() {
    // Reserve a port, then close it so nothing is listening there.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let closed = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let frontend = start_frontend(&closed, Config::DEFAULT_TIMEOUT).await;

    let health = client()
        .get(format!("{frontend}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::BAD_GATEWAY);
    let decision = client()
        .post(format!("{frontend}/v1/systemone"))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(decision.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn slow_worker_returns_504_before_or_after_headers() {
    let slow = Duration::from_secs(2);
    for reply in [
        Reply {
            header_delay: slow,
            ..Default::default()
        },
        Reply {
            body_delay: slow,
            ..Default::default()
        },
    ] {
        let (worker, seen) = start_worker(reply).await;
        let frontend = start_frontend(&worker, Duration::from_millis(200)).await;

        let health = client()
            .get(format!("{frontend}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::GATEWAY_TIMEOUT);
        let decision = client()
            .post(format!("{frontend}/v1/systemone"))
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(decision.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            seen.lock().unwrap().len(),
            2,
            "timed-out request was retried"
        );
    }
}

/// Runs the compiled binary as its own process, the way it is deployed.
#[cfg(unix)]
#[tokio::test]
async fn binary_serves_requests_and_exits_cleanly_on_sigterm() {
    use tokio::io::AsyncBufReadExt;

    let decision = r#"{"answers":{"refund":{"type":"noul","noul":0.9}}}"#;
    let (worker, _seen) = start_worker(Reply {
        body: Some(Bytes::from_static(decision.as_bytes())),
        ..Default::default()
    })
    .await;
    let mut child = tokio::process::Command::new(env!("CARGO_BIN_EXE_omni-jev"))
        .env("OMNI_JEV_BIND", "127.0.0.1:0")
        .env("OMNI_JEV_BACKEND_URL", &worker)
        .env("OMNI_JEV_HEALTH_TIMEOUT_MS", "1000")
        .env("OMNI_JEV_MAX_RESPONSE_BYTES", "65536")
        // A proxy that does not exist: backend traffic must bypass it.
        .env("HTTP_PROXY", "http://127.0.0.1:9")
        .env("http_proxy", "http://127.0.0.1:9")
        .env_remove("NO_PROXY")
        .env_remove("no_proxy")
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
        .to_owned();

    let health = client()
        .get(format!("http://{address}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::OK);
    let response = client()
        .post(format!("http://{address}/v1/systemone"))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.text().await.unwrap(), decision);

    let pid = child.id().unwrap().to_string();
    let killed = std::process::Command::new("kill")
        .args(["-TERM", &pid])
        .status()
        .unwrap();
    assert!(killed.success());
    let exit = tokio::time::timeout(Duration::from_secs(5), child.wait())
        .await
        .expect("binary did not exit after SIGTERM")
        .unwrap();
    assert!(exit.success(), "exit status {exit}");
}
