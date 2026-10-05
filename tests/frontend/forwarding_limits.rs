//! Forwarding limits tested over real sockets: client -> omni-jev -> mock worker.

use std::{convert::Infallible, time::Duration};

use axum::{
    Router,
    body::{Body, Bytes, to_bytes},
    extract::Request,
    http::StatusCode,
    response::Response,
};
use omni_jev::{Config, app};
mod common;

use common::{client, listen};

#[derive(Clone, Default)]
struct Reply {
    header_delay: Duration,
    body_delay: Duration,
}

async fn start_worker(reply: Reply) -> String {
    listen(Router::new().fallback(move |request: Request| {
        let reply = reply.clone();
        async move {
            let body = to_bytes(request.into_body(), usize::MAX).await.unwrap();
            tokio::time::sleep(reply.header_delay).await;
            Body::from_stream(futures_util::stream::once(async move {
                tokio::time::sleep(reply.body_delay).await;
                Ok::<_, Infallible>(body)
            }))
        }
    }))
    .await
}

#[test]
fn rejects_zero_health_timeout_or_response_limit() {
    let mut config = Config::new(Config::DEFAULT_BIND, Config::DEFAULT_BACKEND_URL).unwrap();
    config.health_timeout = Duration::ZERO;
    assert!(app(&config).is_err());
    config.health_timeout = Config::DEFAULT_HEALTH_TIMEOUT;
    config.max_response_bytes = 0;
    assert!(app(&config).is_err());
}

#[tokio::test]
async fn health_timeout_is_shorter_than_inference_before_or_after_headers() {
    let delay = Duration::from_millis(300);
    for reply in [
        Reply {
            header_delay: delay,
            ..Default::default()
        },
        Reply {
            body_delay: delay,
            ..Default::default()
        },
    ] {
        let worker = start_worker(reply).await;
        let mut config = Config::new("127.0.0.1:0", &worker).unwrap();
        config.timeout = Duration::from_secs(2);
        config.health_timeout = Duration::from_millis(50);
        let frontend = listen(omni_jev::app(&config).unwrap()).await;
        let health = tokio::time::timeout(
            Duration::from_secs(1),
            client().get(format!("{frontend}/health?verbose=1")).send(),
        )
        .await
        .expect("health request did not finish")
        .unwrap();
        assert_eq!(health.status(), StatusCode::GATEWAY_TIMEOUT);

        let decision = client()
            .post(format!("{frontend}/v1/systemone"))
            .body("decision")
            .send()
            .await
            .unwrap();
        assert_eq!(decision.status(), StatusCode::OK);
        assert_eq!(decision.bytes().await.unwrap(), "decision");
    }
}

#[tokio::test]
async fn response_limit_covers_declared_and_chunked_bodies_on_both_routes() {
    for declared_length in [false, true] {
        for length in [0, 7, 8, 9] {
            let payload: Vec<u8> = (0..length)
                .map(|i| if i % 2 == 0 { 0 } else { 255 })
                .collect();
            let expected = payload.clone();
            let worker = listen(Router::new().fallback(move || {
                let chunks: Vec<Result<Bytes, Infallible>> = payload
                    .chunks(3)
                    .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
                    .collect();
                async move {
                    let mut response =
                        Response::new(Body::from_stream(futures_util::stream::iter(chunks)));
                    *response.status_mut() = StatusCode::CREATED;
                    response
                        .headers_mut()
                        .insert("content-type", "application/octet-stream".parse().unwrap());
                    if declared_length {
                        response
                            .headers_mut()
                            .insert("content-length", length.to_string().parse().unwrap());
                    }
                    response
                }
            }))
            .await;
            let mut config = Config::new("127.0.0.1:0", &worker).unwrap();
            config.max_response_bytes = 8;
            let frontend = listen(omni_jev::app(&config).unwrap()).await;
            for request in [
                client().get(format!("{frontend}/health")),
                client().post(format!("{frontend}/v1/systemone")),
            ] {
                let response = request.send().await.unwrap();
                if length > 8 {
                    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
                    assert_eq!(
                        response.bytes().await.unwrap(),
                        "backend response too large\n"
                    );
                } else {
                    assert_eq!(response.status(), StatusCode::CREATED);
                    assert_eq!(
                        response.headers()["content-type"],
                        "application/octet-stream"
                    );
                    assert_eq!(response.bytes().await.unwrap(), expected);
                }
            }
        }
    }
}

#[tokio::test]
async fn oversized_response_is_rejected_before_the_worker_finishes() {
    use futures_util::StreamExt;

    for declared_length in [false, true] {
        let worker = listen(Router::new().fallback(move || async move {
            // Known oversize: send headers only. Unknown size: overflow, then stall.
            let chunks = if declared_length {
                vec![]
            } else {
                vec![Ok::<_, Infallible>(Bytes::from_static(b"123456789"))]
            };
            let stream = futures_util::stream::iter(chunks).chain(futures_util::stream::pending());
            let mut response = Response::new(Body::from_stream(stream));
            if declared_length {
                response
                    .headers_mut()
                    .insert("content-length", "9".parse().unwrap());
            }
            response
        }))
        .await;
        let mut config = Config::new("127.0.0.1:0", &worker).unwrap();
        config.max_response_bytes = 8;
        let frontend = listen(omni_jev::app(&config).unwrap()).await;
        let response = tokio::time::timeout(
            Duration::from_secs(1),
            client().post(format!("{frontend}/v1/systemone")).send(),
        )
        .await
        .expect("waited for an oversized response to finish")
        .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    }
}

#[tokio::test]
async fn binary_rejects_invalid_forwarding_limits() {
    for name in ["OMNI_JEV_HEALTH_TIMEOUT_MS", "OMNI_JEV_MAX_RESPONSE_BYTES"] {
        for value in ["0", "invalid"] {
            let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_omni-jev"));
            command
                .env("OMNI_JEV_BIND", "127.0.0.1:0")
                .env("OMNI_JEV_BACKEND_URL", Config::DEFAULT_BACKEND_URL)
                .env("OMNI_JEV_HEALTH_TIMEOUT_MS", "2000")
                .env("OMNI_JEV_MAX_RESPONSE_BYTES", "16777216")
                .env(name, value)
                .kill_on_drop(true);
            let output = tokio::time::timeout(Duration::from_secs(5), command.output())
                .await
                .expect("binary accepted invalid limits")
                .unwrap();
            assert!(!output.status.success());
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(
                error.contains(if value == "0" {
                    "must be positive"
                } else {
                    name
                }),
                "{error}"
            );
        }
    }
}
