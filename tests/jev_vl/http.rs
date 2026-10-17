//! Worker fallback contract only: the frontend does not forward this route.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use super::*;

#[test]
fn online_vision_startup_warmup_is_inline_image_decision() {
    let labels = vec!["OK".to_owned(), "Wait".to_owned()];
    let compiled = omni_jev_vl_native::contract::compile(ONLINE_VISION_WARMUP, &labels).unwrap();
    assert_eq!(compiled.images.len(), 1);
    let image = omni_qwen3_5_native::image_decode::decode_data_url(&compiled.images[0]).unwrap();
    let pixels = omni_qwen3_5_native::image_preprocess::preprocess_rgb8(
        image.width,
        image.height,
        &image.rgb,
    )
    .unwrap();
    assert_eq!(pixels.image_grid_thw, [1, 16, 16]);
    assert_eq!(pixels.image_tokens(), 64);
}

/// Raw HTTP/1.0 POST against a bound test router; returns the full response.
async fn post_raw(app: Router, route: &'static str, body: String) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let response = tokio::task::spawn_blocking(move || {
        let mut stream = TcpStream::connect(address).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(15))).unwrap();
        stream.set_write_timeout(Some(Duration::from_secs(15))).unwrap();
        write!(
            stream,
            "POST {route} HTTP/1.0\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    })
    .await
    .unwrap();
    server.abort();
    response
}

fn split_response(response: &str) -> (&str, &str) {
    response.split_once("\r\n\r\n").unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_model_chat_probe_returns_canonical_worker_error() {
    // This route needs no model or GPU; exercise the production fallback over HTTP.
    let app = Router::new().fallback(fallback);
    let body =
        r#"{"model":"no-such-model","messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#;
    let response = post_raw(app, "/v1/chat/completions", body.to_owned()).await;
    let (headers, body) = split_response(&response);
    assert_eq!(headers.split_whitespace().nth(1), Some("404"));
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("content-type: application/json")
    );
    assert_eq!(
        serde_json::from_str::<Value>(body).unwrap(),
        json!({"error": {
            "message": "The model `no-such-model` does not exist.",
            "type": "NotFoundError",
            "param": "model",
            "code": 404
        }})
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn over_limit_body_returns_envelope_413() {
    // The production body-cap layer and rejection mapping, minus the Engine:
    // an over-limit body must surface the worker's error envelope, not axum's
    // plain-text 413.
    let app = Router::new()
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(
            omni_jev_vl_native::contract::MAX_BODY_BYTES,
        ));
    let body = format!(
        r#"{{"model":"no-such-model","pad":"{}"}}"#,
        "A".repeat(omni_jev_vl_native::contract::MAX_BODY_BYTES)
    );
    let response = post_raw(app, "/v1/systemone", body).await;
    let (headers, body) = split_response(&response);
    assert_eq!(headers.split_whitespace().nth(1), Some("413"));
    let error = serde_json::from_str::<Value>(body).unwrap();
    let error = error["error"].clone();
    assert_eq!(error["type"], json!("BadRequestError"));
    assert_eq!(error["param"], json!("body"));
    assert_eq!(error["code"], json!(413));
    assert!(error["message"].as_str().unwrap().contains("8 MiB"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn body_within_transport_limit_reaches_request_validation() {
    // Bodies above the old 4 MiB layer but within the transport cap must reach
    // the structured handlers, so the advertised inline-image caps stay usable.
    let app = Router::new()
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(
            omni_jev_vl_native::contract::MAX_BODY_BYTES,
        ));
    let body = format!(
        r#"{{"model":"no-such-model","pad":"{}"}}"#,
        "A".repeat(5 << 20)
    );
    let response = post_raw(app, "/v1/systemone", body).await;
    let (headers, body) = split_response(&response);
    assert_eq!(headers.split_whitespace().nth(1), Some("404"));
    let error = serde_json::from_str::<Value>(body).unwrap();
    assert_eq!(error["error"]["code"], json!(404));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn truncated_body_keeps_axums_400_status() {
    // A truncated upload (declared length never delivered, then EOF) hits
    // axum's UnknownBodyError, which is 400 — mapping it must not relabel the
    // client's interrupted delivery as a server fault.
    let app = Router::new()
        .fallback(fallback)
        .layer(DefaultBodyLimit::max(
            omni_jev_vl_native::contract::MAX_BODY_BYTES,
        ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let response = tokio::task::spawn_blocking(move || {
        let mut stream = TcpStream::connect(address).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(15)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(15)))
            .unwrap();
        write!(
            stream,
            "POST /v1/systemone HTTP/1.0\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n"
        )
        .unwrap();
        stream.write_all(b"{}").unwrap();
        stream.shutdown(std::net::Shutdown::Write).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    })
    .await
    .unwrap();
    server.abort();
    let (headers, body) = split_response(&response);
    assert_eq!(headers.split_whitespace().nth(1), Some("400"));
    let error = serde_json::from_str::<Value>(body).unwrap();
    let error = error["error"].clone();
    assert_eq!(error["type"], json!("BadRequestError"));
    assert_eq!(error["code"], json!(400));
}
