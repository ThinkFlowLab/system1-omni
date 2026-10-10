//! Worker fallback contract only: the frontend does not forward this route.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_model_chat_probe_returns_canonical_worker_error() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    // This route needs no model or GPU; exercise the production fallback over HTTP.
    let app = Router::new().fallback(fallback);
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let response = tokio::task::spawn_blocking(move || {
        let body = r#"{"model":"no-such-model","messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#;
        let mut stream = TcpStream::connect(address).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        stream.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
        write!(
            stream,
            "POST /v1/chat/completions HTTP/1.0\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
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
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
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
