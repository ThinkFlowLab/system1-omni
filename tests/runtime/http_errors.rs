use super::*;

fn expected_body(message: &str) -> serde_json::Value {
    match env!("CARGO_PKG_NAME") {
        "omni-cua-s1-native" => json!({"detail": message}),
        "omni-open-jev-native" => json!({"error": message}),
        other => panic!("unexpected worker {other}"),
    }
}

#[tokio::test]
async fn admission_overload_returns_503_with_the_worker_error_shape() {
    for error in [
        anyhow::Error::new(Overloaded),
        anyhow::Error::new(Overloaded).context("executor context"),
    ] {
        let response = inference_error(error);
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["content-type"], "application/json");
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body, expected_body("native execution capacity exhausted"));
    }
}

#[tokio::test]
async fn executor_failures_keep_the_existing_500_response() {
    let response = inference_error(anyhow::anyhow!("private forward failure"));
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let bytes = axum::body::to_bytes(response.into_body(), 1024)
        .await
        .unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let message = match env!("CARGO_PKG_NAME") {
        "omni-cua-s1-native" => "inference failed",
        "omni-open-jev-native" => "model inference failed",
        other => panic!("unexpected worker {other}"),
    };
    assert_eq!(body, expected_body(message));
}
