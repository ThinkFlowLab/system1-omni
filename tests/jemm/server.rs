use super::*;

#[test]
fn readiness_warmup_includes_a_decodable_minimal_image() {
    let request: Value = serde_json::from_slice(WARMUP).unwrap();
    let images = crate::processing::decode_images(&request["images"]).unwrap();
    assert_eq!(
        images.len(),
        1,
        "readiness must exercise vision as well as language"
    );
    assert_eq!(images[0].dimensions(), (32, 32));
    assert_eq!(crate::contract::compile(WARMUP).unwrap().len(), 1);
    let text: Value = serde_json::from_slice(TEXT_WARMUP).unwrap();
    assert!(
        crate::processing::decode_images(&text["images"])
            .unwrap()
            .is_empty()
    );
    assert_eq!(crate::contract::compile(TEXT_WARMUP).unwrap().len(), 1);
}

#[tokio::test]
async fn error_envelope_is_typed_and_bounds_unicode_detail() {
    for (status, class) in [
        (StatusCode::UNPROCESSABLE_ENTITY, "ValueError"),
        (StatusCode::PAYLOAD_TOO_LARGE, "ValueError"),
        (StatusCode::UNSUPPORTED_MEDIA_TYPE, "TypeError"),
        (StatusCode::SERVICE_UNAVAILABLE, "RuntimeError"),
    ] {
        let response = error(status, "界".repeat(250));
        assert_eq!(response.status(), status);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"], class);
        assert_eq!(body["detail"].as_str().unwrap().chars().count(), 200);
    }
}
