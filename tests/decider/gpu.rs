use omni_decider_native::engine::Engine;
use std::path::Path;

#[tokio::test]
async fn missing_artifacts_fail_before_cuda_initialization() {
    let dir = std::env::temp_dir().join(format!("missing-decider-{}", std::process::id()));
    let result = Engine::load(&dir, Path::new("unavailable-cuda-library.so")).await;
    assert!(result.is_err());
    assert!(format!("{:#}", result.err().unwrap()).contains("No such file"));
}

#[tokio::test]
#[ignore = "requires pinned Decider-2B checkpoint and NVIDIA CUDA"]
async fn complete_eager_requests_preserve_order_and_recover_from_validation_errors() {
    let dir = std::env::var("DECIDER_MODEL").unwrap();
    let library = std::env::var("DECIDER_CUDA_LIB").unwrap();
    let engine = Engine::load(Path::new(&dir), Path::new(&library))
        .await
        .unwrap();
    let raw = br#"{"state":"The ticket requests a refund.","questions":{"category":{"type":"choice","instructions":"Choose the category.","criteria":{"refund":"Refund","shipping":"Shipping"}},"refund":{"type":"noul","instructions":"The customer requests a refund."},"urgency":{"type":"score","instructions":"How urgent is the ticket?","criteria":["low urgency","high urgency"]}}}"#;
    let first = engine.predict(raw).await.unwrap();
    assert_eq!(
        first["answers"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        vec!["category", "refund", "urgency"]
    );
    assert_eq!(first["usage"]["output_tokens"], 0);
    assert!(
        engine
            .predict(br#"{"state":0,"questions":{"bad":{"criteria":["one"]}}}"#)
            .await
            .is_err()
    );
    assert!(engine.executor.is_ready());
    assert_eq!(engine.predict(raw).await.unwrap(), first);
    let empty = engine
        .predict(br#"{"state":{},"questions":{}}"#)
        .await
        .unwrap();
    assert!(empty["answers"].as_object().unwrap().is_empty());
    assert_eq!(empty["usage"]["input_tokens"], 0);
    assert_eq!(engine.health()["status"], "ready");
}

#[tokio::test]
#[ignore = "requires pinned Decider-2B checkpoint and NVIDIA CUDA; near-limit allocation"]
async fn maximum_complete_row_grows_scratch_without_leaking_state() {
    use omni_decider_native::Limits;
    use serde_json::json;
    let dir = std::env::var("DECIDER_MODEL").unwrap();
    let library = std::env::var("DECIDER_CUDA_LIB").unwrap();
    let engine = Engine::load(Path::new(&dir), Path::new(&library))
        .await
        .unwrap();
    let short = br#"{"state":"A customer requests a refund.","questions":{"q":{"instructions":"Choose a team.","criteria":["billing","shipping"]}}}"#;
    let baseline = engine.predict(short).await.unwrap();
    let request = |n: usize| {
        json!({"state":"hello ".repeat(45000),"questions":{"q":{"instructions":format!("Choose.{}"," x".repeat(n)),"criteria":["billing","shipping"]}}}).to_string()
    };
    let (mut low, mut high) = (0usize, 8000usize);
    while low < high {
        let mid = (low + high).div_ceil(2);
        if engine.processor.prepare(request(mid).as_bytes()).is_ok() {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    let raw = request(low);
    let prepared = engine.processor.prepare(raw.as_bytes()).unwrap();
    assert_eq!(prepared.rows[0].ids.len(), Limits::default().max_row_tokens);
    assert!(
        engine
            .processor
            .prepare(request(low + 1).as_bytes())
            .is_err()
    );
    let first = engine.predict(raw.as_bytes()).await.unwrap();
    assert!(
        first["answers"]["q"]["probabilities"]
            .as_object()
            .unwrap()
            .values()
            .all(|value| value.as_f64().unwrap().is_finite())
    );
    assert_eq!(engine.predict(raw.as_bytes()).await.unwrap(), first);
    assert_eq!(engine.predict(short).await.unwrap(), baseline);
    assert!(engine.executor.is_ready());
}
