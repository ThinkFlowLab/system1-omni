use super::device_feature_tests::{load, questions};

#[test]
#[ignore = "requires test-only queued-copy failure wrapper, zero-entry cache and CUA_S1_* weights"]
fn device_features_production_transient_failure_drains_and_releases_before_return() {
    assert_eq!(
        std::env::var("CUA_S1_TEST_COPY_FAILURE").as_deref(),
        Ok("1")
    );
    assert_eq!(
        std::env::var("CUA_S1_VISION_CACHE_ENTRIES").as_deref(),
        Ok("0")
    );
    let evidence =
        std::path::PathBuf::from(std::env::var_os("CUA_S1_TEST_COPY_FAILURE_EVIDENCE").unwrap());
    let release = std::path::PathBuf::from(
        std::env::var_os("CUA_S1_TEST_COPY_FAILURE_RELEASE_EVIDENCE").unwrap(),
    );
    assert!(
        !evidence.exists() && !release.exists(),
        "use fresh evidence paths"
    );
    let mut engine = load();
    let first = engine
        .prepare(256, 256, &vec![80; 256 * 256 * 3], &questions(1))
        .unwrap();
    let changed = engine
        .prepare(512, 256, &vec![120; 512 * 256 * 3], &questions(2))
        .unwrap();
    engine.device_features = false;
    let expected_first = engine.predict_prepared(&first).unwrap();
    let expected_changed = engine.predict_prepared(&changed).unwrap();
    engine.device_features = true;
    let error = engine
        .predict_prepared(&first)
        .expect_err("queued D2D failure must surface");
    assert!(error.to_string().contains("device copy"), "{error:#}");
    // Check before recovery: the next enqueue would otherwise mask a missed release.
    assert_eq!(std::fs::read_to_string(&evidence).unwrap(), "drained\n");
    assert_eq!(std::fs::read_to_string(&release).unwrap(), "released\n");
    assert_eq!(engine.predict_prepared(&changed).unwrap(), expected_changed);
    assert_eq!(engine.predict_prepared(&first).unwrap(), expected_first);
}
