use super::device_feature_tests::{load, questions};
use super::*;

#[test]
#[ignore = "requires test-only queued-copy failure wrapper and CUA_S1_* weights"]
fn device_features_queued_copy_failure_drains_before_release_and_recovers() {
    assert_eq!(
        std::env::var("CUA_S1_TEST_COPY_FAILURE").as_deref(),
        Ok("1")
    );
    let evidence =
        std::path::PathBuf::from(std::env::var_os("CUA_S1_TEST_COPY_FAILURE_EVIDENCE").unwrap());
    assert!(!evidence.exists(), "use a fresh copy-drain evidence path");
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
    {
        let features = engine.vision.forward_device(&first.image).unwrap();
        let error = VisionEngine::raw_score_device(
            &mut engine.language,
            &engine.letters,
            &first.prompts[0],
            &features,
            2,
        )
        .expect_err("injected D2D failure must surface");
        assert!(error.to_string().contains("device copy"), "{error:#}");
        // The wrapper records a successful real stream drain after queuing D2D,
        // before the producer's borrowed allocation is allowed to be released.
        assert_eq!(std::fs::read_to_string(&evidence).unwrap(), "drained\n");
    }
    engine.device_features = true;
    assert_eq!(engine.predict_prepared(&changed).unwrap(), expected_changed);
    assert_eq!(engine.predict_prepared(&first).unwrap(), expected_first);
}
