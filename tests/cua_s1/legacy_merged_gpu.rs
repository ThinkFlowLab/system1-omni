//! Frozen A800 compatibility oracle, separate from unmerged/reference parity.
use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::Read;

#[tokio::test]
#[ignore = "requires CUDA and the frozen A800 legacy merged artifact; run serially at Graph 0/1"]
async fn legacy_merged_upgrade_preserves_four_decisions() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/legacy_merged_readout.json")).unwrap();
    let model = std::path::PathBuf::from(
        std::env::var_os("CUA_S1_MERGED_TEST_MODEL").expect("CUA_S1_MERGED_TEST_MODEL"),
    );
    let library =
        std::path::PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").expect("CUA_S1_CUDA_LIB"));
    assert!(matches!(
        crate::provenance::text_export(&model).unwrap(),
        crate::provenance::TextExport::Merged
    ));
    // Freeze weights/tokenizer as well as requests; another merged checkpoint
    // must not accidentally qualify against this A800 compatibility oracle.
    for (name, expected) in fixture["artifact_files"].as_object().unwrap() {
        let path = model.join(name);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            expected["size"].as_u64().unwrap(),
            "{name}"
        );
        let mut file = std::fs::File::open(&path).unwrap();
        let mut hash = Sha256::new();
        let mut buffer = vec![0; 64 * 1024];
        loop {
            let n = file.read(&mut buffer).unwrap();
            if n == 0 {
                break;
            }
            hash.update(&buffer[..n]);
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            expected["sha256"].as_str().unwrap(),
            "{name}"
        );
    }
    let engine = Engine::load(&model, &library).await.unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    let mut first_pass = Vec::new();
    for pass in 0..2 {
        for (index, case) in cases.iter().enumerate() {
            let prepared = engine
                .processor
                .prepare(&serde_json::to_vec(&case["request"]).unwrap())
                .ok()
                .unwrap();
            let rows = engine
                .executor
                .execute(&engine.scheduler, prepared.inputs)
                .await
                .unwrap();
            let response = prepared.context.finish(rows).unwrap();
            let expected = &case["expected_response"];
            assert_eq!(response["model"], expected["model"]);
            assert_eq!(response["usage"], expected["usage"]);
            let actual = &response["answers"]["q"];
            let baseline = &expected["answers"]["q"];
            assert_eq!(actual["type"], baseline["type"]);
            assert_eq!(
                actual["choice"], baseline["choice"],
                "{} pass {pass}",
                case["id"]
            );
            let actual_probs = actual["probabilities"].as_object().unwrap();
            let baseline_probs = baseline["probabilities"].as_object().unwrap();
            assert_eq!(
                actual_probs.keys().collect::<Vec<_>>(),
                baseline_probs.keys().collect::<Vec<_>>()
            );
            let max_error = baseline_probs
                .iter()
                .map(|(key, value)| {
                    let p = actual_probs[key].as_f64().unwrap();
                    assert!(p.is_finite() && (0.0..=1.0).contains(&p));
                    (p - value.as_f64().unwrap()).abs()
                })
                .fold(0.0_f64, f64::max);
            assert!(
                max_error <= 0.02,
                "{} pass {pass}: drift {max_error}",
                case["id"]
            );
            assert!(
                (actual["confidence"].as_f64().unwrap() - baseline["confidence"].as_f64().unwrap())
                    .abs()
                    <= 0.02
            );
            let bytes = serde_json::to_vec(&response).unwrap();
            if pass == 0 {
                first_pass.push(bytes);
            } else {
                assert_eq!(bytes, first_pass[index], "{} repeat drift", case["id"]);
            }
            eprintln!(
                "legacy merged pass={pass} id={} choice={} max_probability_error={max_error}",
                case["id"], actual["choice"]
            );
        }
    }
}
