use omni_laya::{
    config::Config,
    preprocess::{Preprocessor, Request},
};
use serde_json::Value;
use std::path::PathBuf;
#[test]
#[ignore = "requires LAYA_CHECKPOINT at the frozen English checkpoint; no GPU"]
fn official_packing_golden() {
    let checkpoint =
        PathBuf::from(std::env::var_os("LAYA_CHECKPOINT").expect("set LAYA_CHECKPOINT"));
    Config::load(&checkpoint).unwrap();
    let pre = Preprocessor::load(&checkpoint).unwrap();
    let cases: Vec<Value> = serde_json::from_str(include_str!(
        "../../../../recipe/laya/native/packing-golden.json"
    ))
    .unwrap();
    for c in cases {
        let request: Request = serde_json::from_value(c["request"].clone()).unwrap();
        let got = serde_json::to_value(pre.prepare(&request).unwrap()).unwrap();
        let expected = &c["expected"];
        for key in ["b", "l", "input_ids", "lens", "qtypes", "usage"] {
            assert_eq!(got[key], expected[key], "{} {key}", c["name"]);
        }
        for (g, w) in got["questions"]
            .as_array()
            .unwrap()
            .iter()
            .zip(expected["items"].as_array().unwrap())
        {
            for key in ["ids", "markers", "qtype"] {
                assert_eq!(g[key], w[key], "{} {key}", c["name"]);
            }
        }
    }
}

#[test]
#[ignore = "requires LAYA_CHECKPOINT; no GPU"]
fn oversized_options_are_a_client_error() {
    let checkpoint = PathBuf::from(std::env::var_os("LAYA_CHECKPOINT").unwrap());
    let pre = Preprocessor::load(&checkpoint).unwrap();
    let criteria: Vec<_> = (0..129).map(|i| i.to_string()).collect();
    let questions: serde_json::Map<String, Value> = (0..16)
        .map(|i| {
            (
                i.to_string(),
                serde_json::json!({"type":"choice","instructions":"Pick", "criteria":criteria}),
            )
        })
        .collect();
    let request: Request = serde_json::from_value(
        serde_json::json!({"state":"", "model":"english", "questions":questions}),
    )
    .unwrap();
    assert!(
        pre.prepare(&request)
            .unwrap_err()
            .to_string()
            .contains("2048")
    );
    let valid: Request=serde_json::from_value(serde_json::json!({"state":"refund", "model":"english", "questions":{"q":{"type":"noul","instructions":"Refund?"}}})).unwrap();
    assert!(pre.prepare(&valid).is_ok());
}
