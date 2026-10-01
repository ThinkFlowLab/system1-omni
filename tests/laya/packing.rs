use omni_laya::preprocess::{Preprocessor, Request};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

fn checked_file(variable: &str, sha256: &str) -> PathBuf {
    let path =
        PathBuf::from(std::env::var_os(variable).unwrap_or_else(|| panic!("set {variable}")));
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        sha256,
        "{variable}"
    );
    path
}

#[test]
#[ignore = "requires pinned tokenizer and packing oracle; run by CPU CI, no GPU"]
fn official_packing_parity() {
    let tokenizer = checked_file(
        "LAYA_TOKENIZER",
        "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
    );
    let oracle = checked_file(
        "LAYA_PACKING_ORACLE",
        "8cbaa311a59924ac9f9f2ba6f8438f82dafbd08476f64e86868289c2ede67f60",
    );
    let pre = Preprocessor::load(&tokenizer).unwrap();
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(oracle).unwrap()).unwrap();
    assert_eq!(cases.len(), 17);
    for case in cases {
        let request: Request = serde_json::from_value(case["request"].clone()).unwrap();
        let got = pre.prepare(&request).unwrap();
        let expected = &case["expected"];
        let items = expected["items"].as_array().unwrap();
        assert_eq!(
            got.questions.len(),
            items.len(),
            "{} row count",
            case["name"]
        );
        assert_eq!(got.usage, expected["usage"].as_u64().unwrap() as usize);
        for (i, ((question, item), id)) in got
            .questions
            .iter()
            .zip(items)
            .zip(request.questions.keys())
            .enumerate()
        {
            assert_eq!(&question.id, id, "{} row order", case["name"]);
            let actual = serde_json::to_value(question).unwrap();
            for key in ["ids", "markers", "qtype"] {
                assert_eq!(actual[key], item[key], "{} row {i} {key}", case["name"]);
            }
            assert_eq!(
                question.ids.len(),
                expected["lens"][i].as_u64().unwrap() as usize
            );
        }
    }
}
