use super::*;
use crate::contract;

#[test]
#[ignore = "needs the pinned export at OPEN_JEV_MODEL; no GPU needed"]
fn tokenization_matches_reference_and_rejects_oversize_prompts() {
    let dir = std::env::var_os("OPEN_JEV_MODEL").expect("OPEN_JEV_MODEL");
    let dir = Path::new(&dir);
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(dir.join("open_jev_export.json")).unwrap()).unwrap();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let prefix = manifest["chat_prefix"].as_str().unwrap();
    let suffix = manifest["chat_suffix"].as_str().unwrap();
    let cases: Value = serde_json::from_str(include_str!("data/tokenization.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let questions = contract::compile(&serde_json::to_vec(&case["request"]).unwrap()).unwrap();
        let got = encode_questions(&tokenizer, prefix, suffix, 4096, &questions).unwrap();
        assert_eq!(serde_json::json!(got), case["ids"]);
        let limit = got[0][0].len() - 1;
        assert!(encode_questions(&tokenizer, prefix, suffix, limit, &questions).is_err());
    }
}
