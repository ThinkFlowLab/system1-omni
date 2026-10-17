use std::path::Path;

use omni_openjev_native::processing::Processor;
use serde_json::{Value, json};

#[test]
#[ignore = "needs the pinned tokenizer.json at OPENJEV_MODEL; no weights, no GPU"]
fn token_ids_match_the_helpers_chat_template_and_oversize_prompts_fail() {
    let dir = std::env::var_os("OPENJEV_MODEL").expect("OPENJEV_MODEL");
    let processor = Processor::load(Path::new(&dir)).unwrap();
    let fixtures: Value = serde_json::from_str(include_str!("data/contract.json")).unwrap();
    for case in fixtures["cases"].as_array().unwrap() {
        let raw = serde_json::to_vec(&case["request"]).unwrap();
        let prepared = processor.prepare(&raw).unwrap();
        for (p, call) in prepared.iter().zip(case["calls"].as_array().unwrap()) {
            assert_eq!(
                json!(p.input_ids),
                call["input_ids"],
                "{}: {}",
                case["name"],
                p.question.id
            );
            assert_eq!(json!(p.candidate_ids), call["candidate_token_ids"]);
        }
        let usage: usize = prepared.iter().map(|p| p.input_ids.len()).sum();
        assert_eq!(
            json!(usage),
            case["response"]["usage"]["input_tokens"],
            "{}",
            case["name"]
        );
        let longest = prepared.iter().map(|p| p.input_ids.len()).max().unwrap();
        let tight = Processor::load(Path::new(&dir))
            .unwrap()
            .with_max_tokens(longest - 1);
        assert!(
            tight
                .prepare(&raw)
                .unwrap_err()
                .to_string()
                .contains("the limit is")
        );
    }
    let letters: Vec<u32> = (0..52)
        .map(|i| {
            let l = omni_openjev_native::contract::letter(i).to_string();
            fixtures["meta"]["letter_token_ids"][&l].as_u64().unwrap() as u32
        })
        .collect();
    let request = json!({"state": "x", "questions": {"q": {"type": "choice", "instructions": "x",
        "criteria": (0..52).map(|i| (format!("o{i}"), Value::Null)).collect::<serde_json::Map<_, _>>()}}});
    let prepared = processor
        .prepare(&serde_json::to_vec(&request).unwrap())
        .unwrap();
    assert_eq!(prepared[0].candidate_ids, letters);
}
