//! Preparation against fixtures from the pinned reference (tests/omnijev/generate_fixtures.py).
mod common;

use omni_omnijev_native::contract;
use omni_omnijev_native::processing::{self, Processor};
use omni_qwen3_5_native::inputs::image_positions;
use sha2::{Digest, Sha256};

fn positions_sha256(positions: &[Vec<i64>; 3]) -> String {
    let mut hash = Sha256::new();
    for axis in positions {
        for v in axis {
            hash.update(v.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

#[test]
fn questions_prompts_and_layout_match_reference() {
    let fixture = common::fixture("processing.json");
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let request = contract::compile(&common::request(case)).unwrap();
        let grid = processing::pixels(&request.image).unwrap().image_grid_thw;
        assert_eq!(serde_json::json!(grid), case["image_grid_thw"], "{name}");
        let expected = case["questions"].as_array().unwrap();
        assert_eq!(request.questions.len(), expected.len(), "{name}");
        let mut sequences = Vec::new();
        for (q, e) in request.questions.iter().zip(expected) {
            assert_eq!(q.id, e["id"].as_str().unwrap(), "{name}");
            assert_eq!(serde_json::json!(q.keys), e["keys"], "{name} {}", q.id);
            assert_eq!(
                serde_json::json!(q.options),
                e["option_text"],
                "{name} {}",
                q.id
            );
            assert_eq!(
                processing::prompt(q),
                e["prompt"].as_str().unwrap(),
                "{name} {}",
                q.id
            );
            let ids = common::token_ids(&e["token_ids"]);
            let (opens, closes) = processing::spans(&ids);
            assert_eq!(serde_json::json!(opens), e["opens"], "{name} {}", q.id);
            assert_eq!(serde_json::json!(closes), e["closes"], "{name} {}", q.id);
            let positions = image_positions(&ids, processing::IMAGE_TOKEN, grid).unwrap();
            assert_eq!(
                positions_sha256(&positions),
                e["positions_sha256"].as_str().unwrap(),
                "{name} {}",
                q.id
            );
            sequences.push(ids);
        }
        let (layout, rows) = processing::layout(&sequences, grid).unwrap();
        assert_eq!(
            layout.prefix as u64,
            case["prefix_length"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(
            layout.processed_tokens as u64,
            case["input_tokens"].as_u64().unwrap(),
            "{name}"
        );
        for (question_rows, e) in rows.iter().zip(expected) {
            let e = e["rows"].as_array().unwrap();
            assert_eq!(question_rows.len(), e.len(), "{name}");
            for (row, e) in question_rows.iter().zip(e) {
                assert_eq!(serde_json::json!(row.tokens), e["tokens"], "{name}");
                assert_eq!(row.zq as u64, e["zq"].as_u64().unwrap(), "{name}");
                assert_eq!(row.u as u64, e["u"].as_u64().unwrap(), "{name}");
                let targets: Vec<[u64; 2]> = row
                    .targets
                    .iter()
                    .map(|&(p, t)| [p as u64, t as u64])
                    .collect();
                assert_eq!(serde_json::json!(targets), e["targets"], "{name}");
                assert_eq!(serde_json::json!(row.first), e["first"], "{name}");
            }
        }
        // Rows continue the prefix: the first row position is the prefix's largest plus one.
        let positions = image_positions(&sequences[0], processing::IMAGE_TOKEN, grid).unwrap();
        assert_eq!(layout.next_position, positions[0][layout.prefix], "{name}");
    }
}

/// Needs the checkpoint's tokenizer: `OMNIJEV_CHECKPOINT=<tinnel123/OmniJev @ ffe5f43>`.
#[test]
#[ignore]
fn tokenization_matches_reference() {
    let dir = std::env::var("OMNIJEV_CHECKPOINT").expect("set OMNIJEV_CHECKPOINT");
    let processor = Processor::load(std::path::Path::new(&dir)).unwrap();
    let fixture = common::fixture("processing.json");
    for case in fixture["cases"].as_array().unwrap() {
        let raw = common::request(case);
        let prepared = processor.prepare(&raw).unwrap();
        for (q, e) in prepared
            .inputs
            .iter()
            .zip(case["questions"].as_array().unwrap())
        {
            assert_eq!(
                q.token_ids,
                common::token_ids(&e["token_ids"]),
                "{}",
                case["name"]
            );
        }
        assert_eq!(
            prepared.layout.processed_tokens as u64,
            case["input_tokens"].as_u64().unwrap()
        );
    }
}
