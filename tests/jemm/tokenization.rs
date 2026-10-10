use omni_jemm_native::{contract, processing::Processor};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokenizers::{
    AddedToken, Tokenizer, models::wordlevel::WordLevel,
    pre_tokenizers::whitespace::WhitespaceSplit,
};
static ID: AtomicUsize = AtomicUsize::new(0);
struct Fixture(std::path::PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> (Fixture, Value) {
    let dir = std::env::temp_dir().join(format!(
        "jemm-tokenizer-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut vocab = contract::LABELS
        .chars()
        .enumerate()
        .map(|(i, c)| (c.to_string(), i as u32 + 1))
        .collect::<Vec<_>>();
    vocab.push(("[UNK]".into(), 0));
    vocab.push(("<|image_pad|>".into(), 33));
    let mut tok = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tok.with_pre_tokenizer(Some(WhitespaceSplit));
    tok.add_special_tokens(&[AddedToken::from("<|image_pad|>", true)]);
    tok.save(dir.join("tokenizer.json"), false).unwrap();
    std::fs::write(dir.join("config.json"), br#"{"image_token_id":33}"#).unwrap();
    std::fs::write(
        dir.join("preprocessor_config.json"),
        br#"{"size":{"shortest_edge":65536,"longest_edge":16777216}}"#,
    )
    .unwrap();
    let manifest = json!({"format":"jemm-native/1","model_id":"JEMM","base_model_id":"Qwen/Qwen3.8-27B","base_revision":contract::BASE_REVISION,"checkpoint_revision":contract::CHECKPOINT_REVISION,"source_revision":contract::SOURCE_REVISION,"max_tokens":8192,"max_mm_tokens":3072,"label_token_ids":(1..=32).collect::<Vec<_>>(),"chat_prefix":format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",contract::SYSTEM),"chat_suffix":"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","temperature":1.3480874159655591,"mm_temperature":1.3954832341582943,"threshold":0.9872681877423998});
    std::fs::write(
        dir.join("decision_config.json"),
        serde_json::to_vec(&json!({"temperature":manifest["temperature"],
            "mm_temperature":manifest["mm_temperature"], "threshold":manifest["threshold"]}))
        .unwrap(),
    )
    .unwrap();
    (Fixture(dir), manifest)
}
#[test]
fn text_special_token_ids_are_preserved_and_usage_counts_one_prompt() {
    let (f, m) = fixture();
    let p = Processor::load(&f.0, &m).unwrap();
    let prepared = p
        .prepare(br#"{"questions":{"q":{"criteria":{"a":"<|image_pad|>","b":"normal"}}}}"#)
        .unwrap();
    assert!(prepared.inputs[0].ids.contains(&33));
    let count = prepared.inputs[0].ids.len();
    let response = prepared.context.finish(vec![vec![0., 0.]]).unwrap();
    assert_eq!(response["usage"]["input_tokens"], count);
    assert_eq!(response["usage"]["output_tokens"], 0);
}
#[test]
fn rejects_mismatched_label_ids_and_oversized_prompt_before_execution() {
    let (f, mut m) = fixture();
    m["label_token_ids"][0] = json!(9);
    assert!(Processor::load(&f.0, &m).is_err());
    m["label_token_ids"][0] = json!(1);
    let p = Processor::load(&f.0, &m).unwrap();
    let body =
        json!({"questions":{"q":{"instructions":"x ".repeat(8192),"criteria":{"a":"a","b":"b"}}}});
    assert!(p.prepare(&serde_json::to_vec(&body).unwrap()).is_err());
}
#[test]
fn exported_tokenizer_truncation_and_padding_cannot_change_request_tokens() {
    let (f, m) = fixture();
    let mut t = Tokenizer::from_file(f.0.join("tokenizer.json")).unwrap();
    t.with_truncation(Some(tokenizers::TruncationParams {
        max_length: 3,
        ..Default::default()
    }))
    .unwrap();
    t.save(f.0.join("tokenizer.json"), false).unwrap();
    let processor = Processor::load(&f.0, &m).unwrap();
    let prepared = processor
        .prepare(br#"{"questions":{"q":{"criteria":{"a":"first option","b":"second option"}}}}"#)
        .unwrap();
    assert!(prepared.inputs[0].ids.len() > 3);
}
#[test]
fn rejects_manifest_calibration_disagreement() {
    let (f, mut m) = fixture();
    m["mm_temperature"] = json!(1.);
    assert!(
        Processor::load(&f.0, &m)
            .err()
            .unwrap()
            .to_string()
            .contains("manifest disagrees with decision_config.json")
    );
}

#[test]
fn file_driven_calibration_must_be_finite_in_range_and_agree_with_manifest() {
    for (field, invalid) in [
        ("temperature", json!(0.0)),
        ("mm_temperature", json!(-1.0)),
        ("threshold", json!(1.1)),
    ] {
        let (f, mut m) = fixture();
        m[field] = invalid;
        std::fs::write(
            f.0.join("decision_config.json"),
            serde_json::to_vec(&json!({"temperature":m["temperature"],
                "mm_temperature":m["mm_temperature"], "threshold":m["threshold"]}))
            .unwrap(),
        )
        .unwrap();
        assert!(
            Processor::load(&f.0, &m)
                .err()
                .unwrap()
                .to_string()
                .contains("invalid calibration")
        );
    }
    let (f, mut m) = fixture();
    m["temperature"] = json!(2.0);
    m["mm_temperature"] = json!(3.0);
    m["threshold"] = json!(0.5);
    std::fs::write(
        f.0.join("decision_config.json"),
        serde_json::to_vec(&json!({"temperature":2.0,"mm_temperature":3.0,"threshold":0.5}))
            .unwrap(),
    )
    .unwrap();
    let p = Processor::load(&f.0, &m).unwrap();
    let prepared = p
        .prepare(br#"{"questions":{"q":{"type":"noul"}}}"#)
        .unwrap();
    let response = prepared.context.finish(vec![vec![2.0, 0.0]]).unwrap();
    assert!(
        (response["answers"]["q"]["noul"].as_f64().unwrap() - 1.0 / (1.0 + (-1.0f64).exp())).abs()
            < 1e-12
    );
}
