use super::*;
use crate::caches::CacheCfg;
use tokenizers::{
    AddedToken, models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
};

fn processor() -> Processor {
    let vocab = [("[UNK]".to_owned(), 0)].into_iter().collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer.add_special_tokens(&[
        AddedToken::from("<|vision_start|>", true),
        AddedToken::from("<|image_pad|>", true),
        AddedToken::from("<|vision_end|>", true),
    ]);
    let caches = Caches::new(CacheCfg {
        enabled: true,
        l1: true,
        l2: true,
        l3: false,
        l1_max: 8,
        l2_bytes: 1 << 20,
        l3_bytes: 0,
    });
    caches.l2_insert(
        Processor::url_key("prepared://image"),
        Arc::new(ImageAsset {
            grid_thw: [1, 16, 16],
            embeddings: vec![half::bf16::ONE; 64 * 5120],
        }),
    );
    Processor {
        image_pad: tokenizer.token_to_id("<|image_pad|>").unwrap(),
        vision_end: tokenizer.token_to_id("<|vision_end|>").unwrap(),
        tokenizer,
        labels: vec!["A".into(), "B".into()],
        max_length: 256,
        imgcache: None,
        model_index_hash: "test".into(),
        caches,
    }
}

#[test]
fn image_placeholders_are_rejected_before_and_after_l1_warmup() {
    let processor = processor();
    let valid = serde_json::json!({
        "kind": "choice", "state": [{"image": "prepared://image"}],
        "question": "Pick one.", "options": ["yes", "no"],
    });
    let prepare = |request: &Value| {
        let compiled =
            contract::compile(&serde_json::to_vec(request).unwrap(), &processor.labels).unwrap();
        processor.prepare_cached(
            &compiled,
            Readout {
                token_ids: vec![0, 1],
                bias: vec![0.0; 2],
                temperature: 1.0,
            },
            Instant::now(),
        )
    };
    let mut bad_question = valid.clone();
    bad_question["question"] = serde_json::json!("Pick <|image_pad|>.");
    let mut bad_option = valid.clone();
    bad_option["options"][0] = serde_json::json!("<|image_pad|>");
    for request in [&bad_question, &bad_option] {
        assert_eq!(
            prepare(request)
                .err()
                .expect("cold request must fail")
                .status,
            400
        );
    }
    assert_eq!(processor.caches.snapshot().l1_records, 0);
    assert_eq!(prepare(&valid).unwrap().cache_note, "l1=miss,l3=off,p=-");
    assert_eq!(prepare(&valid).unwrap().cache_note, "l1=hit,l3=off,p=-");
    for request in [&bad_question, &bad_option] {
        assert_eq!(
            prepare(request)
                .err()
                .expect("warm request must fail")
                .status,
            400
        );
    }
}
