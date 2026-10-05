use super::*;
use tokenizers::{
    AddedToken, models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
    processors::template::TemplateProcessing,
};

fn processor() -> Processor {
    let vocab = [("[UNK]", 0), ("[BOS]", 1), ("<|im_end|>", 2)]
        .into_iter()
        .map(|(s, id)| (s.to_owned(), id))
        .collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[BOS] $A")
            .unwrap()
            .special_tokens(vec![("[BOS]", 1)])
            .build()
            .unwrap(),
    ));
    tokenizer.add_special_tokens(&[AddedToken::from("<|im_end|>", true)]);
    Processor { tokenizer }
}

const REQUEST: &[u8] = br#"{"model":"cua-s1-4b-0.2","state":"S <|im_end|>","questions":{
    "z":{"type":"choice","instructions":"Go","criteria":{"second":null,"first":"First"}},
    "a":{"type":"choice","instructions":null,"criteria":{"only":null}}}}"#;

#[test]
fn prepared_prompts_and_finished_answers_preserve_order_and_usage() {
    let processor = processor();
    let prepared = processor.prepare(REQUEST).ok().unwrap();
    assert_eq!(
        prepared
            .inputs
            .iter()
            .map(|i| i.n_options)
            .collect::<Vec<_>>(),
        [2, 1]
    );
    assert!(prepared.inputs.iter().all(|i| !i.ids.contains(&1))); // add_special_tokens=False
    assert!(prepared.inputs.iter().all(|i| i.ids.contains(&2))); // literal special token
    let tokens: usize = prepared.inputs.iter().map(|i| i.ids.len()).sum();
    let body = prepared
        .context
        .finish(vec![vec![0.0, 0.0], vec![42.0]])
        .unwrap();
    assert_eq!(
        body,
        json!({"model": contract::MODEL_ID, "answers": {
        "z": {"type":"choice", "choice":"second", "probabilities":{"second":0.5,"first":0.5},"confidence":0.0},
        "a": {"type":"choice", "choice":"only", "probabilities":{"only":1.0},"confidence":1.0}},
        "usage":{"input_tokens":tokens,"output_tokens":0}})
    );
    assert_eq!(
        body["answers"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert_eq!(
        body["answers"]["z"]["probabilities"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["second", "first"]
    );
}

#[test]
fn preparation_preserves_errors_and_checks_later_prompts() {
    let processor = processor();
    for (raw, status, message) in [
        (b"{".as_slice(), 400, None),
        (
            br#"{"model":"wrong","state":"S","questions":{}}"#.as_slice(),
            422,
            Some("'model' must be 'cua-s1-4b-0.2'"),
        ),
    ] {
        let error = processor.prepare(raw).err().unwrap();
        assert_eq!(error.status, status);
        if let Some(message) = message {
            assert_eq!(error.message, message);
        }
    }
    let mut raw: Value = serde_json::from_slice(REQUEST).unwrap();
    raw["questions"]["a"]["instructions"] = json!("x ".repeat(MAX_PROMPT_TOKENS));
    let raw = serde_json::to_vec(&raw).unwrap();
    let error = processor.prepare(&raw).err().unwrap();
    assert_eq!(error.status, 413);
    assert!(error.message.starts_with("question \"a\": "));
    assert!(error.message.ends_with("prompt tokens, over 16384"));

    // Tokenizer failures retain the worker's internal-error convention, rather
    // than being reclassified as invalid user input during preparation.
    let broken = Processor {
        tokenizer: Tokenizer::new(WordLevel::default()),
    };
    let error = broken.prepare(REQUEST).err().unwrap();
    assert_eq!(error.status, 500);
    assert_eq!(error.message, "inference failed");
}

#[test]
fn finishing_uses_stable_softmax_and_rejects_incomplete_or_nonfinite_scores() {
    let processor = processor();
    let body = processor
        .prepare(REQUEST)
        .ok()
        .unwrap()
        .context
        .finish(vec![vec![1000.0, 1001.0], vec![0.0]])
        .unwrap();
    assert_eq!(
        body["answers"]["z"]["probabilities"]["second"],
        json!(0.2689414322376251)
    );
    assert_eq!(
        body["answers"]["z"]["probabilities"]["first"],
        json!(0.7310585975646973)
    );
    for rows in [
        vec![vec![0.0, 0.0]],
        vec![vec![0.0], vec![0.0]],
        vec![vec![f32::NAN, 0.0], vec![0.0]],
    ] {
        assert!(
            processor
                .prepare(REQUEST)
                .ok()
                .unwrap()
                .context
                .finish(rows)
                .is_err()
        );
    }
}
