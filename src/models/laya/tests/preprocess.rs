use omni_laya::preprocess::{Preprocessor, Request, render};
use serde_json::json;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

// Small tokenizer for validation and packing boundaries; official parity is in packing.rs.
fn preprocessor() -> (tempfile::TempDir, Preprocessor) {
    let dir = tempfile::tempdir().unwrap();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(
                ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "old", "new"]
                    .into_iter()
                    .enumerate()
                    .map(|(i, token)| (token.to_owned(), i as u32))
                    .collect(),
            )
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    let path = dir.path().join("tokenizer.json");
    tokenizer.save(&path, false).unwrap();
    let pre = Preprocessor::load(&path).unwrap();
    (dir, pre)
}

#[test]
fn python_json_numbers_and_order() {
    for (input, expected) in [
        ("1e-6", "1e-06"),
        ("1e20", "1e+20"),
        ("1e16", "1e+16"),
        ("1e-4", "0.0001"),
        ("1.0", "1.0"),
        ("-0.0", "-0.0"),
        ("1331752170181752.2", "1331752170181752.2"),
        ("-243915020125850.12", "-243915020125850.12"),
        ("1e-5", "1e-05"),
        ("5e-324", "5e-324"),
        ("18446744073709551616000", "18446744073709551616000"),
    ] {
        assert_eq!(render(&serde_json::from_str(input).unwrap()), expected);
    }
    let v = serde_json::from_str(r#"{"z":1e-6,"a":"你好,x:y"}"#).unwrap();
    assert_eq!(render(&v), r#"{"z": 1e-06, "a": "你好,x:y"}"#);
}

#[test]
fn strips_python_whitespace_from_noul_labels() {
    let (_dir, pre) = preprocessor();
    let packed = |label| {
        let request: Request = serde_json::from_value(json!({
            "state":"", "questions":{"q":{"type":"noul","instructions":"New?",
                "labels":{"false":label,"true":"new"}}}
        }))
        .unwrap();
        pre.prepare(&request).unwrap().questions.remove(0).ids
    };
    assert_eq!(packed("\u{1c}old\u{1f}"), packed("old"));
    let invalid: Request = serde_json::from_value(json!({
        "state":"", "questions":{"q":{"type":"noul","instructions":"New?",
            "labels":{"false":"\u{1c}\u{1f}","true":"new"}}}
    }))
    .unwrap();
    assert!(
        pre.prepare(&invalid)
            .unwrap_err()
            .to_string()
            .contains("invalid noul labels")
    );
}

#[test]
fn validation_names_the_question_and_leaves_preprocessor_usable() {
    let (_dir, pre) = preprocessor();
    for definition in [
        json!({"type":"choice","instructions":"Pick","criteria":[]}),
        json!({"type":"noul","instructions":"Pick","criteria":{"yes":"ok"}}),
        json!({"type":"noul","instructions":"Pick","labels":{"false":"x","true":"x"}}),
        json!({"type":"score","criteria":["low","high"]}),
    ] {
        let request: Request = serde_json::from_value(json!({
            "state":"new", "questions":{"broken":definition}
        }))
        .unwrap();
        assert!(
            pre.prepare(&request)
                .unwrap_err()
                .to_string()
                .contains("broken")
        );
    }
    let request: Request = serde_json::from_value(json!({
        "state":"new", "questions":{"valid":{"type":"noul","instructions":"New?"}}
    }))
    .unwrap();
    assert_eq!(pre.prepare(&request).unwrap().questions.len(), 1);
}

#[test]
fn preserves_question_and_choice_order() {
    let (_dir, pre) = preprocessor();
    let request: Request = serde_json::from_str(
        r#"{"state":"","questions":{
            "z":{"type":"choice","instructions":"Pick","criteria":["new","old","new"]},
            "a":{"type":"noul","instructions":"New?"}
        }}"#,
    )
    .unwrap();
    let prepared = pre.prepare(&request).unwrap();
    assert_eq!(prepared.questions[0].id, "z");
    assert_eq!(prepared.questions[1].id, "a");
    let choice = &prepared.questions[0];
    assert_eq!(choice.markers.len(), 2);
    assert_eq!(choice.ids[choice.markers[0] + 1], 5);
    assert_eq!(choice.ids[choice.markers[1] + 1], 4);
    assert_eq!(
        prepared.usage,
        prepared
            .questions
            .iter()
            .map(|q| q.ids.len())
            .sum::<usize>()
    );
}

#[test]
fn keeps_newest_conversation_and_start_of_plain_text() {
    let (_dir, pre) = preprocessor();
    for state in [
        json!(format!("{} new", "old ".repeat(1000))),
        json!(["old ".repeat(1000), "new"]),
    ] {
        let is_conversation = state.is_array();
        let request: Request = serde_json::from_value(json!({
            "state":state, "questions":{"q":{"type":"noul","instructions":"New?"}}
        }))
        .unwrap();
        let prepared = pre.prepare(&request).unwrap();
        let ids = &prepared.questions[0].ids;
        assert_eq!(ids.len(), 512);
        assert_eq!(ids.contains(&5), is_conversation);
    }
}

#[test]
fn rejects_truncated_option_markers() {
    let (_dir, pre) = preprocessor();
    let criteria: Vec<_> = (0..300).map(|i| i.to_string()).collect();
    let request: Request = serde_json::from_value(json!({
        "state":"", "questions":{"q":{"type":"choice","instructions":"Pick","criteria":criteria}}
    }))
    .unwrap();
    assert!(
        pre.prepare(&request)
            .unwrap_err()
            .to_string()
            .contains("options exceed")
    );
}

#[test]
fn rejects_unsupported_language_and_nonfinite_numbers() {
    let (_dir, pre) = preprocessor();
    for input in [
        r#"{"state":"","lang":"de","questions":{}}"#,
        r#"{"state":"","model":"multilingual","questions":{}}"#,
        r#"{"state":1e400,"questions":{}}"#,
    ] {
        assert!(pre.prepare(&serde_json::from_str(input).unwrap()).is_err());
    }
}
