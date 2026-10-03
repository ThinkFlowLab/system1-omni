use omni_laya::preprocess::{Preprocessor, Request, render};
use serde::Deserialize;
use serde_json::{Map, Value, json};
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
        let request: Request = Request::from_value(json!({
            "state":"", "questions":{"q":{"type":"noul","instructions":"New?",
                "labels":{"false":label,"true":"new"}}}
        }))
        .unwrap();
        pre.prepare(&request).unwrap().questions.remove(0).ids
    };
    assert_eq!(packed("\u{1c}old\u{1f}"), packed("old"));
    let invalid: Request = Request::from_value(json!({
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
        let request: Request = Request::from_value(json!({
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
    let request: Request = Request::from_value(json!({
        "state":"new", "questions":{"valid":{"type":"noul","instructions":"New?"}}
    }))
    .unwrap();
    assert_eq!(pre.prepare(&request).unwrap().questions.len(), 1);
}

#[test]
fn preserves_question_and_choice_order() {
    let (_dir, pre) = preprocessor();
    let request: Request = Request::from_json(
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
        let request: Request = Request::from_value(json!({
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
    let request: Request = Request::from_value(json!({
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
        assert!(pre.prepare(&Request::from_json(input).unwrap()).is_err());
    }
}

#[test]
fn private_json_keys_stay_objects_in_raw_requests() {
    let (_dir, pre) = preprocessor();
    for key in [
        "$serde_json::private::Number",
        "$serde_json::private::RawValue",
    ] {
        let state = json!({"outer": [{(key): "1.5"}]});
        let questions = json!({
            "z": {"type": "choice", "instructions": {(key): "2"},
                "criteria": {"new": [{(key): "3"}], "old": null}},
            "a": {"type": "noul", "instructions": "New?"}
        })
        .as_object()
        .unwrap()
        .clone();
        let expected = Request {
            state,
            model: None,
            questions,
            lang: None,
        };
        let encoded = serde_json::to_string(&expected).unwrap();
        let request: Request = Request::from_json(&encoded).unwrap();
        assert_eq!(request.state, expected.state, "{key}");
        assert_eq!(request.questions, expected.questions, "{key}");
        assert_eq!(serde_json::to_string(&request).unwrap(), encoded);
        assert_eq!(
            render(&request.state),
            format!(r#"{{"outer": [{{"{key}": "1.5"}}]}}"#)
        );
        let packed = pre.prepare(&request).unwrap();
        let expected_packed = pre.prepare(&expected).unwrap();
        assert_eq!(
            packed.questions[0].criteria,
            expected_packed.questions[0].criteria
        );
        assert_eq!(packed.questions[0].ids, expected_packed.questions[0].ids);
        assert_eq!(packed.questions[0].id, "z");
        assert_eq!(packed.questions[1].id, "a");
    }
}

#[test]
fn private_json_keys_stay_objects_from_value() {
    for key in [
        "$serde_json::private::RawValue",
        "$serde_json::private::Number",
    ] {
        let value = json!({"state": {(key): "1.5"}, "questions": {
            "q": {"type": "score", "instructions": "New?",
                "criteria": [{"outer": {(key): "2"}}]}
        }});
        let request: Request = Request::from_value(value.clone()).unwrap();
        assert_eq!(request.state, value["state"]);
        assert_eq!(Value::Object(request.questions), value["questions"]);
    }
}

#[test]
fn request_numbers_keep_arbitrary_precision_and_syntax() {
    for number in [
        "18446744073709551616000",
        "-18446744073709551616000",
        "1e+03",
        "1.2300",
        "-0",
        "1e400",
    ] {
        let encoded =
            format!(r#"{{"state":[{number}],"questions":{{"q":{{"criteria":[{number}]}}}}}}"#);
        let request: Request = Request::from_json(&encoded).unwrap();
        let scalar: Value = serde_json::from_str(number).unwrap();
        assert_eq!(request.state[0], scalar);
        assert_eq!(request.questions["q"]["criteria"][0], scalar);
        let roundtrip: Request =
            Request::from_value(serde_json::to_value(&request).unwrap()).unwrap();
        assert_eq!(roundtrip.state, request.state);
        assert_eq!(roundtrip.questions, request.questions);
    }
}

#[test]
fn request_keeps_default_json_recursion_limit() {
    #[derive(Deserialize)]
    struct Reference {
        #[serde(rename = "state")]
        _state: Value,
        #[serde(rename = "questions")]
        _questions: Map<String, Value>,
    }
    for depth in [124, 125, 126, 127, 128, 200] {
        let nested = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        for encoded in [
            format!(r#"{{"state":{nested},"questions":{{}}}}"#),
            format!(r#"{{"state":null,"questions":{{"q":{{"criteria":{nested}}}}}}}"#),
        ] {
            let expected = serde_json::from_str::<Reference>(&encoded).is_ok();
            let actual = Request::from_json(&encoded);
            assert_eq!(actual.is_ok(), expected, "depth {depth}: {encoded}");
            if !expected {
                assert!(actual.unwrap_err().to_string().contains("recursion limit"));
            }
        }
    }
}

#[test]
fn request_from_value_preserves_deep_values() {
    let mut nested = json!({"$serde_json::private::Number": "1.5"});
    for _ in 0..200 {
        nested = Value::Array(vec![nested]);
    }
    let value = json!({"state": nested.clone(), "questions": {
        "q": {"criteria": nested.clone()}
    }});
    let request = Request::from_value(value).unwrap();
    assert_eq!(request.state, nested);
    assert_eq!(request.questions["q"]["criteria"], nested);
    let request = Request::from_value(serde_json::to_value(&request).unwrap()).unwrap();
    assert_eq!(request.state, nested);
}

#[test]
fn request_rejects_invalid_json_values() {
    for state in ["NaN", "01", "[1,]", r#""\ud800""#] {
        let encoded = format!(r#"{{"state":{state},"questions":{{}}}}"#);
        assert!(Request::from_json(&encoded).is_err(), "{state}");
    }
    assert!(Request::from_json(r#"{"state":null,"questions":[]}"#).is_err());
    for encoded in [
        r#"{"state":null,"questions":{},"extra":1}"#,
        r#"{"state":null,"state":1,"questions":{}}"#,
        r#"{"state":null}"#,
        r#"{"questions":{}}"#,
        r#"{"state":null,"questions":{},"lang":1}"#,
        r#"[{"state":null,"questions":{}}]"#,
    ] {
        assert!(Request::from_json(encoded).is_err(), "{encoded}");
    }
    let request = Request::from_json(r#"{"state":{"x":1,"x":2},"questions":{}}"#).unwrap();
    assert_eq!(request.state["x"], 2);
}
