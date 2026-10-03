//! The text the two heads see, checked byte-for-byte against the reference.
//!
//! `state_text`, `candidates` and `to_text` decide what reaches the encoder, and getting
//! one separator wrong shifts every probability without failing anything. The oracle is
//! produced by the reference implementation itself
//! (`recipe/clm/native/text_oracle.py`, which imports `clm.schema`), so this compares
//! against the real thing rather than a transcription of it.
use omni_clm::serve::{answer_json, candidates, state_text, to_text};
use omni_clm::{Kind, Question, answer, serve::QuestionRequest};
use serde_json::{Value, json};

fn oracle() -> Value {
    let path = std::env::var_os("CLM_TEXT_ORACLE")
        .expect("set CLM_TEXT_ORACLE to the file text_oracle.py writes");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn states() -> Vec<Value> {
    vec![
        json!("I was charged twice."),
        json!({"body": "Charged twice", "order": 4411, "urgent": true}),
        json!({"ticket": {"id": 7, "tags": ["a", "b"]}, "note": null}),
        json!([{"k": 1}, {"k": 2}]),
        json!({"empty_obj": {}, "empty_arr": [], "n": 0.5}),
        json!({"nested": {"deep": {"x": "y"}}}),
        // `str(float)` switches to exponent form outside [1e-4, 1e16) and keeps a `.0` on
        // an integral float, so these pin the number spelling the reference produces.
        json!({"tiny": 1e-5, "smaller": 1e-7, "edge": 1e-4, "round": 1e15, "huge": 1e16, "neg": -1e-6}),
    ]
}

fn questions() -> Vec<QuestionRequest> {
    vec![
        QuestionRequest {
            kind: Kind::Choice,
            instructions: "Which team?".into(),
            criteria: Some(json!({"billing": "Charges and refunds", "tech": "Software problems"})),
        },
        QuestionRequest {
            kind: Kind::Choice,
            instructions: "Pick".into(),
            criteria: Some(json!({"a": "", "b": null})),
        },
        QuestionRequest {
            kind: Kind::Score,
            instructions: "How urgent?".into(),
            criteria: Some(json!(["Not urgent", "Soon", "Now"])),
        },
        QuestionRequest {
            kind: Kind::Noul,
            instructions: "Does the customer ask for a refund?".into(),
            criteria: None,
        },
        QuestionRequest {
            kind: Kind::Noul,
            instructions: "Refund?".into(),
            criteria: Some(json!({"true": "Yes they do", "false": "No they do not"})),
        },
        // An empty container renders to nothing, which is not the same as being absent.
        QuestionRequest {
            kind: Kind::Choice,
            instructions: "Pick a bucket".into(),
            criteria: Some(json!({"empty_obj": {}, "empty_list": []})),
        },
        QuestionRequest {
            kind: Kind::Noul,
            instructions: "Is it so?".into(),
            criteria: Some(json!({"true": {}, "false": []})),
        },
        // Numbers are spelled by `str(float)` in criteria too, not only in the state.
        QuestionRequest {
            kind: Kind::Score,
            instructions: "How much?".into(),
            criteria: Some(json!([1e-5, 0.5, 1e16])),
        },
    ]
}

#[test]
#[ignore = "requires CLM_TEXT_ORACLE from recipe/clm/native/text_oracle.py; CPU only"]
fn to_text_matches_the_reference_byte_for_byte() {
    let oracle = oracle();
    let expected = oracle["to_text"].as_array().unwrap();
    let got = states();
    assert_eq!(
        got.len(),
        expected.len(),
        "the oracle was built from another case list"
    );
    for (i, state) in got.iter().enumerate() {
        assert_eq!(
            to_text(state),
            expected[i].as_str().unwrap(),
            "to_text case {i} for {state}"
        );
    }
}

#[test]
#[ignore = "requires CLM_TEXT_ORACLE from recipe/clm/native/text_oracle.py; CPU only"]
fn state_text_and_candidates_match_the_reference_byte_for_byte() {
    let oracle = oracle();
    let cases = oracle["cases"].as_array().unwrap();
    let states = states();
    let questions = questions();
    assert_eq!(
        cases.len(),
        states.len() * questions.len(),
        "the oracle was built from another case list"
    );

    let mut i = 0;
    for state in &states {
        for q in &questions {
            let case = &cases[i];
            let (keys, texts) = candidates(q).unwrap();
            assert_eq!(
                state_text(state, &q.instructions),
                case["state_text"].as_str().unwrap(),
                "case {i} state_text"
            );
            assert_eq!(
                keys,
                case["keys"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect::<Vec<_>>(),
                "case {i} keys"
            );
            assert_eq!(
                texts,
                case["candidate_texts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect::<Vec<_>>(),
                "case {i} candidate_texts"
            );
            i += 1;
        }
    }
}

#[test]
fn text_construction_handles_the_edges_the_oracle_does_not() {
    // No state in the oracle is empty, and no question leaves `instructions` blank.
    assert_eq!(state_text(&json!("context"), "  "), "context");
    assert_eq!(state_text(&json!(""), "question"), "question");
    // A state that is a bare scalar is not a container, so it renders unindented.
    assert_eq!(to_text(&json!(true)), "true");
    assert_eq!(to_text(&json!(null)), "");
}

#[test]
fn a_score_answer_carries_its_level_legend() {
    // `answer_from_probs` returns the criteria text next to the score so a consumer can
    // read a level back without the request; the serialized answer must not drop it.
    let request = QuestionRequest {
        kind: Kind::Score,
        instructions: "How urgent?".into(),
        criteria: Some(json!(["Not urgent", "Soon", "Now"])),
    };
    let (keys, texts) = candidates(&request).unwrap();
    let question = Question {
        id: "urgency".into(),
        kind: Kind::Score,
        keys,
    };
    let answer = answer(&question, &texts, &[0.1, 0.7, 0.2]).unwrap();
    let json = answer_json(&answer);
    assert_eq!(json["type"], "score");
    assert_eq!(
        json["legend"],
        json!({"0": "Not urgent", "1": "Soon", "2": "Now"})
    );
    assert!((json["score"].as_f64().unwrap() - 1.1).abs() < 1e-6);
}
