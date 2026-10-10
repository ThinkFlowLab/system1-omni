use omni_open_jev_native::contract::{CHECKPOINTS, Checkpoint, answer, compile};
use serde_json::{Value, json};

static OPEN_JEV_27B: &Checkpoint = &CHECKPOINTS[0];
static OPEN_JEV_9B: &Checkpoint = &CHECKPOINTS[1];

fn close(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            assert!(
                (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-12,
                "{a} != {b}"
            );
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len());
            for (key, value) in b {
                close(&a[key], value);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

#[test]
fn prompts_and_typed_answers_match_open_jev_reference() {
    // Generated with jev.api and jev.metrics @ 3308a15ccd7eea1df7a37d6ddc39b023b801ba16.
    let cases: Value = serde_json::from_str(include_str!("data/contract.json")).unwrap();
    for case in cases.as_array().unwrap() {
        let raw = serde_json::to_vec(&case["request"]).unwrap();
        let questions = compile(&raw, OPEN_JEV_27B).unwrap();
        let prompts: Vec<_> = questions.iter().map(|q| &q.prompts).collect();
        assert_eq!(json!(prompts), case["prompts"]);
        let mut answers = serde_json::Map::new();
        for (q, row) in questions.iter().zip(case["logits"].as_array().unwrap()) {
            let logits: Vec<f32> = row
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            answers.insert(
                q.id.clone(),
                answer(q, &logits, case["temperature"].as_f64().unwrap()).unwrap(),
            );
        }
        close(&json!({"answers": answers}), &case["response"]);
    }
}

#[test]
fn rejects_malformed_and_unsupported_requests() {
    for raw in [
        r#"{"state":"x","state":"y","questions":{}}"#,
        r#"{"state":NaN,"questions":{}}"#,
        r#"{"state":12,"questions":{"q":{"type":"noul","instructions":"x"}}}"#,
        r#"{"state":"x","questions":{}}"#,
        r#"{"model":"other","state":"x","questions":{"q":{"type":"noul","instructions":"x"}}}"#,
        r#"{"state":"x","questions":{"q":{"type":"generation","instructions":"x"}}}"#,
        r#"{"state":"x","questions":{"q":{"type":"choice","instructions":"x","criteria":{}}}}"#,
        r#"{"state":"x","questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":4}}}}"#,
        r#"{"state":"x","questions":{"q":{"type":"score","instructions":"x","criteria":["one"]}}}"#,
        r#"{"state":"x","questions":{"q":{"type":"noul","instructions":"x","criteria":{"true":"yes"}}}}"#,
    ] {
        assert!(
            compile(raw.as_bytes(), OPEN_JEV_27B).is_err(),
            "accepted {raw}"
        );
    }
    let q = compile(
        br#"{"state":"x","questions":{"q":{"type":"noul","instructions":"x"}}}"#,
        OPEN_JEV_27B,
    )
    .unwrap();
    for temperature in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(answer(&q[0], &[0.0, 1.0], temperature).is_err());
    }
    assert!(answer(&q[0], &[1.0], 1.0).is_err());
    assert!(answer(&q[0], &[0.0, f32::NAN], 1.0).is_err());
}

#[test]
fn choice_and_score_limits_match_reference() {
    let mut candidates = serde_json::Map::new();
    for i in 0..255 {
        candidates.insert(i.to_string(), Value::Null);
    }
    let mut body = json!({"state": "x", "questions": {"q": {
        "type": "choice", "instructions": "Pick", "criteria": candidates
    }}});
    let questions = compile(&serde_json::to_vec(&body).unwrap(), OPEN_JEV_27B).unwrap();
    assert_eq!(questions[0].prompts.len(), 255);
    body["questions"]["q"]["criteria"]["extra"] = Value::Null;
    assert!(compile(&serde_json::to_vec(&body).unwrap(), OPEN_JEV_27B).is_err());
    body["questions"]["q"] =
        json!({"type": "score", "instructions": "Rate", "criteria": vec!["level"; 10]});
    assert_eq!(
        compile(&serde_json::to_vec(&body).unwrap(), OPEN_JEV_27B).unwrap()[0]
            .prompts
            .len(),
        10
    );
    body["questions"]["q"]["criteria"] = json!(vec!["level"; 11]);
    assert!(compile(&serde_json::to_vec(&body).unwrap(), OPEN_JEV_27B).is_err());
}

fn export(checkpoint: &Checkpoint) -> Value {
    json!({"format": "open-jev-text-merged/1", "model_id": checkpoint.model_id,
        "base_revision": checkpoint.base_revision,
        "checkpoint_revision": checkpoint.checkpoint_revision})
}

#[test]
fn exports_select_their_pinned_checkpoint() {
    for checkpoint in [OPEN_JEV_27B, OPEN_JEV_9B] {
        let selected = Checkpoint::from_export(&export(checkpoint)).unwrap();
        assert_eq!(selected.model_id, checkpoint.model_id);
    }
    let mut manifest = export(OPEN_JEV_9B);
    manifest["checkpoint_revision"] = json!(OPEN_JEV_27B.checkpoint_revision);
    assert!(Checkpoint::from_export(&manifest).is_err());
    let mut manifest = export(OPEN_JEV_9B);
    manifest["base_revision"] = json!(OPEN_JEV_27B.base_revision);
    assert!(Checkpoint::from_export(&manifest).is_err());
    let mut manifest = export(OPEN_JEV_27B);
    manifest["model_id"] = json!("Qwen/Qwen3.5-2B");
    assert!(Checkpoint::from_export(&manifest).is_err());
    let mut manifest = export(OPEN_JEV_27B);
    manifest["format"] = json!("open-jev-text-merged/2");
    assert!(Checkpoint::from_export(&manifest).is_err());
}

#[test]
fn only_validated_checkpoints_pack_candidates() {
    assert!(OPEN_JEV_27B.pack_candidates);
    assert!(!OPEN_JEV_9B.pack_candidates);
}

#[test]
fn request_models_follow_the_loaded_checkpoint() {
    let request = |model: &str| {
        serde_json::to_vec(&json!({"model": model, "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "x"}}}))
        .unwrap()
    };
    for (checkpoint, other) in [(OPEN_JEV_27B, OPEN_JEV_9B), (OPEN_JEV_9B, OPEN_JEV_27B)] {
        for model in [
            checkpoint.model_id,
            checkpoint.alias,
            "open-jev",
            "jev-latest",
        ] {
            assert!(
                compile(&request(model), checkpoint).is_ok(),
                "rejected {model}"
            );
        }
        for model in [other.model_id, other.alias] {
            assert!(
                compile(&request(model), checkpoint).is_err(),
                "accepted {model}"
            );
        }
    }
}
