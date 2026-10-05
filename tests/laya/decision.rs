use omni_laya::{
    config::AgentConfig,
    decision::decode,
    preprocess::{Prepared, Question},
};
use serde_json::{Value, json};
use std::collections::HashMap;

fn config(value: &Value) -> AgentConfig {
    AgentConfig {
        max_len: 512,
        head_max_len: 192,
        head_layers: 2,
        temperature: serde_json::from_value(value["temperature"].clone()).unwrap(),
        temperature_by_options: serde_json::from_value(value["temperature_by_options"].clone())
            .unwrap(),
    }
}

fn prepared(questions: &Value) -> Prepared {
    Prepared {
        questions: questions
            .as_array()
            .unwrap()
            .iter()
            .map(|q| {
                let kind = q["kind"].as_str().unwrap();
                let (qtype, count) = match kind {
                    "choice" => (0, q["criteria"].as_object().unwrap().len()),
                    "score" => (1, q["criteria"].as_array().unwrap().len()),
                    "noul" => (2, 2),
                    _ => panic!("unknown fixture type"),
                };
                Question {
                    id: q["id"].as_str().unwrap().to_owned(),
                    kind: kind.to_owned(),
                    criteria: q["criteria"].clone(),
                    ids: Vec::new(),
                    markers: (0..count).collect(),
                    qtype,
                }
            })
            .collect(),
        usage: 0,
    }
}

fn valid() -> (Prepared, AgentConfig) {
    (
        prepared(&json!([{"id":"q", "kind":"choice", "criteria":{"z":null,"a":null}}])),
        AgentConfig {
            max_len: 512,
            head_max_len: 192,
            head_layers: 2,
            temperature: vec![1.0; 3],
            temperature_by_options: HashMap::new(),
        },
    )
}

#[test]
fn matches_official_decode_reference() {
    let fixture: Value = serde_json::from_str(include_str!("data/decisions.json")).unwrap();
    assert_eq!(fixture["reference"]["laya"], "0.3.20");
    for case in fixture["cases"].as_array().unwrap() {
        let input = prepared(&case["questions"]);
        let cfg = config(case.get("config").unwrap_or(&fixture["config"]));
        let logits: Vec<Vec<f32>> = serde_json::from_value(case["logits"].clone()).unwrap();
        let actions: Vec<[f32; 2]> = serde_json::from_value(case["action_logits"].clone()).unwrap();
        let answers = decode(&input, &cfg, &logits, &actions).unwrap();
        assert_eq!(
            Value::Object(answers.clone()),
            case["answers"],
            "{}",
            case["name"]
        );
        assert_eq!(
            answers.keys().collect::<Vec<_>>(),
            input.questions.iter().map(|q| &q.id).collect::<Vec<_>>()
        );
        for question in &input.questions {
            if question.kind == "choice" {
                assert_eq!(
                    answers[&question.id]["probabilities"]
                        .as_object()
                        .unwrap()
                        .keys()
                        .collect::<Vec<_>>(),
                    question
                        .criteria
                        .as_object()
                        .unwrap()
                        .keys()
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn fp32_reduction_boundary_preserves_answer() {
    let fixture: Value = serde_json::from_str(include_str!("data/decisions.json")).unwrap();
    let case = &fixture["rounding_probe"];
    let input = prepared(&case["questions"]);
    let cfg = config(&case["config"]);
    let logits: Vec<Vec<f32>> = serde_json::from_value(case["logits"].clone()).unwrap();
    let actions: Vec<[f32; 2]> = serde_json::from_value(case["action_logits"].clone()).unwrap();
    let mut actual = Value::Object(decode(&input, &cfg, &logits, &actions).unwrap());
    let expected = &case["answers"];
    let probabilities = actual["q"]["probabilities"].as_object_mut().unwrap();
    let reference = expected["q"]["probabilities"].as_object().unwrap();
    assert_eq!(
        probabilities.keys().collect::<Vec<_>>(),
        reference.keys().collect::<Vec<_>>()
    );
    for (key, value) in probabilities {
        let error = (value.as_f64().unwrap() - reference[key].as_f64().unwrap()).abs();
        assert!(error <= 0.0001 + 1e-12, "option {key}: {error}");
        // Only this probe's probabilities allow one displayed decimal unit.
        *value = reference[key].clone();
    }
    assert_eq!(&actual, expected);
}

#[test]
fn rejects_bad_row_counts_and_nonfinite_outputs() {
    let (input, cfg) = valid();
    assert!(decode(&input, &cfg, &[], &[]).is_err());
    assert!(decode(&input, &cfg, &[vec![0.0, 1.0]], &[]).is_err());
    for logits in [
        vec![],
        vec![1.0],
        vec![0.0, f32::NAN],
        vec![0.0, f32::INFINITY],
    ] {
        assert!(decode(&input, &cfg, &[logits], &[[0.0, 0.0]]).is_err());
    }
    assert!(decode(&input, &cfg, &[vec![0.0, 1.0]], &[[f32::NEG_INFINITY, 0.0]]).is_err());
}

#[test]
fn rejects_inconsistent_question_metadata() {
    for mutation in 0..7 {
        let (mut input, cfg) = valid();
        let q = &mut input.questions[0];
        match mutation {
            0 => q.qtype = -1,
            1 => q.kind = "unknown".into(),
            2 => q.criteria = json!(["z", "a"]),
            3 => q.criteria = json!({"z":null}),
            4 => {
                q.kind = "score".into();
                q.qtype = 1;
            }
            5 => {
                q.kind = "noul".into();
                q.qtype = 2;
                q.markers.push(2);
            }
            _ => q.markers.clear(),
        }
        let raw = vec![0.0; q.markers.len()];
        assert!(
            decode(&input, &cfg, &[raw], &[[0.0, 0.0]]).is_err(),
            "mutation {mutation}"
        );
    }
    let (mut input, cfg) = valid();
    input.questions.extend(valid().0.questions);
    assert!(
        decode(
            &input,
            &cfg,
            &[vec![0.0, 1.0], vec![0.0, 1.0]],
            &[[0.0, 0.0]; 2]
        )
        .is_err()
    );
}

#[test]
fn rejects_invalid_temperatures_and_scaling_overflow() {
    for temperatures in [
        vec![],
        vec![1.0; 2],
        vec![1.0; 4],
        vec![0.0; 3],
        vec![f32::NAN; 3],
    ] {
        let (input, mut cfg) = valid();
        cfg.temperature = temperatures;
        assert!(decode(&input, &cfg, &[vec![0.0, 1.0]], &[[0.0, 0.0]]).is_err());
    }
    let (input, mut cfg) = valid();
    cfg.temperature_by_options
        .insert("choice:2".into(), f32::INFINITY);
    assert!(decode(&input, &cfg, &[vec![0.0, 1.0]], &[[0.0, 0.0]]).is_err());
    cfg.temperature_by_options.insert("choice:2".into(), 0.5);
    assert!(
        decode(&input, &cfg, &[vec![f32::MAX, 0.0]], &[[0.0, 0.0]])
            .unwrap_err()
            .to_string()
            .contains("overflow")
    );
    assert!(decode(&input, &cfg, &[vec![0.0, 1.0]], &[[0.0, 0.0]]).is_ok());
}
