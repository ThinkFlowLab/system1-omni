//! GPU checks of the native execution on an export. They need a GPU, OMNIJEV_EXPORT
//! pointing at an export from recipe/omnijev/export.py and OMNIJEV_CUDA_LIB at
//! libqwen3_5_cuda.so, both absolute:
//!
//!     OMNIJEV_EXPORT=<export> OMNIJEV_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
//!       cargo test --release -p omni-omnijev-native --test worker -- --ignored --test-threads=1
mod common;

use std::path::PathBuf;

use omni_omnijev_native::executor::{self, Executor};
use serde_json::{Value, json};

fn executor() -> (Executor, omni_omnijev_native::processing::Processor) {
    let var =
        |name: &str| PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("set {name}")));
    Executor::load(&var("OMNIJEV_EXPORT"), &var("OMNIJEV_CUDA_LIB")).unwrap()
}

fn request(questions: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "model": "tinnel123/OmniJev",
        "state": {"images": [common::png_url(320, 200, [200, 40, 40])]},
        "questions": questions,
    }))
    .unwrap()
}

/// The answers without the timing fields.
fn decisions(response: &Value) -> Value {
    let mut answers = response["answers"].clone();
    for answer in answers.as_object_mut().unwrap().values_mut() {
        let answer = answer.as_object_mut().unwrap();
        answer.remove("latency_s");
        answer.remove("latency_total_s");
    }
    answers
}

#[test]
#[ignore = "needs a GPU, OMNIJEV_EXPORT and OMNIJEV_CUDA_LIB"]
fn answers_are_repeatable_and_independent_of_question_order() {
    let (mut executor, processor) = executor();
    let questions = json!({
        "red": {"type": "noul", "instructions": "The image is mostly red."},
        "where": {"type": "noul", "instructions": "There is something in this region.",
                  "region": {"box": [0, 0, 160, 100]}},
        "colour": {"type": "choice", "instructions": "Which colour dominates the image?",
                   "options": [{"key": "red", "text": "red"}, {"key": "green", "text": "green"},
                               {"key": "blue", "text": "blue"}, {"text": "none", "abstain": true}]},
        "warm": {"type": "score", "instructions": "How warm is the colour?",
                 "levels": ["cold", "neutral", "warm", "hot"]},
    });
    let run = |executor: &mut Executor, questions: &Value| {
        let prepared = processor.prepare(&request(questions.clone())).unwrap();
        let answers = executor.execute(&prepared).unwrap();
        executor::response(&prepared, answers)
    };
    let first = run(&mut executor, &questions);
    assert_eq!(first["model"], "tinnel123/OmniJev");
    assert!(first["usage"]["input_tokens"].as_u64().unwrap() > 0);
    let answers = first["answers"].as_object().unwrap();
    assert_eq!(
        answers.keys().collect::<Vec<_>>(),
        ["red", "where", "colour", "warm"]
    );
    for (id, field) in [
        ("red", "noul"),
        ("where", "noul"),
        ("colour", "choice"),
        ("warm", "score"),
    ] {
        assert!(answers[id].get(field).is_some(), "{id}: {}", answers[id]);
        assert!(answers[id]["latency_total_s"].is_number(), "{id}");
    }
    assert_eq!(
        decisions(&run(&mut executor, &questions)),
        decisions(&first)
    );
    // each row is its own pass, so the order of the questions changes nothing
    let reversed: serde_json::Map<String, Value> = questions
        .as_object()
        .unwrap()
        .iter()
        .rev()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut back = decisions(&run(&mut executor, &Value::Object(reversed)));
    let mut want = decisions(&first);
    back.as_object_mut().unwrap().sort_keys();
    want.as_object_mut().unwrap().sort_keys();
    assert_eq!(back, want);
}
