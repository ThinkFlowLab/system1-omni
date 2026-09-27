use omni_laya::{
    config::AgentConfig,
    decision::decode,
    preprocess::{Batch, Question},
};
use serde_json::{Map, Value, json};
use std::collections::HashMap;

#[test]
fn choice_11_bucket_uses_official_temperature_bounds() {
    // Deterministic decoder regression, not a model-quality fixture. The expected
    // p0 values are exp(1/T) / (exp(1/T) + 10), rounded to four decimal places,
    // using Laya 0.3.20 common.clamp_temperature's effective T.
    let criteria: Map<String, Value> = (0..11)
        .map(|i| (i.to_string(), json!(format!("option {i}"))))
        .collect();
    let batch = Batch {
        questions: vec![Question {
            id: "q".into(),
            kind: "choice".into(),
            criteria: Value::Object(criteria),
            ids: vec![],
            markers: (0..11).collect(),
            qtype: 0,
        }],
        input_ids: vec![],
        lens: vec![],
        qtypes: vec![],
        b: 1,
        l: 16,
        usage: 0,
    };
    let mut logits = vec![0.0; 11];
    logits[0] = 1.0;
    for (fitted, expected) in [(0.100_582_81, 0.4249), (1.0, 0.2137), (9.0, 0.1088)] {
        let config = AgentConfig {
            max_len: 512,
            head_max_len: 192,
            head_layers: 2,
            temperature: vec![1.0; 3],
            temperature_by_options: HashMap::from([("choice:11+".into(), fitted)]),
        };
        let response = decode(&batch, &config, &[logits.clone()], &[[0.0, 0.0]]).unwrap();
        assert_eq!(response["answers"]["q"]["choice"], "0");
        assert_eq!(
            response["answers"]["q"]["probabilities"]["0"],
            json!(expected),
            "fitted temperature {fitted}"
        );
    }
}
