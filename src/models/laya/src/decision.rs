//! Decode raw model logits using Laya's fitted temperatures and typed response schema.
use crate::{config::AgentConfig, preprocess::Batch};
use anyhow::{Result, ensure};
use serde_json::{Map, Value, json};
fn round4(v: f64) -> f64 {
    (v * 10000.0).round_ties_even() / 10000.0
}
fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<_> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    for v in &mut p {
        *v /= sum;
    }
    p
}

pub fn decode(
    batch: &Batch,
    cfg: &AgentConfig,
    logits: &[Vec<f32>],
    action_logits: &[[f32; 2]],
) -> Result<Value> {
    ensure!(
        logits.len() == batch.questions.len() && action_logits.len() == logits.len(),
        "output row count mismatch"
    );
    let mut answers = Map::new();
    for ((q, raw), action) in batch.questions.iter().zip(logits).zip(action_logits) {
        let k = q.markers.len();
        ensure!(
            raw.len() == k && raw.iter().chain(action.iter()).all(|x| x.is_finite()),
            "invalid model logits"
        );
        let bucket = if k <= 2 {
            "2"
        } else if k <= 5 {
            "3-5"
        } else if k <= 10 {
            "6-10"
        } else {
            "11+"
        };
        // Laya 0.3.20 Agent clamps config temperatures at load via common.py's
        // clamp_temperature ([0.5, 5.0]), including the shipped choice:11+ ~0.1006.
        let temp = cfg
            .temperature_by_options
            .get(&format!("{}:{bucket}", q.kind))
            .copied()
            .unwrap_or(cfg.temperature[q.qtype as usize])
            .clamp(0.5, 5.0);
        let p = softmax(&raw.iter().map(|x| x / temp).collect::<Vec<_>>());
        // argmax keeps the first option on a tie, matching NumPy.
        let mut winner = 0;
        for i in 1..k {
            if p[i] > p[winner] {
                winner = i;
            }
        }
        let confidence = if k < 2 {
            1.0
        } else {
            let ent: f32 = p.iter().map(|x| -x * x.clamp(1e-12, 1.0).ln()).sum();
            (1.0 - f64::from(ent) / (k as f64).ln()).clamp(0.0, 1.0)
        };
        let act = round4(f64::from(softmax(action)[0]));
        let mut answer = json!({
            "type": q.kind,
            "confidence": round4(confidence),
            "answer_confidence": round4(f64::from(p[winner])),
            "action": {"act_probability": act}
        });
        if q.kind == "noul" {
            answer["noul"] = json!(round4(f64::from(p[1])));
            answer["confidence"] = json!(round4(f64::from(p[1]).max(1.0 - f64::from(p[1]))));
        } else {
            let keys: Vec<String> = if q.kind == "choice" {
                q.criteria.as_object().unwrap().keys().cloned().collect()
            } else {
                (0..k).map(|i| i.to_string()).collect()
            };
            let probs: Map<String, Value> = keys
                .iter()
                .zip(&p)
                .map(|(k, v)| (k.clone(), json!(round4(f64::from(*v)))))
                .collect();
            answer["probabilities"] = Value::Object(probs);
            if q.kind == "choice" {
                answer["choice"] = json!(keys[winner]);
            } else {
                answer["score"] = json!(round4(
                    p.iter()
                        .enumerate()
                        .map(|(i, p)| i as f64 * f64::from(*p))
                        .sum()
                ));
                answer["legend"] = Value::Object(
                    q.criteria
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i.to_string(), v.clone()))
                        .collect(),
                );
            }
        }
        answers.insert(q.id.clone(), answer);
    }
    Ok(json!({
        "model": "laya-rl-agent",
        "answers": answers,
        "usage": {"input_tokens": batch.usage, "output_tokens": 0},
        "routing": {
            "model": "english",
            "repo": "convaiinnovations/laya",
            "reason": "explicit model='english'",
            "detection": null,
            "workflow": null
        }
    }))
}
