//! Decode Laya's raw option and action logits without a GPU or HTTP response wrapper.
use crate::{config::AgentConfig, preprocess::Prepared};
use anyhow::{Result, bail, ensure};
use serde_json::{Map, Value, json};

fn round4(value: f64) -> f64 {
    // Match Python round(value, 4), without rounding an intermediate value * 10000.
    format!("{value:.4}").parse().unwrap()
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities: Vec<_> = values.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = probabilities.iter().sum();
    for probability in &mut probabilities {
        *probability /= sum;
    }
    probabilities
}

/// Rows follow `prepared.questions`; each option row follows its marker order.
/// Action rows contain two raw logits, with the first meaning "act".
pub fn decode(
    prepared: &Prepared,
    config: &AgentConfig,
    logits: &[Vec<f32>],
    action_logits: &[[f32; 2]],
) -> Result<Map<String, Value>> {
    ensure!(
        logits.len() == prepared.questions.len() && action_logits.len() == logits.len(),
        "output row count mismatch"
    );
    ensure!(
        config.temperature.len() == 3
            && config
                .temperature
                .iter()
                .chain(config.temperature_by_options.values())
                .all(|t| t.is_finite() && *t > 0.0),
        "invalid temperatures"
    );
    let mut answers = Map::new();
    for ((question, raw), action) in prepared.questions.iter().zip(logits).zip(action_logits) {
        let k = question.markers.len();
        ensure!(
            k > 0 && raw.len() == k && raw.iter().chain(action).all(|v| v.is_finite()),
            "question {:?}: invalid model logits",
            question.id
        );
        let (qtype, keys): (_, Vec<String>) = match question.kind.as_str() {
            "choice" => {
                let criteria = question.criteria.as_object();
                ensure!(
                    criteria.is_some_and(|c| c.len() == k),
                    "invalid choice criteria"
                );
                (0, criteria.unwrap().keys().cloned().collect())
            }
            "score" => {
                ensure!(
                    question.criteria.as_array().is_some_and(|c| c.len() == k),
                    "invalid score criteria"
                );
                (1, (0..k).map(|i| i.to_string()).collect())
            }
            "noul" => {
                ensure!(k == 2, "noul requires two options");
                (2, Vec::new())
            }
            _ => bail!("unknown question type {:?}", question.kind),
        };
        ensure!(question.qtype == qtype as i64, "question type ID mismatch");
        ensure!(!answers.contains_key(&question.id), "duplicate question ID");
        let bucket = match k {
            1..=2 => "2",
            3..=5 => "3-5",
            6..=10 => "6-10",
            _ => "11+",
        };
        let temperature = config
            .temperature_by_options
            .get(&format!("{}:{bucket}", question.kind))
            .copied()
            .unwrap_or(config.temperature[qtype])
            .clamp(0.5, 5.0);
        let scaled: Vec<_> = raw.iter().map(|v| v / temperature).collect();
        ensure!(
            scaled.iter().all(|v| v.is_finite()),
            "scaled logits overflow"
        );
        let probabilities = softmax(&scaled);
        // Strict comparison keeps the first option on a tie, as NumPy does.
        let mut winner = 0;
        for i in 1..k {
            if probabilities[i] > probabilities[winner] {
                winner = i;
            }
        }
        let confidence = if question.kind == "noul" {
            f64::from(probabilities[1]).max(1.0 - f64::from(probabilities[1]))
        } else if k == 1 {
            1.0
        } else {
            let entropy: f32 = probabilities
                .iter()
                .map(|p| -p * p.clamp(1e-12, 1.0).ln())
                .sum();
            f64::from((1.0 - entropy / (k as f64).ln() as f32).clamp(0.0, 1.0))
        };
        let mut answer = json!({
            "type": question.kind,
            "confidence": round4(confidence),
            "answer_confidence": round4(f64::from(probabilities[winner])),
            "action": {"act_probability": round4(f64::from(softmax(action)[0]))}
        });
        if question.kind == "noul" {
            answer["noul"] = json!(round4(f64::from(probabilities[1])));
        } else {
            answer["probabilities"] = Value::Object(
                keys.iter()
                    .zip(&probabilities)
                    .map(|(key, probability)| (key.clone(), json!(round4(f64::from(*probability)))))
                    .collect(),
            );
            if question.kind == "choice" {
                answer["choice"] = json!(keys[winner]);
            } else {
                answer["score"] = json!(round4(
                    probabilities
                        .iter()
                        .enumerate()
                        .map(|(i, p)| i as f64 * f64::from(*p))
                        .sum()
                ));
                answer["legend"] = Value::Object(
                    question
                        .criteria
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, value)| (i.to_string(), value.clone()))
                        .collect(),
                );
            }
        }
        answers.insert(question.id.clone(), answer);
    }
    Ok(answers)
}

#[cfg(test)]
mod tests {
    use super::round4;

    #[test]
    fn rounding_matches_python_at_decimal_boundaries() {
        for (value, expected) in [(0.00035, 0.0003), (0.12345, 0.1235), (-0.00035, -0.0003)] {
            assert_eq!(round4(value), expected);
        }
    }
}
