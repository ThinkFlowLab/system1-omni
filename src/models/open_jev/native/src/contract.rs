//! Open-Jev's typed request and response contract, without generated text.
//! Adapted from Open-Jev @ 3308a15; see ../LICENSE.open-jev.

use anyhow::{Context, Result, bail, ensure};
use omni_qwen3_5_native::json;
use serde_json::{Map, Value, json};

/// A pinned Open-Jev checkpoint, selected by its export's `model_id`.
pub struct Checkpoint {
    pub name: &'static str,
    /// The base model, which Open-Jev also returns as the response model.
    pub model_id: &'static str,
    pub base_revision: &'static str,
    pub checkpoint_revision: &'static str,
    /// Accepted as the request model, besides `model_id`, `open-jev` and `jev-latest`.
    pub alias: &'static str,
    /// Hidden, intermediate, layers, attention, KV, linear key and linear value heads.
    pub backbone: (usize, usize, usize, usize, usize, usize, usize),
    /// Pack a request's candidates for shared GEMMs. Packing was validated against
    /// per-candidate execution on 27B only, so 9B runs candidates one at a time.
    pub pack_candidates: bool,
}

pub static CHECKPOINTS: [Checkpoint; 2] = [
    Checkpoint {
        name: "Open-Jev-27B-v1.1",
        model_id: "Qwen/Qwen3.8-27B",
        base_revision: "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        checkpoint_revision: "28cf73067d5b337860bbef3c85b8b82ba8730956",
        alias: "open-jev-27b-v1.1",
        backbone: (5120, 17408, 64, 24, 4, 16, 48),
        pack_candidates: true,
    },
    Checkpoint {
        name: "Open-Jev-9B",
        model_id: "Qwen/Qwen3.5-9B",
        base_revision: "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        checkpoint_revision: "47e966881e489511c0c7f5633a9e1960a676a551",
        alias: "open-jev-9b",
        backbone: (4096, 12288, 32, 16, 4, 16, 32),
        pack_candidates: false,
    },
];

impl Checkpoint {
    /// The pinned checkpoint that produced a merged export manifest.
    pub fn from_export(manifest: &Value) -> Result<&'static Self> {
        ensure!(
            manifest["format"] == "open-jev-text-merged/1",
            "expected an Open-Jev merged text export"
        );
        let checkpoint = CHECKPOINTS
            .iter()
            .find(|c| manifest["model_id"] == c.model_id)
            .context("expected an Open-Jev-27B-v1.1 or Open-Jev-9B export")?;
        ensure!(
            manifest["base_revision"] == checkpoint.base_revision
                && manifest["checkpoint_revision"] == checkpoint.checkpoint_revision,
            "expected a pinned {} export",
            checkpoint.name
        );
        Ok(checkpoint)
    }

    fn serves(&self, model: &str) -> bool {
        [self.model_id, "open-jev", "jev-latest", self.alias].contains(&model)
    }
}

#[derive(Debug, PartialEq)]
pub enum Kind {
    Choice,
    Score,
    Noul,
}

pub struct Question {
    pub id: String,
    pub kind: Kind,
    pub keys: Vec<String>,
    pub prompts: Vec<String>,
    pub legend: Option<Value>,
}

/// Python json.dumps(ensure_ascii=False, sort_keys=True), or the string itself.
fn render(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_owned();
    }
    let mut sorted = value.clone();
    sorted.sort_all_objects();
    json::dumps(&sorted)
}

fn description(value: &Value) -> Result<String> {
    ensure!(
        value.is_string() || value.is_object() || value.is_array(),
        "instructions and descriptions must be text, an object, or an array"
    );
    Ok(render(value))
}

pub fn compile(raw: &[u8], checkpoint: &Checkpoint) -> Result<Vec<Question>> {
    let request = json::parse(raw).map_err(anyhow::Error::msg)?;
    ensure!(
        matches!(request.get("model"), None | Some(Value::Null))
            || request["model"]
                .as_str()
                .is_some_and(|model| checkpoint.serves(model)),
        "requested model is not loaded"
    );
    let state = request.get("state").context("request requires state")?;
    ensure!(
        state.is_string() || state.is_object() || state.is_array(),
        "state must be text, a JSON object, or an array"
    );
    let questions = request
        .get("questions")
        .and_then(Value::as_object)
        .context("questions must be an object")?;
    ensure!(
        !questions.is_empty() && questions.len() <= 4096,
        "questions must be nonempty and within the server question limit"
    );
    let mut result = Vec::with_capacity(questions.len());
    let mut candidates = 0;
    for (id, definition) in questions {
        let d = definition
            .as_object()
            .context("question definition must be an object")?;
        let mut instructions = description(d.get("instructions").unwrap_or(&Value::Null))?;
        let criteria = d.get("criteria").unwrap_or(&Value::Null);
        let (kind, keys, options, legend) = match d.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let criteria = criteria.as_object().context("Choice requires an object")?;
                ensure!(
                    (1..=255).contains(&criteria.len()),
                    "Choice requires between 1 and 255 candidates"
                );
                let mut options = Vec::with_capacity(criteria.len());
                for (name, value) in criteria {
                    options.push(if value.is_null() {
                        name.clone()
                    } else {
                        format!("{name}: {}", description(value)?)
                    });
                }
                (
                    Kind::Choice,
                    criteria.keys().cloned().collect::<Vec<_>>(),
                    options,
                    None,
                )
            }
            Some("score") => {
                let criteria = criteria.as_array().context("Score requires an array")?;
                ensure!(
                    (2..=10).contains(&criteria.len()),
                    "Score requires 2 to 10 descriptive levels"
                );
                let keys: Vec<String> = (0..criteria.len()).map(|i| i.to_string()).collect();
                let legend: Map<String, Value> =
                    keys.iter().cloned().zip(criteria.iter().cloned()).collect();
                (
                    Kind::Score,
                    keys,
                    criteria
                        .iter()
                        .map(description)
                        .collect::<Result<Vec<_>>>()?,
                    Some(Value::Object(legend)),
                )
            }
            Some("noul") => {
                if !criteria.is_null() {
                    let criteria = criteria
                        .as_object()
                        .context("Noul criteria must contain true and false descriptions")?;
                    ensure!(
                        criteria.len() == 2
                            && criteria.contains_key("true")
                            && criteria.contains_key("false"),
                        "Noul criteria must contain true and false descriptions"
                    );
                    instructions += &format!(
                        "\nYes means: {}\nNo means: {}",
                        description(&criteria["true"])?,
                        description(&criteria["false"])?
                    );
                }
                (
                    Kind::Noul,
                    vec!["false".into(), "true".into()],
                    Vec::new(),
                    None,
                )
            }
            _ => bail!("question type must be choice, score, or noul"),
        };
        let prefix = format!("Context:\n{}\n\nQuestion: {instructions}\n", render(state));
        let prompts = if kind == Kind::Noul {
            vec![prefix + "Is the answer to this question yes? Answer Yes or No."]
        } else {
            options
                .into_iter()
                .map(|option| {
                    format!(
                        "{prefix}Proposed answer: {option}\nIs this proposed answer correct? Answer Yes or No."
                    )
                })
                .collect()
        };
        candidates += prompts.len();
        ensure!(candidates <= 65536, "request exceeds the candidate limit");
        result.push(Question {
            id: id.clone(),
            kind,
            keys,
            prompts,
            legend,
        });
    }
    Ok(result)
}

/// Calibrated normalization is across the complete question's candidates.
pub fn answer(question: &Question, logits: &[f32], temperature: f64) -> Result<Value> {
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "invalid temperature"
    );
    ensure!(
        logits.len() == question.keys.len() && logits.iter().all(|v| v.is_finite()),
        "invalid logits"
    );
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut probs: Vec<f64> = logits
        .iter()
        .map(|&v| ((v as f64 - max) / temperature).exp())
        .collect();
    let total: f64 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= total);
    if question.kind == Kind::Noul {
        return Ok(json!({"type": "noul", "noul": probs[1]}));
    }
    // First maximum wins ties, like Python's max(..., key=...).
    let mode = (1..probs.len()).fold(0, |m, i| if probs[i] > probs[m] { i } else { m });
    let probabilities: Map<String, Value> = question
        .keys
        .iter()
        .cloned()
        .zip(probs.iter().map(|p| json!(p)))
        .collect();
    if question.kind == Kind::Choice {
        let n = probs.len() as f64;
        let confidence = if probs.len() == 1 {
            1.0
        } else {
            (probs[mode] - 1.0 / n) / (1.0 - 1.0 / n)
        };
        Ok(
            json!({"type": "choice", "choice": question.keys[mode], "probabilities": probabilities, "confidence": confidence}),
        )
    } else {
        let score: f64 = probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum();
        let distance: f64 = probs
            .iter()
            .enumerate()
            .map(|(i, p)| i.abs_diff(mode) as f64 * p)
            .sum();
        let center = (probs.len() - 1) as f64 / 2.0;
        let uniform_distance = (0..probs.len())
            .map(|i| (i as f64 - center).abs())
            .sum::<f64>()
            / probs.len() as f64;
        Ok(
            json!({"type": "score", "score": score, "probabilities": probabilities,
            "confidence": (1.0 - distance / uniform_distance).max(0.0), "legend": question.legend}),
        )
    }
}
