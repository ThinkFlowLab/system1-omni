//! openjev/openjev's typed request and response contract, after the pinned helper.
//!
//! Follows `helper/shim.py` at openjev/openjev@ac97900 (sha256 81a22f1b...) under the serving guide's settings
//! (`serve/SERVE.md`): `READOUT_T=0.85`, `READOUT_NOUL_T=1.829074`, `READOUT_NOUL_BIAS=0`, `READOUT_TARGETED=1`,
//! `READOUT_INSTR_STYLE=pyrepr`, every other knob at its default. Text lane only: one prompt per question,
//! at most [`MAX_OPTIONS`] options, a state that is a string or an object without a screenshot.

use anyhow::{Result, bail};
use omni_qwen3_5_native::json;
use serde_json::{Map, Value, json};

use crate::pyrepr;

pub const MODEL_ID: &str = "openjev/openjev";
pub const REVISION: &str = "ac97900fd034fdd7e7e536f3d4c21b836cae0750";
/// Temperature for choice and score: each candidate's score is divided by it before the softmax.
pub const TEMPERATURE: f64 = 0.85;
/// Yes/no calibration: `sigmoid(logit(p_yes) / NOUL_T + NOUL_BIAS)`.
pub const NOUL_T: f64 = 1.829074;
pub const NOUL_BIAS: f64 = 0.0;
/// One letter per option; the helper's multi-pass handling of more options is a later milestone.
pub const MAX_OPTIONS: usize = 52;
/// The model card's prompt limit.
pub const MAX_PROMPT_TOKENS: usize = 16384;
/// `A`..`Z` then `a`..`z`, the helper's `LETTERS`.
pub fn letter(i: usize) -> char {
    (if i < 26 {
        b'A' + i as u8
    } else {
        b'a' + (i - 26) as u8
    }) as char
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Choice,
    Score,
    Noul,
}

/// One question compiled into the user content the model reads, with its options in request order.
#[derive(Debug)]
pub struct Question {
    pub id: String,
    pub kind: Kind,
    /// Option keys: the choice keys, the score levels `"0"`, `"1"`, ... or `yes`, `no`.
    pub keys: Vec<String>,
    /// The score levels as sent, for the response legend.
    pub levels: Vec<Value>,
    /// The chat user content; the chat template wraps it (see `processing`).
    pub content: String,
}

/// A request that compiles; the error message is the 422 text, matching the helper's where it has one.
pub fn compile(raw: &[u8]) -> Result<Vec<Question>> {
    let body = json::parse(raw).map_err(anyhow::Error::msg)?;
    let state = match body.get("state") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(map)) => state_text(map)?,
        Some(Value::Null) | None => bail!("state and a non-empty questions map are required"),
        Some(_) => bail!("state must be a string or an object"),
    };
    let questions = match body.get("questions") {
        Some(Value::Object(qs)) if !qs.is_empty() => qs,
        _ => bail!("state and a non-empty questions map are required"),
    };
    for (id, q) in questions {
        let kind = q.get("type").and_then(Value::as_str);
        if !matches!(kind, Some("choice" | "score" | "noul")) {
            bail!("questions.{id}.type must be one of choice, score, noul");
        }
    }
    questions
        .iter()
        .map(|(id, q)| question(id, q, &state))
        .collect()
}

/// `with_image`: an object state is its `json.dumps(ensure_ascii=False)`. A screenshot (`screenshot` or `image`
/// holding a data URL or more than 2000 characters) would switch the helper to its image lane: not served here.
fn state_text(map: &Map<String, Value>) -> Result<String> {
    for key in ["screenshot", "image"] {
        if let Some(Value::String(v)) = map.get(key)
            && (v.starts_with("data:image") || v.chars().count() > 2000)
        {
            bail!("state.{key} carries an image; this worker reads text only");
        }
    }
    Ok(json::dumps(&Value::Object(map.clone())))
}

fn question(id: &str, q: &Value, state: &str) -> Result<Question> {
    let kind = match q["type"].as_str() {
        Some("choice") => Kind::Choice,
        Some("score") => Kind::Score,
        _ => Kind::Noul,
    };
    let mut instructions = match q.get("instructions") {
        None | Some(Value::Null) => bail!("instructions is required"),
        Some(Value::String(s)) => s.clone(),
        Some(other) => pyrepr::repr(other),
    };
    let (keys, descriptions, levels) = match kind {
        Kind::Choice => match q.get("criteria") {
            Some(Value::Object(c)) if !c.is_empty() => (
                c.keys().cloned().collect(),
                c.values().map(description).collect(),
                vec![],
            ),
            _ => bail!("choice.criteria must be a non-empty map of option -> description|null"),
        },
        Kind::Score => match q.get("criteria") {
            Some(Value::Array(levels)) if levels.len() >= 2 => {
                instructions.push_str(" Rate along the ordered levels below (lowest first).");
                let keys = (0..levels.len()).map(|i| i.to_string()).collect();
                (
                    keys,
                    levels.iter().map(description).collect(),
                    levels.clone(),
                )
            }
            _ => bail!("score.criteria must be an ordered array of at least two levels"),
        },
        Kind::Noul => {
            let empty = Map::new();
            let criteria = match q.get("criteria") {
                None | Some(Value::Null) => &empty,
                Some(Value::Object(c)) => c,
                // The helper drops the connection on a non-object here (`crit.get`); a 422 is clearer.
                Some(_) => bail!(
                    "noul.criteria must be an object with optional true and false descriptions"
                ),
            };
            let side = |key: &str, default: &str| {
                let d = criteria.get(key).map(description).unwrap_or_default();
                if d.is_empty() { default.to_owned() } else { d }
            };
            let yes = side("true", "The statement is true.");
            let no = side("false", "The statement is false.");
            (
                vec!["yes".to_owned(), "no".to_owned()],
                vec![yes, no],
                vec![],
            )
        }
    };
    if keys.len() > MAX_OPTIONS {
        bail!(
            "questions.{id} has {} options; this worker reads at most {MAX_OPTIONS} (one letter each)",
            keys.len()
        );
    }
    let lines: Vec<String> = keys
        .iter()
        .zip(&descriptions)
        .enumerate()
        .map(|(i, (k, d))| format!("[{}] {k}: {d}", letter(i)))
        .collect();
    let content = format!(
        "State:\n{state}\n\nQuestion: {instructions}\nOptions:\n{}\n\nAnswer with the letter of the best option only.",
        lines.join("\n")
    );
    Ok(Question {
        id: id.to_owned(),
        kind,
        keys,
        levels,
        content,
    })
}

/// `_desc`: null is empty, a string is itself, anything else its `json.dumps(ensure_ascii=False)`.
fn description(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        other => json::dumps(other),
    }
}

/// The typed answer from the candidates' scores (log-probabilities or logits: a shift common to all candidates
/// does not change the result), in option order. Probabilities and confidences are rounded to 4 decimals; the
/// choice is picked before rounding, the first maximum winning.
pub fn answer(q: &Question, scores: &[f64]) -> Result<Value> {
    // The helper fails the request rather than flooring a missing score (READOUT_TARGETED=1).
    let bad = q.keys.len().saturating_sub(scores.len())
        + scores.iter().filter(|s| !s.is_finite()).count();
    if scores.len() > q.keys.len() || bad > 0 {
        bail!(
            "{bad} of {} candidate scores missing or not finite",
            q.keys.len()
        );
    }
    let p = softmax(scores, TEMPERATURE);
    let probabilities = |keys: &[String]| -> Map<String, Value> {
        keys.iter()
            .zip(&p)
            .map(|(k, pi)| (k.clone(), json!(round4(*pi))))
            .collect()
    };
    Ok(match q.kind {
        Kind::Choice => {
            let best = argmax(&p);
            json!({"type": "choice", "choice": q.keys[best], "probabilities": probabilities(&q.keys),
                   "confidence": round4(choice_confidence(&p))})
        }
        Kind::Score => {
            let expected: f64 = p.iter().enumerate().map(|(i, pi)| i as f64 * pi).sum();
            let legend: Map<String, Value> = q
                .levels
                .iter()
                .enumerate()
                .map(|(i, l)| (i.to_string(), l.clone()))
                .collect();
            json!({"type": "score", "score": round4(expected), "legend": legend,
                   "probabilities": probabilities(&q.keys), "confidence": round4(score_confidence(&p))})
        }
        Kind::Noul => {
            let py = p[0].clamp(1e-4, 1.0 - 1e-4);
            let z = (py / (1.0 - py)).ln() / NOUL_T + NOUL_BIAS;
            json!({"type": "noul", "noul": round4(1.0 / (1.0 + (-z).exp()))})
        }
    })
}

/// `exp(v / t - max)` normalised, as the helper computes it.
fn softmax(scores: &[f64], t: f64) -> Vec<f64> {
    let z: Vec<f64> = scores.iter().map(|s| s / t).collect();
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in p.iter().enumerate() {
        if *v > p[best] {
            best = i;
        }
    }
    best
}

/// `(max p - 1/N) / (1 - 1/N)`, floored at 0; 1 for a single option.
fn choice_confidence(p: &[f64]) -> f64 {
    if p.len() == 1 {
        return 1.0;
    }
    let u = 1.0 / p.len() as f64;
    let max = p.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    ((max - u) / (1.0 - u)).max(0.0)
}

/// One minus the expected distance from the most likely level, relative to a uniform answer.
fn score_confidence(p: &[f64]) -> f64 {
    if p.len() == 1 {
        return 1.0;
    }
    let mode = argmax(p) as f64;
    let dist: f64 = p
        .iter()
        .enumerate()
        .map(|(i, pi)| pi * (i as f64 - mode).abs())
        .sum();
    let c = (p.len() - 1) as f64 / 2.0;
    let umad: f64 = (0..p.len()).map(|i| (i as f64 - c).abs()).sum::<f64>() / p.len() as f64;
    (1.0 - dist / umad).max(0.0)
}

/// Python's `round(x, 4)`: the exact binary value rounded to 4 decimals, ties to even, which is also how Rust
/// formats with a precision.
pub fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().expect("a formatted float parses")
}
