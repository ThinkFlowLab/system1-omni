//! The typed decision computation: project embeddings, score candidates, assemble answers.
//!
//! This mirrors `src/clm/engine.py` and `src/clm/schema.py`: a state and each candidate
//! are embedded elsewhere (Qwen3-8B behind `/v1/embeddings`) and arrive here as vectors.
//! The state head sees the state, the action head sees every candidate, both projections
//! are L2-normalised, and the score of a pair is `exp(logit_scale) * cos(state, candidate)`,
//! divided by the request temperature and softmaxed across the question's candidates.
//!
//! The three question types differ only after the distribution exists.
use anyhow::{Result, ensure};

use crate::weights::{Head, Heads};

/// One question: its type and its candidate keys in the order the caller offered them.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub id: String,
    pub kind: Kind,
    /// Candidate keys, in order. For `Noul` these are `["false", "true"]`.
    pub keys: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Choice,
    Noul,
    Score,
}

/// One answer, matching the shapes `client.py` parses.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Choice {
        choice: String,
        confidence: f32,
        probabilities: Vec<(String, f32)>,
    },
    Noul {
        noul: f32,
    },
    Score {
        score: f32,
        confidence: f32,
        probabilities: Vec<(String, f32)>,
    },
}

impl Answer {
    /// The discrete label, as `schema.label_of` defines it.
    pub fn label(&self) -> String {
        match self {
            Answer::Choice { choice, .. } => choice.clone(),
            Answer::Noul { noul } => if *noul >= 0.5 { "true" } else { "false" }.to_string(),
            // The first maximum, as Python's `max` keeps. `max_by` would return the
            // last of several equal values, which is a different label for the same
            // distribution.
            Answer::Score { probabilities, .. } => probabilities
                .iter()
                .fold(None, |best: Option<&(String, f32)>, kv| match best {
                    Some(b) if b.1 >= kv.1 => Some(b),
                    _ => Some(kv),
                })
                .map(|(k, _)| k.clone())
                .unwrap_or_default(),
        }
    }
}

/// L2-normalise one row in place, the way `torch.nn.functional.normalize` does.
///
/// Shared with [`crate::embedding`], which normalises what the encoder hands back.
pub fn normalize(row: &mut [f32]) {
    let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Abramowitz & Stegun 7.1.26. The coefficients are kept at full precision through named
/// constants: rounding one to a shorter f32 literal changes the value, and
/// `recipe/clm/native/head_oracle.py` carries the same digits so that both sides remain
/// the same arithmetic rather than merely similar.
#[allow(clippy::excessive_precision)]
const ERF_P: f32 = 0.3275911;
#[allow(clippy::excessive_precision)]
const ERF_A1: f32 = 0.254829592;
#[allow(clippy::excessive_precision)]
const ERF_A2: f32 = -0.284496736;
#[allow(clippy::excessive_precision)]
const ERF_A3: f32 = 1.421413741;
#[allow(clippy::excessive_precision)]
const ERF_A4: f32 = -1.453152027;
#[allow(clippy::excessive_precision)]
const ERF_A5: f32 = 1.061405429;

fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + ERF_P * x);
    let y = 1.0
        - (((((ERF_A5 * t + ERF_A4) * t + ERF_A3) * t + ERF_A2) * t + ERF_A1) * t * (-x * x).exp());
    sign * y
}

fn activate(kind: &str, x: f32) -> f32 {
    match kind {
        "relu" => relu(x),
        "silu" => silu(x),
        _ => gelu(x),
    }
}

#[inline]
fn linear_into(x: &[f32], w: &[f32], b: &[f32], n: usize, k: usize, out: &mut [f32]) {
    for i in 0..n {
        let row = &w[i * k..(i + 1) * k];
        let mut acc = b[i];
        for (a, wv) in x.iter().zip(row) {
            acc += a * wv;
        }
        out[i] = acc;
    }
}

fn layer_norm(x: &mut [f32], weight: &[f32], bias: &[f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for i in 0..x.len() {
        x[i] = (x[i] - mean) * inv * weight[i] + bias[i];
    }
}

/// Project one embedding through a head, returning the L2-normalised vector.
pub fn project(head: &Head, cfg: &crate::config::HeadConfig, x: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        x.len() == cfg.hidden_size,
        "embedding has {} values, the head expects {}",
        x.len(),
        cfg.hidden_size
    );
    let w = cfg.width;
    let mut h = vec![0.0f32; w];
    linear_into(
        x,
        &head.inp_weight,
        &head.inp_bias,
        w,
        cfg.hidden_size,
        &mut h,
    );
    for v in h.iter_mut() {
        *v = activate(&cfg.activation, *v);
    }
    if !head.hidden_weight.is_empty() {
        let mut hidden = vec![0.0f32; w];
        linear_into(
            &h,
            &head.hidden_weight,
            &head.hidden_bias,
            w,
            w,
            &mut hidden,
        );
        if let (Some(nw), Some(nb)) = (&head.norm_weight, &head.norm_bias) {
            layer_norm(&mut hidden, nw, nb);
        }
        for v in hidden.iter_mut() {
            *v = activate(&cfg.activation, *v);
        }
        if cfg.residual {
            for i in 0..w {
                hidden[i] += h[i];
            }
        }
        h = hidden;
    }
    let p = cfg.projection_dim;
    let mut out = vec![0.0f32; p];
    linear_into(&h, &head.out_weight, &head.out_bias, p, w, &mut out);
    normalize(&mut out);
    Ok(out)
}

/// `softmax(scale * cos / temperature)`, the distribution every question type starts from.
pub fn distribution(
    heads: &Heads,
    state: &[f32],
    candidates: &[Vec<f32>],
    temperature: f32,
) -> Result<Vec<f32>> {
    ensure!(
        !candidates.is_empty(),
        "a question needs at least one candidate"
    );
    ensure!(
        temperature > 0.0 && temperature <= 100.0,
        "temperature must be in (0, 100], got {temperature}"
    );
    let cfg = &heads.config.head;
    let zs = project(&heads.state, cfg, state)?;
    let mut logits = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let zc = project(&heads.action, cfg, candidate)?;
        let cos: f32 = zs.iter().zip(&zc).map(|(a, b)| a * b).sum();
        logits.push(heads.config.scale() * cos / temperature);
    }
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let exp: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    ensure!(sum > 0.0, "softmax denominator is zero");
    Ok(exp.iter().map(|v| v / sum).collect())
}

/// `schema.confidence`: the top probability minus the mean of the rest, clamped to [0, 1].
/// A single candidate is fully decided by definition.
pub fn confidence(probs: &[f32]) -> f32 {
    if probs.len() < 2 {
        return 1.0;
    }
    let (j, top) =
        probs.iter().enumerate().fold(
            (0usize, f32::MIN),
            |best, (i, p)| {
                if *p > best.1 { (i, *p) } else { best }
            },
        );
    let rest = (probs.len() - 1) as f32;
    let mean_rest = probs
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != j)
        .map(|(_, p)| *p)
        .sum::<f32>()
        / rest;
    (top - mean_rest).clamp(0.0, 1.0)
}

/// Assemble the answer for one question from its distribution.
pub fn answer(question: &Question, probs: &[f32]) -> Result<Answer> {
    ensure!(
        question.keys.len() == probs.len(),
        "{}: {} keys but {} probabilities",
        question.id,
        question.keys.len(),
        probs.len()
    );
    let pairs: Vec<(String, f32)> = question
        .keys
        .iter()
        .cloned()
        .zip(probs.iter().copied())
        .collect();
    Ok(match question.kind {
        Kind::Noul => {
            // `answer_from_probs` reads the `true` entry, which is the last key by
            // convention; a caller that orders them differently still gets a stable
            // answer because the key is looked up by name.
            let p = pairs
                .iter()
                .find(|(k, _)| k == "true")
                .map(|(_, p)| *p)
                .unwrap_or_else(|| *probs.last().expect("checked non-empty"));
            Answer::Noul { noul: p }
        }
        Kind::Choice => {
            let (choice, _) = pairs
                .iter()
                .fold(None, |best: Option<&(String, f32)>, kv| match best {
                    Some(b) if b.1 >= kv.1 => Some(b),
                    _ => Some(kv),
                })
                .expect("checked non-empty");
            Answer::Choice {
                choice: choice.clone(),
                confidence: confidence(probs),
                probabilities: pairs,
            }
        }
        Kind::Score => {
            // The expected level index, `sum(i * p_i)`.
            let score = probs
                .iter()
                .enumerate()
                .map(|(i, p)| i as f32 * p)
                .sum::<f32>();
            Answer::Score {
                score,
                confidence: confidence(probs),
                probabilities: pairs,
            }
        }
    })
}
