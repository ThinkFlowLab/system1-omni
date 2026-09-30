//! The request path: a `/v1/systemone` body to typed answers.
//!
//! Mirrors `src/clm/schema.py` and `src/clm/engine.py`. Two text functions matter and
//! both are reproduced exactly, down to the separator, because the heads were trained on
//! this layout:
//!
//! - the **state head** sees the context and the question joined by a blank line
//!   (`state_text`), so a question belongs in `instructions`, not repeated in the state;
//! - the **action head** sees each candidate's own text with nothing prefixed for
//!   `choice`, but `"<key>: <text>"` for `noul`.
//!
//! A mismatch here does not fail loudly — it shifts every probability — so both are
//! pinned by tests against the reference implementation's own output.
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};

use crate::embedding::Encoder;
use crate::scoring::{self, Answer, Kind, Question};
use crate::weights::Heads;

/// The candidate keys of a `noul` question, in the order the reference uses.
pub const NOUL_KEYS: [&str; 2] = ["false", "true"];

/// A request body: a state and the questions asked of it.
#[derive(Debug, Clone)]
pub struct Request {
    pub state: Value,
    pub model: Option<String>,
    /// Question id to question object, in insertion order.
    pub questions: Vec<(String, QuestionRequest)>,
}

#[derive(Debug, Clone)]
pub struct QuestionRequest {
    pub kind: Kind,
    pub instructions: String,
    pub criteria: Option<Value>,
}

/// One question prepared for the encoder and the scorer.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub id: String,
    pub question: Question,
    /// The text the state head embeds: context and question, blank-line separated.
    pub state_text: String,
    /// The text the action head embeds per candidate, in `question.keys` order.
    pub candidate_texts: Vec<String>,
}

impl Request {
    /// Parse a `/v1/systemone` body. Unknown top-level fields are ignored, as the
    /// reference does.
    pub fn parse(body: &Value) -> Result<Self> {
        let object = body.as_object().context("body must be an object")?;
        let state = object
            .get("state")
            .context("body must have a state")?
            .clone();
        let raw = object
            .get("questions")
            .and_then(Value::as_object)
            .context("body must have a questions object")?;
        ensure!(!raw.is_empty(), "questions must not be empty");

        let mut questions = Vec::with_capacity(raw.len());
        for (id, q) in raw {
            let q = q
                .as_object()
                .with_context(|| format!("question {id:?} is not an object"))?;
            let kind = match q.get("type").and_then(Value::as_str) {
                Some("choice") => Kind::Choice,
                Some("score") => Kind::Score,
                Some("noul") => Kind::Noul,
                other => bail!("question {id:?}: unknown type {other:?}"),
            };
            let instructions = q
                .get("instructions")
                .map(to_text)
                .unwrap_or_default()
                .trim()
                .to_string();
            questions.push((
                id.clone(),
                QuestionRequest {
                    kind,
                    instructions,
                    criteria: q.get("criteria").cloned(),
                },
            ));
        }
        Ok(Self {
            state,
            model: object
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned),
            questions,
        })
    }

    /// Turn each question into the keys, the state text and the candidate texts.
    pub fn prepare(&self) -> Result<Vec<Prepared>> {
        self.questions
            .iter()
            .map(|(id, q)| {
                let (keys, candidate_texts) = candidates(q)
                    .with_context(|| format!("question {id:?} has invalid criteria"))?;
                Ok(Prepared {
                    id: id.clone(),
                    question: Question {
                        id: id.clone(),
                        kind: q.kind,
                        keys,
                    },
                    state_text: state_text(&self.state, &q.instructions),
                    candidate_texts,
                })
            })
            .collect()
    }
}

/// Context first, question last — the layout the heads were trained on.
pub fn state_text(state: &Value, instructions: &str) -> String {
    let s = to_text(state).trim().to_string();
    let i = instructions.trim();
    if !s.is_empty() && !i.is_empty() {
        format!("{s}\n\n{i}")
    } else if !s.is_empty() {
        s
    } else {
        i.to_string()
    }
}

/// Option keys in answer order, and the candidate text per option.
pub fn candidates(q: &QuestionRequest) -> Result<(Vec<String>, Vec<String>)> {
    match q.kind {
        Kind::Choice => {
            let crit = q
                .criteria
                .as_ref()
                .and_then(Value::as_object)
                .context("choice question needs a non-empty 'criteria' object")?;
            ensure!(
                !crit.is_empty(),
                "choice question needs a non-empty 'criteria' object"
            );
            let keys: Vec<String> = crit.keys().cloned().collect();
            // The option's own text when one is given, else the key. Nothing is prefixed.
            // "Given" is the reference's test — null or the empty string — so an empty
            // container is a description that renders to nothing, not a missing one.
            let texts = keys
                .iter()
                .map(|k| match &crit[k] {
                    v if v.is_null() || v.as_str() == Some("") => k.clone(),
                    v => to_text(v),
                })
                .collect();
            Ok((keys, texts))
        }
        Kind::Score => {
            let crit = q
                .criteria
                .as_ref()
                .and_then(Value::as_array)
                .context("score question needs 'criteria' as an ordered list of levels")?;
            ensure!(crit.len() >= 2, "score question needs at least two levels");
            let keys = (0..crit.len()).map(|i| i.to_string()).collect();
            let texts = crit.iter().map(to_text).collect();
            Ok((keys, texts))
        }
        Kind::Noul => {
            let crit = q.criteria.as_ref().and_then(Value::as_object);
            let ins = &q.instructions;
            let mut texts = Vec::with_capacity(NOUL_KEYS.len());
            for k in NOUL_KEYS {
                // The same "given or not" test as `choice`: `crit.get(k)` in `(None, "")`.
                let body = match crit.and_then(|c| c.get(k)) {
                    Some(v) if !v.is_null() && v.as_str() != Some("") => to_text(v),
                    _ if !ins.is_empty() => {
                        if k == "true" {
                            format!("Yes. This is true: {ins}")
                        } else {
                            format!("No. This is false: {ins}")
                        }
                    }
                    _ => k.to_string(),
                };
                texts.push(format!("{k}: {body}"));
            }
            Ok((NOUL_KEYS.iter().map(|k| k.to_string()).collect(), texts))
        }
    }
}

/// Render a state, description or criteria that may be a string, object or array as
/// plain text. Objects become `key: value` fields — top-level fields separated by a blank
/// line, nested ones indented — and arrays become one `- item` line each. Key order is
/// preserved.
pub fn to_text(x: &Value) -> String {
    render(x, 0)
}

fn render(x: &Value, indent: usize) -> String {
    match x {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Number(n) => n.to_string(),
        Value::Object(map) => {
            let pad = " ".repeat(indent);
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| {
                    if is_nonempty_container(v) {
                        format!("{pad}{k}:\n{}", render(v, indent + 2))
                    } else {
                        format!("{pad}{k}: {}", render(v, indent))
                    }
                })
                .collect();
            parts.join(if indent == 0 { "\n\n" } else { "\n" })
        }
        Value::Array(items) => {
            let pad = " ".repeat(indent);
            let parts: Vec<String> = items
                .iter()
                .map(|v| {
                    if is_nonempty_container(v) {
                        format!("{pad}-\n{}", render(v, indent + 2))
                    } else {
                        format!("{pad}- {}", render(v, indent))
                    }
                })
                .collect();
            parts.join("\n")
        }
    }
}

fn is_nonempty_container(v: &Value) -> bool {
    match v {
        Value::Object(m) => !m.is_empty(),
        Value::Array(a) => !a.is_empty(),
        _ => false,
    }
}

/// One decision, plus what the encoder reported spending on it.
#[derive(Debug, Clone)]
pub struct Decision {
    /// Answers by question id, in request order.
    pub answers: Vec<(String, Answer)>,
    pub encoder_tokens: u64,
}

/// The engine: heads, an encoder and the decision path.
pub struct Engine<E: Encoder> {
    pub heads: Heads,
    pub encoder: E,
}

impl<E: Encoder> Engine<E> {
    pub fn new(heads: Heads, encoder: E) -> Self {
        Self { heads, encoder }
    }

    /// Answer every question in the request.
    ///
    /// Every question's state text and every candidate text is embedded in one pass, the
    /// way the reference batches them, so the encoder sees one request per decision.
    pub fn decide(&self, request: &Request, temperature: f32) -> Result<Decision> {
        ensure!(
            temperature > 0.0 && temperature <= 100.0,
            "temperature must be in (0, 100]"
        );
        let prepared = request.prepare()?;

        let mut texts: Vec<String> = Vec::new();
        for p in &prepared {
            texts.push(p.state_text.clone());
            texts.extend(p.candidate_texts.iter().cloned());
        }
        let (vectors, encoder_tokens) = self.encoder.embed(&texts)?;
        ensure!(
            vectors.len() == texts.len(),
            "encoder returned {} vectors for {} texts",
            vectors.len(),
            texts.len()
        );

        // The vectors came back in the order the texts were sent: one state row followed
        // by that question's candidate rows, per question.
        let mut rows = vectors.into_iter();
        let mut answers = Vec::with_capacity(prepared.len());
        for p in &prepared {
            let state = rows.next().context("missing state vector")?;
            let mut candidates = Vec::with_capacity(p.candidate_texts.len());
            for _ in &p.candidate_texts {
                candidates.push(rows.next().context("missing candidate vector")?);
            }
            let probs = scoring::distribution(&self.heads, &state, &candidates, temperature)
                .with_context(|| format!("question {:?}", p.id))?;
            answers.push((
                p.id.clone(),
                scoring::answer(&p.question, &p.candidate_texts, &probs)?,
            ));
        }

        Ok(Decision {
            answers,
            encoder_tokens,
        })
    }
}

/// The `answers` object of a response, in request order.
pub fn answers_json(answers: &[(String, Answer)]) -> Value {
    let mut out = Map::new();
    for (id, answer) in answers {
        out.insert(id.clone(), answer_json(answer));
    }
    Value::Object(out)
}

/// The level key to level text map a `score` answer carries, in answer order.
fn legend_json(legend: &[(String, String)]) -> Value {
    let mut map = Map::new();
    for (key, text) in legend {
        map.insert(key.clone(), Value::String(text.clone()));
    }
    Value::Object(map)
}

fn probabilities(pairs: &[(String, f32)]) -> Value {
    let mut map = Map::new();
    for (k, p) in pairs {
        map.insert(k.clone(), serde_json::json!(p));
    }
    Value::Object(map)
}

/// One answer in the shape `client.py` parses.
pub fn answer_json(answer: &Answer) -> Value {
    match answer {
        Answer::Choice {
            choice,
            confidence,
            probabilities: p,
        } => serde_json::json!({
            "type": "choice",
            "choice": choice,
            "confidence": confidence,
            "probabilities": probabilities(p),
        }),
        Answer::Noul { noul } => serde_json::json!({"type": "noul", "noul": noul}),
        Answer::Score {
            score,
            confidence,
            legend,
            probabilities: p,
        } => serde_json::json!({
            "type": "score",
            "score": score,
            "confidence": confidence,
            "legend": legend_json(legend),
            "probabilities": probabilities(p),
        }),
    }
}
