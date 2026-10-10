//! Exact segment-aware token compilation and request-local response context.
use crate::{
    Config,
    contract::{self, Kind, Question},
    json,
};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;
use tokenizers::Tokenizer;

const TOKENIZER_SHA256: &str = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523";
const MAX_CONTEXT: usize = 32768;
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub max_rows: usize,
    pub max_row_tokens: usize,
    /// Sum of complete row lengths, not unique-prefix API usage.
    pub max_request_tokens: usize,
    pub max_request_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_rows: 1024,
            max_row_tokens: 36864,
            max_request_tokens: 1048576,
            max_request_bytes: 8 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Debug)]
pub struct Label {
    pub name: String,
    pub id: u32,
}
pub struct Processor {
    tokenizer: Tokenizer,
    labels: Vec<Label>,
    config: Config,
    limits: Limits,
}
/// One complete independent unpadded prefill row. No answer token is inserted.
#[derive(Debug)]
pub struct RowInput {
    pub ids: Vec<u32>,
    /// Candidate output rows A..selected label prefix; Noul and Score use A/B, not no/yes IDs.
    pub candidate_ids: Vec<u32>,
    pub readout_position: usize,
    pub question_id: String,
    pub level_index: Option<usize>,
    pub kind: Kind,
    pub temperature: f32,
}
pub struct PreparedRequest {
    pub rows: Vec<RowInput>,
    pub context: ResponseContext,
}
impl PreparedRequest {
    pub fn processed_tokens(&self) -> usize {
        self.rows.iter().map(|r| r.ids.len()).sum()
    }
}
/// Owns identities, ordering, calibrated readout and usage independently of execution.
pub struct ResponseContext {
    pub(crate) questions: Vec<Question>,
    pub(crate) spans: Vec<(usize, usize)>,
    pub(crate) temperatures: Vec<f32>,
    pub(crate) counts: Vec<usize>,
    pub(crate) input_tokens: usize,
}
impl ResponseContext {
    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }
    /// Consume one BF16-projection-rounded FP32 candidate vector per prepared row.
    pub fn finish(self, logits: Vec<Vec<f32>>) -> Result<Value> {
        ensure!(
            logits.len() == self.counts.len(),
            "invalid output row count"
        );
        for (values, count) in logits.iter().zip(&self.counts) {
            ensure!(
                values.len() == *count && values.iter().all(|v| v.is_finite()),
                "invalid candidate output count or nonfinite logits"
            );
        }
        let rows = logits
            .iter()
            .zip(&self.temperatures)
            .map(|(v, t)| crate::math::softmax(v, *t))
            .collect::<Result<Vec<_>>>()?;
        let mut answers = serde_json::Map::new();
        for (q, (start, count)) in self.questions.iter().zip(&self.spans) {
            let answer = if q.kind == Kind::Score {
                let fit: Vec<f64> = (0..*count).map(|i| rows[start + i][1]).collect();
                let sum: f64 = fit.iter().sum();
                let mass = if sum == 0.0 { 1e-9 } else { sum };
                let probabilities: Vec<f64> = fit.iter().map(|v| v / mass).collect();
                let mut answer = contract::answer(q, &probabilities);
                answer["level_fit"] = Value::Object(
                    fit.iter()
                        .enumerate()
                        .map(|(i, v)| (i.to_string(), serde_json::json!(crate::math::round(*v, 4))))
                        .collect(),
                );
                answer["fit_mass"] = serde_json::json!(crate::math::round(mass, 4));
                answer
            } else {
                contract::answer(q, &rows[*start])
            };
            answers.insert(q.id.clone(), answer);
        }
        Ok(
            serde_json::json!({"model":crate::MODEL_ID,"answers":answers,"usage":{"input_tokens":self.input_tokens,"output_tokens":0}}),
        )
    }
}
impl Processor {
    /// Load the frozen tokenizer/config; this never reads model tensor payloads.
    pub fn load(dir: impl AsRef<Path>, limits: Limits) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Config::load(dir)?;
        let bytes = std::fs::read(dir.join("tokenizer.json"))?;
        ensure!(
            format!("{:x}", Sha256::digest(&bytes)) == TOKENIZER_SHA256,
            "tokenizer does not match pinned Decider-2B revision"
        );
        let tokenizer = Tokenizer::from_bytes(bytes).map_err(anyhow::Error::msg)?;
        let mut labels = Vec::new();
        let names = (b'A'..=b'Z').map(|c| (c as char).to_string()).chain(
            (b'A'..=b'Z')
                .flat_map(|a| (b'A'..=b'Z').map(move |b| format!("{}{}", a as char, b as char))),
        );
        for name in names {
            let ids = encode(&tokenizer, &name)?;
            if ids.len() == 1 {
                labels.push(Label { name, id: ids[0] });
            }
            if labels.len() == 255 {
                break;
            }
        }
        ensure!(
            labels.len() == 255
                && labels
                    .iter()
                    .map(|l| l.id)
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    == 255,
            "invalid 255-label vocabulary"
        );
        Ok(Self {
            tokenizer,
            labels,
            config,
            limits,
        })
    }
    pub fn labels(&self) -> &[Label] {
        &self.labels
    }
    /// Validate and compile the whole request before any executor is submitted work.
    /// Integer range, duplicate-key and body-byte restrictions are native admission policy.
    pub fn prepare(&self, raw: &[u8]) -> Result<PreparedRequest> {
        ensure!(
            raw.len() <= self.limits.max_request_bytes,
            "request exceeds byte budget"
        );
        let request = json::parse(raw)?;
        let r = json::object(&request)?;
        json::default_mode(r, "independent", true)?;
        for key in ["schema_first", "neutralize_none", "chat"] {
            json::default_mode(r, key, false)?;
        }
        json::default_mode(r, "isolated_levels", true)?;
        if let Some(v) = r.get("layout") {
            ensure!(
                v.is_null() || v.as_str() == Some("state_first"),
                "unsupported request layout"
            );
        }
        if let Some(v) = r.get("model") {
            ensure!(v.is_null() || v.is_string(), "model must be text or null");
        }
        let state = json::required(r, "state")?;
        let definitions =
            json::object(json::required(r, "questions")?).context("questions must be an object")?;
        let questions = definitions
            .iter()
            .map(|(k, v)| contract::render_question(k, v))
            .collect::<Result<Vec<_>>>()?;
        let nrows = questions.iter().try_fold(0usize, |n, q| {
            n.checked_add(if q.kind == Kind::Score {
                q.legend.len()
            } else {
                1
            })
            .context("row count overflow")
        })?;
        ensure!(
            nrows <= self.limits.max_rows,
            "request exceeds expanded row budget"
        );
        let mut ctx = encode(
            &self.tokenizer,
            &format!("Context:\n{}", contract::render_state(state)),
        )?;
        ctx.truncate(MAX_CONTEXT);
        let mut rows = Vec::with_capacity(nrows);
        let mut spans = Vec::with_capacity(questions.len());
        let mut total = 0usize;
        for q in &questions {
            let start = rows.len();
            let count = if q.kind == Kind::Score {
                q.legend.len()
            } else {
                1
            };
            for level in 0..count {
                let text = if q.kind == Kind::Score {
                    contract::isolated_text(q, level)
                } else {
                    q.text.clone()
                };
                let options = if q.kind == Kind::Score {
                    vec!["no".to_owned(), "yes".to_owned()]
                } else {
                    q.options.clone()
                };
                let piece = self.question_piece(&text, &options)?;
                let length = ctx
                    .len()
                    .checked_add(piece.len())
                    .context("row length overflow")?;
                ensure!(
                    length <= self.limits.max_row_tokens,
                    "question {:?} exceeds complete-row token budget",
                    q.id
                );
                total = total
                    .checked_add(length)
                    .context("request token overflow")?;
                ensure!(
                    total <= self.limits.max_request_tokens,
                    "request exceeds processed-token budget"
                );
                let mut ids = ctx.clone();
                ids.extend(piece);
                rows.push(RowInput {
                    readout_position: ids.len() - 1,
                    ids,
                    candidate_ids: self.labels[..options.len()].iter().map(|l| l.id).collect(),
                    question_id: q.id.clone(),
                    level_index: (q.kind == Kind::Score).then_some(level),
                    kind: q.kind,
                    temperature: self.config.temperature(q.kind),
                });
            }
            spans.push((start, count));
        }
        let input_tokens = unique_tokens(&rows);
        let context = ResponseContext {
            questions,
            spans,
            temperatures: rows.iter().map(|r| r.temperature).collect(),
            counts: rows.iter().map(|r| r.candidate_ids.len()).collect(),
            input_tokens,
        };
        Ok(PreparedRequest { rows, context })
    }
    fn question_piece(&self, text: &str, options: &[String]) -> Result<Vec<u32>> {
        let head = format!("\n\nQuestion: {text}\nOptions:");
        let tail = "\nAnswer: (";
        if options.len() <= 10 {
            let mut piece = head;
            for (j, option) in options.iter().enumerate() {
                piece += &format!("\n({}) {option}", (b'A' + j as u8) as char);
            }
            piece += tail;
            encode(&self.tokenizer, &piece)
        } else {
            let mut ids = encode(&self.tokenizer, &head)?;
            let open = encode(&self.tokenizer, "\n(")?;
            for (j, option) in options.iter().enumerate() {
                ids.extend(&open);
                ids.push(self.labels[j].id);
                ids.extend(encode(&self.tokenizer, &format!(") {option}"))?);
            }
            ids.extend(encode(&self.tokenizer, tail)?);
            Ok(ids)
        }
    }
}
fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec())
}
fn unique_tokens(rows: &[RowInput]) -> usize {
    if rows.is_empty() {
        return 0;
    }
    let shortest = rows.iter().map(|r| r.ids.len()).min().unwrap();
    let mut prefix = 0;
    while prefix < shortest && rows.iter().all(|r| r.ids[prefix] == rows[0].ids[prefix]) {
        prefix += 1;
    }
    prefix + rows.iter().map(|r| r.ids.len() - prefix).sum::<usize>()
}
