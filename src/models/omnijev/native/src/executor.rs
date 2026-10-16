//! Native execution of a prepared request: the vision tower once, then the request's
//! prefix once, each question's text once and each option's block, as continuations
//! with the fixed GEMM algorithms and 64-token boundaries; the heads on the CPU; and
//! `MSO1._finish`. Every answer equals that of one plain pass per (question, option)
//! row, which `execute_rows` keeps for checking.

use std::ops::Range;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::inputs::{MultimodalInput, image_positions};
use omni_qwen3_5_native::model::{Model, PrefixState};
use omni_qwen3_5_native::vision::VisionModel;
use serde_json::{Map, Value, json};

use crate::contract::{self, Calibration, Kind, round4};
use crate::export;
use crate::heads::{self, Heads};
use crate::processing::{IMAGE_TOKEN, PreparedRequest, Processor, Row};

pub struct Executor {
    vision: VisionModel,
    language: Model,
    heads: Heads,
    calibration: Calibration,
    /// Buffers for the state after the request's prefix and after a question's text. Each
    /// request or question that uses one captures it again; only the allocations outlive a
    /// request, grown as needed.
    request: Option<PrefixState>,
    question: Option<PrefixState>,
}

/// Rows `[start, end)` of a prompt: their token ids, their image placeholders within the
/// window, the rows of the image features those take, and their positions.
struct Window {
    ids: Vec<u32>,
    images: Vec<usize>,
    features: Range<usize>,
    positions: [Vec<i64>; 3],
}

impl Window {
    fn new(ids: &[u32], positions: &[Vec<i64>; 3], range: Range<usize>) -> Self {
        let image_start = ids.iter().position(|&id| id == IMAGE_TOKEN).unwrap_or(0);
        let images: Vec<usize> = range
            .clone()
            .filter(|&i| ids[i] == IMAGE_TOKEN)
            .map(|i| i - range.start)
            .collect();
        let first = images.first().map_or(0, |&i| range.start + i - image_start);
        Self {
            ids: ids[range.clone()].to_vec(),
            features: first..first + images.len(),
            images,
            positions: positions
                .each_ref()
                .map(|axis| axis[range.clone()].to_vec()),
        }
    }

    fn input<'a>(&'a self, features: &'a [half::bf16], hidden: usize) -> MultimodalInput<'a> {
        MultimodalInput {
            token_ids: &self.ids,
            image_token_indices: &self.images,
            image_embeddings: &features[self.features.start * hidden..self.features.end * hidden],
            position_ids: [&self.positions[0], &self.positions[1], &self.positions[2]],
        }
    }
}

/// `slot`'s state, reallocated when it holds fewer than `len` tokens.
fn state<'s>(
    model: &Model,
    slot: &'s mut Option<PrefixState>,
    len: usize,
) -> Result<&'s mut PrefixState> {
    if slot.as_ref().is_none_or(|state| state.capacity() < len) {
        *slot = None;
        let max = model.cfg.max_positions - model.cfg.max_positions % 64;
        *slot = Some(model.alloc_prefix(len.next_multiple_of(1024).min(max).max(len))?);
    }
    Ok(slot.as_mut().unwrap())
}

/// What one row's pass reads: `zq` and `u`, and the option-text tokens'
/// log-probabilities, the first one's from `zq`.
struct RowReadout {
    zq: Vec<f32>,
    u: Vec<f32>,
    picked: Vec<f32>,
    first: Option<f32>,
}

impl Executor {
    /// Check the export at `dir`, then load the processor, the vision tower, the
    /// language model and the heads.
    pub fn load(dir: &Path, library: &Path) -> Result<(Self, Processor)> {
        let export = export::load(dir)?;
        let processor = Processor::load(dir)?;
        let language = Model::load(dir, library)?;
        ensure!(
            language.cfg.image_token_id == Some(IMAGE_TOKEN)
                && language.cfg.hidden == heads::HIDDEN,
            "export config does not match OmniJev-4B"
        );
        let vision = VisionModel::load(dir, library)?;
        let heads = Heads::load(&dir.join("heads.safetensors"))?;
        Ok((
            Self {
                vision,
                language,
                heads,
                calibration: export.calibration,
                request: None,
                question: None,
            },
            processor,
        ))
    }

    /// The answers in question order, each with the reference's `latency_s` and
    /// `latency_total_s`: the request's preparation, its vision and language passes and
    /// its heads, over its questions and in all.
    pub fn execute(&mut self, prepared: &PreparedRequest) -> Result<Vec<Value>> {
        self.execute_with(prepared, true)
    }

    /// `execute` with one plain pass per (question, option) row over the prefix and that
    /// row, which `execute`'s answers equal bit for bit, timing aside.
    pub fn execute_rows(&mut self, prepared: &PreparedRequest) -> Result<Vec<Value>> {
        self.execute_with(prepared, false)
    }

    fn execute_with(&mut self, prepared: &PreparedRequest, reuse: bool) -> Result<Vec<Value>> {
        let start = Instant::now();
        let outputs = self.head_outputs(prepared, reuse)?;
        let elapsed = prepared.preparation_seconds + start.elapsed().as_secs_f64();
        let latency = elapsed / prepared.questions.len() as f64;
        prepared
            .questions
            .iter()
            .zip(outputs)
            .map(|(question, mu)| {
                let mut answer = contract::answer(question, &mu, &self.calibration, latency)?;
                answer
                    .as_object_mut()
                    .context("answers are objects")?
                    .insert("latency_total_s".into(), json!(round4(elapsed)));
                Ok(answer)
            })
            .collect()
    }

    /// Every question's head outputs before finishing, with the prefix reused or with
    /// one plain pass per row; for checking one against the other.
    pub fn head_outputs(
        &mut self,
        prepared: &PreparedRequest,
        reuse: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let result = self.run(prepared, reuse);
        // Also finish queued work on an error before releasing admission.
        let vision = self.vision.synchronize();
        let language = self.language.synchronize();
        let outputs = result?;
        vision?;
        language?;
        Ok(outputs)
    }

    fn run(&mut self, prepared: &PreparedRequest, reuse: bool) -> Result<Vec<Vec<f32>>> {
        let features = self.vision.forward(&prepared.pixels)?;
        let readouts = if reuse {
            let (mut request, mut question) = (self.request.take(), self.question.take());
            let readouts = self.shared(prepared, &features, &mut request, &mut question);
            (self.request, self.question) = (request, question);
            readouts?
        } else {
            let prefix_len = prepared.layout.prefix;
            let prefix = &prepared.inputs.first().context("no questions")?.token_ids[..prefix_len];
            let mut readouts = Vec::with_capacity(prepared.questions.len());
            for (question, input) in prepared.questions.iter().zip(&prepared.inputs) {
                // Score's ordinal head takes no LM features.
                let lm = question.kind != Kind::Score;
                let rows = input
                    .rows
                    .iter()
                    .map(|row| self.row(prefix, row, prepared.grid, &features, lm))
                    .collect::<Result<Vec<_>>>()?;
                readouts.push(rows);
            }
            readouts
        };
        prepared
            .questions
            .iter()
            .zip(readouts)
            .map(|(question, rows)| {
                let zq = &rows[0].zq;
                let u: Vec<Vec<f32>> = rows.iter().map(|r| r.u.clone()).collect();
                if question.kind == Kind::Score {
                    self.heads.ordinal(zq, &u)
                } else {
                    let picked: Vec<Vec<f32>> = rows.iter().map(|r| r.picked.clone()).collect();
                    let first: Vec<Option<f32>> = rows.iter().map(|r| r.first).collect();
                    let features = heads::lm_features(&picked, &first);
                    self.heads
                        .option_probabilities(&u, zq, question.kind.type_id(), &features)
                }
            })
            .collect()
    }

    /// The readouts of every row with the prefix computed once: the request's prefix up
    /// to its last multiple of 64 tokens, then per question its text up to the last
    /// multiple of 64 before its first option, captured when it reaches one, then each
    /// option's block with what remains before it. `zq` is read with the question's text
    /// when that capture ends right after it, and otherwise in every row.
    fn shared(
        &mut self,
        prepared: &PreparedRequest,
        features: &[half::bf16],
        request: &mut Option<PrefixState>,
        question_state: &mut Option<PrefixState>,
    ) -> Result<Vec<Vec<RowReadout>>> {
        let hidden = self.language.cfg.hidden;
        let l = prepared.layout.prefix;
        let p = l - l % 64;
        let prefix = &prepared.inputs.first().context("no questions")?.token_ids[..l];
        let request = if p > 0 {
            let positions = image_positions(prefix, IMAGE_TOKEN, prepared.grid)?;
            let window = Window::new(prefix, &positions, 0..p);
            let state = state(&self.language, request, p)?;
            self.language.readout_window(
                &window.input(features, hidden),
                None,
                Some(&mut *state),
                &[],
                &[],
            )?;
            Some(&*state)
        } else {
            None
        };
        let mut readouts = Vec::with_capacity(prepared.questions.len());
        for (question, input) in prepared.questions.iter().zip(&prepared.inputs) {
            // Score's ordinal head takes no LM features.
            let lm = question.kind != Kind::Score;
            let rows = &input.rows;
            let row0 = rows.first().context("a question without options")?;
            // the first option's marker, and the question's text up to the boundary before it
            let open = l + row0.zq + 1;
            let q = open - open % 64;
            let zq_at = open - 1;
            let mut shared_zq = None;
            // each option's first text token, read from zq with the question's text
            let mut shared_first: Vec<Option<f32>> = Vec::new();
            let base = if q > p {
                let ids: Vec<u32> = prefix.iter().chain(&row0.tokens).copied().collect();
                let positions = image_positions(&ids, IMAGE_TOKEN, prepared.grid)?;
                let window = Window::new(&ids, &positions, p..q);
                // the capture ends right after zq when the marker starts a new chunk
                let read: &[usize] = if zq_at < q { &[zq_at] } else { &[] };
                let targets: Vec<(usize, u32)> = if lm && zq_at < q {
                    rows.iter()
                        .filter_map(|r| r.first.map(|token| (zq_at, token)))
                        .collect()
                } else {
                    Vec::new()
                };
                let state = state(&self.language, question_state, q)?;
                let readout = self.language.readout_window(
                    &window.input(features, hidden),
                    request,
                    Some(&mut *state),
                    read,
                    &targets,
                )?;
                shared_zq = readout.hidden.into_iter().next();
                let mut logprobs = readout.logprobs.into_iter();
                shared_first = rows
                    .iter()
                    .map(|r| r.first.and_then(|_| logprobs.next()))
                    .collect();
                Some(&*state)
            } else {
                request
            };
            let mut question_rows = Vec::with_capacity(rows.len());
            for (i, row) in rows.iter().enumerate() {
                let ids: Vec<u32> = prefix.iter().chain(&row.tokens).copied().collect();
                let positions = image_positions(&ids, IMAGE_TOKEN, prepared.grid)?;
                let window = Window::new(&ids, &positions, q..ids.len());
                let mut read = Vec::with_capacity(2);
                let mut targets = Vec::new();
                if shared_zq.is_none() {
                    read.push(zq_at);
                    if lm {
                        targets.extend(row.first.map(|token| (zq_at, token)));
                    }
                }
                read.push(l + row.u);
                if lm {
                    targets.extend(row.targets.iter().map(|&(p, token)| (l + p, token)));
                }
                let readout = self.language.readout_window(
                    &window.input(features, hidden),
                    base,
                    None,
                    &read,
                    &targets,
                )?;
                let mut hidden_rows = readout.hidden.into_iter();
                let zq = match &shared_zq {
                    Some(zq) => zq.clone(),
                    None => hidden_rows.next().unwrap(),
                };
                let u = hidden_rows.next().unwrap();
                let mut logprobs = readout.logprobs.into_iter();
                let first = match row.first {
                    Some(_) if lm => {
                        if shared_zq.is_some() {
                            shared_first[i]
                        } else {
                            logprobs.next()
                        }
                    }
                    _ => None,
                };
                question_rows.push(RowReadout {
                    zq,
                    u,
                    picked: logprobs.collect(),
                    first,
                });
            }
            readouts.push(question_rows);
        }
        Ok(readouts)
    }

    /// One pass over the prefix and `row`, reading `zq`, `u` and, with `lm`, the
    /// option-text log-probabilities.
    fn row(
        &mut self,
        prefix: &[u32],
        row: &Row,
        grid: [usize; 3],
        features: &[half::bf16],
        lm: bool,
    ) -> Result<RowReadout> {
        let at = prefix.len();
        let ids: Vec<u32> = prefix.iter().chain(&row.tokens).copied().collect();
        let positions = image_positions(&ids, IMAGE_TOKEN, grid)?;
        let image_rows: Vec<usize> = (0..at).filter(|&i| ids[i] == IMAGE_TOKEN).collect();
        let mut targets = Vec::new();
        if lm {
            targets.extend(row.first.map(|token| (at + row.zq, token)));
            targets.extend(row.targets.iter().map(|&(p, token)| (at + p, token)));
        }
        let readout = self.language.forward_multimodal_readout(
            &MultimodalInput {
                token_ids: &ids,
                image_token_indices: &image_rows,
                image_embeddings: features,
                position_ids: [&positions[0], &positions[1], &positions[2]],
            },
            &[at + row.zq, at + row.u],
            &targets,
        )?;
        let mut hidden = readout.hidden.into_iter();
        let (zq, u) = (hidden.next().unwrap(), hidden.next().unwrap());
        let mut logprobs = readout.logprobs.into_iter();
        let first = if lm && row.first.is_some() {
            logprobs.next()
        } else {
            None
        };
        Ok(RowReadout {
            zq,
            u,
            picked: logprobs.collect(),
            first,
        })
    }
}

/// The `/v1/systemone` response: the answers by question id, and the reference's
/// input-token count.
pub fn response(prepared: &PreparedRequest, answers: Vec<Value>) -> Value {
    let answers: Map<String, Value> = prepared
        .questions
        .iter()
        .map(|q| q.id.clone())
        .zip(answers)
        .collect();
    json!({
        "model": contract::MODEL_ID,
        "answers": answers,
        "usage": {"input_tokens": prepared.layout.processed_tokens, "output_tokens": 0},
    })
}
