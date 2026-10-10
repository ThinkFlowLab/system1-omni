//! Native execution of a prepared request: the vision tower once, then one plain pass
//! per (question, option) row over the shared prefix and that row, with the fixed
//! GEMM algorithms; the heads on the CPU; and `MSO1._finish`.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::inputs::{MultimodalInput, image_positions};
use omni_qwen3_5_native::model::Model;
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
            },
            processor,
        ))
    }

    /// The answers in question order, each with the reference's `latency_s` and
    /// `latency_total_s`: the request's preparation and its vision and language passes,
    /// over its questions and in all.
    pub fn execute(&mut self, prepared: &PreparedRequest) -> Result<Vec<Value>> {
        let result = self.run(prepared);
        // Also finish queued work on an error before releasing admission.
        let vision = self.vision.synchronize();
        let language = self.language.synchronize();
        let answers = result?;
        vision?;
        language?;
        Ok(answers)
    }

    fn run(&mut self, prepared: &PreparedRequest) -> Result<Vec<Value>> {
        let start = Instant::now();
        let features = self.vision.forward(&prepared.pixels)?;
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
        let elapsed = prepared.preparation_seconds + start.elapsed().as_secs_f64();
        let latency = elapsed / prepared.questions.len() as f64;
        prepared
            .questions
            .iter()
            .zip(readouts)
            .map(|(question, rows)| {
                let zq = &rows[0].zq;
                let u: Vec<Vec<f32>> = rows.iter().map(|r| r.u.clone()).collect();
                let mu = if question.kind == Kind::Score {
                    self.heads.ordinal(zq, &u)?
                } else {
                    let picked: Vec<Vec<f32>> = rows.iter().map(|r| r.picked.clone()).collect();
                    let first: Vec<Option<f32>> = rows.iter().map(|r| r.first).collect();
                    let features = heads::lm_features(&picked, &first);
                    self.heads
                        .option_probabilities(&u, zq, question.kind.type_id(), &features)?
                };
                let mut answer = contract::answer(question, &mu, &self.calibration, latency)?;
                answer
                    .as_object_mut()
                    .context("answers are objects")?
                    .insert("latency_total_s".into(), json!(round4(elapsed)));
                Ok(answer)
            })
            .collect()
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
