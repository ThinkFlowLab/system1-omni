//! Request/token preparation and calibrated response interpretation.

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, ensure};
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use crate::contract::{self, Checkpoint, Kind, Question};

pub struct Processor {
    checkpoint: &'static Checkpoint,
    tokenizer: Tokenizer,
    prefix: String,
    suffix: String,
    temperature: f64,
    max_length: usize,
}

pub struct PreparedRequest {
    pub inputs: Vec<Vec<Vec<u32>>>,
    pub context: ResponseContext,
}

/// The original question/candidate mapping, calibration, usage, and timing.
pub struct ResponseContext {
    checkpoint: &'static Checkpoint,
    questions: Vec<Question>,
    input_tokens: usize,
    candidates: usize,
    temperature: f64,
    max_length: usize,
    start: Instant,
}

impl Processor {
    pub(crate) fn load(
        dir: &Path,
        checkpoint: &'static Checkpoint,
        prefix: String,
        suffix: String,
        temperature: f64,
        max_length: usize,
    ) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        Ok(Self {
            checkpoint,
            tokenizer,
            prefix,
            suffix,
            temperature,
            max_length,
        })
    }

    /// Validate every candidate length before inference; never truncate.
    pub fn prepare(&self, raw: &[u8]) -> Result<PreparedRequest> {
        let questions = contract::compile(raw, self.checkpoint)?;
        let start = Instant::now();
        let inputs = encode_questions(
            &self.tokenizer,
            &self.prefix,
            &self.suffix,
            self.max_length,
            &questions,
        )?;
        let input_tokens = inputs.iter().flatten().map(Vec::len).sum();
        let candidates = inputs.iter().map(Vec::len).sum();
        Ok(PreparedRequest {
            inputs,
            context: ResponseContext {
                checkpoint: self.checkpoint,
                questions,
                input_tokens,
                candidates,
                temperature: self.temperature,
                max_length: self.max_length,
                start,
            },
        })
    }
}

impl ResponseContext {
    /// Restore the Noul baseline and normalize across each complete question.
    pub fn finish(self, rows: Vec<Vec<f32>>) -> Result<Value> {
        ensure!(rows.len() == self.questions.len(), "invalid logit rows");
        let mut answers = Map::new();
        for (q, mut logits) in self.questions.iter().zip(rows) {
            ensure!(logits.len() == q.prompts.len(), "invalid logits");
            if q.kind == Kind::Noul {
                logits.insert(0, 0.0); // logits [false=0, true=trained score]
            }
            answers.insert(
                q.id.clone(),
                contract::answer(q, &logits, self.temperature)?,
            );
        }
        Ok(
            json!({"model": self.checkpoint.model_id, "answers": answers,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
            "metadata": {"method": "native_merged_lora_decision_head", "temperature": self.temperature,
                "candidate_sequences": self.candidates, "inference_seconds": self.start.elapsed().as_secs_f64(),
                "base_revision": self.checkpoint.base_revision, "max_length": self.max_length,
                "prefix_cache": {"enabled": false, "mode": "independent_candidates"}}}),
        )
    }
}

fn encode_questions(
    tokenizer: &Tokenizer,
    prefix: &str,
    suffix: &str,
    max_length: usize,
    questions: &[Question],
) -> Result<Vec<Vec<Vec<u32>>>> {
    questions
        .iter()
        .map(|q| {
            q.prompts
                .iter()
                .map(|prompt| {
                    let chat = format!("{prefix}{}{suffix}", prompt.trim());
                    let enc = tokenizer.encode(chat, true).map_err(anyhow::Error::msg)?;
                    ensure!(
                        enc.len() <= max_length,
                        "question {}: {} tokens exceeds max_length={max_length}; no silent truncation",
                        q.id,
                        enc.len()
                    );
                    Ok(enc.get_ids().to_vec())
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
#[path = "../../../../../tests/open_jev/tokenization.rs"]
mod tokenization_tests;

#[cfg(test)]
#[path = "../../../../../tests/open_jev/processing.rs"]
mod tests;
