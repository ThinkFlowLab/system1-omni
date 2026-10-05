//! Request preparation and response interpretation, independent of model execution.

use std::path::Path;

use anyhow::{Result, ensure};
use serde_json::{Value, json};
use tokenizers::Tokenizer;

use crate::contract::{self, Question, RequestError, chat_text};
use crate::executor::Input;
use crate::json::quote;

const MAX_PROMPT_TOKENS: usize = 16384;

pub struct Processor {
    pub(crate) tokenizer: Tokenizer,
}

pub struct PreparedRequest {
    pub inputs: Vec<Input>,
    pub context: ResponseContext,
}

/// Question identities and usage stay outside the executor.
pub struct ResponseContext {
    questions: Vec<Question>,
    input_tokens: usize,
}

impl Processor {
    pub fn load(dir: &Path) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        Ok(Self { tokenizer })
    }

    /// Validate and tokenize the entire request before submitting any model work.
    pub fn prepare(&self, raw: &[u8]) -> Result<PreparedRequest, RequestError> {
        let (state, questions) =
            contract::parse_body(raw).and_then(|b| contract::map_request(&b))?;
        let mut inputs = Vec::with_capacity(questions.len());
        for q in &questions {
            let enc = self
                .tokenizer
                .encode(chat_text(&state, q), false)
                .map_err(|e| {
                    eprintln!("inference failed: {e}");
                    RequestError {
                        status: 500,
                        message: "inference failed".into(),
                    }
                })?;
            let ids = enc.get_ids().to_vec();
            if ids.len() > MAX_PROMPT_TOKENS {
                return Err(RequestError {
                    status: 413,
                    message: format!(
                        "question {}: {} prompt tokens, over {MAX_PROMPT_TOKENS}",
                        quote(&q.name),
                        ids.len()
                    ),
                });
            }
            inputs.push(Input {
                ids,
                n_options: q.keys.len(),
            });
        }
        let input_tokens = inputs.iter().map(|input| input.ids.len()).sum();
        Ok(PreparedRequest {
            inputs,
            context: ResponseContext {
                questions,
                input_tokens,
            },
        })
    }
}

impl ResponseContext {
    /// One logit row per question, in prepared order; normalize only that row.
    pub fn finish(self, rows: Vec<Vec<f32>>) -> Result<Value> {
        ensure!(rows.len() == self.questions.len(), "invalid logit rows");
        let mut answers = serde_json::Map::new();
        for (q, logits) in self.questions.iter().zip(rows) {
            ensure!(logits.len() == q.keys.len(), "invalid logits");
            let logits: Vec<f64> = logits.iter().map(|&l| l as f64).collect();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
            let total: f64 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| (e / total) as f32).collect();
            ensure!(
                probs.iter().all(|p| p.is_finite()),
                "non-finite probabilities"
            );
            answers.insert(q.name.clone(), contract::answer(q, &probs));
        }
        Ok(json!({"model": contract::MODEL_ID, "answers": answers,
            "usage": {"input_tokens": self.input_tokens, "output_tokens": 0}}))
    }
}

#[cfg(test)]
#[path = "../../../../../tests/cua_s1/processing.rs"]
mod tests;
