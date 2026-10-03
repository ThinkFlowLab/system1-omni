//! Independent candidates through the shared Qwen3.5/3.8 CUDA prefill.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::model::{Config, Model};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::contract::{Kind, MODEL_ID, Question};

pub const BASE_REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
pub const CHECKPOINT_REVISION: &str = "28cf73067d5b337860bbef3c85b8b82ba8730956";

pub struct Engine {
    tokenizer: Tokenizer,
    model: Arc<Mutex<Model>>,
    head: Vec<f32>,
    bias: f32,
    prefix: String,
    suffix: String,
    pub temperature: f64,
    pub max_length: usize,
}

impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(dir.join("open_jev_export.json"))
                .context("export the merged checkpoint; see recipe/open_jev/native.md")?,
        )?;
        ensure!(
            manifest["format"] == "open-jev-text-merged/1"
                && manifest["model_id"] == MODEL_ID
                && manifest["base_revision"] == BASE_REVISION
                && manifest["checkpoint_revision"] == CHECKPOINT_REVISION,
            "expected a pinned Open-Jev-27B-v1.1 export"
        );
        let temperature = manifest["temperature"].as_f64().context("temperature")?;
        ensure!(
            temperature.is_finite() && temperature > 0.0,
            "invalid temperature"
        );
        let max_length = manifest["max_length"].as_u64().context("max_length")? as usize;
        ensure!(
            (1..=16384).contains(&max_length),
            "max_length must be within 1..=16384"
        );
        let prefix = manifest["chat_prefix"]
            .as_str()
            .context("chat_prefix")?
            .to_owned();
        let suffix = manifest["chat_suffix"]
            .as_str()
            .context("chat_suffix")?
            .to_owned();
        let cfg = Config::load(dir)?;
        ensure!(
            (
                cfg.hidden,
                cfg.intermediate,
                cfg.full_attention.len(),
                cfg.heads,
                cfg.kv_heads,
                cfg.lin_k_heads,
                cfg.lin_v_heads
            ) == (5120, 17408, 64, 24, 4, 16, 48),
            "expected the Qwen3.8-27B backbone dimensions"
        );
        let head: Vec<f32> = manifest["head_weight"]
            .as_array()
            .context("head_weight")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|v| v as f32)
                    .context("head weight must be numeric")
            })
            .collect::<Result<_>>()?;
        let bias = manifest["head_bias"].as_f64().context("head_bias")? as f32;
        ensure!(
            head.len() == cfg.hidden && head.iter().all(|v| v.is_finite()) && bias.is_finite(),
            "invalid scalar decision head"
        );
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let (d, lib) = (dir.to_path_buf(), library.to_path_buf());
        let model = tokio::task::spawn_blocking(move || Model::load(&d, &lib)).await??;
        Ok(Self {
            tokenizer,
            model: Arc::new(Mutex::new(model)),
            head,
            bias,
            prefix,
            suffix,
            temperature,
            max_length,
        })
    }

    /// Validate every length before starting GPU inference; never truncate.
    pub fn encode(&self, questions: &[Question]) -> Result<Vec<Vec<Vec<u32>>>> {
        encode_questions(
            &self.tokenizer,
            &self.prefix,
            &self.suffix,
            self.max_length,
            questions,
        )
    }

    pub async fn score(
        &self,
        ids: Vec<Vec<Vec<u32>>>,
        questions: &[Question],
    ) -> Result<Vec<Vec<f32>>> {
        let model = self.model.clone();
        let head = self.head.clone();
        let bias = self.bias;
        let mut rows = tokio::task::spawn_blocking(move || {
            let mut model = model
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned model"))?;
            ids.iter()
                .map(|candidates| {
                    candidates
                        .iter()
                        .map(|ids| {
                            let last = model.forward(ids)?;
                            // BF16 hidden -> trained FP32 scalar head. FP64 accumulation on the
                            // CPU, rounded to FP32, can differ from PyTorch's FP32 GEMM order.
                            let score = (head
                                .iter()
                                .zip(last)
                                .map(|(&w, h)| w as f64 * h as f64)
                                .sum::<f64>() as f32)
                                + bias;
                            ensure!(score.is_finite(), "non-finite decision score");
                            Ok(score)
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()
        })
        .await??;
        for (row, question) in rows.iter_mut().zip(questions) {
            if question.kind == Kind::Noul {
                row.insert(0, 0.0); // logits [false=0, true=trained score]
            }
        }
        Ok(rows)
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
mod tests;
