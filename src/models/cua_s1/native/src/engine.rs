//! Tokenization and scoring: one prefill-only forward pass per question through the
//! native Qwen3.5 model, scored with the 26 letter rows of the output projection.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use serde_json::Value as Json;
use tokenizers::Tokenizer;

use crate::contract::{LETTERS, Question, chat_text};
use crate::model::Model;

/// Tied with the output projection in Qwen3.5-4B.
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";

pub struct Engine {
    tokenizer: Tokenizer,
    model: Arc<Mutex<Model>>,
    /// the letter rows of the output projection, as float32
    letters: Vec<f32>,
}

impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        // Written by export_text_merged.py; without it `dir` may hold the base model alone.
        ensure!(
            dir.join("cua_s1_export.json").exists(),
            "{} is not a merged text checkpoint; see recipe/cua_s1/native.md",
            dir.display()
        );
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let ids = LETTERS
            .chars()
            .map(|c| {
                tokenizer
                    .token_to_id(&c.to_string())
                    .context("letter token")
            })
            .collect::<Result<Vec<_>>>()?;
        let (d, lib) = (dir.to_path_buf(), library.to_path_buf());
        let model = tokio::task::spawn_blocking(move || Model::load(&d, &lib)).await??;
        let letters = letter_rows(dir, &ids, model.cfg.hidden)?;
        Ok(Self {
            tokenizer,
            model: Arc::new(Mutex::new(model)),
            letters,
        })
    }

    pub fn encode(&self, state: &str, question: &Question) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(chat_text(state, question), false)
            .map_err(anyhow::Error::msg)?;
        Ok(enc.get_ids().to_vec())
    }

    /// Option probabilities for one prompt: the final-norm hidden state at the last
    /// position times the letter rows, in float32 with float64 accumulation, then a
    /// softmax over the first `n_options` letters.
    pub async fn score(&self, ids: Vec<u32>, n_options: usize) -> Result<Vec<f32>> {
        let model = self.model.clone();
        let last = tokio::task::spawn_blocking(move || {
            model
                .lock()
                .map_err(|_| anyhow::anyhow!("poisoned"))?
                .forward(&ids)
        })
        .await??;
        let logits: Vec<f64> = self
            .letters
            .chunks_exact(last.len())
            .take(n_options)
            .map(|w| {
                w.iter()
                    .zip(&last)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>() as f32 as f64
            })
            .collect();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&l| (l - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| (e / total) as f32).collect();
        ensure!(
            probs.iter().all(|p| p.is_finite()),
            "non-finite probabilities"
        );
        Ok(probs)
    }
}

/// The letter rows of the bfloat16 embedding, read from the safetensors files.
fn letter_rows(dir: &Path, ids: &[u32], hidden: usize) -> Result<Vec<f32>> {
    let index: Json = serde_json::from_str(&std::fs::read_to_string(
        dir.join("model.safetensors.index.json"),
    )?)?;
    let file = index["weight_map"][EMBEDDING].as_str().context(EMBEDDING)?;
    let file = std::fs::File::open(dir.join(file))?;
    // SAFETY: the checkpoint is not modified while the worker runs.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
    let view = tensors.tensor(EMBEDDING)?;
    ensure!(
        view.dtype() == safetensors::Dtype::BF16 && view.shape()[1] == hidden,
        "{EMBEDDING}: {:?} {:?}",
        view.dtype(),
        view.shape()
    );
    let mut rows = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        let row = &view.data()[id as usize * hidden * 2..(id as usize + 1) * hidden * 2];
        let (pairs, _) = row.as_chunks::<2>();
        rows.extend(pairs.iter().map(|&b| half::bf16::from_le_bytes(b).to_f32()));
    }
    Ok(rows)
}
