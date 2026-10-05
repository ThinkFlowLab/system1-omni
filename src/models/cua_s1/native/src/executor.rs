//! CUDA prefill and the learned answer-letter projection. No request semantics.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use omni_runtime::SerialScheduler;
use serde_json::Value as Json;

use crate::model::Model;

/// Tied with the output projection in Qwen3.5-4B.
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";

/// One unpadded prompt and the number of answer-letter rows to project.
pub struct Input {
    pub ids: Vec<u32>,
    pub n_options: usize,
}

pub struct Executor {
    model: Arc<Mutex<Model>>,
    letters: Vec<f32>,
}

impl Executor {
    pub async fn load(dir: &Path, library: &Path, letter_ids: &[u32]) -> Result<Self> {
        let (d, lib) = (dir.to_path_buf(), library.to_path_buf());
        let model = tokio::task::spawn_blocking(move || Model::load(&d, &lib)).await??;
        let letters = letter_rows(dir, letter_ids, model.cfg.hidden)?;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            letters,
        })
    }

    /// Admit each question through the worker's shared runtime scheduler.
    /// Returns FP32 letter logits in input order; softmax belongs to processing.
    pub async fn execute(
        &self,
        scheduler: &SerialScheduler,
        inputs: Vec<Input>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut rows = Vec::with_capacity(inputs.len());
        for input in inputs {
            let model = self.model.clone();
            let last = scheduler
                .run(move || {
                    model
                        .lock()
                        .map_err(|_| anyhow::anyhow!("poisoned"))?
                        .forward(&input.ids)
                })
                .await?;
            rows.push(letter_logits(&self.letters, &last, input.n_options)?);
        }
        Ok(rows)
    }
}

fn letter_logits(letters: &[f32], last: &[f32], n_options: usize) -> Result<Vec<f32>> {
    let logits: Vec<f32> = letters
        .chunks_exact(last.len())
        .take(n_options)
        .map(|w| {
            w.iter()
                .zip(last)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>() as f32
        })
        .collect();
    // Abort before submitting a later prompt, as the former per-question softmax did.
    ensure!(
        logits.iter().all(|l| l.is_finite()),
        "non-finite probabilities"
    );
    Ok(logits)
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

#[cfg(test)]
#[path = "../../../../../tests/cua_s1/head.rs"]
mod tests;
