//! Independent CUDA prefill and the learned scalar head, without API semantics.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::model::{Config, Model};
use omni_runtime::SerialScheduler;
use serde_json::Value;

use crate::batching;
use crate::contract::Checkpoint;

#[derive(Clone)]
pub(crate) struct DecisionHead {
    weights: Vec<f32>,
    bias: f32,
}

impl DecisionHead {
    pub(crate) fn load(dir: &Path, manifest: &Value, checkpoint: &Checkpoint) -> Result<Self> {
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
            ) == checkpoint.backbone,
            "expected the {} backbone dimensions",
            checkpoint.model_id
        );
        let weights: Vec<f32> = manifest["head_weight"]
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
            weights.len() == cfg.hidden
                && weights.iter().all(|v| v.is_finite())
                && bias.is_finite(),
            "invalid scalar decision head"
        );
        Ok(Self { weights, bias })
    }

    fn score(&self, last: Vec<f32>) -> Result<f32> {
        // BF16 hidden -> trained FP32 scalar head. FP64 accumulation on the
        // CPU, rounded to FP32, can differ from PyTorch's FP32 GEMM order.
        let score = (self
            .weights
            .iter()
            .zip(last)
            .map(|(&w, h)| w as f64 * h as f64)
            .sum::<f64>() as f32)
            + self.bias;
        ensure!(score.is_finite(), "non-finite decision score");
        Ok(score)
    }
}

pub struct Executor {
    model: Arc<Mutex<Model>>,
    head: DecisionHead,
    pack: bool,
}

impl Executor {
    pub(crate) async fn load(
        dir: &Path,
        library: &Path,
        head: DecisionHead,
        checkpoint: &Checkpoint,
    ) -> Result<Self> {
        let (d, lib) = (dir.to_path_buf(), library.to_path_buf());
        let model = tokio::task::spawn_blocking(move || Model::load(&d, &lib)).await??;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            head,
            pack: checkpoint.pack_candidates,
        })
    }

    /// Inputs and outputs are grouped by question, then candidate, in prepared order.
    /// Admit one whole request; where the checkpoint allows it, pack independent
    /// candidates for shared GEMMs, otherwise run them one at a time.
    pub async fn execute(
        &self,
        scheduler: &SerialScheduler,
        ids: Vec<Vec<Vec<u32>>>,
    ) -> Result<Vec<Vec<f32>>> {
        let model = self.model.clone();
        let head = self.head.clone();
        let pack = self.pack;
        scheduler
            .run(move || {
                let mut model = model
                    .lock()
                    .map_err(|_| anyhow::anyhow!("poisoned model"))?;
                let inputs: Vec<&[u32]> = ids.iter().flatten().map(Vec::as_slice).collect();
                let mut scores = Vec::with_capacity(inputs.len());
                if pack {
                    for range in batching::ranges(&inputs) {
                        for last in model.forward_batch(&inputs[range])? {
                            scores.push(head.score(last)?);
                        }
                    }
                } else {
                    for ids in &inputs {
                        scores.push(head.score(model.forward(ids)?)?);
                    }
                }
                let mut scores = scores.into_iter();
                Ok(ids
                    .iter()
                    .map(|candidates| scores.by_ref().take(candidates.len()).collect())
                    .collect())
            })
            .await
    }
}

#[cfg(test)]
#[path = "../../../../../tests/open_jev/head.rs"]
mod tests;
