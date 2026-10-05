//! Independent CUDA prefill and the learned scalar head, without API semantics.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::model::{Config, Model};
use omni_runtime::SerialScheduler;
use serde_json::Value;

#[derive(Clone)]
pub(crate) struct DecisionHead {
    weights: Vec<f32>,
    bias: f32,
}

impl DecisionHead {
    pub(crate) fn load(dir: &Path, manifest: &Value) -> Result<Self> {
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
}

impl Executor {
    pub(crate) async fn load(dir: &Path, library: &Path, head: DecisionHead) -> Result<Self> {
        let (d, lib) = (dir.to_path_buf(), library.to_path_buf());
        let model = tokio::task::spawn_blocking(move || Model::load(&d, &lib)).await??;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            head,
        })
    }

    /// Inputs and outputs are grouped by question, then candidate, in prepared order.
    /// Admit one whole request; the model lock spans its independent candidate calls.
    pub async fn execute(
        &self,
        scheduler: &SerialScheduler,
        ids: Vec<Vec<Vec<u32>>>,
    ) -> Result<Vec<Vec<f32>>> {
        let model = self.model.clone();
        let head = self.head.clone();
        scheduler
            .run(move || {
                let mut model = model
                    .lock()
                    .map_err(|_| anyhow::anyhow!("poisoned model"))?;
                ids.iter()
                    .map(|candidates| {
                        candidates
                            .iter()
                            .map(|ids| head.score(model.forward(ids)?))
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .await
    }
}

#[cfg(test)]
#[path = "../../../../../tests/open_jev/head.rs"]
mod tests;
