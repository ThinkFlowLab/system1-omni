//! CLM head configuration, read from the exported safetensors metadata.
//!
//! A CLM checkpoint is a `torch.save` dict, so `recipe/clm/native/export_weights.py`
//! converts it first; everything this crate reads is safetensors. The head geometry is
//! not a file of its own, so it travels in the safetensors metadata as `cfg`.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;

/// The head shape the checkpoint was trained with. `depth` counts `inp`, the hidden
/// blocks and `out`, so `depth - 2` is the number of `hidden.N` / `norms.N` pairs.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct HeadConfig {
    pub hidden_size: usize,
    pub projection_dim: usize,
    pub width: usize,
    pub depth: usize,
    pub activation: String,
    pub layernorm: bool,
    pub residual: bool,
    #[serde(default)]
    pub model: Option<String>,
}

impl HeadConfig {
    /// Number of hidden blocks; `depth` includes the input and output projections.
    pub fn hidden_blocks(&self) -> Result<usize> {
        ensure!(
            self.depth >= 2,
            "depth {} cannot cover an input and an output projection",
            self.depth
        );
        Ok(self.depth - 2)
    }
}

/// The reference's cap on `exp(logit_scale)`.
pub const MAX_SCALE: f32 = 100.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub head: HeadConfig,
    /// `exp(logit_scale)` is the inverse InfoNCE temperature the heads were trained with.
    pub logit_scale: f32,
}

impl Config {
    /// Build from the safetensors metadata written by the export tool.
    pub fn from_metadata(
        metadata: Option<&std::collections::HashMap<String, String>>,
    ) -> Result<Self> {
        let metadata = metadata.context("the checkpoint carries no metadata")?;
        ensure!(
            metadata.get("format").map(String::as_str) == Some("clm-heads"),
            "not a converted CLM head checkpoint: format is {:?}",
            metadata.get("format")
        );
        let cfg = metadata.get("cfg").context("metadata has no cfg")?;
        let mut head: HeadConfig =
            serde_json::from_str(cfg).with_context(|| format!("parse cfg {cfg}"))?;
        // hidden_size and projection_dim are repeated at the top level; prefer those
        // when the cfg omits them so an older export still loads.
        head.hidden_size = metadata
            .get("hidden_size")
            .and_then(|v| v.parse().ok())
            .unwrap_or(head.hidden_size);
        head.projection_dim = metadata
            .get("projection_dim")
            .and_then(|v| v.parse().ok())
            .unwrap_or(head.projection_dim);
        let logit_scale: f32 = metadata
            .get("logit_scale")
            .context("metadata has no logit_scale")?
            .parse()
            .context("logit_scale is not a number")?;

        ensure!(head.hidden_size > 0, "hidden_size must be positive");
        ensure!(head.width > 0, "width must be positive");
        ensure!(head.projection_dim > 0, "projection_dim must be positive");
        ensure!(
            head.activation == "gelu" || head.activation == "relu" || head.activation == "silu",
            "unsupported activation {:?}",
            head.activation
        );
        // The upstream head applies LayerNorm before every hidden block and adds the
        // residual after it; the published 0.1 checkpoint sets both to true/false
        // respectively, and the export keeps them so the two paths stay distinguishable.
        head.hidden_blocks()?;
        ensure!(
            logit_scale.is_finite(),
            "logit_scale {logit_scale} is not finite"
        );
        Ok(Self { head, logit_scale })
    }

    /// `exp(logit_scale)`, capped as the reference caps it.
    ///
    /// `heads.py` computes `exp(logit_scale).clamp(max=100.0)`, and the published
    /// checkpoint's `logit_scale` is 4.6132, whose exponential is 100.82 — so the cap
    /// binds and the effective scale is 100, not 100.82. Without it every probability is
    /// off by about 0.8 %, which is what the end-to-end comparison against the reference
    /// caught; no CPU-side oracle can, because both sides of those share this constant.
    pub fn scale(&self) -> f32 {
        self.logit_scale.exp().min(MAX_SCALE)
    }
}
