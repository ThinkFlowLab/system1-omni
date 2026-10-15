//! Cua checkpoint/provenance validation with the shared Qwen execution.
use super::VisionCheckpoint;
use crate::{cuda::DeviceBf16Matrix, image_preprocess::ProcessedImage};
use anyhow::Result;
use half::bf16;
use omni_qwen3_5_native::vision::CacheLimits;
use std::path::Path;
pub struct VisionModel(omni_qwen3_5_native::vision::VisionModel);
impl VisionModel {
    pub fn load(base: impl AsRef<Path>, adapter: impl AsRef<Path>, library: &Path) -> Result<Self> {
        let checkpoint = VisionCheckpoint::load(base, adapter)?;
        let c = checkpoint.config();
        let config = omni_qwen3_5_native::vision::VisionConfig::from_value(serde_json::json!({
            "depth":c.depth,"hidden_size":c.hidden_size,"intermediate_size":c.intermediate_size,
            "num_heads":c.num_heads,"num_position_embeddings":c.num_position_embeddings,
            "out_hidden_size":c.out_hidden_size,"in_channels":c.in_channels,"patch_size":c.patch_size,
            "temporal_patch_size":c.temporal_patch_size,"spatial_merge_size":c.spatial_merge_size,
            "hidden_act":c.hidden_act,"deepstack_visual_indexes":c.deepstack_visual_indexes,"model_type":c.model_type
        }))?;
        let base = checkpoint
            .base_names()
            .map(|n| Ok((n.to_owned(), checkpoint.base_tensor(n)?)))
            .collect::<Result<Vec<_>>>()?;
        let lora = checkpoint
            .adapter_names()
            .map(|n| Ok((n.to_owned(), checkpoint.adapter_tensor(n)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self(
            omni_qwen3_5_native::vision::VisionModel::from_tensors_with_cache(
                config,
                base,
                lora,
                library,
                cache_limits_from_env(),
            )?,
        ))
    }
    pub fn synchronize(&self) -> Result<()> {
        self.0.synchronize()
    }
    pub fn forward(&mut self, image: &ProcessedImage) -> Result<Vec<bf16>> {
        self.0.forward(image)
    }
    /// Borrow synchronized shared-executor output until the language consumer drains.
    pub fn forward_device(&mut self, image: &ProcessedImage) -> Result<DeviceBf16Matrix<'_>> {
        self.0.forward_device(image)
    }
    pub fn release_transient(&mut self) -> Result<()> {
        self.0.release_transient()
    }
    pub fn forward_with_trace(
        &mut self,
        image: &ProcessedImage,
        callback: impl FnMut(&str, &[bf16]) -> Result<()>,
    ) -> Result<Vec<bf16>> {
        self.0.forward_with_trace(image, callback)
    }
}

fn cache_limits_from_env() -> CacheLimits {
    let value = |name: &str, default| match std::env::var_os(name) {
        None => default,
        Some(raw) => match raw.to_str().and_then(|v| v.parse::<usize>().ok()) {
            Some(value) => value,
            None => {
                eprintln!(
                    "Invalid {name}={raw:?}; expected an unsigned integer, using default {default}"
                );
                default
            }
        },
    };
    CacheLimits {
        entries: value("CUA_S1_VISION_CACHE_ENTRIES", 1).min(16),
        bytes: value("CUA_S1_VISION_CACHE_BYTES", 256 << 20),
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/cua_s1/vision_cache_env.rs"]
mod cache_tests;
