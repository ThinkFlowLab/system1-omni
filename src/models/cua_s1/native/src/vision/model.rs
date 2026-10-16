//! Cua checkpoint/provenance validation with the shared Qwen execution.
use super::VisionCheckpoint;
use crate::image_preprocess::ProcessedImage;
use anyhow::Result;
use half::bf16;
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
            omni_qwen3_5_native::vision::VisionModel::from_tensors(config, base, lora, library)?,
        ))
    }
    pub fn synchronize(&self) -> Result<()> {
        self.0.synchronize()
    }
    pub fn forward(&mut self, image: &ProcessedImage) -> Result<Vec<bf16>> {
        self.0.forward(image)
    }
    pub fn forward_with_trace(
        &mut self,
        image: &ProcessedImage,
        callback: impl FnMut(&str, &[bf16]) -> Result<()>,
    ) -> Result<Vec<bf16>> {
        self.0.forward_with_trace(image, callback)
    }
}
