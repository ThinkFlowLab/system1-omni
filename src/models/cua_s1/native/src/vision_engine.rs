//! Native screenshot-to-decision orchestration, with request-local vision reuse.
pub use crate::vision_processing::PreparedRequest;
use crate::vision_processing::{VisionProcessor, probabilities};
use crate::{
    contract::{LETTERS, Question},
    image_request::ImageRequest,
    inputs::MultimodalInput,
    model::Model,
    multimodal::ImagePrompt,
    vision::VisionModel,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::sync::Arc;
use std::{fs::File, path::Path};
use tokenizers::Tokenizer;

pub struct VisionEngine {
    pub processor: Arc<VisionProcessor>,
    pub vision: VisionModel,
    language: Model,
    letters: Vec<f32>,
}

pub struct Readout {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}

impl VisionEngine {
    pub fn load(base: &Path, adapter: &Path, language: &Path, library: &Path) -> Result<Self> {
        let marker: Value = serde_json::from_slice(
            &std::fs::read(language.join("cua_s1_language_export.json"))
                .context("export the multimodal language checkpoint first")?,
        )?;
        ensure!(
            marker["format"] == "cua-s1-multimodal-language-merged/1"
                && marker["base_revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
                && marker["adapter_revision"] == "16818868b0cc7813808aae4e87b417657046ab79",
            "expected pinned multimodal language export"
        );
        crate::provenance::verify_sources(base, adapter)?;
        crate::provenance::verify_export(language, &marker)?;
        let tokenizer =
            Tokenizer::from_file(language.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let ids = LETTERS
            .chars()
            .map(|c| {
                let enc = tokenizer
                    .encode(c.to_string(), false)
                    .map_err(anyhow::Error::msg)?;
                ensure!(enc.len() == 1, "each candidate must be one token");
                Ok(enc.get_ids()[0])
            })
            .collect::<Result<Vec<_>>>()?;
        let model = Model::load(language, library)?;
        ensure!(
            tokenizer.token_to_id("<|image_pad|>") == model.cfg.image_token_id
                && model.cfg.image_token_id.is_some(),
            "tokenizer/image-token mismatch"
        );
        let letters = letter_rows(language, &ids, model.cfg.hidden)?;
        let vision = VisionModel::load(base, adapter, library)?;
        Ok(Self {
            processor: Arc::new(VisionProcessor::new(
                tokenizer,
                model.cfg.image_token_id.unwrap(),
            )),
            vision,
            language: model,
            letters,
        })
    }

    /// Validate all tokenized prompts before the first CUDA forward of a request.
    pub fn prepare(
        &self,
        width: usize,
        height: usize,
        rgb: &[u8],
        questions: &[Question],
    ) -> Result<PreparedRequest> {
        self.processor.prepare(width, height, rgb, questions)
    }

    fn raw_score(
        &mut self,
        prompt: &ImagePrompt,
        features: &[bf16],
        options: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!((1..=26).contains(&options), "expected 1 to 26 options");
        let hidden = self.language.forward_multimodal(&MultimodalInput {
            token_ids: &prompt.token_ids,
            image_token_indices: &prompt.image_token_indices,
            image_embeddings: features,
            position_ids: [
                &prompt.position_ids[0],
                &prompt.position_ids[1],
                &prompt.position_ids[2],
            ],
        })?;
        let logits: Vec<f32> = self
            .letters
            .chunks_exact(hidden.len())
            .take(options)
            .map(|w| {
                w.iter()
                    .zip(&hidden)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        ensure!(
            logits.iter().all(|x| x.is_finite()),
            "non-finite candidate logits"
        );
        Ok((hidden, logits))
    }

    /// Diagnostic readout, including processor normalization of raw model logits.
    pub fn score(
        &mut self,
        prompt: &ImagePrompt,
        features: &[bf16],
        options: usize,
    ) -> Result<Readout> {
        let (hidden, logits) = self.raw_score(prompt, features, options)?;
        let probabilities = probabilities(&logits)?;
        Ok(Readout {
            hidden,
            logits,
            probabilities,
        })
    }

    pub fn predict(&mut self, request: &ImageRequest) -> Result<Value> {
        let prepared = self.prepare(
            request.width,
            request.height,
            &request.rgb,
            &request.questions,
        )?;
        self.predict_prepared(&prepared)
    }

    /// One admitted unit covers vision encoding and every question, keeping
    /// features request-local and model resources alive until both streams finish.
    pub fn execute_prepared(&mut self, prepared: &PreparedRequest) -> Result<Vec<Vec<f32>>> {
        let result = (|| {
            let counts: Vec<usize> = prepared.context.option_counts().collect();
            ensure!(
                prepared.prompts.len() == counts.len() && !counts.is_empty(),
                "question/prompt count mismatch"
            );
            let features = self.vision.forward(&prepared.image)?;
            prepared
                .prompts
                .iter()
                .zip(counts)
                .map(|(p, count)| {
                    self.raw_score(p, &features, count)
                        .map(|(_, logits)| logits)
                })
                .collect()
        })();
        // Also finish queued work on an error before releasing runtime admission.
        let vision_sync = self.vision.synchronize();
        let language_sync = self.language.synchronize();
        vision_sync?;
        language_sync?;
        result
    }

    pub fn predict_prepared(&mut self, prepared: &PreparedRequest) -> Result<Value> {
        let rows = self.execute_prepared(prepared)?;
        prepared.context.finish(rows)
    }
}

fn letter_rows(dir: &Path, ids: &[u32], hidden: usize) -> Result<Vec<f32>> {
    let names = [
        "embed_tokens.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    ];
    let (path, requested) = if dir.join("model.safetensors.index.json").exists() {
        let index: Value =
            serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
        names
            .iter()
            .find_map(|name| {
                index["weight_map"][name]
                    .as_str()
                    .map(|file| (dir.join(file), Some(*name)))
            })
            .context("embedding missing from index")?
    } else {
        (dir.join("model.safetensors"), None)
    };
    let file = File::open(path)?;
    // SAFETY: exported checkpoint files must remain immutable during inference.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = SafeTensors::deserialize(&map)?;
    let name = requested
        .or_else(|| names.iter().copied().find(|n| tensors.tensor(n).is_ok()))
        .context("missing embedding")?;
    let tensor = tensors.tensor(name)?;
    ensure!(
        tensor.dtype() == Dtype::BF16 && tensor.shape().len() == 2 && tensor.shape()[1] == hidden,
        "embedding shape/dtype mismatch"
    );
    let mut result = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        ensure!(
            (id as usize) < tensor.shape()[0],
            "candidate outside vocabulary"
        );
        let row = &tensor.data()[id as usize * hidden * 2..(id as usize + 1) * hidden * 2];
        result.extend(
            row.as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_le_bytes(*b).to_f32()),
        );
    }
    Ok(result)
}
