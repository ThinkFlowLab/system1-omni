//! Native screenshot-to-decision orchestration, with request-local vision reuse.
use crate::{
    contract::{self, LETTERS, Question},
    image_preprocess::{ProcessedImage, preprocess_rgb8},
    image_request::{ImageRequest, MODEL_ID},
    inputs::MultimodalInput,
    model::Model,
    multimodal::{ImagePrompt, prepare_prompt},
    vision::VisionModel,
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use serde_json::{Value, json};
use std::{fs::File, path::Path};
use tokenizers::Tokenizer;

pub struct VisionEngine {
    tokenizer: Tokenizer,
    pub vision: VisionModel,
    language: Model,
    letters: Vec<f32>,
}

pub struct PreparedRequest {
    pub image: ProcessedImage,
    pub prompts: Vec<ImagePrompt>,
    questions: Vec<Question>,
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
            tokenizer,
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
        crate::multimodal::validate_questions(questions)?;
        let image = preprocess_rgb8(width, height, rgb)?;
        let prompts = questions
            .iter()
            .map(|q| {
                prepare_prompt(
                    &self.tokenizer,
                    q,
                    image.image_grid_thw,
                    self.language.cfg.image_token_id.unwrap(),
                )
            })
            .collect::<Result<_>>()?;
        Ok(PreparedRequest {
            image,
            prompts,
            questions: questions.to_vec(),
        })
    }

    pub fn score(
        &mut self,
        prompt: &ImagePrompt,
        features: &[bf16],
        options: usize,
    ) -> Result<Readout> {
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
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exps: Vec<f64> = logits.iter().map(|&x| (x as f64 - max).exp()).collect();
        let total: f64 = exps.iter().sum();
        let probabilities = exps.iter().map(|x| (x / total) as f32).collect();
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

    pub fn predict_prepared(&mut self, prepared: &PreparedRequest) -> Result<Value> {
        let questions = &prepared.questions;
        ensure!(
            prepared.prompts.len() == questions.len() && !questions.is_empty(),
            "question/prompt count mismatch"
        );
        let features = self.vision.forward(&prepared.image)?;
        let mut answers = serde_json::Map::new();
        for (q, p) in questions.iter().zip(&prepared.prompts) {
            answers.insert(
                q.name.clone(),
                contract::answer(q, &self.score(p, &features, q.keys.len())?.probabilities),
            );
        }
        let tokens: usize = prepared.prompts.iter().map(|p| p.token_ids.len()).sum();
        Ok(
            json!({"model": MODEL_ID, "answers": answers, "usage":{"input_tokens":tokens,"output_tokens":0}}),
        )
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
