//! Complete-request validation, image transforms, tokenization and response reconstruction.
use crate::contract::{self, Question};
use anyhow::{Context, Result, ensure};
use base64::Engine as _;
use image::{ImageFormat, ImageReader, RgbImage};
use omni_qwen3_5_native::image_preprocess::{
    ImageLimits, ProcessedImage, preprocess_rgb8_with_limits,
};
use serde_json::{Map, Value, json};
use std::{path::Path, sync::Arc, time::Instant};
use tokenizers::Tokenizer;
pub const MAX_IMAGE_PIXELS: usize = 3072 * 32 * 32;
pub fn decode_images(items: &Value) -> Result<Vec<RgbImage>> {
    if items.is_null() {
        return Ok(Vec::new());
    }
    let items = items
        .as_array()
        .context("images must be a list of at most 4 base64 strings")?;
    ensure!(
        items.len() <= 4,
        "images must be a list of at most 4 base64 strings"
    );
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let mut text = item
                .as_str()
                .with_context(|| format!("image {index} is not a base64 string"))?;
            if text.starts_with("data:") {
                let (head, data) = text
                    .split_once(',')
                    .context("image is not a base64 data URL")?;
                ensure!(head.ends_with(";base64"), "image is not a base64 data URL");
                text = data;
            }
            let cleaned: String = text
                .chars()
                .filter(|c| !c.is_whitespace() && !matches!(c, '\u{1c}'..='\u{1f}'))
                .collect();
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(cleaned)
                .context("image could not be decoded")?;
            let reader = ImageReader::new(std::io::Cursor::new(&bytes)).with_guessed_format()?;
            ensure!(
                matches!(
                    reader.format(),
                    Some(ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP)
                ),
                "image {index} is not PNG, JPEG or WebP"
            );
            let (width, height) = reader.into_dimensions()?;
            ensure!(
                (width as usize)
                    .checked_mul(height as usize)
                    .is_some_and(|n| n <= MAX_IMAGE_PIXELS),
                "image {index} exceeds {MAX_IMAGE_PIXELS} pixels"
            );
            Ok(ImageReader::new(std::io::Cursor::new(bytes))
                .with_guessed_format()?
                .decode()?
                .into_rgb8())
        })
        .collect()
}
/// Qwen's image grids consume merged H/W positions; text resumes after the maximum.
pub fn image_positions(
    ids: &[u32],
    image_token: u32,
    grids: &[[usize; 3]],
) -> Result<[Vec<i64>; 3]> {
    let mut axes: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::with_capacity(ids.len()));
    let (mut cursor, mut base) = (0usize, 0i64);
    for &[t, h, w] in grids {
        ensure!(
            t == 1 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0,
            "invalid image grid"
        );
        let start = ids[cursor..]
            .iter()
            .position(|&id| id == image_token)
            .map(|i| cursor + i)
            .context("missing image placeholders")?;
        for _ in cursor..start {
            for axis in &mut axes {
                axis.push(base);
            }
            base += 1;
        }
        let (h, w) = (h / 2, w / 2);
        let count = h.checked_mul(w).context("image grid overflow")?;
        ensure!(
            start.checked_add(count).is_some_and(|end| end <= ids.len())
                && ids[start..start + count]
                    .iter()
                    .all(|&id| id == image_token),
            "image grid placeholder mismatch"
        );
        for row in 0..h {
            for col in 0..w {
                axes[0].push(base);
                axes[1].push(base + row as i64);
                axes[2].push(base + col as i64);
            }
        }
        base += h.max(w) as i64;
        cursor = start + count;
    }
    ensure!(
        ids[cursor..].iter().all(|&id| id != image_token),
        "unexpected image placeholders"
    );
    for _ in cursor..ids.len() {
        for axis in &mut axes {
            axis.push(base);
        }
        base += 1;
    }
    Ok(axes)
}
pub struct Input {
    pub ids: Vec<u32>,
    pub n_candidates: usize,
    pub images: Arc<Vec<ProcessedImage>>,
    pub image_indices: Vec<usize>,
    pub positions: [Vec<i64>; 3],
}
pub struct PreparedRequest {
    pub inputs: Vec<Input>,
    pub context: ResponseContext,
}
pub struct ResponseContext {
    questions: Vec<Question>,
    input_tokens: usize,
    temperature: f64,
    start: Instant,
}
pub struct Processor {
    tokenizer: Tokenizer,
    prefix: String,
    suffix: String,
    temperature: f64,
    mm_temperature: f64,
    image_token: u32,
    limits: ImageLimits,
}
impl Processor {
    pub fn load(dir: &Path, manifest: &Value) -> Result<Self> {
        ensure!(
            manifest["format"] == "jemm-native/1"
                && manifest["model_id"] == "JEMM"
                && manifest["base_model_id"] == "Qwen/Qwen3.8-27B"
                && manifest["base_revision"] == contract::BASE_REVISION
                && manifest["checkpoint_revision"] == contract::CHECKPOINT_REVISION
                && manifest["source_revision"] == contract::SOURCE_REVISION,
            "expected pinned JEMM native export"
        );
        ensure!(
            manifest["max_tokens"] == 8192 && manifest["max_mm_tokens"] == 3072,
            "expected JEMM token budgets"
        );
        let mut tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        tokenizer
            .with_truncation(None)
            .map_err(anyhow::Error::msg)?;
        tokenizer.with_padding(None);
        let label_ids = manifest["label_token_ids"]
            .as_array()
            .context("label_token_ids")?;
        ensure!(
            label_ids.len() == 32
                && label_ids
                    .iter()
                    .filter_map(Value::as_u64)
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    == 32,
            "expected 32 distinct label IDs"
        );
        for (label, id) in contract::LABELS.chars().zip(label_ids) {
            let encoded = tokenizer
                .encode(label.to_string(), false)
                .map_err(anyhow::Error::msg)?;
            ensure!(
                encoded.get_ids().len() == 1 && Some(encoded.get_ids()[0] as u64) == id.as_u64(),
                "label must be exactly one token matching export"
            );
        }
        let prefix = manifest["chat_prefix"]
            .as_str()
            .context("chat_prefix")?
            .to_owned();
        let suffix = manifest["chat_suffix"]
            .as_str()
            .context("chat_suffix")?
            .to_owned();
        ensure!(
            prefix.contains(contract::SYSTEM) && suffix.contains("<think>\n\n</think>"),
            "expected JEMM system and disabled-thinking template"
        );
        let calibration: Value = serde_json::from_slice(
            &std::fs::read(dir.join("decision_config.json")).context("decision_config.json")?,
        )?;
        let mut values = [0.0; 3];
        for (index, key) in ["temperature", "mm_temperature", "threshold"]
            .iter()
            .enumerate()
        {
            let value = calibration[key]
                .as_f64()
                .with_context(|| format!("calibration {key}"))?;
            ensure!(
                value.is_finite()
                    && if index < 2 {
                        value > 0.0
                    } else {
                        (0.0..=1.0).contains(&value)
                    },
                "invalid calibration {key}"
            );
            ensure!(
                manifest[key].as_f64() == Some(value),
                "manifest disagrees with decision_config.json: {key}"
            );
            values[index] = value;
        }
        let [temperature, mm_temperature, _threshold] = values;
        let config: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let image_token = config["image_token_id"]
            .as_u64()
            .and_then(|x| u32::try_from(x).ok())
            .context("image_token_id")?;
        let preprocessor: Value =
            serde_json::from_slice(&std::fs::read(dir.join("preprocessor_config.json"))?)?;
        ensure!(
            preprocessor["size"]["shortest_edge"] == 65536
                && preprocessor["size"]["longest_edge"] == 16777216,
            "expected pinned image processor size"
        );
        Ok(Self {
            tokenizer,
            prefix,
            suffix,
            temperature,
            mm_temperature,
            image_token,
            limits: ImageLimits {
                max_source_side: 32768,
                max_source_pixels: MAX_IMAGE_PIXELS,
                min_pixels: 65536,
                max_pixels: 16777216,
                max_patches: 12288,
            },
        })
    }
    pub fn prepare(&self, raw: &[u8]) -> Result<PreparedRequest> {
        let start = Instant::now();
        let request = contract::parse(raw)?;
        let questions = contract::compile(raw)?;
        let images = decode_images(request.get("images").unwrap_or(&Value::Null))?
            .into_iter()
            .map(|rgb| {
                preprocess_rgb8_with_limits(
                    rgb.width() as usize,
                    rgb.height() as usize,
                    rgb.as_raw(),
                    &self.limits,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let images = Arc::new(images);
        let grids = images.iter().map(|i| i.image_grid_thw).collect::<Vec<_>>();
        let mut inputs = Vec::new();
        let mut input_tokens = 0;
        for q in &questions {
            let image_prefix = images
                .iter()
                .map(|image| {
                    format!(
                        "<|vision_start|>{}<|vision_end|>",
                        "<|image_pad|>".repeat(image.image_tokens())
                    )
                })
                .collect::<String>();
            let chat = format!("{}{image_prefix}{}{}", self.prefix, q.prompt, self.suffix);
            let encoded = self
                .tokenizer
                .encode(chat, false)
                .map_err(anyhow::Error::msg)?;
            let ids = encoded.get_ids().to_vec();
            let budget = if images.is_empty() { 8192 } else { 3072 };
            ensure!(
                !ids.is_empty() && ids.len() <= budget,
                "UNSUPPORTED: input exceeds token budget"
            );
            let image_indices = ids
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (!images.is_empty() && v == self.image_token).then_some(i))
                .collect::<Vec<_>>();
            let positions = if images.is_empty() {
                std::array::from_fn(|_| (0..ids.len()).map(|i| i as i64).collect())
            } else {
                image_positions(&ids, self.image_token, &grids)?
            };
            input_tokens += ids.len();
            inputs.push(Input {
                ids,
                n_candidates: q.keys.len(),
                images: images.clone(),
                image_indices,
                positions,
            });
        }
        Ok(PreparedRequest {
            inputs,
            context: ResponseContext {
                questions,
                input_tokens,
                temperature: if images.is_empty() {
                    self.temperature
                } else {
                    self.mm_temperature
                },
                start,
            },
        })
    }
}
impl ResponseContext {
    pub fn finish(self, rows: Vec<Vec<f32>>) -> Result<Value> {
        ensure!(rows.len() == self.questions.len(), "invalid logit rows");
        let mut answers = Map::new();
        for (q, row) in self.questions.iter().zip(rows) {
            answers.insert(q.id.clone(), contract::answer(q, &row, self.temperature)?);
        }
        Ok(
            json!({"model":"JEMM","answers":answers,"usage":{"input_tokens":self.input_tokens,"output_tokens":0,"latency_ms":self.start.elapsed().as_secs_f64()*1000.}}),
        )
    }
}
