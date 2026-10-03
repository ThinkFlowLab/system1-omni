//! Bounded PNG/JPEG screenshot request mapping for the native worker.
use crate::contract::Question;
use anyhow::{Context, Result, ensure};
use base64::Engine;
use serde_json::{Map, Value};
use std::io::Cursor;

pub const MAX_BODY: usize = 8 * 1024 * 1024;
pub const MAX_IMAGE_BYTES: usize = 4 * 1024 * 1024;
pub const MODEL_ID: &str =
    "cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79:multimodal";

pub struct ImageRequest {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub questions: Vec<Question>,
}

pub fn parse_image_body(body: &Map<String, Value>) -> Result<ImageRequest> {
    ensure!(
        body.len() == 3
            && ["model", "state", "questions"]
                .iter()
                .all(|k| body.contains_key(*k)),
        "request must contain model, state and questions only"
    );
    let questions = body
        .get("questions")
        .and_then(Value::as_object)
        .context("questions must be an object")?;
    ensure!(
        (1..=8).contains(&questions.len()),
        "questions must contain 1 to 8 questions"
    );
    for (name, q) in questions {
        bounded_key(name)?;
        let q = q.as_object().context("question must be an object")?;
        ensure!(
            q.keys()
                .all(|k| ["type", "instructions", "criteria"].contains(&k.as_str())),
            "unsupported question fields"
        );
        let goal_len = match q.get("instructions") {
            Some(Value::Null) => 0,
            Some(v) => checked_text(v)?.chars().count(),
            None => anyhow::bail!("instructions is required"),
        };
        let criteria = q
            .get("criteria")
            .and_then(Value::as_object)
            .context("criteria must be an object")?;
        let mut length = goal_len;
        for (key, value) in criteria {
            bounded_key(key)?;
            let label = if value.is_null() {
                checked_text(&Value::String(key.clone()))?
            } else {
                checked_text(value)?
            };
            // Match Python's escaped-label character budget.
            length += crate::json::quote(&label).chars().count() - 2;
        }
        ensure!(
            length <= 16384,
            "combined question text exceeds 16384 characters"
        );
    }
    let mut mapped = body.clone();
    mapped.insert("state".into(), Value::String("image".into()));
    let (_, questions) =
        crate::contract::map_request(&mapped).map_err(|e| anyhow::anyhow!(e.message))?;
    let state = body["state"]
        .as_object()
        .context("state must contain exactly one image")?;
    ensure!(state.len() == 1, "state must contain exactly one image");
    let url = state
        .get("image")
        .and_then(Value::as_str)
        .context("state.image must be a data URL")?;
    let (prefix, encoded) = url.split_once(',').context("invalid image data URL")?;
    let format = match prefix {
        "data:image/png;base64" => image::ImageFormat::Png,
        "data:image/jpeg;base64" => image::ImageFormat::Jpeg,
        _ => anyhow::bail!("only inline PNG/JPEG images are supported"),
    };
    ensure!(
        encoded.len() <= 4 * MAX_IMAGE_BYTES.div_ceil(3),
        "encoded image exceeds 4 MiB"
    );
    let raw = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .context("invalid base64 image")?;
    ensure!(raw.len() <= MAX_IMAGE_BYTES, "image exceeds 4 MiB");
    ensure!(
        image::guess_format(&raw)? == format,
        "image format does not match MIME type"
    );
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(2048);
    limits.max_image_height = Some(2048);
    limits.max_alloc = Some(32 * 1024 * 1024);
    let mut header = image::ImageReader::with_format(Cursor::new(&raw), format);
    header.limits(limits.clone());
    let (width, height) = header.into_dimensions()?;
    let (w, h) = (width as usize, height as usize);
    ensure!(
        w > 0 && h > 0 && w <= 2048 && h <= 2048 && w * h <= 1048576 && w.max(h) <= 200 * w.min(h),
        "image dimensions exceed supported limits"
    );
    let decoded = if format == image::ImageFormat::Png {
        let decoder = image::codecs::png::PngDecoder::with_limits(Cursor::new(&raw), limits)?;
        ensure!(!decoder.is_apng()?, "image must be single-frame");
        image::DynamicImage::from_decoder(decoder)?
    } else {
        let mut reader = image::ImageReader::with_format(Cursor::new(&raw), format);
        reader.limits(limits);
        reader.decode()?
    };
    Ok(ImageRequest {
        width: w,
        height: h,
        rgb: pillow_rgb(
            decoded,
            format == image::ImageFormat::Png && raw.get(25) == Some(&0),
        ),
        questions,
    })
}

fn pillow_rgb(image: image::DynamicImage, png_grayscale: bool) -> Vec<u8> {
    use image::DynamicImage::*;
    match image {
        ImageLuma16(p) => p.pixels().flat_map(|v| [v[0].min(255) as u8; 3]).collect(),
        ImageLumaA16(p) if png_grayscale => {
            p.pixels().flat_map(|v| [v[0].min(255) as u8; 3]).collect()
        }
        ImageLumaA16(p) => p.pixels().flat_map(|v| [(v[0] >> 8) as u8; 3]).collect(),
        ImageRgb16(p) => p
            .pixels()
            .flat_map(|v| [(v[0] >> 8) as u8, (v[1] >> 8) as u8, (v[2] >> 8) as u8])
            .collect(),
        ImageRgba16(p) => p
            .pixels()
            .flat_map(|v| [(v[0] >> 8) as u8, (v[1] >> 8) as u8, (v[2] >> 8) as u8])
            .collect(),
        other => other.to_rgb8().into_raw(),
    }
}

fn bounded_key(key: &str) -> Result<()> {
    ensure!(
        (1..=256).contains(&key.chars().count()),
        "names and option keys must contain 1 to 256 characters"
    );
    Ok(())
}

fn checked_text(value: &Value) -> Result<String> {
    let text = match value {
        Value::String(s) => s.clone(),
        Value::Object(_) | Value::Array(_) => crate::json::dumps(value),
        _ => anyhow::bail!("text must be a string, object or array"),
    };
    ensure!(
        text.chars().count() <= 16384,
        "text exceeds 16384 characters"
    );
    ensure!(
        ![
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>"
        ]
        .iter()
        .any(|t| text.contains(t)),
        "unsupported media control token"
    );
    Ok(text)
}
