//! Bounded PNG/JPEG screenshot request mapping for the native worker.
use crate::contract::Question;
use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value};

pub const MAX_BODY: usize = 8 * 1024 * 1024;
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
    let decoded = omni_qwen3_5_native::image_decode::decode_data_url(url)?;
    Ok(ImageRequest {
        width: decoded.width,
        height: decoded.height,
        rgb: decoded.rgb,
        questions,
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use serde_json::json;

    fn parse_jpeg(raw: &[u8]) -> Result<ImageRequest> {
        let body = json!({
            "model": "cua-s1-4b-0.2",
            "state": {"image": format!("data:image/jpeg;base64,{}", base64::engine::general_purpose::STANDARD.encode(raw))},
            "questions": {"pick": {"type": "choice", "instructions": "Choose next action", "criteria": {"a": "Continue"}}}
        });
        parse_image_body(body.as_object().unwrap())
    }

    #[test]
    fn request_rejects_pillow_two_frame_mpo() {
        // Pillow: RGB 32x32 red.save(format="MPO", save_all=True,
        // append_images=[RGB 32x32 blue]). This is an actual two-picture file.
        let raw = base64::engine::general_purpose::STANDARD.decode(concat!(
            "/9j/4AAQSkZJRgABAQAAAQABAAD/4gBoTVBGAElJKgAIAAAAAwAAsAcABAAAADAxMDABsAQAAQAAAAIAAAACsAcAIAAAADIAAAAA",
            "AAAAAAADAO8CAAAAAAAAAAAAAAAAAACFAgAA0wIAAAAAAAAgICAgICAgICAgICAgICAg/9sAQwAIBgYHBgUIBwcHCQkICgwUDQwL",
            "CwwZEhMPFB0aHx4dGhwcICQuJyAiLCMcHCg3KSwwMTQ0NB8nOT04MjwuMzQy/9sAQwEJCQkMCwwYDQ0YMiEcITIyMjIyMjIyMjIy",
            "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy/8AAEQgAIAAgAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAA",
            "AAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJ",
            "ChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeo",
            "qaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgME",
            "BQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBka",
            "JicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2",
            "t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A4uiiivmT9xCiiigAooooAKKKKAP/",
            "2f/Y/+AAEEpGSUYAAQEAAAEAAQAA/9sAQwAIBgYHBgUIBwcHCQkICgwUDQwLCwwZEhMPFB0aHx4dGhwcICQuJyAiLCMcHCg3KSww",
            "MTQ0NB8nOT04MjwuMzQy/9sAQwEJCQkMCwwYDQ0YMiEcITIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy",
            "MjIyMjIyMjIy/8AAEQgAIAAgAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMF",
            "BQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RV",
            "VldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh",
            "4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwAB",
            "AgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2Rl",
            "ZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp",
            "6vLz9PX29/j5+v/aAAwDAQACEQMRAD8A8cooor9xPMCiiigAooooAKKKKAP/2Q==",
        )).unwrap();
        let error = parse_jpeg(&raw)
            .err()
            .expect("multi-picture input must be rejected");
        assert!(error.to_string().contains("single-frame"), "{error}");
    }

    fn jpeg() -> Vec<u8> {
        let mut raw = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut raw)
            .encode(&[20, 40, 60], 1, 1, image::ExtendedColorType::Rgb8)
            .unwrap();
        raw
    }

    fn with_segment(marker: u8, payload: &[u8]) -> Vec<u8> {
        let mut raw = jpeg();
        let mut segment = vec![0xff, marker];
        segment.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        segment.extend_from_slice(payload);
        raw.splice(2..2, segment);
        raw
    }

    #[test]
    fn request_accepts_single_jpeg_and_unrelated_metadata() {
        for raw in [
            jpeg(),
            with_segment(0xe2, b"ICC_PROFILE\0MPF\0"),
            with_segment(0xfe, b"MPF\0"),
        ] {
            let request = parse_jpeg(&raw).unwrap();
            assert_eq!((request.width, request.height), (1, 1));
            assert_eq!(request.rgb.len(), 3);
        }
    }

    #[test]
    fn request_rejects_truncated_jpeg_segment() {
        assert!(parse_jpeg(&[0xff, 0xd8, 0xff, 0xe2, 0, 20, b'M', b'P', b'F', 0]).is_err());
    }
}
