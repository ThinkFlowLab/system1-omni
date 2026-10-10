//! OmniJev's typed request contract: one image and Choice, Noul or Score questions.
//! Question shapes and option rendering follow OmniJev @ 14dbec4 (`mso/infer.py`
//! `MSO1._options`, `mso/records.py` `render_option`); see ../LICENSE.omnijev.

use std::io::Cursor;

use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use omni_qwen3_5_native::json;
use serde_json::{Map, Value};

pub const MODEL_ID: &str = "tinnel123/OmniJev";
const ALIASES: [&str; 2] = ["omnijev", "omnijev-4b-v1.1"];

pub const MAX_BODY: usize = 12 * 1024 * 1024;
pub const MAX_IMAGE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_IMAGE_SIDE: usize = 4096;
pub const MAX_QUESTIONS: usize = 64;
pub const MAX_OPTIONS: usize = 255;
pub const MAX_ROWS: usize = 1024;
pub const MAX_TEXT_CHARS: usize = 8192;
pub const MAX_ID_CHARS: usize = 256;
/// Rendered option text is cut to this many characters, as the reference does.
pub const OPTION_CHARS: usize = 200;

/// Text the reference would read as image or option markers, which would move
/// the readout positions; requests containing it are refused.
pub const RESERVED_TOKENS: [&str; 6] = [
    "<|opt|>",
    "<|/opt|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

impl Kind {
    /// The decision head's question-type index.
    pub fn type_id(self) -> usize {
        match self {
            Kind::Noul => 0,
            Kind::Choice => 1,
            Kind::Score => 2,
        }
    }
}

pub struct Question {
    pub id: String,
    pub kind: Kind,
    pub instructions: String,
    /// Answer keys, one per option, in caller order.
    pub keys: Vec<String>,
    /// Each option's text as written between the option markers.
    pub options: Vec<String>,
}

/// The request's image: its encoded bytes and the dimensions from its header.
/// `processing::pixels` decodes it.
pub struct Image {
    pub format: image::ImageFormat,
    pub width: usize,
    pub height: usize,
    pub bytes: Vec<u8>,
}

pub struct Request {
    pub image: Image,
    pub questions: Vec<Question>,
}

pub fn compile(raw: &[u8]) -> Result<Request> {
    ensure!(raw.len() <= MAX_BODY, "request body exceeds 12 MiB");
    let request = json::parse(raw).map_err(anyhow::Error::msg)?;
    ensure!(
        request
            .keys()
            .all(|k| ["model", "state", "questions"].contains(&k.as_str())),
        "request fields must be model, state and questions"
    );
    ensure!(
        match request.get("model") {
            None | Some(Value::Null) => true,
            Some(Value::String(model)) => model == MODEL_ID || ALIASES.contains(&model.as_str()),
            Some(_) => false,
        },
        "requested model is not loaded"
    );
    let image = image(request.get("state").context("request requires state")?)?;
    let questions = request
        .get("questions")
        .and_then(Value::as_object)
        .context("questions must be an object")?;
    ensure!(
        (1..=MAX_QUESTIONS).contains(&questions.len()),
        "questions must contain 1 to {MAX_QUESTIONS} questions"
    );
    let questions = questions
        .iter()
        .map(|(id, q)| {
            ensure!(
                id.chars().count() <= MAX_ID_CHARS,
                "question ids must be at most {MAX_ID_CHARS} characters"
            );
            question(id, q).with_context(|| format!("question {id}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let rows: usize = questions.iter().map(|q| q.options.len()).sum();
    ensure!(
        rows <= MAX_ROWS,
        "request exceeds {MAX_ROWS} options in all"
    );
    Ok(Request { image, questions })
}

fn image(state: &Value) -> Result<Image> {
    let state = state
        .as_object()
        .context("state must be an object with images")?;
    ensure!(
        state.keys().all(|k| k == "images"),
        "state supports images only"
    );
    let images = state
        .get("images")
        .and_then(Value::as_array)
        .context("state.images must be an array")?;
    ensure!(
        images.len() == 1,
        "state.images must contain exactly one image"
    );
    let url = images[0].as_str().context("an image must be a data URL")?;
    let (prefix, encoded) = url.split_once(',').context("invalid image data URL")?;
    let format = match prefix {
        "data:image/png;base64" => image::ImageFormat::Png,
        "data:image/jpeg;base64" => image::ImageFormat::Jpeg,
        _ => bail!("only inline PNG/JPEG images are supported"),
    };
    ensure!(
        encoded.len() <= 4 * MAX_IMAGE_BYTES.div_ceil(3),
        "image exceeds 8 MiB"
    );
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .context("invalid base64 image")?;
    ensure!(bytes.len() <= MAX_IMAGE_BYTES, "image exceeds 8 MiB");
    ensure!(
        image::guess_format(&bytes).ok() == Some(format),
        "image format does not match its MIME type"
    );
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_IMAGE_SIDE as u32);
    limits.max_image_height = Some(MAX_IMAGE_SIDE as u32);
    let mut reader = image::ImageReader::with_format(Cursor::new(&bytes), format);
    reader.limits(limits);
    let (width, height) = reader.into_dimensions().context("invalid image")?;
    let (width, height) = (width as usize, height as usize);
    ensure!(
        width > 0 && height > 0 && width <= MAX_IMAGE_SIDE && height <= MAX_IMAGE_SIDE,
        "image must be at most {MAX_IMAGE_SIDE} pixels per side"
    );
    // The processor refuses aspect ratios above 200.
    ensure!(
        width.max(height) <= 200 * width.min(height),
        "image aspect ratio must be at most 200"
    );
    Ok(Image {
        format,
        width,
        height,
        bytes,
    })
}

fn text<'a>(value: Option<&'a Value>, what: &str) -> Result<&'a str> {
    let text = value
        .and_then(Value::as_str)
        .with_context(|| format!("{what} must be a string"))?;
    ensure!(
        text.chars().count() <= MAX_TEXT_CHARS,
        "{what} exceeds {MAX_TEXT_CHARS} characters"
    );
    Ok(text)
}

/// Python's truthiness for the JSON values a question may carry.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `region [x1,y1,x2,y2]`, each coordinate rounded as Python's `round` does.
fn region(value: &Value) -> Result<String> {
    let corners = value
        .as_object()
        .filter(|r| r.len() == 1)
        .and_then(|r| r.get("box"))
        .and_then(Value::as_array)
        .filter(|b| b.len() == 4)
        .context("a region must be {\"box\": [x1, y1, x2, y2]}")?;
    let mut rounded = Vec::with_capacity(4);
    for v in corners {
        let v = v.as_f64().context("region coordinates must be numbers")?;
        ensure!(v.abs() <= 1e9, "region coordinates must be within ±1e9");
        rounded.push(v.round_ties_even() as i64);
    }
    Ok(format!(
        "region [{},{},{},{}]",
        rounded[0], rounded[1], rounded[2], rounded[3]
    ))
}

fn cut(text: &str) -> String {
    text.chars().take(OPTION_CHARS).collect()
}

fn question(id: &str, q: &Value) -> Result<Question> {
    let q = q.as_object().context("a question must be an object")?;
    let kind = match q.get("type").and_then(Value::as_str) {
        Some("noul") => Kind::Noul,
        Some("choice") => Kind::Choice,
        Some("score") => Kind::Score,
        _ => bail!("question type must be choice, score, or noul"),
    };
    let instructions = text(q.get("instructions"), "instructions")?.to_owned();
    let (keys, options) = match kind {
        Kind::Noul => {
            ensure!(
                q.keys()
                    .all(|k| ["type", "instructions", "region"].contains(&k.as_str())),
                "Noul supports type, instructions and region"
            );
            let option = if truthy(q.get("region")) {
                region(&q["region"])?
            } else {
                String::new()
            };
            (vec!["yes".to_owned()], vec![option])
        }
        Kind::Score => {
            ensure!(
                q.keys()
                    .all(|k| ["type", "instructions", "levels", "criteria"].contains(&k.as_str())),
                "Score supports type, instructions, levels and criteria"
            );
            ensure!(
                !(q.contains_key("levels") && q.contains_key("criteria")),
                "Score takes levels or criteria, not both"
            );
            let levels: Vec<String> = if q.contains_key("levels") {
                q["levels"]
                    .as_array()
                    .context("levels must be an array of strings")?
                    .iter()
                    .map(|l| text(Some(l), "a level").map(str::to_owned))
                    .collect::<Result<_>>()?
            } else {
                q.get("criteria")
                    .and_then(Value::as_object)
                    .context("Score requires levels, or criteria as an object keyed by level")?
                    .keys()
                    .map(|k| {
                        ensure!(
                            k.chars().count() <= MAX_TEXT_CHARS,
                            "a level exceeds {MAX_TEXT_CHARS} characters"
                        );
                        Ok(k.clone())
                    })
                    .collect::<Result<_>>()?
            };
            let options = levels.iter().map(|l| cut(l)).collect();
            (levels, options)
        }
        Kind::Choice => {
            ensure!(
                q.keys()
                    .all(|k| ["type", "instructions", "options", "criteria"].contains(&k.as_str())),
                "Choice supports type, instructions, options and criteria"
            );
            ensure!(
                !(q.contains_key("options") && q.contains_key("criteria")),
                "Choice takes options or criteria, not both"
            );
            if q.contains_key("options") {
                choice_options(&q["options"])?
            } else {
                choice_criteria(q.get("criteria"))?
            }
        }
    };
    ensure!(
        (1..=MAX_OPTIONS).contains(&options.len()),
        "a question needs 1 to {MAX_OPTIONS} options"
    );
    for (i, key) in keys.iter().enumerate() {
        ensure!(!keys[..i].contains(key), "answer key {key:?} is repeated");
    }
    for text in std::iter::once(&instructions).chain(&options) {
        if let Some(token) = RESERVED_TOKENS.iter().find(|t| text.contains(*t)) {
            bail!("text must not contain {token}");
        }
    }
    Ok(Question {
        id: id.to_owned(),
        kind,
        instructions,
        keys,
        options,
    })
}

/// The reference's option list: `key`, `text`, `region` and `abstain` entries.
fn choice_options(list: &Value) -> Result<(Vec<String>, Vec<String>)> {
    let list = list.as_array().context("options must be an array")?;
    ensure!(list.len() <= 4 * MAX_OPTIONS, "too many options");
    let (mut keys, mut options) = (Vec::new(), Vec::new());
    for (i, o) in list.iter().enumerate() {
        let o: &Map<String, Value> = o.as_object().context("an option must be an object")?;
        ensure!(
            o.keys()
                .all(|k| ["key", "text", "region", "abstain"].contains(&k.as_str())),
            "an option supports key, text, region and abstain"
        );
        ensure!(
            matches!(o.get("abstain"), None | Some(Value::Bool(_))),
            "abstain must be a boolean"
        );
        if truthy(o.get("abstain")) {
            continue; // the reference's abstain probability is the head's own
        }
        for field in ["key", "text"] {
            if o.contains_key(field) {
                text(o.get(field), field)?;
            }
        }
        let key = [o.get("key"), o.get("text")]
            .into_iter()
            .find(|v| truthy(*v))
            .and_then(|v| v.and_then(Value::as_str))
            .map_or_else(|| format!("option_{i}"), str::to_owned);
        let option = if o.contains_key("region") {
            region(&o["region"])?
        } else {
            cut(o.get("text").and_then(Value::as_str).unwrap_or(""))
        };
        keys.push(key);
        options.push(option);
    }
    Ok((keys, options))
}

/// Jev's criteria map: a key alone, or `key: rubric`.
fn choice_criteria(criteria: Option<&Value>) -> Result<(Vec<String>, Vec<String>)> {
    let criteria = criteria
        .and_then(Value::as_object)
        .context("Choice requires options or criteria")?;
    let mut options = Vec::with_capacity(criteria.len());
    for (key, rubric) in criteria {
        let rubric = match rubric {
            Value::Null => "",
            Value::String(s) => s.as_str(),
            _ => bail!("criteria values must be strings or null"),
        };
        ensure!(
            key.chars().count() + rubric.chars().count() <= MAX_TEXT_CHARS,
            "criteria exceed {MAX_TEXT_CHARS} characters"
        );
        options.push(if rubric.is_empty() {
            cut(key)
        } else {
            cut(&format!("{key}: {rubric}"))
        });
    }
    Ok((criteria.keys().cloned().collect(), options))
}

/// Post-hoc calibration saved with the heads (`head_meta.json`).
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// Noul, Choice and Score temperatures.
    pub temperatures: [f64; 3],
    /// Noul's log-odds bias, applied before its temperature.
    pub noul_bias: f64,
}

impl Calibration {
    /// `MSO1._scale`: `p^(1/T)`, renormalized.
    fn scale(&self, kind: Kind, values: &[f64]) -> Vec<f64> {
        let t = self.temperatures[kind.type_id()];
        if (t - 1.0).abs() < 1e-6 {
            return values.to_vec();
        }
        let w: Vec<f64> = values.iter().map(|&x| x.max(1e-9).powf(1.0 / t)).collect();
        let z: f64 = w.iter().sum();
        w.iter().map(|x| x / z).collect()
    }
}

/// Python's `round(x, 4)`: the nearest four-decimal value, ties to even.
pub fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

/// Jev's confidence `(K p_max - 1) / (K - 1)`, in float32 as the reference computes it.
fn confidence(probs: &[f32]) -> f64 {
    let k = probs.len();
    if k < 2 {
        return 1.0;
    }
    let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (((k as f32) * max - 1.0) / (k as f32 - 1.0)).clamp(0.0, 1.0) as f64
}

/// The first maximum, as Python's `max(range(n), key=...)` picks it.
fn first_max(values: &[f64]) -> usize {
    (1..values.len()).fold(0, |m, i| if values[i] > values[m] { i } else { m })
}

/// `MSO1._finish`: one question's answer from its head output `mu`.
pub fn answer(
    question: &Question,
    mu: &[f32],
    calibration: &Calibration,
    latency: f64,
) -> Result<Value> {
    ensure!(
        !mu.is_empty() && mu.len() == question.keys.len() && mu.iter().all(|v| v.is_finite()),
        "invalid head output"
    );
    ensure!(
        calibration
            .temperatures
            .iter()
            .all(|t| t.is_finite() && *t > 0.0)
            && calibration.noul_bias.is_finite(),
        "invalid calibration"
    );
    let latency = round4(latency);
    if question.kind == Kind::Noul {
        let mut p0 = mu[0] as f64;
        if calibration.noul_bias != 0.0 {
            p0 = p0.clamp(1e-9, 1.0 - 1e-9);
            let z = ((p0 / (1.0 - p0)).ln() + calibration.noul_bias).clamp(-40.0, 40.0);
            p0 = 1.0 / (1.0 + (-z).exp());
        }
        let p = calibration.scale(Kind::Noul, &[p0, 1.0 - p0])[0];
        return Ok(serde_json::json!({"noul": round4(p), "latency_s": latency}));
    }
    // choice_outputs, in float32
    let sum = mu.iter().fold(0.0f32, |a, &x| a + x);
    let valid = sum <= 1.0 + 1e-5;
    let (probs, abstain): (Vec<f32>, f32) = if question.kind == Kind::Score {
        (mu.iter().map(|&x| x / sum.max(1e-6)).collect(), 0.0)
    } else if valid {
        (mu.to_vec(), (1.0 - sum).max(0.0))
    } else {
        (mu.iter().map(|&x| x / sum).collect(), 0.0)
    };
    let mut full: Vec<f64> = probs.iter().map(|&p| p as f64).collect();
    full.push(abstain as f64);
    let full = calibration.scale(question.kind, &full);
    // The reference stores the scaled values back into float32 tensors.
    let probs: Vec<f32> = full[..full.len() - 1].iter().map(|&p| p as f32).collect();
    let abstain = full[full.len() - 1] as f32;
    let mut probs64: Vec<f64> = probs.iter().map(|&p| p as f64).collect();
    let probabilities = |values: &[f64]| -> Map<String, Value> {
        question
            .keys
            .iter()
            .zip(values)
            .map(|(k, &p)| (k.clone(), serde_json::json!(round4(p))))
            .collect()
    };
    if question.kind == Kind::Score {
        let total: f64 = probs64.iter().sum();
        let total = if total == 0.0 { 1.0 } else { total };
        probs64.iter_mut().for_each(|p| *p /= total);
        let mode = first_max(&probs64);
        let as_f32: Vec<f32> = probs64.iter().map(|&p| p as f32).collect();
        return Ok(serde_json::json!({
            "score": question.keys[mode], "probabilities": probabilities(&probs64),
            "confidence": round4(confidence(&as_f32)), "latency_s": latency}));
    }
    let mode = first_max(&probs64);
    let mut with_abstain = probs.clone();
    with_abstain.push(abstain);
    Ok(serde_json::json!({
        "choice": question.keys[mode], "probabilities": probabilities(&probs64),
        "abstain": round4(abstain as f64), "valid": valid,
        "confidence": round4(confidence(&with_abstain)), "latency_s": latency}))
}
