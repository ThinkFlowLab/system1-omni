//! JEV-27B-VL System-1 request contract and prompt compilation.
//!
//! The request is the official /v1/decide shape ({kind, state, question,
//! options}); the raw System-1 prompt mirrors serve_decide.py's `s1_pass`:
//! `[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n{lines}\n[decision]:`
//! with Python `json.dumps(..., ensure_ascii=False)` rendering for structured
//! state parts. Rejections carry the official response envelope
//! ({"error": {message, type, param, code}}) with the upstream status codes, so
//! the frozen boundary probes compare equal under digit normalization.

use omni_qwen3_5_native::json;
use serde_json::{Map, Value, json};

pub const MODEL_ID: &str = "autotrust/JEV-27B-VL";
pub const PROTOCOL: &str = "jev27-bare-v1";
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// A rejection keeping the official status code and body envelope.
#[derive(Debug)]
pub struct Reject {
    pub status: u16,
    pub body: Value,
}

impl Reject {
    fn new(status: u16, message: String, type_: &str, param: Option<&str>) -> Self {
        let mut error = json!({"message": message, "type": type_});
        if let Some(param) = param {
            error["param"] = json!(param);
        }
        error["code"] = json!(status);
        Self {
            status,
            body: json!({"error": error}),
        }
    }

    /// serve_decide.py's `_err`: semantic rejections from the decide layer.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(400, message.into(), "BadRequestError", None)
    }

    /// The upstream pydantic-validation envelope ("Bad Request").
    pub fn validation(message: String, param: &str) -> Self {
        Self::new(400, message, "Bad Request", Some(param))
    }

    fn missing(field: &str, items: usize) -> Self {
        Self::validation(
            format!(
                "1 validation error:\n  {{'type': 'missing', 'loc': 'body.{field}', 'msg': 'Field required', 'input': '<dict of {items} items>'}}"
            ),
            field,
        )
    }

    pub fn unknown_model(model: &str) -> Self {
        Self::new(
            404,
            format!("The model `{model}` does not exist."),
            "NotFoundError",
            Some("model"),
        )
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Noul,
    Score,
    Choice,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Noul => "noul",
            Kind::Score => "score",
            Kind::Choice => "choice",
        }
    }
}

#[derive(Clone, Debug)]
pub enum Part {
    Text(String),
    Image(String),
}

#[derive(Debug)]
pub struct Compiled {
    pub kind: Kind,
    pub parts: Vec<Part>,
    pub question: String,
    pub options: Vec<String>,
    /// The raw System-1 prompt; image parts become the placeholder text.
    pub prompt: String,
    pub images: Vec<String>,
}

/// Python repr() of a JSON scalar, for pydantic-style error interpolation.
fn py_repr(value: &Value) -> String {
    match value {
        Value::String(s) => {
            if !s.contains('\'') {
                format!("'{s}'")
            } else {
                format!("\"{}\"", s.replace('"', "\\\""))
            }
        }
        Value::Null => "None".into(),
        other => json::dumps(other),
    }
}

/// `json.dumps(value, ensure_ascii=False)` for dict/other state parts.
fn render(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_owned();
    }
    json::dumps(value)
}

/// serve_decide.py's `_parts`: state -> text/image parts + has_image.
pub fn parts(state: &Value) -> (Vec<Part>, Vec<String>) {
    match state {
        Value::String(text) => (vec![Part::Text(text.clone())], Vec::new()),
        Value::Object(_) => (vec![Part::Text(render(state))], Vec::new()),
        Value::Array(list) => {
            let mut out = Vec::with_capacity(list.len());
            let mut images = Vec::new();
            for p in list {
                match p {
                    Value::String(text) => out.push(Part::Text(text.clone())),
                    Value::Object(map) if map.contains_key("image") => {
                        let url = map["image"].as_str().unwrap_or_default().to_owned();
                        images.push(url);
                        out.push(Part::Image(images.last().unwrap().clone()));
                    }
                    Value::Object(map)
                        if map.get("type").and_then(Value::as_str) == Some("image_url") =>
                    {
                        let url = map
                            .get("image_url")
                            .and_then(|v| v.get("url"))
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned();
                        images.push(url.clone());
                        out.push(Part::Image(url));
                    }
                    Value::Object(map)
                        if map.get("type").and_then(Value::as_str) == Some("text") =>
                    {
                        out.push(Part::Text(
                            map.get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                        ));
                    }
                    other => out.push(Part::Text(render(other))),
                }
            }
            (out, images)
        }
        _ => (vec![Part::Text(render(state))], Vec::new()),
    }
}

/// Validate and compile a request; mirrors the /v1/decide entrance checks.
/// `labels` is the exported single-token option-label list (its length is the
/// advertised option bound, 256).
pub fn compile(raw: &[u8], labels: &[String]) -> Result<Compiled, Reject> {
    let request: Map<String, Value> = match json::parse(raw) {
        Ok(map) => map,
        Err(_) => {
            return Err(Reject::validation(
                format!(
                    "1 validation error:\n  {{'type': 'model_attributes_type', 'loc': 'body', 'msg': 'Input should be a valid dictionary or object to extract fields from', 'input': '<bytes of {} bytes>'}}",
                    raw.len()
                ),
                "body",
            ));
        }
    };
    let items = request.len();
    if let Some(model) = request.get("model").and_then(Value::as_str)
        && model != MODEL_ID
    {
        return Err(Reject::unknown_model(model));
    }
    let kind = match request.get("kind") {
        None => return Err(Reject::missing("kind", items)),
        Some(v) => match v.as_str() {
            Some("noul") => Kind::Noul,
            Some("score") => Kind::Score,
            Some("choice") => Kind::Choice,
            Some(_) => {
                return Err(Reject::validation(
                    format!(
                        "1 validation error:\n  {{'type': 'literal_error', 'loc': 'body.kind', 'msg': \"Input should be 'noul', 'score' or 'choice'\", 'input': {}, 'ctx': {{'expected': \"'noul', 'score' or 'choice'\"}}}}",
                        py_repr(v)
                    ),
                    "kind",
                ));
            }
            None => {
                return Err(Reject::validation(
                    format!(
                        "1 validation error:\n  {{'type': 'literal_error', 'loc': 'body.kind', 'msg': \"Input should be 'noul', 'score' or 'choice'\", 'input': {}, 'ctx': {{'expected': \"'noul', 'score' or 'choice'\"}}}}",
                        py_repr(v)
                    ),
                    "kind",
                ));
            }
        },
    };
    let question = match request.get("question") {
        None => return Err(Reject::missing("question", items)),
        Some(Value::String(q)) => q.clone(),
        _ => {
            return Err(Reject::bad_request("question must be a string".to_owned()));
        }
    };
    if let Some(thinking) = request.get("thinking") {
        let thinking = thinking
            .as_str()
            .ok_or_else(|| Reject::bad_request("thinking must be a string"))?;
        match thinking {
            "default" | "off" => {}
            "auto" | "on" => {
                if kind == Kind::Score {
                    return Err(Reject::bad_request(
                        "thinking is supported for noul and choice",
                    ));
                }
                return Err(Reject::bad_request(
                    "adaptive thinking is not implemented by this worker",
                ));
            }
            _ => {
                return Err(Reject::bad_request(format!(
                    "thinking must be one of default, off, auto, on; got {thinking:?}"
                )));
            }
        }
    }
    if request
        .get("system2_only")
        .is_some_and(|value| value != &Value::Bool(false))
    {
        return Err(Reject::bad_request(
            "system2_only is not implemented by this worker",
        ));
    }
    if let Some(strategy) = request.get("strategy") {
        let strategy = strategy
            .as_str()
            .ok_or_else(|| Reject::bad_request("strategy must be a string"))?;
        if !matches!(strategy, "auto" | "single") {
            return Err(Reject::bad_request(format!(
                "strategy {strategy:?} is not implemented by this worker"
            )));
        }
    }
    let options: Vec<String> = match kind {
        Kind::Noul => vec!["false".into(), "true".into()],
        Kind::Score => (0..6).map(|i| i.to_string()).collect(),
        Kind::Choice => match request.get("options") {
            Some(Value::Array(list)) => {
                let mut opts = Vec::with_capacity(list.len());
                for (i, v) in list.iter().enumerate() {
                    match v.as_str() {
                        Some(o) => opts.push(o.to_owned()),
                        None => {
                            return Err(Reject::bad_request(format!(
                                "options[{i}] must be a string"
                            )));
                        }
                    }
                }
                opts
            }
            Some(Value::Null) | None => Vec::new(),
            Some(_) => {
                return Err(Reject::bad_request("options must be a list of strings"));
            }
        },
    };
    if kind == Kind::Choice && !(2..=labels.len()).contains(&options.len()) {
        return Err(Reject::bad_request(format!(
            "choice needs 2-{} options, got {}",
            labels.len(),
            options.len()
        )));
    }
    let state = request
        .get("state")
        .cloned()
        .unwrap_or(Value::String(String::new()));
    let (parts, images) = parts(&state);
    let lines: Vec<String> = match kind {
        Kind::Choice => options
            .iter()
            .enumerate()
            .map(|(i, o)| format!("{}) {o}", labels[i]))
            .collect(),
        _ => options.clone(),
    };
    let mut prompt = format!("[kind] {}\n[state] ", kind.as_str());
    for part in &parts {
        match part {
            Part::Text(t) => prompt.push_str(t),
            Part::Image(_) => prompt.push_str(IMAGE_PLACEHOLDER),
        }
    }
    prompt.push_str(&format!(
        "\n[question] {question}\n[options]\n{}\n[decision]:",
        lines.join("\n")
    ));
    Ok(Compiled {
        kind,
        parts,
        question,
        options,
        prompt,
        images,
    })
}

/// The prompt tail after the last image placeholder, rebuilt exactly as
/// `compile` would have it. The L1 hit path tokenizes only this suffix;
/// `<|vision_end|>` is a special token (always atomic to BPE), so the split
/// concatenation equals a whole-prompt tokenization bit for bit.
/// None when the state carries no image (=> no prefix structure).
pub fn tail_after_last_image(compiled: &Compiled, labels: &[String]) -> Option<String> {
    let idx = compiled
        .parts
        .iter()
        .rposition(|p| matches!(p, Part::Image(_)))?;
    let mut tail = String::new();
    for part in &compiled.parts[idx + 1..] {
        match part {
            Part::Text(t) => tail.push_str(t),
            Part::Image(_) => return None,
        }
    }
    let lines: Vec<String> = match compiled.kind {
        Kind::Choice => compiled
            .options
            .iter()
            .enumerate()
            .map(|(i, o)| format!("{}) {o}", labels[i]))
            .collect(),
        _ => compiled.options.clone(),
    };
    tail.push_str(&format!(
        "\n[question] {}\n[options]\n{}\n[decision]:",
        compiled.question,
        lines.join("\n")
    ));
    Some(tail)
}

/// The /v1/systemone answer, field-aligned with the official /v1/decide body.
pub fn answer(
    kind: Kind,
    options: &[String],
    probabilities: &[f64],
    input_tokens: usize,
    elapsed_seconds: f64,
) -> Value {
    let k = (1..probabilities.len()).fold(0, |m, i| {
        if probabilities[i] > probabilities[m] {
            i
        } else {
            m
        }
    });
    let adaptation = if kind != Kind::Choice || options.len() <= 16 {
        "native"
    } else {
        "single"
    };
    json!({
        "kind": kind.as_str(),
        "effective_kind": kind.as_str(),
        "options": options,
        "probabilities": probabilities,
        "choice_index": k,
        "choice": options[k],
        "adaptation": adaptation,
        "protocol": PROTOCOL,
        "model": MODEL_ID,
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": 1,
            "total_tokens": input_tokens + 1,
        },
        "elapsed_seconds": elapsed_seconds,
        "num_model_requests": 1,
    })
}
