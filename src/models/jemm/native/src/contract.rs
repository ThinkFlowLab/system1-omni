//! JEMM's ordered single-question prompts and calibrated decision responses.
//! Adapted from ypcypc/JEMM @ 6822fe0 (Apache-2.0).
use anyhow::{Context, Result, bail, ensure};
use omni_qwen3_5_native::json;
use serde_json::{Map, Value, json};

pub const MAX_BODY_BYTES: usize = 16_000_000;
pub const MAX_QUESTIONS: usize = 64;
pub const MAX_COMPILED_PROMPT_BYTES: usize = 16 * 1024 * 1024;
pub const LABELS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345";
pub const SYSTEM: &str = "Choose the best available candidate for the question using only the supplied state. Return exactly one candidate label.";
pub const BASE_REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
pub const CHECKPOINT_REVISION: &str = "76e3c209e8441fa658221c7ba2725bad2f811176";
pub const SOURCE_REVISION: &str = "6822fe0fd53c5e6670af6ba99fb2c857a661e532";
#[derive(Debug)]
pub struct Question {
    pub id: String,
    pub kind: String,
    pub keys: Vec<String>,
    pub prompt: String,
}
pub fn flat(s: &str) -> String {
    s.split(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}
/// Validate integer lexemes before serde can coerce oversized integers to f64.
/// Integer -0 is normalized to Python's integer 0; floating -0.0 stays negative.
pub fn normalize_json_integers(raw: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(raw.len());
    let (mut cursor, mut inside, mut escaped) = (0usize, false, false);
    while cursor < raw.len() {
        let byte = raw[cursor];
        if inside {
            out.push(byte);
            cursor += 1;
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                inside = false;
            }
            continue;
        }
        if byte == b'"' {
            inside = true;
            out.push(byte);
            cursor += 1;
            continue;
        }
        if byte == b'-' || byte.is_ascii_digit() {
            let start = cursor;
            while cursor < raw.len()
                && !matches!(
                    raw[cursor],
                    b' ' | b'\t' | b'\r' | b'\n' | b',' | b']' | b'}' | b':'
                )
            {
                cursor += 1;
            }
            let token = &raw[start..cursor];
            let digits = token.strip_prefix(b"-").unwrap_or(token);
            if !digits.is_empty() && digits.iter().all(u8::is_ascii_digit) {
                let text = std::str::from_utf8(token)?;
                ensure!(
                    if token[0] == b'-' {
                        text.parse::<i64>().is_ok()
                    } else {
                        text.parse::<u64>().is_ok()
                    },
                    "integer outside i64/u64 range is unsupported"
                );
                if token == b"-0" {
                    out.push(b'0');
                } else {
                    out.extend(token);
                }
            } else {
                out.extend(token);
            }
            continue;
        }
        out.push(byte);
        cursor += 1;
    }
    Ok(out)
}
pub fn parse(raw: &[u8]) -> Result<Map<String, Value>> {
    ensure!(
        raw.len() <= MAX_BODY_BYTES,
        "request exceeds 16000000-byte body limit"
    );
    json::parse(&normalize_json_integers(raw)?).map_err(anyhow::Error::msg)
}
/// Preserve Python float formatting and insertion order, with compact separators.
pub fn compact(v: &Value) -> String {
    let spaced = json::dumps(v);
    let mut out = String::new();
    let (mut inside, mut escape) = (false, false);
    for c in spaced.chars() {
        if inside {
            out.push(c);
            if escape {
                escape = false;
            } else if c == '\\' {
                escape = true;
            } else if c == '"' {
                inside = false;
            }
        } else if c == '"' {
            inside = true;
            out.push(c);
        } else if c != ' ' {
            out.push(c);
        }
    }
    out
}
fn py_repr(v: &Value) -> String {
    match v {
        Value::String(s) => {
            let quote = if s.contains('\'') && !s.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut out = String::from(quote);
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == quote => {
                        out.push('\\');
                        out.push(c);
                    }
                    c if !crate::unicode::is_printable(c) => {
                        let n = c as u32;
                        out.push_str(&if n <= 255 {
                            format!("\\x{n:02x}")
                        } else if n <= 65535 {
                            format!("\\u{n:04x}")
                        } else {
                            format!("\\U{n:08x}")
                        });
                    }
                    c => out.push(c),
                }
            }
            out.push(quote);
            out
        }
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}: {}", py_repr(&Value::String(k.clone())), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => py_str(v),
    }
}
pub fn py_str(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        Value::Number(_) => json::dumps(v),
        _ => py_repr(v),
    }
}
fn description(spec: &Value) -> String {
    if spec.is_object() {
        py_str(spec.get("description").unwrap_or(&json!("")))
    } else if spec.is_null() {
        String::new()
    } else {
        py_str(spec)
    }
}
fn render_tool(text: &str) -> Result<String> {
    // Inspect syntax and the name key without interpreting numeric values.
    // RawValue preserves 1e400 even under workspace arbitrary_precision.
    let Ok(raw) = serde_json::from_str::<Box<serde_json::value::RawValue>>(text) else {
        return Ok(flat(text));
    };
    let Ok(tool) = serde_json::from_str::<
        std::collections::HashMap<String, Box<serde_json::value::RawValue>>,
    >(raw.get()) else {
        return Ok(flat(text));
    };
    if !tool.contains_key("name") {
        return Ok(flat(text));
    }
    let tool = parse(text.as_bytes())?;
    let mut result = flat(&py_str(&tool["name"]));
    let summary = flat(&py_str(tool.get("description").unwrap_or(&json!(""))));
    if !summary.is_empty() {
        result.push_str(" — ");
        result.push_str(&summary);
    }
    if let Some(params) = tool.get("parameters").and_then(Value::as_object) {
        let props = params
            .get("properties")
            .and_then(Value::as_object)
            .unwrap_or(params);
        let required = params.get("required").and_then(Value::as_array);
        let mut names = Vec::new();
        for (key, spec) in props {
            if ["properties", "required", "type"].contains(&key.as_str()) && !spec.is_object() {
                continue;
            }
            let mut piece = key.clone();
            if spec.is_object() {
                if required.is_some_and(|a| a.iter().any(|v| v.as_str() == Some(key))) {
                    piece.push('*');
                }
                if let Some(default) = spec
                    .get("default")
                    .filter(|v| !v.is_null() && v.as_str() != Some(""))
                {
                    piece.push('=');
                    piece.extend(flat(&py_str(default)).chars().take(24));
                }
            }
            names.push(piece);
        }
        if !names.is_empty() {
            result.push_str(" | params: ");
            result.push_str(&names.join(", "));
        }
    }
    Ok(result)
}
pub fn compile(raw: &[u8]) -> Result<Vec<Question>> {
    let request = parse(raw)?;
    ensure!(
        matches!(request.get("model"), None | Some(Value::Null))
            || request["model"]
                .as_str()
                .is_some_and(|m| ["JEMM", "jemm", "MaestroYan/JEMM"].contains(&m)),
        "requested model is not loaded"
    );
    let questions = request
        .get("questions")
        .and_then(Value::as_object)
        .context("no questions")?;
    ensure!(
        !questions.is_empty() && questions.len() <= MAX_QUESTIONS,
        "questions must contain 1 to 64 entries"
    );
    let state = match request.get("state") {
        None => std::borrow::Cow::Borrowed(""),
        Some(Value::String(s)) => std::borrow::Cow::Borrowed(s.as_str()),
        Some(v) => std::borrow::Cow::Owned(compact(v)),
    };
    let state = state.trim_matches(|c: char| c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}'));
    let mut out = Vec::with_capacity(questions.len());
    let mut prompt_bytes = 0;
    for (id, q) in questions {
        ensure!(q.is_object(), "question {id:?} is not an object");
        let kind = match q.get("type") {
            None => "choice",
            Some(v) => v.as_str().context("question type must be a string")?,
        };
        let criteria = q.get("criteria").unwrap_or(&Value::Null);
        let mut candidates: Vec<(String, String, bool)> = Vec::new();
        match kind {
            "choice" => {
                let map = criteria
                    .as_object()
                    .context("choice question needs a criteria map with at least two options")?;
                ensure!(
                    (2..=32).contains(&map.len()),
                    "UNSUPPORTED: candidate count"
                );
                for (key, spec) in map {
                    let tool = spec
                        .get("action")
                        .and_then(Value::as_object)
                        .is_some_and(|a| a.contains_key("tool_name"));
                    candidates.push((key.clone(), description(spec), tool));
                }
            }
            "score" => {
                let levels = criteria
                    .as_array()
                    .context("score question needs a list of at least two levels")?;
                ensure!(
                    (2..=32).contains(&levels.len()),
                    "UNSUPPORTED: candidate count"
                );
                for (i, l) in levels.iter().enumerate() {
                    let d = description(l);
                    candidates.push((
                        i.to_string(),
                        if d.is_empty() {
                            format!("level {i}")
                        } else {
                            d
                        },
                        false,
                    ));
                }
            }
            "noul" => {
                for (key, desc, fallback) in [("yes", "true", "yes"), ("no", "false", "no")] {
                    let d = description(criteria.get(desc).unwrap_or(&Value::Null));
                    candidates.push((
                        key.into(),
                        if d.is_empty() { fallback.into() } else { d },
                        false,
                    ));
                }
            }
            _ => bail!("unsupported question type: {kind:?}"),
        }
        ensure!(
            (2..=32).contains(&candidates.len()),
            "UNSUPPORTED: candidate count"
        );
        let lines = candidates
            .iter()
            .enumerate()
            .map(|(i, (_, d, tool))| {
                Ok(format!(
                    "{}) {}",
                    LABELS.as_bytes()[i] as char,
                    if *tool { render_tool(d)? } else { flat(d) }
                ))
            })
            .collect::<Result<Vec<_>>>()?
            .join("\n");
        let instructions = flat(&py_str(q.get("instructions").unwrap_or(&json!(""))));
        const OVERHEAD: usize =
            "State:\n\n\nQuestion: \n\nCandidates:\n\n\nAnswer with exactly one candidate label."
                .len();
        let bytes = [state.len(), instructions.len(), lines.len(), OVERHEAD]
            .into_iter()
            .try_fold(0usize, |sum, n| {
                sum.checked_add(n)
                    .context("compiled prompt length overflow")
            })?;
        reserve_prompt_bytes(&mut prompt_bytes, bytes, MAX_COMPILED_PROMPT_BYTES)?;
        let prompt = format!(
            "State:\n{}\n\nQuestion: {}\n\nCandidates:\n{lines}\n\nAnswer with exactly one candidate label.",
            state, instructions
        );
        out.push(Question {
            id: id.clone(),
            kind: kind.into(),
            keys: candidates.into_iter().map(|c| c.0).collect(),
            prompt,
        });
    }
    Ok(out)
}
pub fn answer(q: &Question, logits: &[f32], temperature: f64) -> Result<Value> {
    ensure!(
        temperature.is_finite() && temperature > 0.,
        "invalid temperature"
    );
    ensure!(
        logits.len() == q.keys.len() && logits.iter().all(|v| v.is_finite()),
        "invalid logits"
    );
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p = logits
        .iter()
        .map(|&v| ((v as f64 - max) / temperature).exp())
        .collect::<Vec<_>>();
    let sum = p.iter().sum::<f64>();
    p.iter_mut().for_each(|v| *v /= sum);
    let mode = (1..p.len()).fold(0, |m, i| if p[i] > p[m] { i } else { m });
    let probs: Map<String, Value> = q
        .keys
        .iter()
        .cloned()
        .zip(p.iter().map(|v| json!(v)))
        .collect();
    let confidence = p[mode];
    Ok(match q.kind.as_str() {
        "noul" => json!({"type":"noul","noul":p[0],"probabilities":probs,"confidence":confidence}),
        "score" => {
            json!({"type":"score","expected_value":p.iter().enumerate().map(|(i,p)|i as f64*p).sum::<f64>(),"probabilities":probs,"confidence":confidence})
        }
        _ => {
            json!({"type":"choice","choice":q.keys[mode],"probabilities":probs,"confidence":confidence})
        }
    })
}
/// Reserve aggregate compiled prompt storage before allocating the next prompt.
pub fn reserve_prompt_bytes(total: &mut usize, additional: usize, limit: usize) -> Result<()> {
    let reserved = total
        .checked_add(additional)
        .context("compiled prompt length overflow")?;
    ensure!(
        reserved <= limit,
        "aggregate compiled prompts exceed byte limit {limit}"
    );
    *total = reserved;
    Ok(())
}
