//! Laya 0.3.20 packing. One question is one model row; option and question order are semantic.
use anyhow::{Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub state: Value,
    #[serde(default)]
    pub model: Option<String>,
    pub questions: Map<String, Value>,
    #[serde(default)]
    pub lang: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Question {
    pub id: String,
    pub kind: String,
    pub criteria: Value,
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
    pub qtype: i64,
}

#[derive(Debug, Serialize)]
pub struct Batch {
    pub questions: Vec<Question>,
    pub input_ids: Vec<i64>,
    pub lens: Vec<i32>,
    pub qtypes: Vec<i64>,
    pub b: usize,
    pub l: usize,
    pub usage: usize,
}

pub struct Preprocessor {
    tokenizer: Tokenizer,
    cls: u32,
    sep: u32,
    mask: u32,
}

// Python json.dumps(..., ensure_ascii=False) uses spaces after commas/colons.
// Walk values so punctuation inside strings remains untouched.
pub fn render(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        _ => spaced_json(value),
    }
}
fn spaced_json(value: &Value) -> String {
    match value {
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(spaced_json).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), spaced_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Number(n) => python_number(n),
        _ => serde_json::to_string(value).unwrap(),
    }
}

impl Preprocessor {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer/tokenizer.json"))
            .map_err(|e| anyhow!("{e}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow!("{e}"))?;
        let token = |s| {
            tokenizer
                .token_to_id(s)
                .ok_or_else(|| anyhow!("missing token {s}"))
        };
        Ok(Self {
            cls: token("[CLS]")?,
            sep: token("[SEP]")?,
            mask: token("[MASK]")?,
            tokenizer,
        })
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text.replace("[MASK]", " "), false)
            .map_err(|e| anyhow!("{e}"))?
            .get_ids()
            .to_vec())
    }
    pub fn prepare(&self, request: &Request) -> Result<Batch> {
        validate_numbers(&request.state)?;
        validate_numbers(&Value::Object(request.questions.clone()))?;
        ensure!(
            request.model.as_deref().is_none_or(|s| s == "english"),
            "only model=english is supported"
        );
        ensure!(
            request
                .lang
                .as_deref()
                .is_none_or(|s| s == "en" || s == "english"),
            "only English is supported; use lang=en"
        );
        ensure!(
            request.questions.len() <= 16,
            "at most 16 questions per request"
        );
        let state = self.encode(&render(&request.state))?;
        let mut questions = Vec::new();
        for (id, definition) in &request.questions {
            let (kind, criteria, opts) =
                options(definition).map_err(|e| anyhow!("question {id:?}: {e}"))?;
            let ins = definition
                .get("instructions")
                .ok_or_else(|| anyhow!("question {id:?}: missing instructions"))?;
            let mut head = self.encode(&format!("{kind} question: {}", render(ins)))?;
            let mut opt_ids = Vec::new();
            for opt in &opts {
                let mut ids = vec![self.mask];
                ids.extend(self.encode(&format!(" {opt}"))?.into_iter().take(48));
                opt_ids.push(ids);
            }
            let mut budget = 192isize - opt_ids.iter().map(Vec::len).sum::<usize>() as isize;
            if budget < 16 {
                let per = (176 / opt_ids.len()).max(4);
                for o in &mut opt_ids {
                    o.truncate(per);
                }
                budget = 192 - opt_ids.iter().map(Vec::len).sum::<usize>() as isize;
            }
            head.truncate(budget.max(8) as usize);
            let mut ids = vec![self.cls];
            ids.extend(head);
            ids.push(self.sep);
            let mut markers = Vec::new();
            for opt in opt_ids {
                markers.push(ids.len());
                ids.extend(opt);
            }
            ids.push(self.sep);
            let room = 512usize.saturating_sub(ids.len() + 1);
            if request.state.is_array() {
                ids.extend_from_slice(&state[state.len().saturating_sub(room)..]);
            } else {
                ids.extend_from_slice(&state[..room.min(state.len())]);
            }
            ids.push(self.sep);
            ids.truncate(512);
            ensure!(
                markers.iter().all(|&m| m < 512),
                "question {id:?}: options exceed head_max_len=192"
            );
            let qtype = match kind.as_str() {
                "choice" => 0,
                "score" => 1,
                _ => 2,
            };
            questions.push(Question {
                id: id.clone(),
                kind,
                criteria,
                ids,
                markers,
                qtype,
            });
        }
        ensure!(
            questions.iter().map(|q| q.markers.len()).sum::<usize>() <= 2048,
            "at most 2048 options across all questions"
        );
        let n = questions.len();
        let max_l = questions.iter().map(|q| q.ids.len()).max().unwrap_or(0);
        let l = if max_l <= 256 {
            max_l.div_ceil(16) * 16
        } else {
            max_l.div_ceil(64) * 64
        };
        let b = if n == 0 { 0 } else { n.next_power_of_two() };
        let mut input_ids = vec![0; b * l];
        let mut lens = vec![0; b];
        let mut qtypes = vec![0; b];
        for (i, q) in questions.iter().enumerate() {
            input_ids[i * l..i * l + max_l].fill(50283);
            for (j, &v) in q.ids.iter().enumerate() {
                input_ids[i * l + j] = i64::from(v);
            }
            lens[i] = q.ids.len() as i32;
            qtypes[i] = q.qtype;
        }
        let usage = questions.iter().map(|q| q.ids.len()).sum();
        Ok(Batch {
            questions,
            input_ids,
            lens,
            qtypes,
            b,
            l,
            usage,
        })
    }
}

fn options(q: &Value) -> Result<(String, Value, Vec<String>)> {
    ensure!(q.is_object(), "definition must be an object");
    let kind = q["type"].as_str().ok_or_else(|| anyhow!("missing type"))?;
    ensure!(
        ["choice", "score", "noul"].contains(&kind),
        "unknown type {kind}"
    );
    ensure!(
        kind == "noul" || q.get("labels").is_none(),
        "labels only apply to noul"
    );
    let mut criteria = q.get("criteria").cloned().unwrap_or(Value::Null);
    let opts = match kind {
        "choice" => {
            if let Some(a) = criteria.as_array() {
                let mut o = Map::new();
                for key in a {
                    o.insert(
                        key.as_str()
                            .ok_or_else(|| anyhow!("choice labels must be strings"))?
                            .to_owned(),
                        Value::Null,
                    );
                }
                criteria = Value::Object(o);
            }
            let o = criteria
                .as_object()
                .ok_or_else(|| anyhow!("choice criteria must be object or list"))?;
            ensure!(!o.is_empty(), "at least one choice required");
            o.iter()
                .map(|(k, v)| {
                    if v.is_null() || v.as_str() == Some("") {
                        k.clone()
                    } else {
                        format!("{k}: {}", render(v))
                    }
                })
                .collect()
        }
        "score" => {
            let a = criteria
                .as_array()
                .ok_or_else(|| anyhow!("score criteria must be a list"))?;
            ensure!(!a.is_empty(), "at least one level required");
            a.iter()
                .enumerate()
                .map(|(i, v)| format!("level {i}: {}", render(v)))
                .collect()
        }
        _ => {
            if criteria.is_null() {
                criteria = Value::Object(Map::new());
            }
            let o = criteria
                .as_object()
                .ok_or_else(|| anyhow!("noul criteria must be object"))?;
            let mut normalized = Map::new();
            for (k, v) in o {
                let key = k.to_lowercase();
                ensure!(
                    key == "true" || key == "false",
                    "noul criteria keys must be true/false"
                );
                normalized.insert(key, v.clone());
            }
            criteria = Value::Object(normalized);
            let labels = match q.get("labels") {
                None | Some(Value::Null) => ["false", "true"],
                Some(Value::Object(o)) if o.len() == 2 => [
                    o.get("false").and_then(Value::as_str).unwrap_or("").trim(),
                    o.get("true").and_then(Value::as_str).unwrap_or("").trim(),
                ],
                _ => bail!("noul labels must map true/false to distinct non-empty strings"),
            };
            ensure!(
                !labels[0].is_empty() && !labels[1].is_empty() && labels[0] != labels[1],
                "invalid noul labels"
            );
            ["false", "true"]
                .iter()
                .enumerate()
                .map(|(i, k)| {
                    let v = &criteria[*k];
                    let desc = if v.is_null() || v.as_str() == Some("") {
                        if i == 0 {
                            "no, the statement does not hold".to_owned()
                        } else {
                            "yes, the statement holds".to_owned()
                        }
                    } else {
                        render(v)
                    };
                    format!("{}: {desc}", labels[i])
                })
                .collect()
        }
    };
    Ok((kind.to_owned(), criteria, opts))
}

// Python repr(float) uses scientific notation below 1e-4 and from 1e16,
// with a signed exponent of at least two digits. Preserve arbitrary-size integers.
fn python_number(n: &serde_json::Number) -> String {
    let raw = n.to_string();
    if !raw.contains(['.', 'e', 'E']) {
        return raw;
    }
    let Some(value) = n.as_f64().filter(|v| v.is_finite()) else {
        return raw;
    };
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let e: i32 = exponent.parse().unwrap();
    if !(-4..16).contains(&e) {
        return format!("{mantissa}e{e:+03}");
    }
    let mut plain = value.to_string();
    if !plain.contains('.') {
        plain.push_str(".0");
    }
    plain
}

fn validate_numbers(value: &Value) -> Result<()> {
    match value {
        Value::Number(n) if n.to_string().contains(['.', 'e', 'E']) => ensure!(
            n.as_f64().is_some_and(f64::is_finite),
            "floating point value outside supported finite range"
        ),
        Value::Array(a) => {
            for v in a {
                validate_numbers(v)?;
            }
        }
        Value::Object(o) => {
            for v in o.values() {
                validate_numbers(v)?;
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn python_json_numbers_and_order() {
        for (input, expected) in [
            ("1e-6", "1e-06"),
            ("1e20", "1e+20"),
            ("1e16", "1e+16"),
            ("1e-4", "0.0001"),
            ("1.0", "1.0"),
            ("-0.0", "-0.0"),
            ("18446744073709551616000", "18446744073709551616000"),
        ] {
            assert_eq!(render(&serde_json::from_str(input).unwrap()), expected);
        }
        let v = serde_json::from_str(r#"{"z":1e-6,"a":"x,y:z"}"#).unwrap();
        assert_eq!(render(&v), r#"{"z": 1e-06, "a": "x,y:z"}"#);
    }
    #[test]
    fn reject_invalid_questions() {
        for q in [
            serde_json::json!({"type":"choice","criteria":[]}),
            serde_json::json!({"type":"noul","criteria":{"yes":"ok"}}),
            serde_json::json!({"type":"noul","labels":{"false":"x","true":"x"}}),
        ] {
            assert!(options(&q).is_err());
        }
    }
}
