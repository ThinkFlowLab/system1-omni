//! Adapted from Mapika/decider systemone.py at 50d0be0; Apache-2.0.
use crate::json;
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Choice,
    Noul,
    Score,
}
impl Kind {
    pub(crate) fn parse(t: &str) -> Result<Self> {
        match t {
            "choice" => Ok(Self::Choice),
            "noul" | "bool" => Ok(Self::Noul),
            "score" => Ok(Self::Score),
            _ => bail!("unknown question type {t:?}"),
        }
    }
    pub(crate) fn index(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Noul => 1,
            Self::Score => 2,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct Question {
    pub id: String,
    pub kind: Kind,
    pub text: String,
    pub options: Vec<String>,
    pub names: Vec<String>,
    pub legend: Vec<String>,
}
fn empty(v: &Value) -> bool {
    v.is_null() || v.as_str() == Some("")
}
fn annotate(v: &Value) -> Value {
    match v {
        Value::Array(a) => Value::Array(
            a.iter()
                .enumerate()
                .map(|(i, x)| {
                    let x = annotate(x);
                    if a.len() < 8 {
                        x
                    } else {
                        let mut m = Map::new();
                        m.insert("_index".into(), Value::from(i));
                        if let Value::Object(fields) = x {
                            for (k, v) in fields {
                                m.insert(k, v);
                            }
                        } else {
                            m.insert("value".into(), x);
                        }
                        Value::Object(m)
                    }
                })
                .collect(),
        ),
        Value::Object(m) => {
            Value::Object(m.iter().map(|(k, v)| (k.clone(), annotate(v))).collect())
        }
        _ => v.clone(),
    }
}
pub(crate) fn render_state(v: &Value) -> String {
    if v.is_string() {
        json::text(v)
    } else {
        json::dumps(&annotate(v))
    }
}
pub(crate) fn render_question(id: &str, spec: &Value) -> Result<Question> {
    let s = json::object(spec).context("question definition must be an object")?;
    json::default_mode(s, "isolated", true)?;
    let kind = Kind::parse(
        s.get("type")
            .map(|v| v.as_str().context("type must be text"))
            .transpose()?
            .unwrap_or("choice"),
    )?;
    let none = Value::Null;
    let blank = Value::String(String::new());
    let criteria = s
        .get("criteria")
        .or_else(|| s.get("options"))
        .unwrap_or(&none);
    let raw = s
        .get("instructions")
        .or_else(|| s.get("question"))
        .unwrap_or(&blank);
    let text = if kind == Kind::Noul && empty(raw) {
        let described = criteria.as_object().is_some_and(|m| {
            ["true", "false"]
                .iter()
                .any(|k| m.get(*k).is_some_and(|v| !empty(v)))
        });
        ensure!(
            described,
            "noul without instructions requires true or false description"
        );
        "Which answer fits the context?".into()
    } else {
        json::text(raw)
    };
    ensure!(!text.is_empty(), "question without instructions");
    let mut names = Vec::new();
    let mut options = Vec::new();
    let mut legend = Vec::new();
    match kind {
        Kind::Choice => {
            let fields = if let Some(a) = criteria.as_array() {
                let mut fields = Map::new();
                for v in a {
                    let name = v.as_str().context(
                        "Choice list alias requires string names; use a map for other JSON values",
                    )?;
                    fields.insert(name.into(), Value::Null);
                }
                fields
            } else {
                json::object(criteria)
                    .context("Choice requires map or string list")?
                    .clone()
            };
            ensure!(
                (2..=255).contains(&fields.len()),
                "Choice requires 2..255 options"
            );
            for (name, v) in fields {
                options.push(if empty(&v) {
                    name.clone()
                } else {
                    format!("{name}: {}", json::text(&v))
                });
                names.push(name);
            }
        }
        Kind::Score => {
            let levels = if let Some(a) = criteria.as_array() {
                a.clone()
            } else {
                let mut ordered = json::object(criteria)?
                    .iter()
                    .map(|(k, v)| {
                        let number =
                            json::python_float(k).context("Score legend keys must be numeric")?;
                        ensure!(number.is_finite(), "Score legend keys must be finite");
                        Ok((number, v.clone()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                ordered.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite key"));
                ordered.into_iter().map(|(_, v)| v).collect()
            };
            ensure!(
                (2..=10).contains(&levels.len()),
                "Score requires 2..10 levels"
            );
            legend = levels.iter().map(json::text).collect();
            for (i, v) in legend.iter().enumerate() {
                names.push(i.to_string());
                options.push(format!("{i}: {v}"));
            }
        }
        Kind::Noul => {
            ensure!(
                criteria.is_null() || criteria.is_object(),
                "Noul criteria must be a map"
            );
            for (key, label) in [("false", "no"), ("true", "yes")] {
                let v = criteria.get(key).unwrap_or(&none);
                options.push(if empty(v) {
                    label.into()
                } else {
                    format!("{label}: {}", json::text(v))
                });
                names.push(key.into());
            }
        }
    }
    Ok(Question {
        id: id.into(),
        kind,
        text,
        options,
        names,
        legend,
    })
}
pub(crate) fn isolated_text(q: &Question, level: usize) -> String {
    use crate::text::{decimal, whitespace};
    let original = &q.legend[level];
    let trimmed = original.trim_start_matches(whitespace);
    let unsigned = trimmed.strip_prefix('-').unwrap_or(trimmed);
    let digits = unsigned
        .char_indices()
        .find(|(_, c)| decimal(*c).is_none())
        .map_or(unsigned.len(), |(i, _)| i);
    let stripped = if digits > 0 {
        unsigned[digits..]
            .trim_start_matches(whitespace)
            .strip_prefix(':')
            .map(|v| v.trim_start_matches(whitespace))
    } else {
        None
    };
    format!(
        "{}\nProposed answer: {}\nDoes the proposed answer fit?",
        q.text,
        stripped.unwrap_or(original)
    )
}

fn normalized(p: &[f64]) -> Vec<f64> {
    let sum: f64 = p.iter().sum();
    if sum == 0.0 {
        vec![1.0 / p.len() as f64; p.len()]
    } else {
        p.iter().map(|v| v / sum).collect()
    }
}
fn mode(p: &[f64]) -> usize {
    (1..p.len()).fold(0, |m, i| if p[i] > p[m] { i } else { m })
}
pub(crate) fn answer(q: &Question, probabilities: &[f64]) -> Value {
    use crate::math::round;
    use serde_json::json;
    let sum: f64 = probabilities.iter().take(q.options.len()).sum();
    let divisor = if sum == 0.0 { 1.0 } else { sum };
    let p: Vec<f64> = probabilities
        .iter()
        .take(q.options.len())
        .map(|v| v / divisor)
        .collect();
    if q.kind == Kind::Noul {
        return json!({"type":"noul","noul":round(p[1],4)});
    }
    let j = mode(&p);
    let n = p.len();
    let entropy: f64 = p.iter().filter(|v| **v > 0.0).map(|v| -v * v.ln()).sum();
    let certainty = (1.0 - entropy / (n as f64).ln()).max(0.0);
    // Confidence's zero-mass uniform fallback never replaces emitted probabilities.
    let norm = normalized(&p);
    let confidence = if q.kind == Kind::Choice {
        ((n as f64 * norm[mode(&norm)] - 1.0) / (n - 1) as f64).clamp(0.0, 1.0)
    } else {
        let modal = mode(&norm);
        let spread: f64 = norm
            .iter()
            .enumerate()
            .map(|(i, v)| v * i.abs_diff(modal) as f64)
            .sum();
        let center = (n - 1) as f64 / 2.0;
        let uniform = (0..n).map(|i| (i as f64 - center).abs()).sum::<f64>() / n as f64;
        (1.0 - spread / uniform).clamp(0.0, 1.0)
    };
    let probabilities: Map<String, Value> = q
        .names
        .iter()
        .cloned()
        .zip(p.iter().map(|v| json!(round(*v, 4))))
        .collect();
    let mut out = json!({"type":if q.kind==Kind::Choice {"choice"} else {"score"},"confidence":round(confidence,4),"x_p_max":round(p[j],4),"certainty":round(certainty,4),"probabilities":probabilities});
    if q.kind == Kind::Choice {
        out["choice"] = json!(q.names[j]);
    } else {
        let score: f64 = p.iter().enumerate().map(|(i, v)| i as f64 * v).sum();
        out["score"] = json!(round(score, 2));
        out["legend"] = Value::Object(
            q.legend
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), json!(v)))
                .collect(),
        );
    }
    out
}
