//! Request mapping, prompts and answers for Cua-S1 4B 0.2, as in
//! `src/models/cua_s1/README.md` and the reference worker's `text/contract.py`.

use serde_json::{Map, Value, json};

use crate::json::{dumps, parse, quote};

pub const MODEL_NAME: &str = "cua-s1-4b-0.2";
pub const MODEL_ID: &str = "cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79:text";
pub const LETTERS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const MAX_QUESTIONS: usize = 64;

// The system message and the user message layout are copied from trycua/cua at
// 0e75660ce4c2edda519e0c795fa3ad98abf4e76f (`libs/cua-s1/python/src/cua_s1/four_b.py`
// and `libs/cua-driver/examples/jev-use/python/decision_models.py`).
//
// MIT License
//
// Copyright (c) 2025 Cua AI, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
pub const SYSTEM_PROMPT: &str = "You are a one-pass computer-use decision model. You are shown the \
current state of a screen and a fixed, closed list of candidate \
(element, action) options, each given a single letter. Choose exactly \
one option: the single best next action to take. Answer with ONLY that \
option's letter -- no words, no punctuation, no explanation.";

pub struct RequestError {
    pub status: u16,
    pub message: String,
}

fn error(status: u16, message: impl Into<String>) -> RequestError {
    RequestError {
        status,
        message: message.into(),
    }
}

#[derive(Clone)]
pub struct Question {
    pub name: String,
    pub goal: String,
    pub keys: Vec<String>,
    pub labels: Vec<String>,
}

pub fn parse_body(raw: &[u8]) -> Result<Map<String, Value>, RequestError> {
    parse(raw).map_err(|message| error(400, message))
}

/// A string as is; an object or array as Python's `json.dumps` writes it.
fn text(value: &Value, place: &str) -> Result<String, RequestError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Object(_) | Value::Array(_) => Ok(dumps(value)),
        _ => Err(error(
            422,
            format!("{place} must be a string, an object or an array"),
        )),
    }
}

pub fn map_request(body: &Map<String, Value>) -> Result<(String, Vec<Question>), RequestError> {
    if body.get("model").and_then(Value::as_str) != Some(MODEL_NAME) {
        return Err(error(422, format!("'model' must be '{MODEL_NAME}'")));
    }
    let state = body.get("state").unwrap_or(&Value::Null);
    if [json!(""), json!({}), json!([])].contains(state) {
        return Err(error(422, "'state' must not be empty"));
    }
    let state = text(state, "'state'")?;
    let questions = match body.get("questions") {
        Some(Value::Object(q)) if !q.is_empty() => q,
        _ => return Err(error(422, "'questions' must be a non-empty object")),
    };
    if questions.len() > MAX_QUESTIONS {
        return Err(error(413, format!("more than {MAX_QUESTIONS} questions")));
    }
    let mut mapped = Vec::with_capacity(questions.len());
    for (name, q) in questions {
        let place = format!("question {}", quote(name));
        let Value::Object(q) = q else {
            return Err(error(422, format!("{place} must be an object")));
        };
        match q.get("type").unwrap_or(&Value::Null) {
            Value::String(t) if t == "choice" => {}
            Value::String(t) if t == "score" || t == "noul" => {
                return Err(error(422, format!("{place}: type '{t}' is not supported")));
            }
            other => return Err(error(422, format!("{place}: unknown type {other}"))),
        }
        let goal = match q.get("instructions") {
            None => return Err(error(422, format!("{place}: 'instructions' is required"))),
            Some(Value::Null) => String::new(),
            Some(value) => text(value, &place)?,
        };
        let criteria = match q.get("criteria") {
            Some(Value::Object(c)) if (1..=LETTERS.len()).contains(&c.len()) => c,
            _ => {
                let message = format!("{place}: 'criteria' must be an object with 1 to 26 options");
                return Err(error(422, message));
            }
        };
        let mut labels = Vec::with_capacity(criteria.len());
        for (key, value) in criteria {
            let label = match value {
                Value::Null => key.clone(),
                value => text(value, &format!("{place}: {}", quote(key)))?,
            };
            // escaped as `json.dumps(label, ensure_ascii=False)[1:-1]`, like upstream's chooser
            let quoted = quote(&label);
            labels.push(quoted[1..quoted.len() - 1].to_string());
        }
        let keys = criteria.keys().cloned().collect();
        mapped.push(Question {
            name: name.clone(),
            goal,
            keys,
            labels,
        });
    }
    Ok((state, mapped))
}

/// The prompt the Qwen3.5 chat template renders for the system and user messages with
/// `add_generation_prompt=True`. The template trims message content, which changes
/// nothing here: the user message starts with "Goal: " or "App: " and ends with "letter.".
pub fn chat_text(state: &str, question: &Question) -> String {
    let goal = match question.goal.as_str() {
        "" => String::new(),
        goal => format!("Goal: {goal}\n\n"),
    };
    let options: Vec<String> = LETTERS
        .chars()
        .zip(&question.labels)
        .map(|(letter, label)| format!("{letter}. Decision \"{label}\" -> select"))
        .collect();
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{goal}App: Cua Driver\n\
         Task family: closed-candidate decision\n\nAccessibility tree:\n{state}\n\nOptions:\n{}\n\n\
         Answer with a single letter.<|im_end|>\n<|im_start|>assistant\n<think>\n",
        options.join("\n")
    )
}

/// The Jev choice answer; ties go to the earliest option. `confidence` is the
/// normalized entropy `1 - H(p) / ln(n)`, as the LAYA worker reports it.
pub fn answer(question: &Question, probabilities: &[f32]) -> Value {
    let p: Vec<f64> = probabilities.iter().map(|&x| x as f64).collect();
    let best = (0..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b });
    let entropy: f64 = -p
        .iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x * x.ln())
        .sum::<f64>();
    let n = p.len() as f64;
    let probs: Map<String, Value> = question
        .keys
        .iter()
        .cloned()
        .zip(p.iter().map(|&x| json!(x)))
        .collect();
    json!({
        "type": "choice",
        "choice": question.keys[best],
        "probabilities": probs,
        "confidence": if p.len() > 1 { (1.0 - entropy / n.ln()).max(0.0) } else { 1.0 },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(body: &str) -> Result<(String, Vec<Question>), RequestError> {
        map_request(&parse_body(body.as_bytes())?)
    }

    #[test]
    fn maps_a_request() {
        let body = r#"{"model": "cua-s1-4b-0.2", "state": {"a": [1.5, "é"]}, "questions": {"q": {"type": "choice", "instructions": "go", "criteria": {"a": "Say \"hi\"\n", "b": null}}}}"#;
        let (state, questions) = map(body).ok().unwrap();
        assert_eq!(state, r#"{"a": [1.5, "é"]}"#);
        assert_eq!(questions[0].keys, ["a", "b"]);
        assert_eq!(questions[0].labels, [r#"Say \"hi\"\n"#, "b"]);
        let text = chat_text(&state, &questions[0]);
        assert!(text.contains("<|im_start|>user\nGoal: go\n\nApp: Cua Driver\n"));
        assert!(text.ends_with(
            "Options:\nA. Decision \"Say \\\"hi\\\"\\n\" -> select\nB. Decision \"b\" -> select\n\n\
             Answer with a single letter.<|im_end|>\n<|im_start|>assistant\n<think>\n"
        ));
    }

    #[test]
    fn errors() {
        let q = |question: &str| {
            format!(
                r#"{{"model": "cua-s1-4b-0.2", "state": "S", "questions": {{"q": {question}}}}}"#
            )
        };
        let cases = [
            (r#"{"state": "S"}"#.to_string(), 422),
            (
                r#"{"model": "cua-s1-4b-0.2", "state": {}}"#.to_string(),
                422,
            ),
            (
                r#"{"model": "cua-s1-4b-0.2", "state": 1.5}"#.to_string(),
                422,
            ),
            (q(r#"{"type": "noul"}"#), 422),
            (q(r#"{"type": "choice"}"#), 422),
            (
                q(r#"{"type": "choice", "instructions": null, "criteria": {"a": true}}"#),
                422,
            ),
            (
                q(r#"{"type": "choice", "instructions": null, "criteria": {}}"#),
                422,
            ),
            (r#"{"a": 1, "a": 2}"#.to_string(), 400),
        ];
        for (body, status) in cases {
            assert_eq!(map(&body).err().map(|e| e.status), Some(status), "{body}");
        }
    }

    #[test]
    fn answer_and_confidence() {
        let (_, questions) = map(r#"{"model": "cua-s1-4b-0.2", "state": "S", "questions": {"q": {"type": "choice", "instructions": null, "criteria": {"a": "A", "b": "B"}}}}"#).ok().unwrap();
        let tie = answer(&questions[0], &[0.5, 0.5]);
        assert_eq!(tie["choice"], "a");
        assert!(tie["confidence"].as_f64().unwrap().abs() < 1e-12);
        assert_eq!(answer(&questions[0], &[0.25, 0.75])["choice"], "b");
    }
}
