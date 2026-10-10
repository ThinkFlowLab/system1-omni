//! The request path: a `/v1/systemone` body to typed answers.
//!
//! Mirrors `src/clm/schema.py` and `src/clm/engine.py`. Two text functions matter and
//! both are reproduced exactly, down to the separator, because the heads were trained on
//! this layout:
//!
//! - the **state head** sees the context and the question joined by a blank line
//!   (`state_text`), so a question belongs in `instructions`, not repeated in the state;
//! - the **action head** sees each candidate's own text with nothing prefixed for
//!   `choice`, but `"<key>: <text>"` for `noul`.
//!
//! A mismatch here does not fail loudly — it shifts every probability — so both are
//! pinned by tests against the reference implementation's own output.
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};

/// The fields of a request that get rendered, kept as the text they arrived as.
///
/// `serde_json` has no way to hand back what an integer literal said: one that does not
/// fit an `i64` or a `u64` is parsed straight to a double, and `18446744073709551616`
/// comes back as `1.8446744073709552e19`. Python gets an arbitrary-precision `int`
/// instead, and `str` prints every digit. `RawValue` is the only place a literal
/// survives, so the fields that reach the encoder are deserialized twice — once for the
/// structure and once for the text — and rendered from the text.
#[derive(Deserialize)]
struct RawBody {
    state: Box<RawValue>,
    /// Each question object as its own text. A map, not `serde_json::Map`: with
    /// `preserve_order` that one is `IndexMap<String, Value>` and cannot hold another
    /// type. The request order comes from the parsed `Value` instead, and this is
    /// looked up by id.
    #[serde(default)]
    questions: std::collections::HashMap<String, Box<RawValue>>,
}

/// The two rendered fields of one question.
#[derive(Deserialize)]
struct RawQuestion {
    #[serde(default)]
    instructions: Option<Box<RawValue>>,
    #[serde(default)]
    criteria: Option<Box<RawValue>>,
}

/// What each rendered field arrived as.
#[derive(Debug, Clone, Default)]
struct RawFields {
    state: String,
    /// Question id to (instructions, criteria).
    questions: std::collections::HashMap<String, (Option<String>, Option<String>)>,
}

impl RawFields {
    fn of(line: &str) -> Result<Self> {
        let raw: RawBody = serde_json::from_str(line).context("read the request text")?;
        let mut questions = std::collections::HashMap::with_capacity(raw.questions.len());
        for (id, object) in &raw.questions {
            let q: RawQuestion = serde_json::from_str(object.get())
                .with_context(|| format!("read the text of question {id:?}"))?;
            questions.insert(
                id.clone(),
                (
                    q.instructions.map(|v| v.get().to_string()),
                    q.criteria.map(|v| v.get().to_string()),
                ),
            );
        }
        Ok(Self {
            state: raw.state.get().to_string(),
            questions,
        })
    }

    fn question(&self, id: &str) -> (Option<&str>, Option<&str>) {
        self.questions
            .get(id)
            .map(|(i, c)| (i.as_deref(), c.as_deref()))
            .unwrap_or((None, None))
    }
}

/// Render one field from the text it arrived as. Each call gets its own literals, so
/// rendering the same field twice — the state is rendered once per question — is safe,
/// and nothing depends on the order the fields appear in.
fn render_raw(raw: &str) -> Result<String> {
    let value: Value = serde_json::from_str(raw).context("render a request field")?;
    render_with(raw, &value)
}

/// Render `value` from the text it arrived as. The literals are the value's own, so this
/// can be called per key: a number is always spelled by the literal it was written with,
/// whatever order the keys are read in.
///
/// Object keys must be unique before pairing: replacing a nonnumeric value with a
/// number can reorder the numeric walk even when the two number counts agree.
fn render_with(raw: &str, value: &Value) -> Result<String> {
    ensure_unique_keys(raw)?;
    let mut numbers = NumberLiterals::of(raw);
    let text = render(value, 0, &mut numbers);
    numbers.ensure_paired()?;
    Ok(text)
}

/// Inspect object entries before serde's maps can collapse duplicate keys. Children
/// stay as raw JSON so validating them never rounds an arbitrary-precision integer.
fn ensure_unique_keys(raw: &str) -> Result<()> {
    struct ObjectFields(Vec<Box<RawValue>>);

    impl<'de> Deserialize<'de> for ObjectFields {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct FieldsVisitor;
            impl<'de> serde::de::Visitor<'de> for FieldsVisitor {
                type Value = ObjectFields;

                fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                    formatter.write_str("a JSON object with unique keys")
                }

                fn visit_map<M: serde::de::MapAccess<'de>>(
                    self,
                    mut map: M,
                ) -> Result<Self::Value, M::Error> {
                    let mut keys = HashSet::new();
                    let mut fields = Vec::new();
                    while let Some(key) = map.next_key::<String>()? {
                        if !keys.insert(key.clone()) {
                            return Err(serde::de::Error::custom(format!(
                                "a JSON object repeats a key: {key:?}"
                            )));
                        }
                        fields.push(map.next_value::<Box<RawValue>>()?);
                    }
                    Ok(ObjectFields(fields))
                }
            }
            deserializer.deserialize_map(FieldsVisitor)
        }
    }

    let children = match raw.trim_start().as_bytes().first() {
        Some(b'{') => serde_json::from_str::<ObjectFields>(raw)?.0,
        Some(b'[') => serde_json::from_str::<Vec<Box<RawValue>>>(raw)?,
        _ => return Ok(()),
    };
    for child in children {
        ensure_unique_keys(child.get())?;
    }
    Ok(())
}

use crate::embedding::Encoder;
use crate::scoring::{self, Answer, Kind, Question};
use crate::weights::Heads;

/// The candidate keys of a `noul` question, in the order the reference uses.
pub const NOUL_KEYS: [&str; 2] = ["false", "true"];

/// A request body: a state and the questions asked of it.
#[derive(Debug, Clone)]
pub struct Request {
    pub state: Value,
    pub model: Option<String>,
    /// Question id to question object, in insertion order.
    pub questions: Vec<(String, QuestionRequest)>,
    /// What the rendered fields arrived as, when the caller had the text. `parse`
    /// cannot fill this in; `parse_line` can, and it is what keeps an integer literal
    /// above `u64::MAX` from being rounded.
    raw: Option<RawFields>,
}

#[derive(Debug, Clone)]
pub struct QuestionRequest {
    pub kind: Kind,
    pub instructions: String,
    pub criteria: Option<Value>,
}

/// One question prepared for the encoder and the scorer.
#[derive(Debug, Clone)]
pub struct Prepared {
    pub id: String,
    pub question: Question,
    /// The text the state head embeds: context and question, blank-line separated.
    pub state_text: String,
    /// The text the action head embeds per candidate, in `question.keys` order.
    pub candidate_texts: Vec<String>,
}

impl Request {
    /// Parse a `/v1/systemone` body from the text it arrived as.
    ///
    /// This is the entry point a request path should use: numbers are rendered from
    /// their literals, which [`Request::parse`] cannot do once the text is gone.
    pub fn parse_line(line: &str) -> Result<Self> {
        let body: Value = serde_json::from_str(line).context("request is not JSON")?;
        let mut request = Self::parse(&body)?;
        request.raw = Some(RawFields::of(line)?);
        Ok(request)
    }

    /// Parse a `/v1/systemone` body. Unknown top-level fields are ignored, as the
    /// reference does.
    pub fn parse(body: &Value) -> Result<Self> {
        let object = body.as_object().context("body must be an object")?;
        let state = object
            .get("state")
            .context("body must have a state")?
            .clone();
        let raw = object
            .get("questions")
            .and_then(Value::as_object)
            .context("body must have a questions object")?;
        ensure!(!raw.is_empty(), "questions must not be empty");

        let mut questions = Vec::with_capacity(raw.len());
        for (id, q) in raw {
            let q = q
                .as_object()
                .with_context(|| format!("question {id:?} is not an object"))?;
            let kind = match q.get("type").and_then(Value::as_str) {
                Some("choice") => Kind::Choice,
                Some("score") => Kind::Score,
                Some("noul") => Kind::Noul,
                other => bail!("question {id:?}: unknown type {other:?}"),
            };
            let instructions = q
                .get("instructions")
                .map(to_text)
                .unwrap_or_default()
                .trim()
                .to_string();
            questions.push((
                id.clone(),
                QuestionRequest {
                    kind,
                    instructions,
                    criteria: q.get("criteria").cloned(),
                },
            ));
        }
        Ok(Self {
            state,
            model: object
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned),
            questions,
            raw: None,
        })
    }

    /// Turn each question into the keys, the state text and the candidate texts.
    pub fn prepare(&self) -> Result<Vec<Prepared>> {
        match &self.raw {
            Some(raw) => self.prepare_from_text(raw),
            None => Ok(self
                .questions
                .iter()
                .map(|(id, q)| {
                    let (keys, candidate_texts) =
                        candidates_with(q, None, &q.instructions, &mut NumberLiterals::default())
                            .with_context(|| format!("question {id:?} has invalid criteria"))?;
                    Ok(Prepared {
                        id: id.clone(),
                        question: Question {
                            id: id.clone(),
                            kind: q.kind,
                            keys,
                        },
                        state_text: state_text(&self.state, &q.instructions),
                        candidate_texts,
                    })
                })
                .collect::<Result<Vec<_>>>()?),
        }
    }

    /// The same, rendering each field from the text it arrived as.
    ///
    /// The state is rendered once rather than once per question, and each rendered
    /// field gets its own literals, so neither the number of questions nor the order
    /// the fields were written in can shift which literal belongs to which number.
    fn prepare_from_text(&self, raw: &RawFields) -> Result<Vec<Prepared>> {
        let state = render_with(&raw.state, &self.state)?.trim().to_string();

        let mut prepared = Vec::with_capacity(self.questions.len());
        for (id, q) in &self.questions {
            let (instructions_raw, criteria_raw) = raw.question(id);
            let instructions = match instructions_raw {
                Some(text) => render_raw(text)?.trim().to_string(),
                None => q.instructions.clone(),
            };
            let mut numbers = match criteria_raw {
                Some(text) => {
                    ensure_unique_keys(text)
                        .with_context(|| format!("question {id:?} has invalid criteria"))?;
                    NumberLiterals::of(text)
                }
                None => NumberLiterals::default(),
            };
            let (keys, candidate_texts) =
                candidates_with(q, criteria_raw, &instructions, &mut numbers)
                    .with_context(|| format!("question {id:?} has invalid criteria"))?;
            let state_text = if !state.is_empty() && !instructions.is_empty() {
                format!("{state}\n\n{instructions}")
            } else if !state.is_empty() {
                state.clone()
            } else {
                instructions
            };
            prepared.push(Prepared {
                id: id.clone(),
                question: Question {
                    id: id.clone(),
                    kind: q.kind,
                    keys,
                },
                state_text,
                candidate_texts,
            });
        }
        Ok(prepared)
    }
}

/// Context first, question last — the layout the heads were trained on.
pub fn state_text(state: &Value, instructions: &str) -> String {
    state_text_with(state, instructions, &mut NumberLiterals::default())
}

fn state_text_with(state: &Value, instructions: &str, numbers: &mut NumberLiterals) -> String {
    let s = render(state, 0, numbers).trim().to_string();
    let i = instructions.trim();
    if !s.is_empty() && !i.is_empty() {
        format!("{s}\n\n{i}")
    } else if !s.is_empty() {
        s
    } else {
        i.to_string()
    }
}

/// Option keys in answer order, and the candidate text per option.
pub fn candidates(q: &QuestionRequest) -> Result<(Vec<String>, Vec<String>)> {
    candidates_with(q, None, &q.instructions, &mut NumberLiterals::default())
}

/// The same, rendering each criteria value from the text it arrived as.
///
/// `criteria_raw` is the criteria field's own text when the request had one. Binding each
/// value to the key it was written under is what keeps a literal with its number: a
/// `choice` is read in source order but a `noul` in [`NOUL_KEYS`] order, so
/// `{"true": 1, "false": 2}` has to render as `false: 2` — counting positions gives
/// `false: 1`, which is a different answer, not a different spelling.
///
/// `instructions` is `to_text` of the question's instructions, rendered from their text
/// when there was one. The `noul` defaults are built from it, so a literal in the
/// statement reaches the candidate too.
fn candidates_with(
    q: &QuestionRequest,
    criteria_raw: Option<&str>,
    instructions: &str,
    numbers: &mut NumberLiterals,
) -> Result<(Vec<String>, Vec<String>)> {
    // Only an object has values to find by key. Anything else — a `score`'s list, an
    // absent or null criteria — renders in the order it was written, which is what the
    // cursor is for. The map is keyed, so it does not matter that it is unordered: the
    // keys come from the question, not from here.
    let by_key: Option<HashMap<String, &RawValue>> = match (criteria_raw, q.criteria.as_ref()) {
        (Some(text), Some(Value::Object(_))) => {
            Some(serde_json::from_str(text).context("read the criteria text of a question")?)
        }
        _ => None,
    };
    let value_text = |key: &str, v: &Value, numbers: &mut NumberLiterals| match by_key
        .as_ref()
        .and_then(|values| values.get(key))
    {
        Some(raw) => render_with(raw.get(), v),
        None => Ok(render(v, 0, numbers)),
    };
    match q.kind {
        Kind::Choice => {
            let crit = q
                .criteria
                .as_ref()
                .and_then(Value::as_object)
                .context("choice question needs a non-empty 'criteria' object")?;
            ensure!(
                !crit.is_empty(),
                "choice question needs a non-empty 'criteria' object"
            );
            let keys: Vec<String> = crit.keys().cloned().collect();
            // The option's own text when one is given, else the key. Nothing is prefixed.
            // "Given" is the reference's test — null or the empty string — so an empty
            // container is a description that renders to nothing, not a missing one.
            let mut texts = Vec::with_capacity(keys.len());
            for k in &keys {
                let v = &crit[k];
                texts.push(match v {
                    v if v.is_null() || v.as_str() == Some("") => k.clone(),
                    v => value_text(k, v, numbers)?,
                });
            }
            Ok((keys, texts))
        }
        Kind::Score => {
            let crit = q
                .criteria
                .as_ref()
                .and_then(Value::as_array)
                .context("score question needs 'criteria' as an ordered list of levels")?;
            ensure!(crit.len() >= 2, "score question needs at least two levels");
            let keys = (0..crit.len()).map(|i| i.to_string()).collect();
            let texts: Vec<String> = crit.iter().map(|c| render(c, 0, numbers)).collect();
            // A `score`'s levels are a list, so they share one cursor over the list's own text
            // rather than binding per key the way `choice` does. A level that is an object
            // repeating a key can still leave the cursor short of it.
            numbers.ensure_paired()?;
            Ok((keys, texts))
        }
        Kind::Noul => {
            let crit = q.criteria.as_ref().and_then(Value::as_object);
            let mut texts = Vec::with_capacity(NOUL_KEYS.len());
            for k in NOUL_KEYS {
                // The same "given or not" test as `choice`: `crit.get(k)` in `(None, "")`.
                let body = match crit.and_then(|c| c.get(k)) {
                    Some(v) if !v.is_null() && v.as_str() != Some("") => value_text(k, v, numbers)?,
                    _ if !instructions.is_empty() => {
                        if k == "true" {
                            format!("Yes. This is true: {instructions}")
                        } else {
                            format!("No. This is false: {instructions}")
                        }
                    }
                    _ => k.to_string(),
                };
                texts.push(format!("{k}: {body}"));
            }
            Ok((NOUL_KEYS.iter().map(|k| k.to_string()).collect(), texts))
        }
    }
}

/// The number literals of one JSON document, in the order they appear.
///
/// `serde_json` cannot hand back what an integer literal said. One that does not fit an
/// `i64` or a `u64` is parsed straight to a double — `18446744073709551616` becomes
/// `1.8446744073709552e19` — where `json.loads` gives Python an arbitrary-precision `int`
/// and `str` prints every digit. The literal is the only place those digits survive.
///
/// After duplicate keys are rejected, a depth-first walk of the parsed value meets
/// numbers in the same order as the lexer. The walk below pairs them;
/// [`NumberLiterals::of`] documents how strings are skipped.
#[derive(Debug, Clone, Default)]
pub struct NumberLiterals {
    literals: Vec<String>,
    at: usize,
}

impl NumberLiterals {
    /// Collect the literals of `raw`, which must already have parsed as JSON.
    ///
    /// Strings are skipped whole, escapes included, so a digit inside one is not a
    /// number. Everything else that looks like a number is one: the document is valid
    /// JSON, which is what makes a lexer sufficient here.
    pub fn of(raw: &str) -> Self {
        let bytes = raw.as_bytes();
        let mut literals = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            match bytes[at] {
                b'"' => {
                    at += 1;
                    while at < bytes.len() && bytes[at] != b'"' {
                        at += if bytes[at] == b'\\' { 2 } else { 1 };
                    }
                    at += 1;
                }
                b'-' | b'0'..=b'9' => {
                    let start = at;
                    while at < bytes.len()
                        && matches!(bytes[at], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                    {
                        at += 1;
                    }
                    literals.push(raw[start..at].to_string());
                }
                _ => at += 1,
            }
        }
        Self { literals, at: 0 }
    }

    fn next(&mut self) -> Option<&str> {
        let literal = self.literals.get(self.at).map(String::as_str);
        self.at += 1;
        literal
    }

    /// Check the cursor consumed the literals after structural duplicate-key
    /// validation. Cardinality alone cannot prove the numbers occur in the same order.
    fn ensure_paired(&self) -> Result<()> {
        ensure!(
            self.at >= self.literals.len(),
            "a JSON object repeats a key, so the numbers in this text cannot be paired with \
             the value it parsed to"
        );
        Ok(())
    }

    /// How many literals were collected, for the test that keeps this in step with a walk.
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }
}

/// Render a state, description or criteria that may be a string, object or array as
/// plain text. Objects become `key: value` fields — top-level fields separated by a blank
/// line, nested ones indented — and arrays become one `- item` line each. Key order is
/// preserved.
pub fn to_text(x: &Value) -> String {
    render(x, 0, &mut NumberLiterals::default())
}

/// [`to_text`] for a JSON document, rendered from its text so an integer keeps every
/// digit. See [`NumberLiterals`] for why the text has to come along.
pub fn to_text_json(raw: &str) -> Result<String> {
    let value: Value = serde_json::from_str(raw).context("render a JSON document")?;
    render_with(raw, &value)
}

fn render(x: &Value, indent: usize, numbers: &mut NumberLiterals) -> String {
    match x {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Number(n) => match numbers.next() {
            // The literal decides which of `json.loads`'s two types this is, not the
            // value: a point or an exponent makes it a float, and anything else is an
            // arbitrary-precision `int` that `str` prints digit for digit.
            Some(literal) if !literal.contains(['.', 'e', 'E']) => {
                // `json.loads("-0")` is the int 0 and `str(0)` is "0". JSON forbids
                // leading zeros and a leading "+", so this is the only integer spelling
                // `str(int(...))` would change.
                if literal == "-0" {
                    "0".to_string()
                } else {
                    literal.to_string()
                }
            }
            Some(_) => python_float(n.as_f64().expect("a JSON number is an integer or a float")),
            // No literal: the caller had only a `Value`, so `serde_json`'s integer types
            // are all that is left to tell the two apart. Digits beyond a `u64` are
            // already gone by then, which is why the request path carries the text.
            None => match n.as_i64() {
                Some(i) => i.to_string(),
                None => match n.as_u64() {
                    Some(u) => u.to_string(),
                    None => {
                        python_float(n.as_f64().expect("a JSON number is an integer or a float"))
                    }
                },
            },
        },
        Value::Object(map) => {
            let pad = " ".repeat(indent);
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| {
                    if is_nonempty_container(v) {
                        format!("{pad}{k}:\n{}", render(v, indent + 2, numbers))
                    } else {
                        format!("{pad}{k}: {}", render(v, indent, numbers))
                    }
                })
                .collect();
            parts.join(if indent == 0 { "\n\n" } else { "\n" })
        }
        Value::Array(items) => {
            let pad = " ".repeat(indent);
            let parts: Vec<String> = items
                .iter()
                .map(|v| {
                    if is_nonempty_container(v) {
                        format!("{pad}-\n{}", render(v, indent + 2, numbers))
                    } else {
                        format!("{pad}- {}", render(v, indent, numbers))
                    }
                })
                .collect();
            parts.join("\n")
        }
    }
}

/// `str(float)` as CPython writes it, which is what the reference's `to_text` produces.
///
/// Rust and Python disagree on both ends of the range: an integral float keeps its `.0`
/// here but not in `serde_json`, and a float outside `[1e-4, 1e16)` is exponent form with
/// a signed, at-least-two-digit exponent (`1e-05`, `1e+16`). The heads are trained on
/// these strings, so a differently spelled number is a different input.
fn python_float(x: f64) -> String {
    // The shortest representation that round-trips, which is what `repr` uses. Rust's
    // shortest digits agree with CPython's on *how many* there are but not always on the
    // last one: for a value exactly between two equally short decimals CPython rounds to
    // even and Rust away from zero. Re-rendering at that precision uses the same correct
    // rounding as CPython's `%.*e`, so the digits agree.
    let shortest = format!("{x:e}");
    let (shortest_mantissa, _) = shortest
        .split_once('e')
        .expect("a scientific format always writes an exponent");
    let precision = shortest_mantissa
        .chars()
        .filter(char::is_ascii_digit)
        .count()
        - 1;
    let rendered = format!("{:.*e}", precision, x);
    let (mantissa, exponent) = rendered
        .split_once('e')
        .expect("`{:e}` always writes an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` writes a plain exponent");
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    // CPython writes the exponent when the decimal point lands at or before -4, or past
    // 16 digits.
    let point = exponent + 1;
    let body = if point <= -4 || point > 16 {
        let (head, tail) = digits.split_at(1);
        let e = point - 1;
        let (esign, e) = if e < 0 { ('-', -e) } else { ('+', e) };
        if tail.is_empty() {
            format!("{head}e{esign}{e:02}")
        } else {
            format!("{head}.{tail}e{esign}{e:02}")
        }
    } else if point <= 0 {
        format!("0.{}{digits}", "0".repeat(-point as usize))
    } else if point as usize >= digits.len() {
        format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
    } else {
        let (head, tail) = digits.split_at(point as usize);
        format!("{head}.{tail}")
    };
    format!("{sign}{body}")
}

fn is_nonempty_container(v: &Value) -> bool {
    match v {
        Value::Object(m) => !m.is_empty(),
        Value::Array(a) => !a.is_empty(),
        _ => false,
    }
}

/// One decision, plus what the encoder reported spending on it.
#[derive(Debug, Clone)]
pub struct Decision {
    /// Answers by question id, in request order.
    pub answers: Vec<(String, Answer)>,
    pub encoder_tokens: u64,
}

/// The engine: heads, an encoder and the decision path.
pub struct Engine<E: Encoder> {
    pub heads: Heads,
    pub encoder: E,
}

impl<E: Encoder> Engine<E> {
    pub fn new(heads: Heads, encoder: E) -> Self {
        Self { heads, encoder }
    }

    /// Answer every question in the request.
    ///
    /// Every question's state text and every candidate text is embedded in one pass, the
    /// way the reference batches them, so the encoder sees one request per decision.
    pub fn decide(&self, request: &Request, temperature: f32) -> Result<Decision> {
        ensure!(
            temperature > 0.0 && temperature <= 100.0,
            "temperature must be in (0, 100]"
        );
        let prepared = request.prepare()?;

        let mut texts: Vec<String> = Vec::new();
        for p in &prepared {
            texts.push(p.state_text.clone());
            texts.extend(p.candidate_texts.iter().cloned());
        }
        let (vectors, encoder_tokens) = self.encoder.embed(&texts)?;
        ensure!(
            vectors.len() == texts.len(),
            "encoder returned {} vectors for {} texts",
            vectors.len(),
            texts.len()
        );

        // The vectors came back in the order the texts were sent: one state row followed
        // by that question's candidate rows, per question.
        let mut rows = vectors.into_iter();
        let mut answers = Vec::with_capacity(prepared.len());
        for p in &prepared {
            let state = rows.next().context("missing state vector")?;
            let mut candidates = Vec::with_capacity(p.candidate_texts.len());
            for _ in &p.candidate_texts {
                candidates.push(rows.next().context("missing candidate vector")?);
            }
            let probs = scoring::distribution(&self.heads, &state, &candidates, temperature)
                .with_context(|| format!("question {:?}", p.id))?;
            answers.push((
                p.id.clone(),
                scoring::answer(&p.question, &p.candidate_texts, &probs)?,
            ));
        }

        Ok(Decision {
            answers,
            encoder_tokens,
        })
    }
}

/// The `answers` object of a response, in request order.
pub fn answers_json(answers: &[(String, Answer)]) -> Value {
    let mut out = Map::new();
    for (id, answer) in answers {
        out.insert(id.clone(), answer_json(answer));
    }
    Value::Object(out)
}

/// The level key to level text map a `score` answer carries, in answer order.
fn legend_json(legend: &[(String, String)]) -> Value {
    let mut map = Map::new();
    for (key, text) in legend {
        map.insert(key.clone(), Value::String(text.clone()));
    }
    Value::Object(map)
}

fn probabilities(pairs: &[(String, f32)]) -> Value {
    let mut map = Map::new();
    for (k, p) in pairs {
        map.insert(k.clone(), serde_json::json!(p));
    }
    Value::Object(map)
}

/// One answer in the shape `client.py` parses.
pub fn answer_json(answer: &Answer) -> Value {
    match answer {
        Answer::Choice {
            choice,
            confidence,
            probabilities: p,
        } => serde_json::json!({
            "type": "choice",
            "choice": choice,
            "confidence": confidence,
            "probabilities": probabilities(p),
        }),
        Answer::Noul { noul } => serde_json::json!({"type": "noul", "noul": noul}),
        Answer::Score {
            score,
            confidence,
            legend,
            probabilities: p,
        } => serde_json::json!({
            "type": "score",
            "score": score,
            "confidence": confidence,
            "legend": legend_json(legend),
            "probabilities": probabilities(p),
        }),
    }
}
