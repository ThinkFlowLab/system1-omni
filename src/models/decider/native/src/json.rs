//! Model-private strict JSON decoding avoids arbitrary_precision feature-unification traps.
use anyhow::{Context, Result, bail, ensure};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value, value::RawValue};
use std::fmt;

pub fn parse(raw: &[u8]) -> Result<Value> {
    let raw: Box<RawValue> = serde_json::from_slice(raw).context("invalid request JSON")?;
    parse_value(&raw, 0)
}
fn parse_value(raw: &RawValue, depth: usize) -> Result<Value> {
    let text = raw.get();
    if !matches!(text.as_bytes()[0], b'{' | b'[') {
        if text.as_bytes()[0].is_ascii_digit() || text.starts_with('-') {
            let number = if text.contains(['.', 'e', 'E']) {
                let x: f64 = text.parse()?;
                Number::from_f64(x).context("nonfinite or out-of-range JSON float")?
            } else if text.starts_with('-') {
                Number::from(
                    text.parse::<i64>()
                        .context("JSON integer outside i64/u64 range")?,
                )
            } else {
                Number::from(
                    text.parse::<u64>()
                        .context("JSON integer outside i64/u64 range")?,
                )
            };
            return Ok(Value::Number(number));
        }
        return Ok(serde_json::from_str(text)?);
    }
    ensure!(depth < 127, "JSON nesting exceeds supported depth");
    struct Container(usize);
    impl<'de> Visitor<'de> for Container {
        type Value = Value;
        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("JSON array or object")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Value, A::Error> {
            let mut out = Vec::new();
            while let Some(raw) = seq.next_element::<Box<RawValue>>()? {
                out.push(parse_value(&raw, self.0 + 1).map_err(de::Error::custom)?);
            }
            Ok(Value::Array(out))
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Value, A::Error> {
            let mut out = Map::new();
            while let Some((key, raw)) = map.next_entry::<String, Box<RawValue>>()? {
                if out.contains_key(&key) {
                    return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
                }
                out.insert(
                    key,
                    parse_value(&raw, self.0 + 1).map_err(de::Error::custom)?,
                );
            }
            Ok(Value::Object(out))
        }
    }
    use serde::Deserializer;
    Ok(serde_json::Deserializer::from_str(text).deserialize_any(Container(depth))?)
}
pub fn object(value: &Value) -> Result<&Map<String, Value>> {
    value.as_object().context("expected JSON object")
}
pub fn text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| dumps(value))
}
pub fn dumps(value: &Value) -> String {
    match value {
        Value::String(s) => serde_json::to_string(s).expect("string serialization"),
        Value::Array(a) => format!("[{}]", a.iter().map(dumps).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!(
                    "{}: {}",
                    serde_json::to_string(k).expect("key serialization"),
                    dumps(v)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Number(n) => {
            let s = n.to_string();
            if s.contains(['.', 'e', 'E']) {
                float_repr(n.as_f64().expect("validated float"))
            } else {
                s
            }
        }
        _ => value.to_string(),
    }
}
// Python repr notation thresholds, exponent sign/padding, and integer-valued floats.
// Uses Rust's shortest round-trip digits; rare shortest-decimal tie choices may differ.
fn float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let sci = format!("{x:e}");
    let (m, e) = sci.split_once('e').expect("scientific exponent");
    let exponent: i32 = e.parse().expect("integer exponent");
    let neg = m.starts_with('-');
    let digits = m.trim_start_matches('-').replace('.', "");
    let decpt = exponent + 1;
    let sign = if neg { "-" } else { "" };
    if decpt <= -4 || decpt > 16 {
        let fraction = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        format!(
            "{sign}{}{fraction}e{}{:02}",
            &digits[..1],
            if exponent < 0 { '-' } else { '+' },
            exponent.unsigned_abs()
        )
    } else if decpt <= 0 {
        format!("{sign}0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt as usize >= digits.len() {
        format!(
            "{sign}{}{}.0",
            digits,
            "0".repeat(decpt as usize - digits.len())
        )
    } else {
        format!(
            "{sign}{}.{}",
            &digits[..decpt as usize],
            &digits[decpt as usize..]
        )
    }
}
pub fn required<'a>(m: &'a Map<String, Value>, key: &str) -> Result<&'a Value> {
    m.get(key).with_context(|| format!("missing {key}"))
}
pub fn default_mode(m: &Map<String, Value>, key: &str, expected: bool) -> Result<()> {
    if let Some(v) = m.get(key)
        && v.as_bool() != Some(expected)
    {
        bail!("unsupported {key}: expected {expected}");
    }
    Ok(())
}

// Python float(string) accepts surrounding Python whitespace, digit separators,
// and Unicode decimal (Nd) digits. Normalize these before Rust's numeric parser.
pub fn python_float(key: &str) -> Result<f64> {
    let normalized: String = key
        .trim_matches(crate::text::float_whitespace)
        .chars()
        .map(|c| crate::text::decimal(c).unwrap_or(c))
        .collect();
    let bytes = normalized.as_bytes();
    for (i, c) in bytes.iter().enumerate() {
        if *c == b'_' {
            ensure!(
                i > 0
                    && i + 1 < bytes.len()
                    && bytes[i - 1].is_ascii_digit()
                    && bytes[i + 1].is_ascii_digit(),
                "invalid digit separator in numeric text"
            );
        }
    }
    normalized
        .replace('_', "")
        .parse::<f64>()
        .context("numeric text must parse as float")
}
