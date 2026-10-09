//! Request JSON, on serde_json.
//!
//! - [`parse`] rejects what the contract rejects with a 400: invalid JSON or UTF-8,
//!   `NaN`/`Infinity`, numbers out of range, lone surrogates and nesting deeper than
//!   serde_json's limit (all serde_json errors), and keys repeated in an object.
//! - [`dumps`] writes `json.dumps(value, ensure_ascii=False)`: separators `, ` and
//!   `: `, key order kept, floats as Python's `repr`.

use std::fmt::Write as _;
use std::io;

use serde::Serialize;
use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::value::RawValue;
use serde_json::{Map, Number, Value};

/// Decode a request body into its top-level object; the error is the 400 message.
pub fn parse(raw: &[u8]) -> Result<Map<String, Value>, String> {
    let value = serde_json::from_slice::<Box<RawValue>>(raw)
        .and_then(|raw| parse_value(&raw, 0))
        .map_err(|e| format!("request body is not valid JSON: {e}"))?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err("request body must be a JSON object".into()),
    }
}

// Read containers as raw JSON so feature unification with arbitrary_precision
// cannot confuse numeric values or legitimate private-marker object keys.
fn parse_value(raw: &RawValue, depth: usize) -> serde_json::Result<Value> {
    let text = raw.get();
    if !matches!(text.as_bytes()[0], b'{' | b'[') {
        let value: Value = serde_json::from_str(text)?;
        return if let Value::Number(n) = value {
            if n.is_i64() || n.is_u64() {
                Ok(Value::Number(n))
            } else {
                n.as_f64()
                    .and_then(Number::from_f64)
                    .map(Value::Number)
                    .ok_or_else(|| de::Error::custom("number out of range"))
            }
        } else {
            Ok(value)
        };
    }
    if depth >= 127 {
        return Err(de::Error::custom("recursion limit exceeded"));
    }
    struct Container(usize);
    impl<'de> Visitor<'de> for Container {
        type Value = Value;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a JSON object or array")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
            let mut items = Vec::new();
            while let Some(raw) = seq.next_element::<Box<RawValue>>()? {
                items.push(parse_value(&raw, self.0 + 1).map_err(de::Error::custom)?);
            }
            Ok(Value::Array(items))
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
            let mut obj = Map::new();
            while let Some((key, raw)) = map.next_entry::<String, Box<RawValue>>()? {
                if obj.contains_key(&key) {
                    return Err(de::Error::custom(format_args!(
                        "duplicate key {}",
                        quote(&key)
                    )));
                }
                obj.insert(
                    key,
                    parse_value(&raw, self.0 + 1).map_err(de::Error::custom)?,
                );
            }
            Ok(Value::Object(obj))
        }
    }
    serde_json::Deserializer::from_str(text).deserialize_any(Container(depth))
}

/// A string as a JSON literal, which is also how error messages quote names.
pub fn quote(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}

/// `json.dumps(value, ensure_ascii=False)`.
pub fn dumps(value: &Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyFormatter);
    value.serialize(&mut ser).expect("a Value serializes");
    String::from_utf8(out).expect("serde_json writes UTF-8")
}

/// serde_json's compact output with Python's separators and float format; its string
/// escaping (`"`, `\\` and control characters, `\u00XX` in lowercase) is Python's.
struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
    fn write_number_str<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        value: &str,
    ) -> io::Result<()> {
        if value.contains(['.', 'e', 'E']) {
            let x: f64 = value.parse().map_err(io::Error::other)?;
            w.write_all(float_repr(x).as_bytes())
        } else {
            w.write_all(value.as_bytes())
        }
    }
    fn write_f64<W: ?Sized + io::Write>(&mut self, w: &mut W, x: f64) -> io::Result<()> {
        w.write_all(float_repr(x).as_bytes())
    }
}

/// Python's `repr(float)`: the shortest digits that round-trip, in fixed notation
/// for exponents from -5 to 15 and scientific notation otherwise. (Python breaks the
/// rare exact ties between two shortest candidates to even; this does not.)
pub fn float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let (neg, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    let decpt = exp + 1;
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let sign = if exp < 0 { '-' } else { '+' };
        write!(out, "e{sign}{:02}", exp.unsigned_abs()).unwrap();
    } else if decpt <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-decpt) as usize));
        out.push_str(&digits);
    } else if decpt as usize >= digits.len() {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', decpt as usize - digits.len()));
        out.push_str(".0");
    } else {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    out
}

#[cfg(test)]
#[path = "../../../../../tests/qwen3_5/json.rs"]
mod tests;
