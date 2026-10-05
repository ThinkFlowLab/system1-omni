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
use serde_json::{Map, Number, Value};

/// Decode a request body into its top-level object; the error is the 400 message.
pub fn parse(raw: &[u8]) -> Result<Map<String, Value>, String> {
    let mut de = serde_json::Deserializer::from_slice(raw);
    let value = de
        .deserialize_any(NoDuplicates)
        .and_then(|v| de.end().map(|()| v))
        .map_err(|e| format!("request body is not valid JSON: {e}"))?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err("request body must be a JSON object".into()),
    }
}

/// Builds a `Value` like serde_json does, but fails on a repeated key.
struct NoDuplicates;

impl<'de> de::Deserialize<'de> for Wrapped {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(NoDuplicates).map(Wrapped)
    }
}

struct Wrapped(Value);

impl<'de> Visitor<'de> for NoDuplicates {
    type Value = Value;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a JSON value")
    }
    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_bool<E>(self, b: bool) -> Result<Value, E> {
        Ok(Value::Bool(b))
    }
    fn visit_i64<E>(self, n: i64) -> Result<Value, E> {
        Ok(Value::Number(n.into()))
    }
    fn visit_u64<E>(self, n: u64) -> Result<Value, E> {
        Ok(Value::Number(n.into()))
    }
    fn visit_f64<E: de::Error>(self, x: f64) -> Result<Value, E> {
        Number::from_f64(x)
            .map(Value::Number)
            .ok_or_else(|| E::custom("number out of range"))
    }
    fn visit_str<E>(self, s: &str) -> Result<Value, E> {
        Ok(Value::String(s.to_owned()))
    }
    fn visit_string<E>(self, s: String) -> Result<Value, E> {
        Ok(Value::String(s))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(Wrapped(v)) = seq.next_element()? {
            items.push(v);
        }
        Ok(Value::Array(items))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut obj = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let Wrapped(v) = map.next_value()?;
            if obj.contains_key(&key) {
                return Err(de::Error::custom(format_args!(
                    "duplicate key {}",
                    quote(&key)
                )));
            }
            obj.insert(key, v);
        }
        Ok(Value::Object(obj))
    }
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
