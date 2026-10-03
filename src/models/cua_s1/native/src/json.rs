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
use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};

/// Decode a request body into its top-level object; the error is the 400 message.
pub fn parse(raw: &[u8]) -> Result<Map<String, Value>, String> {
    let mut de = serde_json::Deserializer::from_slice(raw);
    let value = NoDuplicates(0)
        .deserialize(&mut de)
        .and_then(|v| de.end().map(|()| v))
        .map_err(|e| format!("request body is not valid JSON: {e}"))?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err("request body must be a JSON object".into()),
    }
}

/// Builds a `Value` like serde_json does, but fails on a repeated key.
struct NoDuplicates(usize);

impl<'de> DeserializeSeed<'de> for NoDuplicates {
    type Value = Value;

    fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
        let raw = <&serde_json::value::RawValue as de::Deserialize>::deserialize(d)?;
        let text = raw.get();
        // Another workspace member may enable arbitrary_precision. Read number
        // tokens directly so its private map encoding cannot become user data.
        if matches!(text.as_bytes()[0], b'-' | b'0'..=b'9') {
            if text != "-0" {
                if let Ok(n) = text.parse::<i64>() {
                    return self.visit_i64(n);
                }
                if let Ok(n) = text.parse::<u64>() {
                    return self.visit_u64(n);
                }
            }
            let n = serde_json::from_str::<f64>(text).map_err(de::Error::custom)?;
            return self.visit_f64(n);
        }
        if self.0 >= 127 && matches!(text.as_bytes()[0], b'{' | b'[') {
            return Err(de::Error::custom("recursion limit exceeded"));
        }
        serde_json::Deserializer::from_str(text)
            .deserialize_any(self)
            .map_err(de::Error::custom)
    }
}

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
        while let Some(v) = seq.next_element_seed(NoDuplicates(self.0 + 1))? {
            items.push(v);
        }
        Ok(Value::Array(items))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut obj = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let v = map.next_value_seed(NoDuplicates(self.0 + 1))?;
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
    fn write_number_str<W: ?Sized + io::Write>(&mut self, w: &mut W, n: &str) -> io::Result<()> {
        if n.contains(['.', 'e', 'E']) {
            let x = serde_json::from_str::<f64>(n).map_err(io::Error::other)?;
            self.write_f64(w, x)
        } else {
            w.write_all(n.as_bytes())
        }
    }
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
mod tests {
    use super::*;

    fn err(body: &str) -> String {
        parse(body.as_bytes()).unwrap_err()
    }

    #[test]
    fn float_repr_matches_python_examples() {
        let cases = [
            (1.0, "1.0"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (-0.0, "-0.0"),
            (3.14e-07, "3.14e-07"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5.960464477539063e-08, "5.960464477539063e-08"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x), want, "{x:e}");
        }
    }

    #[test]
    fn dumps_matches_python() {
        let v = Value::Object(parse(br#"{"a": [1.0, 1e16, 1e-5, 0.0001, -0.0, 123456789012345678, 3.14e-07, true, null], "b": {}}"#).unwrap());
        assert_eq!(
            dumps(&v),
            r#"{"a": [1.0, 1e+16, 1e-05, 0.0001, -0.0, 123456789012345678, 3.14e-07, true, null], "b": {}}"#
        );
        let s = Value::String("\u{0}\u{1f}\u{7f}\u{2028}\"\\/\t\u{8}\u{c}é😀".into());
        assert_eq!(
            dumps(&s),
            "\"\\u0000\\u001f\u{7f}\u{2028}\\\"\\\\/\\t\\b\\fé😀\""
        );
    }

    #[test]
    fn rejects_what_the_contract_rejects() {
        assert_eq!(err("[]"), "request body must be a JSON object");
        for body in [
            r#"{"a": NaN}"#,
            r#"{"a": 1e400}"#,
            r#"{"a": "\ud800x"}"#,
            r#"{"a": 1, "b": 2, "a": 3}"#,
            r#"{"a": [1,]}"#,
            r#"{} x"#,
            "\u{feff}{}",
        ] {
            assert!(
                err(body).starts_with("request body is not valid JSON"),
                "{body}"
            );
        }
        assert!(err(r#"{"a": 1, "a": 2}"#).contains("duplicate key \"a\""));
        assert!(
            err(&format!(
                "{{\"a\": {}1{}}}",
                "[".repeat(200),
                "]".repeat(200)
            ))
            .contains("recursion limit")
        );
        assert!(parse(b"{\"a\": \"\xff\"}").is_err());
    }
}

#[cfg(test)]
#[path = "../../../../../tests/cua_s1/json.rs"]
mod json_regression_tests;
