//! Python's `repr()` of a JSON-decoded value, as `str(instructions)` writes it under the serving guide's
//! `READOUT_INSTR_STYLE=pyrepr`: dicts and lists with `, ` and `: `, strings in Python's quoting, `True`,
//! `False`, `None`, integers as written and floats as `repr(float)`.

use omni_qwen3_5_native::json::float_repr;
use serde_json::Value;
use std::fmt::Write as _;

pub fn repr(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out
}

fn write_value(out: &mut String, value: &Value) {
    match value {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(n) if n.is_i64() || n.is_u64() => out.push_str(&n.to_string()),
        Value::Number(n) => out.push_str(&float_repr(n.as_f64().expect("a finite JSON number"))),
        Value::String(s) => write_str(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(out, key);
                out.push_str(": ");
                write_value(out, item);
            }
            out.push('}');
        }
    }
}

/// `repr(str)`: single quotes unless the text has a `'` and no `"`; backslash, the quote, `\t`, `\n`, `\r`
/// escaped; other non-printable characters as `\xNN`, `\uNNNN` or `\UNNNNNNNN`.
fn write_str(out: &mut String, s: &str) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if !printable(c) => {
                let n = c as u32;
                if n < 0x100 {
                    write!(out, "\\x{n:02x}").unwrap();
                } else if n < 0x10000 {
                    write!(out, "\\u{n:04x}").unwrap();
                } else {
                    write!(out, "\\U{n:08x}").unwrap();
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
}

/// `str.isprintable()` for the characters a request is likely to carry: the space is printable; control
/// characters (Cc), the other separators (Zs, Zl, Zp) and the common format characters (Cf) are not.
/// Unassigned and private-use code points (Cn, Co) are taken as printable here, unlike Python.
fn printable(c: char) -> bool {
    let n = c as u32;
    !(n < 0x20
        || (0x7f..=0xa0).contains(&n)
        || n == 0xad
        || (0x600..=0x605).contains(&n)
        || n == 0x61c
        || n == 0x6dd
        || n == 0x70f
        || n == 0x1680
        || n == 0x180e
        || (0x2000..=0x200f).contains(&n)
        || (0x2028..=0x202f).contains(&n)
        || (0x205f..=0x2064).contains(&n)
        || (0x2066..=0x206f).contains(&n)
        || n == 0x3000
        || (0xd800..=0xdfff).contains(&n)
        || n == 0xfeff
        || (0xfff9..=0xfffb).contains(&n)
        || n == 0x110bd
        || n == 0x110cd
        || (0x1bca0..=0x1bca3).contains(&n)
        || (0x1d173..=0x1d17a).contains(&n)
        || n == 0xe0001
        || (0xe0020..=0xe007f).contains(&n))
}
