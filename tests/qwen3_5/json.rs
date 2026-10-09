use super::*;

#[test]
fn precision_features_preserve_numbers_and_private_marker_keys() {
    let raw = br#"{"a":1.5,"b":1e-5,"$serde_json::private::Number":"literal","nested":{"$serde_json::private::RawValue":2}}"#;
    let values = parse(raw).unwrap();
    assert_eq!(values["a"], serde_json::json!(1.5));
    assert_eq!(values["$serde_json::private::Number"], "literal");
    assert_eq!(values["nested"]["$serde_json::private::RawValue"], 2);
    assert!(dumps(&Value::Object(values)).contains("\"b\": 1e-05"));
    assert!(err(r#"{"a":{"b":1,"b":2}}"#).contains("duplicate key"));
}

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
