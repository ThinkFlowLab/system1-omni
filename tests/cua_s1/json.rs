use super::*;

#[test]
fn number_encoding_does_not_change_user_objects() {
    let input =
        r#"{"a": {"$serde_json::private::Number": "1.5"}, "b": -0, "c": 18446744073709551616}"#;
    let value = Value::Object(parse(input.as_bytes()).unwrap());
    assert_eq!(value["a"]["$serde_json::private::Number"], "1.5");
    assert_eq!(
        dumps(&value),
        r#"{"a": {"$serde_json::private::Number": "1.5"}, "b": -0.0, "c": 1.8446744073709552e+19}"#
    );
    assert!(
        parse(br#"{"a": {"x": 1, "\u0078": 2}}"#)
            .unwrap_err()
            .contains("duplicate key")
    );
}
