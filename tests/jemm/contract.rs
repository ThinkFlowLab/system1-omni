use omni_jemm_native::contract::{answer, compile};
#[test]
fn prompt_preserves_order_and_compact_state() {
    let qs=compile(br#"{"state":{"z":1,"a":true},"questions":{"q":{"instructions":"  choose\n one ","criteria":{"last":"last  option","first":{"description":"first"}}}}}"#).unwrap();
    assert_eq!(qs[0].keys, ["last", "first"]);
    assert_eq!(
        qs[0].prompt,
        "State:\n{\"z\":1,\"a\":true}\n\nQuestion: choose one\n\nCandidates:\nA) last option\nB) first\n\nAnswer with exactly one candidate label."
    );
}

#[test]
fn structured_state_preserves_python_float_shortest_ties() {
    let questions =
        compile(br#"{"state":{"x":1000000000000000.2},"questions":{"q":{"type":"noul"}}}"#)
            .unwrap();
    assert_eq!(
        questions[0].prompt,
        "State:\n{\"x\":1000000000000000.2}\n\nQuestion: \n\nCandidates:\nA) yes\nB) no\n\nAnswer with exactly one candidate label."
    );
}

#[test]
fn numeric_instructions_and_descriptions_preserve_python_float_text() {
    let questions = compile(
        br#"{"questions":{"q":{"instructions":[1000000000000000.2,-0.0],"criteria":{"a":{"description":-250786113075992.62},"b":"fallback"}}}}"#,
    )
    .unwrap();
    assert_eq!(
        questions[0].prompt,
        "State:\n\n\nQuestion: [1000000000000000.2, -0.0]\n\nCandidates:\nA) -250786113075992.62\nB) fallback\n\nAnswer with exactly one candidate label."
    );
}
#[test]
fn noul_uses_yes_first_and_score_expected_value() {
    let qs = compile(
        br#"{"questions":{"n":{"type":"noul"},"s":{"type":"score","criteria":[null,"mid","hi"]}}}"#,
    )
    .unwrap();
    let n = answer(&qs[0], &[0., 0.], 1.3480874159655591).unwrap();
    assert_eq!(n["noul"], 0.5);
    assert_eq!(n["confidence"], 0.5);
    let s = answer(&qs[1], &[0., 0., 0.], 1.).unwrap();
    assert!((s["expected_value"].as_f64().unwrap() - 1.).abs() < 1e-12);
    assert!(s.get("score").is_none());
}
#[test]
fn renders_tool_parameters_and_python_coercion() {
    let qs=compile(br#"{"questions":{"q":{"instructions":null,"criteria":{"t":{"description":"{\"name\":\"run\",\"description\":\" Do  it \",\"parameters\":{\"properties\":{\"x\":{\"default\":true},\"y\":{}},\"required\":[\"y\"]}}","action":{"tool_name":"run"}},"n":false}}}}"#).unwrap();
    assert!(qs[0].prompt.contains("Question: None"));
    assert!(
        qs[0]
            .prompt
            .contains("A) run — Do it | params: x=True, y*\nB) False")
    );
}
#[test]
fn rejects_candidate_limits_bad_kinds_and_nonfinite_scores() {
    for body in [
        br#"{"questions":{"q":{"criteria":{"a":"one"}}}}"#.as_slice(),
        br#"{"questions":{"q":{"type":"unknown"}}}"#,
    ] {
        assert!(compile(body).is_err());
    }
    let q = compile(br#"{"questions":{"q":{"criteria":{"a":"one","b":"two"}}}}"#).unwrap();
    assert!(answer(&q[0], &[f32::NAN, 0.], 1.).is_err());
    let a = answer(&q[0], &[1., 1.], 1.).unwrap();
    assert_eq!(a["choice"], "a");
}
#[test]
fn rejects_non_string_type_and_supports_all_32_labels() {
    assert!(
        compile(br#"{"questions":{"q":{"type":null,"criteria":{"a":"one","b":"two"}}}}"#).is_err()
    );
    let mut criteria = serde_json::Map::new();
    for i in 0..32 {
        criteria.insert(i.to_string(), serde_json::json!(format!("candidate{i}")));
    }
    let raw = serde_json::json!({"questions":{"q":{"criteria":criteria}}});
    let qs = compile(&serde_json::to_vec(&raw).unwrap()).unwrap();
    assert!(
        qs[0]
            .prompt
            .ends_with("5) candidate31\n\nAnswer with exactly one candidate label.")
    );
    let mut raw = raw;
    raw["questions"]["q"]["criteria"]["extra"] = serde_json::json!("33");
    assert!(compile(&serde_json::to_vec(&raw).unwrap()).is_err());
}
#[test]
fn python_container_repr_preserves_combining_marks_and_escapes_private_use() {
    use omni_jemm_native::contract::py_str;
    assert_eq!(
        py_str(&serde_json::json!(["e\u{0300}", "\u{e000}", true, null])),
        "['e\u{0300}', '\\ue000', True, None]"
    );
}
#[test]
fn python_unicode15_repr_handles_nko_arabic_controls_and_unassigned() {
    use omni_jemm_native::contract::py_str;
    assert_eq!(
        py_str(&serde_json::json!([
            "\u{07fd}",
            "\u{08d3}\u{08db}",
            "\u{0001}",
            "\u{e000}",
            "\u{0378}"
        ])),
        "['\u{07fd}', '\u{08d3}\u{08db}', '\\x01', '\\ue000', '\\u0378']"
    );
}
#[test]
fn integer_negative_zero_matches_python_and_big_integers_are_rejected() {
    let q = compile(br#"{"state":-0,"questions":{"q":{"criteria":{"a":"a","b":"b"}}}}"#).unwrap();
    assert!(q[0].prompt.starts_with("State:\n0\n"));
    let error = compile(
        br#"{"state":18446744073709551616,"questions":{"q":{"criteria":{"a":"a","b":"b"}}}}"#,
    )
    .unwrap_err();
    assert!(error.to_string().contains("integer outside i64/u64"));
    assert!(
        compile(
            br#"{"state":-9223372036854775809,"questions":{"q":{"criteria":{"a":"a","b":"b"}}}}"#
        )
        .is_err()
    );
}
#[test]
fn aggregate_prompt_budget_reserves_before_allocation_and_caps_questions() {
    use omni_jemm_native::contract::reserve_prompt_bytes;
    let mut total = 0;
    reserve_prompt_bytes(&mut total, 4, 8).unwrap();
    assert!(reserve_prompt_bytes(&mut total, 5, 8).is_err());
    assert_eq!(total, 4);
    let questions = (0..65)
        .map(|i| {
            (
                i.to_string(),
                serde_json::json!({"criteria":{"a":"a","b":"b"}}),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    assert!(
        compile(&serde_json::to_vec(&serde_json::json!({"questions":questions})).unwrap()).is_err()
    );
}
#[test]
fn recognized_tool_json_rejects_nonfinite_defaults_under_workspace_features() {
    let body = serde_json::json!({"questions":{"q":{"criteria":{"a":{"description":"{\"name\":\"run\",\"parameters\":{\"properties\":{\"x\":{\"default\":1e400}}}}","action":{"tool_name":"run"}},"b":"fallback"}}}});
    assert!(compile(&serde_json::to_vec(&body).unwrap()).is_err());
}
#[test]
fn tool_defaults_preserve_integer_zero_bounds_and_fallbacks() {
    fn tool_body(description: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({"questions":{"q":{"criteria":{"a":{"description":description,"action":{"tool_name":"run"}},"b":"fallback"}}}})).unwrap()
    }
    for (default, expected) in [
        ("-0", "0"),
        ("18446744073709551615", "18446744073709551615"),
    ] {
        let text = r#"{"name":"run","parameters":{"properties":{"x":{"default":DEFAULT}}}}"#
            .replace("DEFAULT", default);
        let q = compile(&tool_body(&text)).unwrap();
        assert!(
            q[0].prompt
                .contains(&format!("A) run | params: x={expected}"))
        );
    }
    assert!(
        compile(&tool_body(
            "{\"name\":\"run\",\"parameters\":{\"x\":{\"default\":18446744073709551616}}}"
        ))
        .is_err()
    );
    for text in ["{broken json", "{\"default\":1e400}"] {
        let q = compile(&tool_body(text)).unwrap();
        assert!(q[0].prompt.contains(&format!("A) {text}")));
    }
}

#[test]
fn systemone_accepts_nine_questions_and_enforces_native_resource_cap() {
    use serde_json::{Map, json};
    for count in [1, 8, 9, 64, 65] {
        let questions: Map<String, serde_json::Value> = (0..count)
            .map(|i| (format!("q{i}"), json!({"type":"noul"})))
            .collect();
        let result = compile(&serde_json::to_vec(&json!({"questions":questions})).unwrap());
        if count <= 64 {
            let compiled = result.unwrap();
            assert_eq!(compiled.len(), count);
            assert_eq!(compiled.last().unwrap().id, format!("q{}", count - 1));
        } else {
            assert!(result.unwrap_err().to_string().contains("1 to 64"));
        }
    }
}
