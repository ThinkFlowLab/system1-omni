//! Golden pins for the JEV-27B-VL System-1 prompt contract and the upstream
//! error envelope; the strings mirror serve_decide.py's s1_pass exactly.

use omni_jev_vl_native::contract::{Kind, compile};

fn labels() -> Vec<String> {
    let all: Vec<String> = ('A'..='Z')
        .map(|c| c.to_string())
        .chain(('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}"))))
        .collect();
    all[..256].to_vec()
}

fn compile_str(raw: &str) -> omni_jev_vl_native::contract::Compiled {
    compile(raw.as_bytes(), &labels()).unwrap()
}

#[test]
fn noul_prompt_matches_serve_decide() {
    let c = compile_str(
        r#"{"kind":"noul","state":"Dialog: Update installed.","question":"Close it?"}"#,
    );
    assert_eq!(c.kind, Kind::Noul);
    assert_eq!(c.options, ["false", "true"]);
    assert_eq!(
        c.prompt,
        "[kind] noul\n[state] Dialog: Update installed.\n[question] Close it?\n[options]\nfalse\ntrue\n[decision]:"
    );
}

#[test]
fn score_prompt_uses_digits_lines() {
    let c = compile_str(r#"{"kind":"score","state":"x","question":"Rate it."}"#);
    assert_eq!(c.options, ["0", "1", "2", "3", "4", "5"]);
    assert_eq!(
        c.prompt,
        "[kind] score\n[state] x\n[question] Rate it.\n[options]\n0\n1\n2\n3\n4\n5\n[decision]:"
    );
}

#[test]
fn choice_prompt_labels_options() {
    let c = compile_str(
        r#"{"kind":"choice","state":"Dialog: x.","question":"Pick one.","options":["yes","no","later"]}"#,
    );
    assert_eq!(
        c.prompt,
        "[kind] choice\n[state] Dialog: x.\n[question] Pick one.\n[options]\nA) yes\nB) no\nC) later\n[decision]:"
    );
}

#[test]
fn dict_state_renders_like_python_json_dumps() {
    let c = compile_str(r#"{"kind":"noul","state":{"b":1,"a":"x"},"question":"q?"}"#);
    // Python json.dumps(ensure_ascii=False): ", " and ": " separators, order kept.
    assert!(
        c.prompt.contains(r#"[state] {"b": 1, "a": "x"}"#),
        "{}",
        c.prompt
    );
}

#[test]
fn list_state_parts_concatenate_without_separator() {
    let c = compile_str(
        r#"{"kind":"score","state":["one","two",{"type":"text","text":"three"}],"question":"q?"}"#,
    );
    assert!(c.prompt.contains("[state] onetwothree\n"), "{}", c.prompt);
}

#[test]
fn image_state_becomes_placeholder_and_records_url() {
    let c = compile_str(
        r#"{"kind":"noul","state":["shot",{"image":"https://x/y.png"}],"question":"q?"}"#,
    );
    assert_eq!(c.images, ["https://x/y.png"]);
    assert!(
        c.prompt
            .contains("[state] shot<|vision_start|><|image_pad|><|vision_end|>\n"),
        "{}",
        c.prompt
    );
}

#[test]
fn image_url_part_matches_upstream_image_shorthand() {
    let shorthand = compile_str(
        r#"{"kind":"noul","state":["shot",{"image":"data:image/png;base64,AAAA"}],"question":"q?"}"#,
    );
    let typed = compile_str(
        r#"{"kind":"noul","state":["shot",{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}],"question":"q?"}"#,
    );
    assert_eq!(typed.images, shorthand.images);
    assert_eq!(typed.prompt, shorthand.prompt);
}

#[test]
fn system2_only_cannot_silently_return_system1() {
    for value in ["true", "1", r#""true""#] {
        let r = reject(&format!(
            r#"{{"kind":"noul","question":"q?","system2_only":{value}}}"#,
        ));
        assert_eq!(r.status, 400);
        assert_eq!(
            r.body["error"]["message"],
            "system2_only is not implemented by this worker"
        );
    }
    compile_str(r#"{"kind":"noul","question":"q?","system2_only":false}"#);
}

fn reject(raw: &str) -> omni_jev_vl_native::contract::Reject {
    compile(raw.as_bytes(), &labels()).unwrap_err()
}

#[test]
fn choice_option_count_error_keeps_upstream_message() {
    let r = reject(r#"{"kind":"choice","state":"x","question":"q?","options":["only"]}"#);
    assert_eq!(r.status, 400);
    assert_eq!(
        r.body["error"]["message"],
        "choice needs 2-256 options, got 1"
    );
    assert_eq!(r.body["error"]["type"], "BadRequestError");
    assert_eq!(r.body["error"]["code"], 400);
}

#[test]
fn kind_literal_error_mirrors_pydantic_envelope() {
    let r = reject(r#"{"kind":"bogus","state":"x","question":"q?"}"#);
    assert_eq!(r.status, 400);
    assert_eq!(
        r.body["error"]["message"],
        "1 validation error:\n  {'type': 'literal_error', 'loc': 'body.kind', 'msg': \"Input should be 'noul', 'score' or 'choice'\", 'input': 'bogus', 'ctx': {'expected': \"'noul', 'score' or 'choice'\"}}"
    );
    assert_eq!(r.body["error"]["type"], "Bad Request");
    assert_eq!(r.body["error"]["param"], "kind");
    assert_eq!(r.body["error"]["code"], 400);
}

#[test]
fn missing_question_mirrors_pydantic_envelope() {
    let r = reject(r#"{"kind":"noul","state":"x"}"#);
    assert_eq!(
        r.body["error"]["message"],
        "1 validation error:\n  {'type': 'missing', 'loc': 'body.question', 'msg': 'Field required', 'input': '<dict of 2 items>'}"
    );
    assert_eq!(r.body["error"]["param"], "question");
}

#[test]
fn score_thinking_on_keeps_upstream_message() {
    let r = reject(r#"{"kind":"score","state":"x","question":"q?","thinking":"on"}"#);
    assert_eq!(
        r.body["error"]["message"],
        "thinking is supported for noul and choice"
    );
}

#[test]
fn decision_controls_reject_non_strings() {
    for field in ["thinking", "strategy"] {
        for value in ["true", "null", "42", "[]", "{}"] {
            let r = reject(&format!(
                r#"{{"kind":"noul","question":"q?","{field}":{value}}}"#,
            ));
            assert_eq!(r.status, 400, "{field}: {value}");
        }
    }
}

#[test]
fn supported_decision_controls_remain_valid() {
    for thinking in ["default", "off"] {
        for strategy in ["auto", "single"] {
            compile_str(&format!(
                r#"{{"kind":"noul","question":"q?","thinking":"{thinking}","strategy":"{strategy}"}}"#,
            ));
        }
    }
}

#[test]
fn wrong_content_type_body_mirrors_bytes_error() {
    let raw = br#"{"kind":"noul","state":"x","question":"q?"}"#;
    // Non-JSON content type: upstream validates the raw bytes object.
    let r = omni_jev_vl_native::contract::Reject::validation(
        format!(
            "1 validation error:\n  {{'type': 'model_attributes_type', 'loc': 'body', 'msg': 'Input should be a valid dictionary or object to extract fields from', 'input': '<bytes of {} bytes>'}}",
            raw.len()
        ),
        "body",
    );
    assert_eq!(
        r.body["error"]["message"],
        "1 validation error:\n  {'type': 'model_attributes_type', 'loc': 'body', 'msg': 'Input should be a valid dictionary or object to extract fields from', 'input': '<bytes of 43 bytes>'}"
    );
}

#[test]
fn unknown_model_keeps_404_envelope() {
    let r = reject(r#"{"model":"no-such-model","kind":"noul","question":"q?"}"#);
    assert_eq!(r.status, 404);
    assert_eq!(
        r.body["error"]["message"],
        "The model `no-such-model` does not exist."
    );
    assert_eq!(r.body["error"]["type"], "NotFoundError");
    assert_eq!(r.body["error"]["param"], "model");
    assert_eq!(r.body["error"]["code"], 404);
}
