//! The text the two heads see, checked byte-for-byte against the reference.
//!
//! `state_text`, `candidates` and `to_text` decide what reaches the encoder, and getting
//! one separator wrong shifts every probability without failing anything. The oracle is
//! produced by the reference implementation itself
//! (`recipe/clm/native/text_oracle.py`, which imports `clm.schema`), so this compares
//! against the real thing rather than a transcription of it.
use omni_clm::serve::{answer_json, candidates, state_text, to_text};
use omni_clm::{Kind, Question, Request, answer, serve::QuestionRequest};
use omni_clm::{NumberLiterals, to_text_json};
use serde_json::{Map, Value, json};

fn oracle() -> Value {
    let path = std::env::var_os("CLM_TEXT_ORACLE")
        .expect("set CLM_TEXT_ORACLE to the file text_oracle.py writes");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn states() -> Vec<Value> {
    vec![
        json!("I was charged twice."),
        json!({"body": "Charged twice", "order": 4411, "urgent": true}),
        json!({"ticket": {"id": 7, "tags": ["a", "b"]}, "note": null}),
        json!([{"k": 1}, {"k": 2}]),
        json!({"empty_obj": {}, "empty_arr": [], "n": 0.5}),
        json!({"nested": {"deep": {"x": "y"}}}),
        // `str(float)` switches to exponent form outside [1e-4, 1e16) and keeps a `.0` on
        // an integral float, so these pin the number spelling the reference produces.
        json!({"tiny": 1e-5, "smaller": 1e-7, "edge": 1e-4, "round": 1e15, "huge": 1e16, "neg": -1e-6}),
        // Parsed from the literal text rather than built from an `f64`, because
        // the parse is the part that was wrong.
        serde_json::from_str(r#"{"seventeen": 7.8190461323667115, "inexact": 9007199254740993.0}"#)
            .unwrap(),
        serde_json::from_str(concat!(
            r#"{"big": 18446744073709551616, "#,
            r#""huge": 340282366920938463463374607431768211456, "#,
            r#""negzero": -0, "negbig": -18446744073709551616}"#
        ))
        .unwrap(),
    ]
}

/// The same states as [`states`], as the JSON text a request arrives as.
fn state_texts() -> Vec<&'static str> {
    vec![
        r#""I was charged twice.""#,
        r#"{"body": "Charged twice", "order": 4411, "urgent": true}"#,
        r#"{"ticket": {"id": 7, "tags": ["a", "b"]}, "note": null}"#,
        r#"[{"k": 1}, {"k": 2}]"#,
        r#"{"empty_obj": {}, "empty_arr": [], "n": 0.5}"#,
        r#"{"nested": {"deep": {"x": "y"}}}"#,
        r#"{"tiny": 1e-5, "smaller": 1e-7, "edge": 1e-4, "round": 1e15, "huge": 1e16, "neg": -1e-6}"#,
        r#"{"seventeen": 7.8190461323667115, "inexact": 9007199254740993.0}"#,
        concat!(
            r#"{"big": 18446744073709551616, "#,
            r#""huge": 340282366920938463463374607431768211456, "#,
            r#""negzero": -0, "negbig": -18446744073709551616}"#
        ),
    ]
}

fn questions() -> Vec<QuestionRequest> {
    vec![
        QuestionRequest {
            kind: Kind::Choice,
            instructions: "Which team?".into(),
            criteria: Some(json!({"billing": "Charges and refunds", "tech": "Software problems"})),
        },
        QuestionRequest {
            kind: Kind::Choice,
            instructions: "Pick".into(),
            criteria: Some(json!({"a": "", "b": null})),
        },
        QuestionRequest {
            kind: Kind::Score,
            instructions: "How urgent?".into(),
            criteria: Some(json!(["Not urgent", "Soon", "Now"])),
        },
        QuestionRequest {
            kind: Kind::Noul,
            instructions: "Does the customer ask for a refund?".into(),
            criteria: None,
        },
        QuestionRequest {
            kind: Kind::Noul,
            instructions: "Refund?".into(),
            criteria: Some(json!({"true": "Yes they do", "false": "No they do not"})),
        },
        // An empty container renders to nothing, which is not the same as being absent.
        QuestionRequest {
            kind: Kind::Choice,
            instructions: "Pick a bucket".into(),
            criteria: Some(json!({"empty_obj": {}, "empty_list": []})),
        },
        QuestionRequest {
            kind: Kind::Noul,
            instructions: "Is it so?".into(),
            criteria: Some(json!({"true": {}, "false": []})),
        },
        // Numbers are spelled by `str(float)` in criteria too, not only in the state.
        QuestionRequest {
            kind: Kind::Score,
            instructions: "How much?".into(),
            criteria: Some(json!([1e-5, 0.5, 1e16])),
        },
    ]
}

#[test]
#[ignore = "requires CLM_TEXT_ORACLE from recipe/clm/native/text_oracle.py; CPU only"]
fn to_text_matches_the_reference_byte_for_byte() {
    let oracle = oracle();
    let expected = oracle["to_text"].as_array().unwrap();
    let got = states();
    assert_eq!(
        got.len(),
        expected.len(),
        "the oracle was built from another case list"
    );
    // All but the last: a `Value` cannot hold an integer above `u64::MAX`, so the digits
    // are already gone before `to_text` is called. That case is the next loop's.
    for (i, state) in got.iter().take(expected.len() - 1).enumerate() {
        assert_eq!(
            to_text(state),
            expected[i].as_str().unwrap(),
            "to_text case {i} for {state}"
        );
    }

    // The same renderings again, from the text, which is the path a request takes and
    // the only one that keeps an integer literal's digits.
    for (i, raw) in state_texts().iter().enumerate() {
        assert_eq!(
            to_text_json(raw).unwrap(),
            expected[i].as_str().unwrap(),
            "to_text_json case {i}"
        );
    }
}

#[test]
#[ignore = "requires CLM_TEXT_ORACLE from recipe/clm/native/text_oracle.py; CPU only"]
fn state_text_and_candidates_match_the_reference_byte_for_byte() {
    let oracle = oracle();
    let cases = oracle["cases"].as_array().unwrap();
    let states = state_texts();
    let questions = questions();
    assert_eq!(
        cases.len(),
        states.len() * questions.len(),
        "the oracle was built from another case list"
    );

    let mut i = 0;
    for raw in &states {
        for q in &questions {
            let case = &cases[i];
            // Through the request path rather than `state_text`/`candidates` on a
            // `Value`: the text is what keeps an integer literal's digits, and a `Value`
            // has already lost the ones above `u64::MAX`.
            let line = format!(
                r#"{{"state": {raw}, "questions": {{"q": {}}}}}"#,
                question_json(q)
            );
            let prepared = Request::parse_line(&line).unwrap().prepare().unwrap();
            assert_eq!(
                prepared[0].state_text,
                case["state_text"].as_str().unwrap(),
                "case {i} state_text"
            );
            assert_eq!(
                prepared[0].question.keys,
                case["keys"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect::<Vec<_>>(),
                "case {i} keys"
            );
            assert_eq!(
                prepared[0].candidate_texts,
                case["candidate_texts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect::<Vec<_>>(),
                "case {i} candidate_texts"
            );
            i += 1;
        }
    }
}

/// A question as the JSON text a request carries it in, for the loop above.
fn question_json(q: &QuestionRequest) -> Value {
    let mut object = Map::new();
    object.insert(
        "type".to_string(),
        json!(match q.kind {
            Kind::Choice => "choice",
            Kind::Score => "score",
            Kind::Noul => "noul",
        }),
    );
    object.insert("instructions".to_string(), json!(q.instructions));
    if let Some(criteria) = &q.criteria {
        object.insert("criteria".to_string(), criteria.clone());
    }
    Value::Object(object)
}

#[test]
fn text_construction_handles_the_edges_the_oracle_does_not() {
    // No state in the oracle is empty, and no question leaves `instructions` blank.
    assert_eq!(state_text(&json!("context"), "  "), "context");
    assert_eq!(state_text(&json!(""), "question"), "question");
    // A state that is a bare scalar is not a container, so it renders unindented.
    assert_eq!(to_text(&json!(true)), "true");
    assert_eq!(to_text(&json!(null)), "");
}

#[test]
fn a_score_answer_carries_its_level_legend() {
    // `answer_from_probs` returns the criteria text next to the score so a consumer can
    // read a level back without the request; the serialized answer must not drop it.
    let request = QuestionRequest {
        kind: Kind::Score,
        instructions: "How urgent?".into(),
        criteria: Some(json!(["Not urgent", "Soon", "Now"])),
    };
    let (keys, texts) = candidates(&request).unwrap();
    let question = Question {
        id: "urgency".into(),
        kind: Kind::Score,
        keys,
    };
    let answer = answer(&question, &texts, &[0.1, 0.7, 0.2]).unwrap();
    let json = answer_json(&answer);
    assert_eq!(json["type"], "score");
    assert_eq!(
        json["legend"],
        json!({"0": "Not urgent", "1": "Soon", "2": "Now"})
    );
    assert!((json["score"].as_f64().unwrap() - 1.1).abs() < 1e-6);
}

/// The whole path: a JSON literal to the string the heads see.
///
/// `serde_json`'s default float parsing is not correctly rounded, so a literal
/// with more digits than a double holds can land on the neighbouring double and
/// render as a different number — `7.8190461323667115` came out
/// `7.819046132366712`, which is a different embedding input. Formatting an
/// already-parsed float correctly does not help; the parse has to be right too,
/// and `float_roundtrip` is what makes it match `json.loads`.
#[test]
fn json_numbers_parse_to_the_same_doubles_as_the_reference() {
    // Both columns are what CPython's `json.loads` then `str` produce.
    for (literal, bits, text) in [
        (
            "7.8190461323667115",
            0x401f_46b4_0781_b8a4,
            "7.8190461323667115",
        ),
        (
            "9007199254740993.0",
            0x4340_0000_0000_0000,
            "9007199254740992.0",
        ),
        ("0.1", 0x3fb9_9999_9999_999a, "0.1"),
        (
            "3.141592653589793238",
            0x4009_21fb_5444_2d18,
            "3.141592653589793",
        ),
    ] {
        let parsed: Value = serde_json::from_str(literal).unwrap();
        assert_eq!(
            parsed.as_f64().unwrap().to_bits(),
            bits,
            "parsing {literal}"
        );
        assert_eq!(to_text(&parsed), text, "rendering {literal}");
    }
}

/// The whole path in one test: a request line as `clm-run` reads it, through
/// parsing and preparation, to the exact strings the encoder is asked for.
///
/// This is the layer the number bug lived at. Checking that `to_text` renders a
/// double correctly says nothing if the double was already the wrong one, so the
/// literals go in as JSON text and the assertion is on what comes out the far end.
#[test]
fn a_request_line_reaches_the_encoder_with_the_reference_text() {
    let line = concat!(
        r#"{"model":"clm-latest","state":{"seventeen":7.8190461323667115,"#,
        r#""inexact":9007199254740993.0,"big":18446744073709551616},"#,
        r#""questions":{"q":{"type":"choice","instructions":"Pick","#,
        r#""criteria":{"a":7.8190461323667115,"b":18446744073709551616}}}}"#
    );
    let request = Request::parse_line(line).unwrap();
    let prepared = request.prepare().unwrap();

    assert_eq!(prepared.len(), 1);
    assert_eq!(
        prepared[0].state_text,
        "seventeen: 7.8190461323667115\n\ninexact: 9007199254740992.0\n\n\
         big: 18446744073709551616\n\nPick"
    );
    assert_eq!(prepared[0].question.keys, ["a", "b"]);
    assert_eq!(
        prepared[0].candidate_texts,
        ["7.8190461323667115", "18446744073709551616"]
    );
}

/// Whole request bodies, as the text a caller sends them in.
///
/// The loop above pairs every state with every question, but a `QuestionRequest` holds
/// `instructions` as a `String` and `criteria` as a `Value`, so it cannot express an
/// integer above `u64::MAX` or a statement that is a number rather than a string. These
/// are the requests that need the literal itself, and they go in as text.
#[test]
#[ignore = "requires CLM_TEXT_ORACLE from recipe/clm/native/text_oracle.py; CPU only"]
fn raw_requests_match_the_reference_byte_for_byte() {
    let oracle = oracle();
    let cases = oracle["raw_cases"].as_array().unwrap();
    assert!(
        !cases.is_empty(),
        "the oracle was built from another case list"
    );

    for case in cases {
        let line = case["line"].as_str().unwrap();
        let prepared = Request::parse_line(line).unwrap().prepare().unwrap();
        let expected = case["questions"].as_object().unwrap();
        assert_eq!(prepared.len(), expected.len(), "question count for {line}");
        for p in &prepared {
            let want = &expected[&p.id];
            assert_eq!(
                p.state_text,
                want["state_text"].as_str().unwrap(),
                "{} state_text for {line}",
                p.id
            );
            assert_eq!(
                p.question.keys,
                strings(&want["keys"]),
                "{} keys for {line}",
                p.id
            );
            assert_eq!(
                p.candidate_texts,
                strings(&want["candidate_texts"]),
                "{} candidate_texts for {line}",
                p.id
            );
        }
    }
}

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect()
}

/// A `noul` question reads its two descriptions in `NOUL_KEYS` order — `false` first —
/// which is not the order they were written in. A literal has to be found by the key it
/// was written under, not by counting: `{"true": 1, "false": 2}` is `false: 2`, and
/// pairing the first literal with the first key gives `false: 1`, a different answer.
#[test]
fn a_noul_literal_follows_its_key_not_its_position() {
    let line = concat!(
        r#"{"state":"s","questions":{"q":{"type":"noul","instructions":"Is it so?","#,
        r#""criteria":{"true":1,"false":2}}}}"#
    );
    let prepared = Request::parse_line(line).unwrap().prepare().unwrap();
    assert_eq!(prepared[0].question.keys, ["false", "true"]);
    assert_eq!(prepared[0].candidate_texts, ["false: 2", "true: 1"]);
}

/// The default `noul` candidates are the question's own statement, so a literal in the
/// statement has to reach them as written rather than as a rounded double. The statement
/// is a value like any other here, not necessarily a string.
#[test]
fn a_default_candidate_keeps_the_instruction_literal() {
    let line = concat!(
        r#"{"state":"s","questions":{"q":{"type":"noul","#,
        r#""instructions":18446744073709551616,"criteria":null}}}"#
    );
    let prepared = Request::parse_line(line).unwrap().prepare().unwrap();
    assert_eq!(
        prepared[0].candidate_texts,
        [
            "false: No. This is false: 18446744073709551616",
            "true: Yes. This is true: 18446744073709551616",
        ]
    );
    // The same statement reaches the state head, so both heads see one text.
    assert_eq!(
        prepared[0].state_text, "s\n\n18446744073709551616",
        "the statement is rendered once and reused"
    );
}

/// Integer literals are arbitrary precision in `json.loads`, so they have to keep every
/// digit here too.
///
/// A JSON integer above `u64::MAX` has no exact `f64`, and `serde_json` parses one
/// straight to a double: `18446744073709551616` was reaching the encoder as
/// `1.8446744073709552e+19`. The literal decides which of `json.loads`'s two types a
/// number is, not the value it holds.
#[test]
fn integer_literals_keep_every_digit() {
    // The expected strings are `str(json.loads(literal))`.
    for (literal, text) in [
        ("18446744073709551616", "18446744073709551616"),
        (
            "340282366920938463463374607431768211456",
            "340282366920938463463374607431768211456",
        ),
        ("-18446744073709551616", "-18446744073709551616"),
        ("0", "0"),
        ("-0", "0"),
        ("9007199254740993", "9007199254740993"),
        // The same digits as a float literal, so it is a float afterwards.
        ("9007199254740993.0", "9007199254740992.0"),
        // And an exponent is a float however integral it looks.
        ("1e5", "100000.0"),
    ] {
        assert_eq!(to_text_json(literal).unwrap(), text, "rendering {literal}");
    }
}

/// The lexer and the value walk have to agree on how many numbers a document has, or a
/// literal would be paired with the wrong number. Both are in document order, and this
/// is what says so.
#[test]
fn the_literals_line_up_with_the_values() {
    fn numbers(value: &Value) -> usize {
        match value {
            Value::Number(_) => 1,
            Value::Array(items) => items.iter().map(numbers).sum(),
            Value::Object(map) => map.values().map(numbers).sum(),
            _ => 0,
        }
    }
    for raw in state_texts() {
        let value: Value = serde_json::from_str(raw).unwrap();
        let literals = NumberLiterals::of(raw);
        assert_eq!(literals.len(), numbers(&value), "counting {raw}");
    }
    for raw in [
        r#"{"a": "not a 1 or a 2", "b": [3, {"c": -4.5e-6}], "d": null}"#,
        "{\"escaped\": \"quote \\\" then 7 and \\\\ then 8\", \"n\": 9}",
        r#"[1, [2, [3]], {"k": 4}]"#,
        r#"{}"#,
        r#"[]"#,
    ] {
        let value: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(
            NumberLiterals::of(raw).len(),
            numbers(&value),
            "counting {raw}"
        );
    }

    // Numeric duplicates can change cardinality. Replacing a nonnumeric value can instead
    // change traversal order while leaving the counts equal; request-path tests below pin
    // that case too. Structural validation rejects both before pairing any literals.
    let raw = r#"{"a": 1, "b": 2, "a": 3}"#;
    let value: Value = serde_json::from_str(raw).unwrap();
    assert_eq!(NumberLiterals::of(raw).len(), 3);
    assert_eq!(numbers(&value), 2);
}

/// A document that repeats an object key is refused, because its numbers cannot be paired
/// with the text they came from.
///
/// `{"a": 1, "b": 2, "a": 3}` parses to a map of two numbers while the text holds three, so
/// the walk meets `b`'s number one literal late and would spell `a` as `1`. Python's
/// `json.loads` collapses the key the same way, so the reference renders such a document
/// without complaining and no oracle case can catch the difference.
#[test]
fn a_repeated_key_is_refused_rather_than_paired_wrongly() {
    for raw in [
        r#"{"a": 1, "b": 2, "a": 3}"#,
        // Nested, so the mispaired literal is not the document's first number.
        r#"{"outer": {"x": 1, "y": 2, "x": 3}, "after": 4}"#,
        // Inside an array, where the wrong pairing reaches past the object.
        r#"[9, {"x": 1, "y": 2, "x": 3}, 4]"#,
    ] {
        let error = to_text_json(raw).unwrap_err().to_string();
        assert!(error.contains("repeats a key"), "{raw}: {error}");
    }

    // Structural validation rejects duplicates independently of numeric counts.
    for raw in [
        r#"{"a":"x","a":"y"}"#,
        r#"{"a":"x","b":3,"a":2}"#,
        r#"{"a":"x","b":3,"\u0061":2}"#,
    ] {
        assert!(
            to_text_json(raw)
                .unwrap_err()
                .to_string()
                .contains("repeats a key")
        );
    }
}

/// The same refusal on the path a whole request takes.
///
/// The state, a `choice`'s values and a `score`'s levels reach the cursor by different
/// routes — the state and each `choice` value render from their own text, a `score`'s levels
/// share one cursor over the list's text — so each route is checked rather than one of them.
#[test]
fn a_request_that_repeats_a_key_is_refused() {
    let line = |state: &str, question: &str| {
        format!(r#"{{"model":"clm-latest","state":{state},"questions":{{"q":{question}}}}}"#)
    };
    let choice = r#"{"type":"choice","instructions":"Pick","criteria":{"a":"A","b":"B"}}"#;
    let score = r#"{"type":"score","instructions":"How much?","criteria":[{"x":1,"x":2},3]}"#;

    for (state, question) in [
        // In the state, which `state_text_with` renders from its own text. `b` follows the
        // repeated `a`, so one wrong literal reaches past the duplicate: this rendered
        // `a: 1\n\nb: 2` where the reference gives `a: 2\n\nb: 3`.
        (r#"{"a":1,"a":2,"b":3}"#, choice),
        // The same shape with the duplicate last, so nothing follows it.
        (r#"{"x": 1, "y": 2, "x": 3}"#, choice),
        // In a `score`'s levels, which share the cursor over the list's text.
        (r#"{"ticket": "t"}"#, score),
    ] {
        let error = Request::parse_line(&line(state, question))
            .unwrap()
            .prepare()
            .unwrap_err();
        // `{:#}` rather than `{}`: a question's failure is wrapped in the context that names
        // the question, so the reason is the second link of the chain.
        let chain = format!("{error:#}");
        assert!(
            chain.contains("repeats a key"),
            "{state} + {question}: {chain}"
        );
    }
}

/// Losing a nonnumeric value can reorder numbers without changing their count.
#[test]
fn request_rejects_duplicate_keys_even_when_number_counts_match() {
    let choice = r#"{"type":"choice","instructions":"Pick","criteria":{"x":"X","y":"Y"}}"#;
    for state in [
        r#"{"a":"x","b":3,"a":2}"#,
        r#"{"a":2,"b":3,"a":"x"}"#,
        r#"{"a":"x","a":"y"}"#,
        r#"{"a":"x","b":3,"\u0061":2}"#,
        r#"{"outer":[{"a":"x","b":3,"a":2}],"after":4}"#,
    ] {
        let line = format!(r#"{{"state":{state},"questions":{{"q":{choice}}}}}"#);
        let result = Request::parse_line(&line).and_then(|request| request.prepare());
        let error = result.expect_err(&line);
        assert!(
            format!("{error:#}").contains("repeats a key"),
            "{line}: {error:#}"
        );
    }
}

#[test]
fn request_rejects_duplicate_keys_in_every_rendered_question_field() {
    for question in [
        r#"{"type":"choice","instructions":{"a":"x","b":3,"a":2},"criteria":{"x":"X","y":"Y"}}"#,
        r#"{"type":"choice","instructions":[{"a":"x","a":"y"}],"criteria":{"x":"X","y":"Y"}}"#,
        r#"{"type":"choice","instructions":"Pick","criteria":{"x":"X","y":"Y","x":"Z"}}"#,
        r#"{"type":"choice","instructions":"Pick","criteria":{"x":{"a":"x","b":3,"a":2},"y":"Y"}}"#,
        r#"{"type":"score","instructions":"Rate","criteria":[{"a":"x","b":3,"a":2},4]}"#,
        r#"{"type":"score","instructions":"Rate","criteria":[{"a":"x","a":"y"},4]}"#,
        r#"{"type":"noul","instructions":"True?","criteria":{"true":"yes","false":"no","true":"maybe"}}"#,
        r#"{"type":"noul","instructions":"True?","criteria":{"true":{"a":"x","b":3,"a":2},"false":"no"}}"#,
        r#"{"type":"noul","instructions":"True?","criteria":{"true":[{"a":"x","\u0061":"y"}],"false":"no"}}"#,
    ] {
        let line = format!(r#"{{"state":{{}},"questions":{{"q":{question}}}}}"#);
        let result = Request::parse_line(&line).and_then(|request| request.prepare());
        let error = result.expect_err(&line);
        assert!(
            format!("{error:#}").contains("repeats a key"),
            "{line}: {error:#}"
        );
    }
}

#[test]
fn unique_keys_keep_scopes_and_numeric_literals_in_request_text() {
    let line = r#"{"state":{"a":2,"b":3,"nested":{"a":18446744073709551616}},"questions":{"q":{"type":"choice","instructions":{"a":340282366920938463463374607431768211456},"criteria":{"x":{"a":2,"b":3},"y":{"a":4}}}}}"#;
    let prepared = Request::parse_line(line).unwrap().prepare().unwrap();
    assert_eq!(
        prepared[0].state_text,
        "a: 2\n\nb: 3\n\nnested:\n  a: 18446744073709551616\n\na: 340282366920938463463374607431768211456"
    );
    assert_eq!(prepared[0].candidate_texts, ["a: 2\n\nb: 3", "a: 4"]);
}
