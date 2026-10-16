import pytest

from models.d3.protocol import (
    MAX_BODY,
    MODEL_NAME,
    MalformedJSON,
    RequestError,
    parse_body,
    parse_request,
)


def body(**overrides):
    value = {
        "model": MODEL_NAME,
        "state": "The card was charged twice for one order.",
        "questions": {
            "refund": {
                "type": "choice",
                "instructions": "Decide the refund action.",
                "criteria": {"refund": "Refund the duplicate charge", "wait": "Wait for review"},
            }
        },
    }
    value.update(overrides)
    return value


def test_parse_body_rejects_duplicate_and_nonfinite_json():
    with pytest.raises(MalformedJSON, match="duplicate"):
        parse_body(b'{"model":"x","model":"y"}')
    with pytest.raises(MalformedJSON, match="non-finite"):
        parse_body(b'{"value":NaN}')
    with pytest.raises(MalformedJSON):
        parse_body(b"{not json")
    with pytest.raises(MalformedJSON):
        parse_body(b"[1, 2]")


def test_parse_body_enforces_the_slice_scoped_body_limit():
    padding = b'{"state":"' + b"a" * (MAX_BODY) + b'"}'
    with pytest.raises(RequestError) as excinfo:
        parse_body(padding)
    assert excinfo.value.status == 413
    assert "64 MiB" in str(excinfo.value)


def test_parse_rejects_unknown_fields_with_the_upstream_message():
    value = body(extra=1)
    with pytest.raises(RequestError, match="Unknown request fields: \\['extra'\\]"):
        parse_request(value)


def test_parse_accepts_and_ignores_any_model_value():
    for model in (MODEL_NAME, "other-model", "", None, 17):
        parsed = parse_request(body(model=model))
        assert parsed.model == model
    parsed = parse_request(
        {k: v for k, v in body().items() if k != "model"}
    )
    assert parsed.model is None


def test_parse_passes_state_through_in_every_wire_shape():
    for state in ("text", {"goal": "route"}, ["part one"], 3.5, None, ""):
        parsed = parse_request(body(state=state))
        assert parsed.state == state


def test_parse_admits_media_lists_and_checks_only_the_shape():
    for field in ("images", "videos"):
        # Absent and null mean "no media of this kind".
        assert getattr(parse_request(body()), field) is None
        assert getattr(parse_request(body(**{field: None})), field) is None
        assert getattr(parse_request(body(**{field: []})), field) == []

        # A non-empty list passes through untouched — order and identity
        # preserved, no data-URL inspection at the envelope layer.
        items = ["data:image/png;base64,AAAA", "not a data url", 17]
        parsed = parse_request(body(**{field: items}))
        assert getattr(parsed, field) == items
        assert getattr(parsed, field)[0] is items[0]

        # Anything but a list is refused with the upstream server's message.
        for bad in ("data:image/png;base64,AAAA", 17, {}, True):
            with pytest.raises(RequestError) as excinfo:
                parse_request(body(**{field: bad}))
            assert excinfo.value.status == 422
            assert str(excinfo.value) == f"{field} must be a list of base64 data URLs"


def test_parse_checks_media_shape_before_questions():
    # Mirroring the upstream decode order: a non-list images field is
    # refused even when the questions field is also broken.
    value = body(images="data:image/png;base64,AAAA", questions={})
    with pytest.raises(RequestError, match="must be a list"):
        parse_request(value)


def test_parse_rejects_empty_or_malformed_questions():
    with pytest.raises(RequestError, match="nonempty mapping"):
        parse_request(body(questions={}))
    with pytest.raises(RequestError, match="nonempty mapping"):
        parse_request(body(questions=[]))
    with pytest.raises(RequestError, match="nonempty strings"):
        parse_request(body(questions={"": {"type": "noul"}}))
    with pytest.raises(RequestError, match="nonempty strings"):
        parse_request(body(questions={3: {"type": "noul"}}))


def test_parse_preserves_question_request_order():
    questions = {
        "zeta": {"type": "noul"},
        "alpha": {"type": "score", "criteria": ["low", "high"]},
        "mid": {"type": "choice", "criteria": {"a": None, "b": None}},
    }
    parsed = parse_request(body(questions=questions))
    assert [name for name, _ in parsed.questions] == ["zeta", "alpha", "mid"]
    assert dict(parsed.questions)["alpha"] is questions["alpha"]


def test_parse_leaves_question_content_to_the_pinned_runtime():
    # The envelope does not judge question bodies: kinds, criteria shapes and
    # sizes are validated by the runtime and reported per question.
    parsed = parse_request(
        body(questions={"weird": {"type": "triangle", "criteria": 17}})
    )
    assert dict(parsed.questions)["weird"] == {"type": "triangle", "criteria": 17}
