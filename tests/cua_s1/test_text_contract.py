"""Contract tests without weights or torch. The tokenizer test also runs when
CUA_S1_BASE points to a local Qwen/Qwen3.5-4B directory (tokenizer files only).

PYTHONPATH=src python -m pytest tests/cua_s1
"""

import json
import math
import os

import pytest

from models.cua_s1.text.contract import (
    RequestError,
    answer,
    build_messages,
    map_request,
    parse_body,
)


def body(state="Screen", **question):
    q = {
        "type": "choice",
        "instructions": "Pick one.",
        "criteria": {"a": "A", "b": "B"},
    }
    q.update(question)
    return {"model": "cua-s1-4b-0.2", "state": state, "questions": {"q": q}}


def mapped(request):
    return map_request(parse_body(json.dumps(request).encode()))


def reject(request, status=422):
    raw = request if isinstance(request, bytes) else json.dumps(request).encode()
    with pytest.raises(RequestError) as info:
        map_request(parse_body(raw))
    assert info.value.status == status
    return str(info.value)


# upstream's libs/cua-driver/examples/jev-use/fixtures/jev-choice-request-v1.json,
# with the chooser's rendered region as `state`, and the user message upstream builds for it
FIXTURE = body(
    'Visual-region-derived observation for capture "capture-fixture-1":\n'
    "\"submit-text\": text 'Submit' at (300,240,100,40) confidence=0.96 interactive=true",
    instructions="Submit the verified form.",
    criteria={
        "submit-form": "Submit using the unique validated visual region.",
        "reobserve": "Discard this decision set and obtain a fresh observation.",
        "abstain": "Stop without acting if no supplied action is safe.",
    },
)
FIXTURE_USER = (
    "Goal: Submit the verified form.\n\n"
    "App: Cua Driver\nTask family: closed-candidate decision\n\n"
    "Accessibility tree:\n"
    'Visual-region-derived observation for capture "capture-fixture-1":\n'
    "\"submit-text\": text 'Submit' at (300,240,100,40) confidence=0.96 interactive=true\n\n"
    "Options:\n"
    'A. Decision "Submit using the unique validated visual region." -> select\n'
    'B. Decision "Discard this decision set and obtain a fresh observation." -> select\n'
    'C. Decision "Stop without acting if no supplied action is safe." -> select\n\n'
    "Answer with a single letter."
)


def test_fixture_prompt_matches_upstream():
    state, (question,) = mapped(FIXTURE)
    system, user = build_messages(state, question)
    assert user == {"role": "user", "content": FIXTURE_USER}
    assert system["content"].startswith(
        "You are a one-pass computer-use decision model."
    )


@pytest.mark.skipif(not os.environ.get("CUA_S1_BASE"), reason="set CUA_S1_BASE to run")
def test_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.environ["CUA_S1_BASE"])
    assert tokenizer.convert_tokens_to_ids(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) == list(
        range(32, 58)
    )
    state, (question,) = mapped(FIXTURE)
    text = tokenizer.apply_chat_template(
        build_messages(state, question), tokenize=False, add_generation_prompt=True
    )
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    ids = tokenizer(text)["input_ids"]
    assert len(ids) == 218
    assert ids == tokenizer(text, add_special_tokens=False)["input_ids"]


def test_goal_left_out_when_empty_or_null():
    for goal in ["", None]:
        state, (question,) = mapped(body(instructions=goal))
        assert build_messages(state, question)[1]["content"].startswith(
            "App: Cua Driver\n"
        )


def test_structured_values_escaping_and_null_label():
    tree = {
        "app": "Settings",
        "elements": [{"id": "e1", "label": "Location", "on": True}],
    }
    state, (question,) = mapped(
        body(
            tree,
            instructions={"question": "Which one?"},
            criteria={
                "e1": {"action": "click", "element": "e1"},
                "e2": ["click", "e2"],
                "quote": 'Click "Submit"\n(tab\there) C:\\Users',
                "save": "点击「保存」",
                "abstain": None,
            },
        )
    )
    assert state == json.dumps(tree, ensure_ascii=False)
    assert question.goal == '{"question": "Which one?"}'
    assert question.labels == (
        '{\\"action\\": \\"click\\", \\"element\\": \\"e1\\"}',
        '[\\"click\\", \\"e2\\"]',
        'Click \\"Submit\\"\\n(tab\\there) C:\\\\Users',
        "点击「保存」",
        "abstain",
    )


def test_request_errors():
    assert "not supported" in reject(body(type="score", criteria=["low", "high"]))
    assert "not supported" in reject(body(type="noul"))
    assert "unknown type" in reject(body(type="rank"))
    assert "1 to 26 options" in reject(body(criteria={}))
    assert "1 to 26 options" in reject(body(criteria={f"o{i}": "x" for i in range(27)}))
    assert (
        len(mapped(body(criteria={f"o{i}": "x" for i in range(26)}))[1][0].keys) == 26
    )
    no_instructions = body()
    del no_instructions["questions"]["q"]["instructions"]
    assert "'instructions' is required" in reject(no_instructions)
    for value in [1, 2.5, True]:
        reject(body(criteria={"a": value}))
    for state in ["", {}, [], None, 3, True]:
        reject(body(state))
    assert "'model'" in reject({**body(), "model": "english"})
    many = body()
    many["questions"] = {f"q{i}": many["questions"]["q"] for i in range(65)}
    reject(many, status=413)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"model": "cua-s1-4b-0.2", "state": {"x": 1, "x": 2}}',
        b'{"model": "cua-s1-4b-0.2", "state": NaN}',
        b'{"model": "cua-s1-4b-0.2", "state": {"x": 1e400}}',
        b'{"model": "cua-s1-4b-0.2", "state": {"x": ' + b"9" * 5000 + b"}}",
        b'{"model": "cua-s1-4b-0.2", "state": "\\ud800"}',
        b"[" * 100000 + b"]" * 100000,
        b"\xff\xfe",
        b"\xef\xbb\xbf{}",
        b"not json",
        b"[1, 2]",
    ],
)
def test_malformed_bodies_are_400(raw):
    reject(raw, status=400)


def test_answer():
    _, (question,) = mapped(body())
    tie = answer(question, [0.5, 0.5])
    assert tie["choice"] == "a" and tie["confidence"] == pytest.approx(0.0, abs=1e-12)
    result = answer(question, [0.12, 0.88])
    assert result["choice"] == "b"
    assert list(result["probabilities"]) == ["a", "b"]
    h = -(0.12 * math.log(0.12) + 0.88 * math.log(0.88))
    assert result["confidence"] == pytest.approx(1 - h / math.log(2))
