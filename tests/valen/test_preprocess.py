import hashlib

from models.valen.preprocess import prepare_request, question_record
from models.valen.protocol import (
    MODEL_NAME,
    NOUL_FALSE_DESCRIPTION,
    NOUL_TRUE_DESCRIPTION,
    parse_request,
)

from .test_protocol import body


def test_prepare_materializes_request_scoped_image_and_valen_record(tmp_path):
    request = parse_request(body())
    prepared = prepare_request(request, tmp_path)

    expected_hash = hashlib.sha256(request.state.data).hexdigest()
    assert prepared.image_path == tmp_path / "input.png"
    assert prepared.image_path.read_bytes() == request.state.data
    assert prepared.image_sha256 == expected_hash
    assert prepared.record["assets"] == [{"path": "input.png", "sha256": expected_hash}]
    assert prepared.record["request"]["state"] == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "input.png"}}
                ],
            }
        ]
    }
    assert prepared.record["request"]["questions"]["move"]["criteria"] == {
        "up": "Move up",
        "down": "Move down",
    }
    assert prepared.response_context.model == MODEL_NAME
    assert [q.name for q in prepared.response_context.questions] == ["move"]


def test_prepare_compiles_text_state_without_media(tmp_path):
    request = parse_request(body(state="Dialog: the update finished."))
    prepared = prepare_request(request, tmp_path)

    assert prepared.image_path is None
    assert prepared.image_sha256 is None
    assert prepared.record["assets"] == []
    assert prepared.record["request"]["state"] == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Dialog: the update finished."}
                ],
            }
        ]
    }
    assert list(prepared.record["request"]["questions"]) == ["move"]
    assert list(tmp_path.iterdir()) == []


def noul_question(criteria=None):
    question = {"type": "noul", "instructions": "Was the card charged twice?"}
    if criteria is not None:
        question["criteria"] = criteria
    request = parse_request(
        body(state="The card was charged twice.", questions={"ask": question})
    )
    return request.questions[0]


def test_question_record_keeps_reference_noul_wording_native():
    record = question_record(noul_question())

    assert record == {"type": "noul", "instructions": "Was the card charged twice?"}

    explicit = question_record(
        noul_question(
            criteria={
                "true": NOUL_TRUE_DESCRIPTION,
                "false": NOUL_FALSE_DESCRIPTION,
            }
        )
    )
    assert explicit == record


def test_question_record_carries_custom_noul_wording_as_choice():
    record = question_record(
        noul_question(criteria={"true": "It was duplicated.", "false": "It was unique."})
    )

    assert record == {
        "type": "choice",
        "instructions": "Was the card charged twice?",
        "criteria": {"true": "It was duplicated.", "false": "It was unique."},
    }

    partial = question_record(noul_question(criteria={"false": "It was unique."}))
    assert partial["criteria"] == {
        "true": NOUL_TRUE_DESCRIPTION,
        "false": "It was unique.",
    }


def test_question_record_passes_score_levels_in_order():
    request = parse_request(
        body(
            state="Dialog: the update finished.",
            questions={
                "grade": {
                    "type": "score",
                    "instructions": "Grade the dialog.",
                    "criteria": ["unclear", "clear"],
                }
            },
        )
    )

    assert question_record(request.questions[0]) == {
        "type": "score",
        "instructions": "Grade the dialog.",
        "criteria": ["unclear", "clear"],
    }


def test_prepare_builds_mixed_type_records(tmp_path):
    request = parse_request(
        body(
            state="The card was charged twice for one order.",
            questions={
                "ask": {
                    "type": "noul",
                    "instructions": "Was the card charged twice?",
                },
                "grade": {
                    "type": "score",
                    "instructions": "Grade the severity.",
                    "criteria": ["No error", "A minor error", "A major error"],
                },
                "act": {
                    "type": "choice",
                    "instructions": "Decide the refund action.",
                    "criteria": {"refund": None, "wait": "Wait for review"},
                },
            },
        )
    )
    prepared = prepare_request(request, tmp_path)

    assert prepared.record["request"]["questions"] == {
        "ask": {"type": "noul", "instructions": "Was the card charged twice?"},
        "grade": {
            "type": "score",
            "instructions": "Grade the severity.",
            "criteria": ["No error", "A minor error", "A major error"],
        },
        "act": {
            "type": "choice",
            "instructions": "Decide the refund action.",
            "criteria": {"refund": "refund", "wait": "Wait for review"},
        },
    }
    assert prepared.record["assets"] == []
    assert [q.kind for q in prepared.response_context.questions] == [
        "noul",
        "score",
        "choice",
    ]
