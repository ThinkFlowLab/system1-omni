import math

import pytest

from models.valen.postprocess import answer, build_response
from models.valen.preprocess import ResponseContext
from models.valen.protocol import Question, RequestError


QUESTION = Question("move", "Choose.", ("up", "down"), ("Move up", "Move down"))


def test_answer_uses_valen_probability_and_confidence_formula():
    result = answer(QUESTION, [0.0, math.log(3.0)])

    assert result["choice"] == "down"
    assert result["probabilities"]["up"] == pytest.approx(0.25)
    assert result["probabilities"]["down"] == pytest.approx(0.75)
    assert result["confidence"] == pytest.approx(0.5)


def test_answer_ties_follow_candidate_order_and_single_option_is_confident():
    assert answer(QUESTION, [1.0, 1.0])["choice"] == "up"
    one = Question("only", "", ("ok",), ("OK",))
    result = answer(one, [0.0])
    assert result["confidence"] == 1.0
    assert result["probabilities"] == {"ok": 1.0}


def test_answer_rejects_invalid_logits_and_temperature():
    with pytest.raises(ValueError, match="one logit"):
        answer(QUESTION, [0.0])
    with pytest.raises(ValueError, match="invalid logits"):
        answer(QUESTION, [float("nan"), 0.0])
    with pytest.raises(RequestError, match="temperature"):
        answer(QUESTION, [0.0, 1.0], temperature=0.0)


def test_build_response_preserves_question_order_and_usage():
    context = ResponseContext("valen-preview-0923", (QUESTION,))
    answer_value = answer(QUESTION, [0.0, 1.0])

    result = build_response(context, {"move": answer_value}, 17, 34)

    assert result == {
        "model": "valen-preview-0923",
        "answers": {"move": answer_value},
        "usage": {"input_tokens": 17, "output_tokens": 0},
        "internal_usage": {"compute_tokens": 34},
    }


NOUL = Question("ask", "Was it duplicated?", ("true", "false"), (None, None), "noul")


def test_noul_answer_reports_true_probability_and_full_pair():
    result = answer(NOUL, [0.0, math.log(3.0)])

    assert result["type"] == "noul"
    assert result["noul"] == pytest.approx(0.25)
    assert result["probabilities"] == {
        "true": pytest.approx(0.25),
        "false": pytest.approx(0.75),
    }
    # Valen's choice confidence formula evaluated on the two candidates.
    assert result["confidence"] == pytest.approx(0.5)

    flipped = answer(NOUL, [math.log(3.0), 0.0])
    assert flipped["noul"] == pytest.approx(0.75)


def test_noul_answer_honors_temperature_and_rejects_bad_logits():
    sharp = answer(NOUL, [1.0, 0.0], temperature=0.5)
    plain = answer(NOUL, [1.0, 0.0], temperature=1.0)
    assert sharp["noul"] > plain["noul"] > 0.5

    with pytest.raises(ValueError, match="one logit"):
        answer(NOUL, [0.0])
    with pytest.raises(ValueError, match="invalid logits"):
        answer(NOUL, [float("nan"), 0.0])


SCORE = Question(
    "grade",
    "Grade the severity.",
    ("0", "1", "2"),
    ("No error", "A minor error", "A major error"),
    "score",
)


def test_score_answer_reports_expected_level_and_distribution():
    result = answer(SCORE, [0.0, 0.0, 0.0])

    assert result["type"] == "score"
    assert result["probabilities"] == {"0": pytest.approx(1 / 3), "1": pytest.approx(1 / 3), "2": pytest.approx(1 / 3)}
    assert result["legend"] == {"0": "No error", "1": "A minor error", "2": "A major error"}
    assert result["score"] == pytest.approx(1.0)
    # A uniform distribution has no concentration.
    assert result["confidence"] == pytest.approx(0.0)

    peaked = answer(SCORE, [-100.0, 3.0, -100.0])
    assert peaked["score"] == pytest.approx(1.0)
    assert peaked["confidence"] == pytest.approx(1.0)


def test_score_answer_confidence_penalizes_split_mass():
    # Mass split between the two endpoints: mode is 0 (ties go to the
    # earliest level), distance = 0.6 * 0 + 0.4 * 2 = 0.8 against a uniform
    # mean absolute deviation of 2/3, so confidence is 0 (clamped).
    result = answer(SCORE, [math.log(1.5), -50.0, 0.0])
    assert result["probabilities"]["0"] == pytest.approx(0.6)
    assert result["probabilities"]["2"] == pytest.approx(0.4, abs=1e-6)
    assert result["score"] == pytest.approx(0.8)
    assert result["confidence"] == 0.0

    # A tilted but concentrated distribution: p = [0.7, 0.2, 0.1] has mode 0,
    # distance 0.2*1 + 0.1*2 = 0.4 and uniform MAD 2/3, so confidence is
    # 1 - 0.4/(2/3) = 0.4.
    tilted = answer(SCORE, [math.log(7.0), math.log(2.0), 0.0])
    assert tilted["probabilities"]["0"] == pytest.approx(0.7)
    assert tilted["score"] == pytest.approx(0.4)
    assert tilted["confidence"] == pytest.approx(0.4)


def test_score_answer_rejects_wrong_logit_counts():
    with pytest.raises(ValueError, match="one logit"):
        answer(SCORE, [0.0, 1.0])
    with pytest.raises(RequestError, match="temperature"):
        answer(SCORE, [0.0, 1.0, 2.0], temperature=-1.0)
