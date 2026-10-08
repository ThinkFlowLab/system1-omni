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
