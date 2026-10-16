"""Pure Valen response reconstruction from executor logits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .preprocess import ResponseContext
from .protocol import Question, RequestError


def _probabilities(logits: Sequence[float], temperature: float) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise RequestError("calibration temperature must be finite and positive")
    values = [float(value) for value in logits]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("model returned invalid logits")
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _choice_confidence(probabilities: Sequence[float]) -> float:
    """Valen's original choice confidence formula (reference answer module)."""

    n = len(probabilities)
    if n == 1:
        return 1.0
    return max(0.0, (max(probabilities) - 1 / n) / (1 - 1 / n))


def _noul_answer(
    question: Question, probabilities: Sequence[float]
) -> dict[str, Any]:
    """The reference's noul answer (P(true)) plus the full candidate pair."""

    pair = dict(zip(question.keys, probabilities))
    return {
        "type": "noul",
        "noul": pair["true"],
        "probabilities": pair,
        "confidence": _choice_confidence(probabilities),
    }


def _score_answer(
    question: Question, probabilities: Sequence[float]
) -> dict[str, Any]:
    """Valen's pinned score decoding: expected level and concentration."""

    mode = max(range(len(probabilities)), key=probabilities.__getitem__)
    count = len(probabilities)
    distance = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
    uniform_mad = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
    return {
        "type": "score",
        "probabilities": dict(zip(question.keys, probabilities)),
        "score": sum(i * p for i, p in enumerate(probabilities)),
        "legend": dict(zip(question.keys, question.descriptions)),
        "confidence": max(0.0, 1.0 - distance / uniform_mad),
    }


def answer(
    question: Question,
    logits: Sequence[float],
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Match Valen's pinned decoding and confidence formulas per type."""

    if len(logits) != len(question.keys):
        raise ValueError("model must return one logit per candidate")
    probabilities = _probabilities(logits, temperature)
    if question.kind == "noul":
        return _noul_answer(question, probabilities)
    if question.kind == "score":
        return _score_answer(question, probabilities)
    mode = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "type": "choice",
        "choice": question.keys[mode],
        "probabilities": dict(zip(question.keys, probabilities)),
        "confidence": _choice_confidence(probabilities),
    }


def build_response(
    context: ResponseContext,
    answers: Mapping[str, dict[str, Any]],
    logical_tokens: int,
    compute_tokens: int,
) -> dict[str, Any]:
    """Reconstruct the public response and Valen's internal usage fields."""

    expected = tuple(question.name for question in context.questions)
    if tuple(answers) != expected:
        raise ValueError("answers do not match request question order")
    if logical_tokens < 0 or compute_tokens < 0:
        raise ValueError("token counts must be non-negative")
    return {
        "model": context.model,
        "answers": dict(answers),
        "usage": {"input_tokens": logical_tokens, "output_tokens": 0},
        "internal_usage": {"compute_tokens": compute_tokens},
    }
