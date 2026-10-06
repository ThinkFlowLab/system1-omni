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


def answer(
    question: Question,
    logits: Sequence[float],
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Match Valen's pinned choice decoding and confidence formula."""

    if len(logits) != len(question.keys):
        raise ValueError("model returned one logit per candidate")
    probabilities = _probabilities(logits, temperature)
    mode = max(range(len(probabilities)), key=probabilities.__getitem__)
    n = len(probabilities)
    confidence = (
        1.0
        if n == 1
        else max(0.0, (max(probabilities) - 1 / n) / (1 - 1 / n))
    )
    return {
        "type": "choice",
        "choice": question.keys[mode],
        "probabilities": dict(zip(question.keys, probabilities)),
        "confidence": confidence,
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
