"""Request mapping, prompts and answers for Cua-S1 4B 0.2, following
`src/models/cua_s1/README.md`. No torch imports, so it can be tested without weights.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

MODEL_NAME = "cua-s1-4b-0.2"
MODEL_ID = "cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79:text"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_QUESTIONS = 64

# The system message and the user message layout are copied from trycua/cua at
# 0e75660ce4c2edda519e0c795fa3ad98abf4e76f (`libs/cua-s1/python/src/cua_s1/four_b.py`
# and `libs/cua-driver/examples/jev-use/python/decision_models.py`).
#
# MIT License
#
# Copyright (c) 2025 Cua AI, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
SYSTEM_PROMPT = (
    "You are a one-pass computer-use decision model. You are shown the "
    "current state of a screen and a fixed, closed list of candidate "
    "(element, action) options, each given a single letter. Choose exactly "
    "one option: the single best next action to take. Answer with ONLY that "
    "option's letter -- no words, no punctuation, no explanation."
)


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Question:
    name: str
    goal: str
    keys: tuple[str, ...]
    labels: tuple[str, ...]


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise RequestError(f"duplicate key {key!r}", 400)
        obj[key] = value
    return obj


def parse_body(raw: bytes) -> dict[str, Any]:
    try:
        body = json.loads(raw.decode(), object_pairs_hook=_unique_keys)
        # NaN, Infinity, numbers out of range and lone surrogates fail here.
        json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, RecursionError) as error:
        if isinstance(error, RequestError):
            raise
        raise RequestError(f"request body is not valid JSON: {error}", 400) from error
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object", 400)
    return body


def _text(value: Any, where: str) -> str:
    """A string as is; an object or array as Python's json.dumps writes it."""
    if not isinstance(value, (str, dict, list)):
        raise RequestError(f"{where} must be a string, an object or an array")
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def map_request(body: dict[str, Any]) -> tuple[str, list[Question]]:
    if body.get("model") != MODEL_NAME:
        raise RequestError(f"'model' must be {MODEL_NAME!r}")
    if body.get("state") in ("", {}, []):
        raise RequestError("'state' must not be empty")
    state = _text(body.get("state"), "'state'")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("'questions' must be a non-empty object")
    if len(questions) > MAX_QUESTIONS:
        raise RequestError(f"more than {MAX_QUESTIONS} questions", 413)
    mapped = []
    for name, q in questions.items():
        where = f"question {name!r}"
        if not isinstance(q, dict):
            raise RequestError(f"{where} must be an object")
        if q.get("type") in ("score", "noul"):
            raise RequestError(f"{where}: type {q['type']!r} is not supported")
        if q.get("type") != "choice":
            raise RequestError(f"{where}: unknown type {q.get('type')!r}")
        if "instructions" not in q:
            raise RequestError(f"{where}: 'instructions' is required")
        goal = "" if q["instructions"] is None else _text(q["instructions"], where)
        criteria = q.get("criteria")
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= len(LETTERS):
            raise RequestError(
                f"{where}: 'criteria' must be an object with 1 to 26 options"
            )
        labels = tuple(
            json.dumps(
                key if value is None else _text(value, f"{where}: {key!r}"),
                ensure_ascii=False,
            )[1:-1]
            for key, value in criteria.items()
        )
        mapped.append(Question(name, goal, tuple(criteria), labels))
    return state, mapped


def build_messages(state: str, question: Question) -> list[dict[str, str]]:
    options = "\n".join(
        f'{letter}. Decision "{label}" -> select'
        for letter, label in zip(LETTERS, question.labels)
    )
    user = (
        (f"Goal: {question.goal}\n\n" if question.goal else "")
        + "App: Cua Driver\nTask family: closed-candidate decision\n\n"
        + f"Accessibility tree:\n{state}\n\nOptions:\n{options}\n\nAnswer with a single letter."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def answer(question: Question, probabilities: list[float]) -> dict[str, Any]:
    """The Jev choice answer; ties go to the earliest option. `confidence` is the
    normalized entropy `1 - H(p) / ln(n)`, as the LAYA worker reports it."""
    n = len(probabilities)
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return {
        "type": "choice",
        "choice": question.keys[max(range(n), key=probabilities.__getitem__)],
        "probabilities": dict(zip(question.keys, probabilities)),
        "confidence": max(0.0, 1 - entropy / math.log(n)) if n > 1 else 1.0,
    }
