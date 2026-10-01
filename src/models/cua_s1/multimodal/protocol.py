"""Bounded screenshot-only extension of the Cua-S1 choice contract in PR #11."""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

MODEL = "cua-s1-4b-0.2"
MAX_BODY = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 1024 * 1024
MAX_SIDE = 2048
MAX_ASPECT_RATIO = 200  # Pinned Transformers Qwen image processor's smart_resize.
MAX_QUESTIONS = 8
MAX_TEXT = 16384

# Prompt and letter layout follow trycua/cua at 0e75660ce4c2edda519e0c795fa3ad98abf4e76f.
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


class InvalidRequest(ValueError):
    """Input cannot be evaluated under the supported contract."""


class MalformedJSON(InvalidRequest):
    """The body is not a usable UTF-8 JSON object (HTTP 400)."""


@dataclass(frozen=True)
class Question:
    name: str
    keys: tuple[str, ...]
    labels: tuple[str, ...]
    goal: str


@dataclass(frozen=True)
class Request:
    image: Image.Image
    questions: tuple[Question, ...]


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MalformedJSON("duplicate JSON keys are not supported")
        result[key] = value
    return result


def _nonfinite(value):
    raise MalformedJSON("non-finite JSON numbers are not supported")


def decode_request(raw: bytes) -> dict:
    if len(raw) > MAX_BODY:
        raise InvalidRequest("request body exceeds 8 MiB")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_nonfinite
        )
        # The decoder accepts lone surrogate escapes and overflowing floats.
        # Reject both before any text can reach the tokenizer or a response.
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except MalformedJSON:
        raise
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise MalformedJSON(
            "request body must contain valid JSON and UTF-8 text"
        ) from exc
    if not isinstance(value, dict):
        raise MalformedJSON("request must be a JSON object")
    return value


def _text(value, field):
    if not isinstance(value, (str, dict, list)):
        raise InvalidRequest(f"{field} must be a string, object or array")
    try:
        result = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise InvalidRequest(f"invalid {field}") from exc
    if len(result) > MAX_TEXT:
        raise InvalidRequest(f"{field} exceeds {MAX_TEXT} characters")
    if any(
        token in result
        for token in (
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        )
    ):
        raise InvalidRequest(f"{field} contains an unsupported media control token")
    return result


def _image(state):
    if not isinstance(state, dict) or set(state) != {"image"}:
        raise InvalidRequest("state must contain exactly one image data URL")
    url = state["image"]
    if not isinstance(url, str):
        raise InvalidRequest("state.image must be a PNG/JPEG base64 data URL")
    prefix, separator, encoded = url.partition(",")
    expected = {"data:image/png;base64": "PNG", "data:image/jpeg;base64": "JPEG"}
    if not separator or prefix not in expected:
        raise InvalidRequest("only inline PNG/JPEG images are supported")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise InvalidRequest("encoded image exceeds 4 MiB")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > MAX_IMAGE_BYTES:
            raise InvalidRequest("image exceeds 4 MiB")
        with Image.open(io.BytesIO(raw)) as source:
            if source.format != expected[prefix]:
                raise InvalidRequest("image format does not match its MIME type")
            w, h = source.size
            if max(w, h) > MAX_ASPECT_RATIO * min(w, h):
                raise InvalidRequest(
                    f"image aspect ratio must not exceed {MAX_ASPECT_RATIO}:1"
                )
            if (
                max(w, h) > MAX_SIDE
                or w * h > MAX_PIXELS
                or getattr(source, "n_frames", 1) != 1
            ):
                raise InvalidRequest(
                    "image must be single-frame, at most 2048 per side and 1048576 pixels"
                )
            source.load()
            return source.convert("RGB")
    except (
        binascii.Error,
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        ValueError,
    ) as exc:
        if isinstance(exc, InvalidRequest):
            raise
        raise InvalidRequest("invalid image data") from exc


def parse_request(value: dict) -> Request:
    if not isinstance(value, dict) or set(value) != {"model", "state", "questions"}:
        raise InvalidRequest("request must contain model, state and questions only")
    if value["model"] != MODEL:
        raise InvalidRequest(f"model must be {MODEL}")
    questions = value["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise InvalidRequest("questions must contain 1 to 8 questions")
    parsed = []
    for name, q in questions.items():
        if not isinstance(name, str) or not name or len(name) > 256:
            raise InvalidRequest("question names must contain 1 to 256 characters")
        if not isinstance(q, dict) or q.get("type") != "choice":
            raise InvalidRequest("only choice questions are supported")
        if set(q) - {"type", "instructions", "criteria"}:
            raise InvalidRequest("unsupported question fields")
        if "instructions" not in q:
            raise InvalidRequest("instructions is required for every question")
        criteria = q.get("criteria")
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 26:
            raise InvalidRequest("choice requires 1 to 26 options")
        labels = []
        for key, label in criteria.items():
            if not isinstance(key, str) or not key or len(key) > 256:
                raise InvalidRequest("option keys must contain 1 to 256 characters")
            text = _text(key if label is None else label, "criteria")
            labels.append(json.dumps(text, ensure_ascii=False)[1:-1])
        goal = (
            ""
            if q["instructions"] is None
            else _text(q["instructions"], "instructions")
        )
        if len(goal) + sum(map(len, labels)) > MAX_TEXT:
            raise InvalidRequest("combined question text exceeds 16384 characters")
        parsed.append(Question(name, tuple(criteria), tuple(labels), goal))
    return Request(_image(value["state"]), tuple(parsed))


def build_messages(q: Question, image_marker: str = "inline.png") -> list[dict]:
    lines = "\n".join(
        f'{chr(65 + i)}. Decision "{label}" -> select'
        for i, label in enumerate(q.labels)
    )
    text = (
        (f"Goal: {q.goal}\n\n" if q.goal else "")
        + "App: Cua Driver\nTask family: closed-candidate decision\n\n"
        + "The current screenshot is attached.\n\n"
        + f"Options:\n{lines}\n\nAnswer with a single letter."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_marker},
                {"type": "text", "text": text},
            ],
        },
    ]


def answer(q: Question, probabilities: list[float]) -> dict:
    if len(probabilities) != len(q.keys) or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities
    ):
        raise ValueError("model returned invalid probabilities")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
        raise ValueError("model probabilities do not sum to one")
    n = len(probabilities)
    confidence = (
        1.0
        if n == 1
        else 1 + sum(p * math.log(p) for p in probabilities if p) / math.log(n)
    )
    return {
        "type": "choice",
        "choice": q.keys[max(range(n), key=probabilities.__getitem__)],
        "probabilities": dict(zip(q.keys, probabilities)),
        "confidence": max(0.0, min(1.0, confidence)),
    }
