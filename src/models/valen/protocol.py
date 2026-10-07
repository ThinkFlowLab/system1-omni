"""Wire contract for the Valen-Preview-0923 reference worker.

This module deliberately has no torch dependency. It validates the public
/v1/systemone request and keeps the model-specific image and text-state limits
separate from the later Valen compiler and executor layers.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
from dataclasses import dataclass
from typing import Any

from PIL import Image, UnidentifiedImageError

MODEL_NAME = "valen-preview-0923"
MAX_BODY = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 1024 * 1024
MAX_SIDE = 2048
MAX_ASPECT_RATIO = 200
MAX_QUESTIONS = 8
MAX_OPTIONS = 255
MAX_TEXT = 16384

_MEDIA_CONTROL_TOKENS = (
    "<|image_pad|>",
    "<|video_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
)


class RequestError(ValueError):
    """A request cannot be evaluated under the reference contract."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


class MalformedJSON(RequestError):
    """The body is not a usable UTF-8 JSON object."""

    def __init__(self, message: str) -> None:
        super().__init__(message, 400)


@dataclass(frozen=True)
class ImageData:
    """Validated inline image bytes ready for request-scoped materialization."""

    data: bytes
    format: str
    width: int
    height: int


@dataclass(frozen=True)
class TextState:
    """Validated text state: a plain or JSON-serialized string."""

    text: str


@dataclass(frozen=True)
class Question:
    name: str
    instructions: str
    keys: tuple[str, ...]
    descriptions: tuple[str, ...]


@dataclass(frozen=True)
class Request:
    model: str
    state: ImageData | TextState
    questions: tuple[Question, ...]


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MalformedJSON(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise MalformedJSON(f"non-finite JSON number {value} is not supported")


def parse_body(raw: bytes) -> dict[str, Any]:
    """Decode a bounded JSON body and reject ambiguous JSON values."""

    if len(raw) > MAX_BODY:
        raise RequestError("request body exceeds 8 MiB", 413)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=_nonfinite,
        )
        # Reject lone surrogates and values that cannot be emitted as UTF-8.
        json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except MalformedJSON:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise MalformedJSON("request body must contain valid JSON and UTF-8 text") from exc
    if not isinstance(value, dict):
        raise MalformedJSON("request must be a JSON object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, (str, dict, list)):
        raise RequestError(f"{field} must be a string, object or array")
    try:
        result = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise RequestError(f"invalid {field}") from exc
    if len(result) > MAX_TEXT:
        raise RequestError(f"{field} exceeds {MAX_TEXT} characters")
    if any(token in result for token in _MEDIA_CONTROL_TOKENS):
        raise RequestError(f"{field} contains an unsupported media control token")
    return result


def _image(state: Any) -> ImageData:
    if not isinstance(state, dict) or set(state) != {"image"}:
        raise RequestError("state must contain exactly one image data URL")
    url = state["image"]
    if not isinstance(url, str):
        raise RequestError("state.image must be a PNG/JPEG base64 data URL")
    prefix, separator, encoded = url.partition(",")
    expected = {
        "data:image/png;base64": "PNG",
        "data:image/jpeg;base64": "JPEG",
    }
    if not separator or prefix not in expected:
        raise RequestError("only inline PNG/JPEG images are supported")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise RequestError("encoded image exceeds 4 MiB")
    try:
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_IMAGE_BYTES:
            raise RequestError("image exceeds 4 MiB")
        with Image.open(io.BytesIO(data)) as source:
            if source.format != expected[prefix]:
                raise RequestError("image format does not match its MIME type")
            width, height = source.size
            if max(width, height) > MAX_ASPECT_RATIO * min(width, height):
                raise RequestError(f"image aspect ratio must not exceed {MAX_ASPECT_RATIO}:1")
            if (
                max(width, height) > MAX_SIDE
                or width * height > MAX_PIXELS
                or getattr(source, "n_frames", 1) != 1
            ):
                raise RequestError(
                    "image must be single-frame, at most 2048 per side and 1048576 pixels"
                )
            source.load()
    except RequestError:
        raise
    except (
        binascii.Error,
        Image.DecompressionBombError,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        raise RequestError("invalid image data") from exc
    return ImageData(data, expected[prefix], width, height)


def _state(value: Any) -> ImageData | TextState:
    """Accept one inline image or one text state.

    A dict carrying an "image" key always enters image validation, so a
    mistyped image request fails loudly instead of silently becoming text.
    Every other string, object or array follows the shared text-state wire
    convention: objects and arrays are serialized to JSON text.
    """

    if isinstance(value, dict) and "image" in value:
        return _image(value)
    if isinstance(value, (str, dict, list)):
        text = _text(value, "state")
        if not text:
            raise RequestError("state must not be empty")
        return TextState(text)
    raise RequestError("state must be an image data URL or text")


def parse_request(value: dict[str, Any]) -> Request:
    """Validate the system1-omni wire request."""

    if not isinstance(value, dict) or set(value) != {"model", "state", "questions"}:
        raise RequestError("request must contain model, state and questions only")
    if value["model"] != MODEL_NAME:
        raise RequestError(f"model must be {MODEL_NAME}")

    questions = value["questions"]
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise RequestError(f"questions must contain 1 to {MAX_QUESTIONS} questions")

    parsed: list[Question] = []
    for name, question in questions.items():
        if not isinstance(name, str) or not name or len(name) > 256:
            raise RequestError("question names must contain 1 to 256 characters")
        if not isinstance(question, dict) or set(question) - {
            "type",
            "instructions",
            "criteria",
        }:
            raise RequestError(f"question {name!r} contains unsupported fields")
        if question.get("type") != "choice":
            raise RequestError(f"question {name!r}: only choice is supported")
        if "instructions" not in question:
            raise RequestError(f"question {name!r}: instructions is required")
        instructions = (
            "" if question["instructions"] is None else _text(question["instructions"], "instructions")
        )
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or not 1 <= len(criteria) <= MAX_OPTIONS:
            raise RequestError(
                f"question {name!r}: choice requires 1 to {MAX_OPTIONS} options"
            )
        keys: list[str] = []
        descriptions: list[str] = []
        for key, label in criteria.items():
            if not isinstance(key, str) or not key or len(key) > 256:
                raise RequestError("option keys must contain 1 to 256 characters")
            keys.append(key)
            descriptions.append(
                key if label is None else _text(label, f"criteria {key!r}")
            )
        if len(instructions) + sum(map(len, descriptions)) > MAX_TEXT:
            raise RequestError("combined question text exceeds 16384 characters")
        parsed.append(Question(name, instructions, tuple(keys), tuple(descriptions)))

    return Request(MODEL_NAME, _state(value["state"]), tuple(parsed))
