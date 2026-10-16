"""End-to-end decide() tests with a CPU-only FakeD3: no weights, no torch.

FakeD3 mirrors the upstream runtime surface the worker drives —
prepare/run/respond plus model_name, max_length, the capability flags and
load_images/load_videos — with the pinned question-validation rules
reimplemented just far enough to produce upstream-shaped per-question
errors, and answer construction delegated to models.d3.postprocess (which
locks the pinned formulas).
"""

import base64
import binascii
import json
import logging

import pytest

from frontend.d3 import decide
from models.d3.postprocess import describe, to_answer
from models.d3.protocol import MODEL_NAME, RequestError

from .test_protocol import body

# Upstream strict-loading budgets (d3_runtime.py), reimplemented only for
# the fake's rejections: real pixel decode and frame sampling belong to the
# pinned runtime and stay author-side.
MAX_IMAGE_BYTES = 8_000_000
MAX_VIDEO_BYTES = 32_000_000


def image_data_url(payload=b"tiny-png-bytes"):
    return f"data:image/png;base64,{base64.b64encode(payload).decode()}"


def video_data_url(payload=b"tiny-mp4-bytes"):
    return f"data:video/mp4;base64,{base64.b64encode(payload).decode()}"


class FakePrepared:
    def __init__(self, keys, questions, errors):
        self.keys = list(keys)
        self.questions = questions
        self.errors = errors


class FakeD3:
    """The pinned runtime's contract, driven without its code.

    Two shapes, switched by ``media``: a vision-capable checkpoint
    (``image_unavailable``/``video_unavailable`` are None) and the text-only
    shape the runtime produces when the vision tower is missing.
    ``load_images``/``load_videos`` reimplement the data-URL slice of the
    runtime's strict loading just far enough to raise upstream-shaped
    rejections; they return ``("image"/"video", payload)`` markers instead
    of decoded pixels or sampled frames.
    """

    model_name = MODEL_NAME

    def __init__(self, probabilities=None, max_length=None, media=False):
        self.probabilities = dict(probabilities or {})
        self.max_length = max_length
        self.warmed = False
        self.closed = False
        self.image_unavailable = (
            None if media else "the checkpoint has no vision tower (visual.* weights)"
        )
        self.video_unavailable = None if media else self.image_unavailable
        self.modalities = ["text", "image", "video"] if media else ["text"]
        self.loaded_images = []
        self.loaded_videos = []
        self.prepare_calls = []

    # -- upstream strict loading, reduced to the data-URL envelope ---------
    @staticmethod
    def _image_payload(value):
        if not isinstance(value, str) or not value.startswith("data:"):
            raise ValueError("images must be base64 PNG, JPEG or WebP data URLs")
        header, separator, encoded = value.partition(",")
        kind = header.strip().lower()
        if (
            not separator
            or not kind.startswith("data:image/")
            or not kind.endswith(";base64")
        ):
            raise ValueError("a data URL image is data:image/<format>;base64,<data>")
        if kind[len("data:image/") : -len(";base64")] not in ("png", "jpeg", "jpg", "webp"):
            raise ValueError("images must be base64 PNG, JPEG or WebP data URLs")
        if len(encoded) > 4 * -(-MAX_IMAGE_BYTES // 3):
            raise ValueError(f"each image must be at most {MAX_IMAGE_BYTES:,} bytes")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise ValueError("invalid base64 image data") from exc
        if len(payload) > MAX_IMAGE_BYTES:
            raise ValueError(f"each image must be at most {MAX_IMAGE_BYTES:,} bytes")
        return payload

    @staticmethod
    def _video_payload(value):
        header, separator, encoded = value.partition(",")
        kind = header.strip().lower()
        if (
            not separator
            or not kind.startswith("data:video/")
            or not kind.endswith(";base64")
        ):
            raise ValueError("a data URL video is data:video/<format>;base64,<data>")
        if kind[len("data:video/") :].split(";")[0] not in (
            "mp4",
            "webm",
            "quicktime",
            "x-matroska",
        ):
            raise ValueError(
                "videos must be base64 MP4, WebM, QuickTime or Matroska data URLs"
            )
        if len(encoded) > 4 * -(-MAX_VIDEO_BYTES // 3):
            raise ValueError(f"each video must be at most {MAX_VIDEO_BYTES:,} bytes")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise ValueError("invalid base64 video data") from exc
        if len(payload) > MAX_VIDEO_BYTES:
            raise ValueError(f"each video must be at most {MAX_VIDEO_BYTES:,} bytes")
        return payload

    def load_images(self, images):
        if not images:
            return []
        if self.image_unavailable is not None:
            raise ValueError(f"image inputs are not available: {self.image_unavailable}")
        self.loaded_images.append(list(images))
        decoded = []
        for number, value in enumerate(images):
            try:
                decoded.append(("image", self._image_payload(value)))
            except ValueError as exc:
                raise ValueError(f"images[{number}]: {exc}") from exc
        return decoded

    def load_videos(self, videos):
        if not videos:
            return []
        if self.video_unavailable is not None:
            raise ValueError(f"video inputs are not available: {self.video_unavailable}")
        self.loaded_videos.append(list(videos))
        decoded = []
        for number, value in enumerate(videos):
            try:
                decoded.append(("video", self._video_payload(value)))
            except ValueError as exc:
                raise ValueError(f"videos[{number}]: {exc}") from exc
        return decoded

    # -- upstream normalize_question, reduced to what the tests exercise ----
    @staticmethod
    def _normalize(question):
        if not isinstance(question, dict):
            raise ValueError("a question is an object with a type")
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                raise ValueError("choice criteria must map 1 to 255 option keys to descriptions")
            if any(not isinstance(key, str) or not key for key in criteria):
                raise ValueError("choice option keys must be nonempty strings")
            return {"kind": "choice", "keys": list(criteria), "levels": None}
        if kind == "noul":
            if criteria is not None and (
                not isinstance(criteria, dict) or set(criteria) - {"false", "true"}
            ):
                raise ValueError('noul criteria may only describe "false" and "true"')
            return {"kind": "noul", "keys": ["false", "true"], "levels": None}
        if kind == "score":
            if not isinstance(criteria, (list, tuple)) or not 2 <= len(criteria) <= 10:
                raise ValueError("score criteria must be an ordered list of 2 to 10 levels")
            levels = [describe(level) for level in criteria]
            if any(not text for text in levels) or len(set(levels)) != len(levels):
                raise ValueError("score levels must be distinct and nonempty")
            return {
                "kind": "score",
                "keys": [str(index) for index in range(len(criteria))],
                "levels": list(criteria),
            }
        raise ValueError(f"unsupported question type {kind!r} (choice, noul or score)")

    def _tokens(self, state, question):
        return 8 + len(json.dumps(state)) // 8 + len(json.dumps(question)) // 8

    def prepare(self, state, questions, images=None, videos=None):
        self.prepare_calls.append((state, questions, images, videos))
        errors = {}
        normalized = {}
        for key, question in questions.items():
            try:
                normalized[key] = self._normalize(question)
            except ValueError as exc:
                kind = question.get("type") if isinstance(question, dict) else None
                errors[key] = {"type": kind, "error": "invalid_question", "message": str(exc)}
        prepared = FakePrepared(questions, normalized, errors)
        if self.max_length is not None:
            for key, question in questions.items():
                if key in errors:
                    continue
                tokens = self._tokens(state, question)
                if tokens > self.max_length:
                    errors[key] = {
                        "type": normalized[key]["kind"],
                        "error": "max_length_exceeded",
                        "message": (
                            f"the question prompt has {tokens} tokens, over the maximum"
                            f" context length of {self.max_length} tokens; nothing was truncated"
                        ),
                    }
        return prepared

    def run(self, prepared):
        probabilities = {}
        tokens = 0
        for key in prepared.keys:
            if key in prepared.errors:
                continue
            probabilities[key] = self.probabilities.get(key) or [
                1.0 for _ in prepared.questions[key]["keys"]
            ]
            tokens += 8 + 4 * len(probabilities[key])
        return probabilities, tokens

    def respond(self, prepared, probabilities, tokens):
        answers = {}
        for key in prepared.keys:
            if key in prepared.errors:
                answers[key] = dict(prepared.errors[key])
                continue
            question = prepared.questions[key]
            try:
                answers[key] = to_answer(
                    question["kind"], question["keys"], question["levels"], probabilities[key]
                )
            except ValueError as exc:
                answers[key] = {
                    "type": question["kind"],
                    "error": "invalid_model_output",
                    "message": str(exc),
                }
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    def warmup(self):
        self.warmed = True

    def close(self):
        self.closed = True


def three_kind_body():
    return {
        "model": MODEL_NAME,
        "state": {"ticket": "double charge", "customer": "vip"},
        "questions": {
            "refund": {
                "type": "choice",
                "instructions": "Decide the refund action.",
                "criteria": {"refund": "Refund the duplicate charge", "wait": "Wait for review"},
            },
            "urgent": {
                "type": "noul",
                "instructions": "Is this urgent for the customer?",
                "criteria": {"true": "Customer hardship"},
            },
            "severity": {
                "type": "score",
                "instructions": "Rate the severity.",
                "criteria": ["minor", "moderate", "major"],
            },
        },
    }


def test_decide_answers_all_three_kinds_in_one_request():
    executor = FakeD3(
        probabilities={
            "refund": [1.0, 3.0],
            "urgent": [1.0, 3.0],
            "severity": [1.0, 2.0, 1.0],
        }
    )
    result = decide(json.dumps(three_kind_body()).encode("utf-8"), executor)

    assert result["model"] == "d3"
    assert list(result["answers"]) == ["refund", "urgent", "severity"]
    assert result["answers"]["refund"]["choice"] == "wait"
    assert result["answers"]["refund"]["probabilities"] == {"refund": 0.25, "wait": 0.75}
    assert result["answers"]["urgent"] == {"type": "noul", "noul": 0.75}
    assert result["answers"]["severity"]["score"] == pytest.approx(1.0)
    assert result["answers"]["severity"]["legend"] == {"0": "minor", "1": "moderate", "2": "major"}
    assert result["usage"]["output_tokens"] == 0
    assert result["usage"]["input_tokens"] == 8 + 4 * 2 + 8 + 4 * 2 + 8 + 4 * 3


def test_decide_aggregates_invalid_questions_into_one_422():
    value = three_kind_body()
    value["questions"]["broken"] = {"type": "triangle"}
    value["questions"]["also_broken"] = {"type": "choice", "criteria": {}}
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(value).encode("utf-8"), FakeD3())

    assert excinfo.value.status == 422
    assert list(excinfo.value.body["invalid_questions"]) == ["broken", "also_broken"]
    assert "unsupported question type" in excinfo.value.body["invalid_questions"]["broken"]
    assert "choice criteria" in excinfo.value.body["invalid_questions"]["also_broken"]


def test_decide_rejects_over_limit_questions_without_truncating():
    value = three_kind_body()
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(value).encode("utf-8"), FakeD3(max_length=16))

    assert excinfo.value.status == 422
    assert excinfo.value.body is None
    assert "tokens, over the maximum context length of 16" in str(excinfo.value)
    assert "nothing was truncated" in str(excinfo.value)


def test_decide_keeps_partial_success_and_warns(caplog):
    executor = FakeD3(
        probabilities={"refund": [1.0, 3.0], "urgent": [0.0, 0.0], "severity": [1.0, 2.0, 1.0]}
    )
    with caplog.at_level(logging.WARNING, logger="models.d3.postprocess"):
        result = decide(json.dumps(three_kind_body()).encode("utf-8"), executor)

    broken = result["answers"]["urgent"]
    assert broken["type"] == "noul"
    assert broken["error"] == "invalid_model_output"
    assert "one finite non-negative probability" in broken["message"]
    assert result["answers"]["refund"]["choice"] == "wait"
    assert result["answers"]["severity"]["type"] == "score"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "urgent" in warnings[0].getMessage()
    assert "invalid_model_output" in warnings[0].getMessage()


def test_decide_maps_runtime_value_errors_to_422():
    class RejectingD3(FakeD3):
        def prepare(self, state, questions):
            raise ValueError("questions must be a nonempty mapping")

    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(body()).encode("utf-8"), RejectingD3())
    assert excinfo.value.status == 422
    assert "pinned runtime" in str(excinfo.value)
    assert "nonempty mapping" in str(excinfo.value)


def test_decide_rejects_responses_that_break_question_order():
    class ReorderedD3(FakeD3):
        def respond(self, prepared, probabilities, tokens):
            response = super().respond(prepared, probabilities, tokens)
            response["answers"] = dict(reversed(list(response["answers"].items())))
            return response

    with pytest.raises(RuntimeError, match="question order"):
        decide(json.dumps(three_kind_body()).encode("utf-8"), ReorderedD3())


def test_decide_decodes_images_and_passes_them_to_prepare():
    executor = FakeD3(media=True)
    value = three_kind_body()
    value["images"] = [image_data_url(b"one"), image_data_url(b"two")]
    result = decide(json.dumps(value).encode("utf-8"), executor)

    state, questions, images, videos = executor.prepare_calls[0]
    assert images == [("image", b"one"), ("image", b"two")]
    assert videos is None
    assert executor.loaded_images == [list(value["images"])]
    assert executor.loaded_videos == []
    assert result["model"] == "d3"
    assert list(result["answers"]) == ["refund", "urgent", "severity"]


def test_decide_routes_media_to_the_upstream_prepare_branches():
    # Videos (with or without images) take the four-argument branch, images
    # alone the three-argument one, and empty or absent media stays text.
    executor = FakeD3(media=True)
    value = three_kind_body()
    value["images"] = [image_data_url()]
    value["videos"] = [video_data_url()]
    decide(json.dumps(value).encode("utf-8"), executor)
    _, _, images, videos = executor.prepare_calls[0]
    assert images == [("image", b"tiny-png-bytes")]
    assert videos == [("video", b"tiny-mp4-bytes")]

    executor = FakeD3(media=True)
    value = three_kind_body()
    value["images"] = []
    value["videos"] = []
    decide(json.dumps(value).encode("utf-8"), executor)
    _, _, images, videos = executor.prepare_calls[0]
    assert images is None and videos is None
    assert executor.loaded_images == [] and executor.loaded_videos == []

    executor = FakeD3(media=True)
    decide(json.dumps(three_kind_body()).encode("utf-8"), executor)
    _, _, images, videos = executor.prepare_calls[0]
    assert images is None and videos is None


def test_decide_refuses_media_a_text_only_checkpoint_cannot_read():
    value = three_kind_body()
    value["images"] = [image_data_url()]
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(value).encode("utf-8"), FakeD3())
    assert excinfo.value.status == 422
    assert str(excinfo.value) == "This model reads text only; images are not supported."

    value = three_kind_body()
    value["videos"] = [video_data_url()]
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(value).encode("utf-8"), FakeD3())
    assert excinfo.value.status == 422
    assert str(excinfo.value) == "This model does not read videos."

    # The gates are independent: an image-capable checkpoint without video
    # support still refuses videos, after the images have decoded.
    executor = FakeD3(media=True)
    executor.video_unavailable = "video inputs need a Qwen3.5 backbone"
    value = three_kind_body()
    value["images"] = [image_data_url()]
    value["videos"] = [video_data_url()]
    with pytest.raises(RequestError) as excinfo:
        decide(json.dumps(value).encode("utf-8"), executor)
    assert str(excinfo.value) == "This model does not read videos."
    assert executor.loaded_images == [list(value["images"])]
    assert executor.loaded_videos == []
    assert executor.prepare_calls == []


def test_decide_rejects_undecodable_media_with_the_runtime_message():
    for field, bad, expected in (
        ("images", "https://example.com/pic.png", "images must be base64 PNG"),
        ("images", "data:image/tiff;base64,AAAA", "images must be base64 PNG"),
        ("images", "data:image/png;base64,!!!!", "invalid base64 image data"),
        ("images", "data:image/png;base64," + "A" * 11_000_000, "8,000,000 bytes"),
        ("videos", "data:video/avi;base64,AAAA", "videos must be base64 MP4"),
        ("videos", "data:video/mp4;base64," + "A" * 43_000_000, "32,000,000 bytes"),
    ):
        value = three_kind_body()
        value[field] = [bad]
        with pytest.raises(RequestError) as excinfo:
            decide(json.dumps(value).encode("utf-8"), FakeD3(media=True))
        assert excinfo.value.status == 422
        assert excinfo.value.body is None
        assert str(excinfo.value).startswith(f"{field}[0]: ")
        assert expected in str(excinfo.value)
