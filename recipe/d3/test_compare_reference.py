#!/usr/bin/env python3
"""Regression tests for compare_reference.py.

Checks the pure comparison and export helpers without the pinned d3 package,
weights or a GPU: the PASS/FAIL verdicts of compare_case, embedded-error
shape agreement, the reference arm's media decode stage, the oracle export
gate, and the oracle export structure.

    python recipe/d3/test_compare_reference.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare_reference as tool


def worker_response(answers, input_tokens=17):
    return {
        "model": "d3",
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }


CHOICE = {
    "type": "choice",
    "choice": "wait",
    "probabilities": {"refund": 0.25, "wait": 0.75},
    "confidence": 0.18872187554086717,
}
NOUL = {"type": "noul", "noul": 0.75}
SCORE = {
    "type": "score",
    "score": 1.0,
    "probabilities": {"0": 0.25, "1": 0.5, "2": 0.25},
    "confidence": 0.18872187554086717,
    "legend": {"0": "minor", "1": "moderate", "2": "major"},
}


def reference():
    return worker_response({"refund": dict(CHOICE), "urgent": dict(NOUL), "severity": dict(SCORE)})


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))

    # 1. An exactly matching worker response passes every verdict.
    lines, ok = tool.compare_case("fixture", worker_response(
        {"refund": dict(CHOICE), "urgent": dict(NOUL), "severity": dict(SCORE)}
    ), reference())
    report = "\n".join(lines)
    check(
        "exact match passes",
        ok
        and "decision agreement: PASS" in report
        and "numeric tolerance <= 1e-06: PASS" in report
        and "token accounting: PASS" in report,
        report,
    )

    # 2. A tiny delta stays inside the declared tolerance.
    drifted = dict(CHOICE)
    drifted["probabilities"] = {"refund": 0.25 + 1e-9, "wait": 0.75 - 1e-9}
    drifted["confidence"] = CHOICE["confidence"] + 1e-9
    lines, ok = tool.compare_case(
        "fixture", worker_response({"refund": drifted, "urgent": dict(NOUL), "severity": dict(SCORE)}),
        reference(),
    )
    check("delta within tolerance passes", ok, "\n".join(lines))

    # 3. Each failure mode fails on its own.
    flipped = dict(CHOICE)
    flipped["choice"] = "refund"
    lines, ok = tool.compare_case(
        "fixture", worker_response({"refund": flipped, "urgent": dict(NOUL), "severity": dict(SCORE)}),
        reference(),
    )
    check(
        "decision mismatch fails",
        not ok and "decision agreement: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    lines, ok = tool.compare_case(
        "fixture", worker_response(
            {"refund": dict(CHOICE), "urgent": dict(NOUL), "severity": dict(SCORE)},
            input_tokens=18,
        ), reference(),
    )
    check(
        "token mismatch fails",
        not ok and "token accounting: FAIL" in "\n".join(lines)
        and "input_tokens: 18 != 17" in "\n".join(lines),
        "\n".join(lines),
    )

    reordered = {"urgent": dict(NOUL), "refund": dict(CHOICE), "severity": dict(SCORE)}
    lines, ok = tool.compare_case("fixture", worker_response(reordered), reference())
    check(
        "answer order mismatch fails",
        not ok and "answer order differs" in "\n".join(lines),
        "\n".join(lines),
    )

    # 4. Embedded invalid_model_output shapes must agree between the paths.
    bad = {
        "type": "noul",
        "error": "invalid_model_output",
        "message": "need one finite non-negative probability per option",
    }
    lines, ok = tool.compare_case(
        "fixture", worker_response({"refund": dict(bad), "urgent": dict(NOUL), "severity": dict(SCORE)}),
        worker_response({"refund": dict(bad), "urgent": dict(NOUL), "severity": dict(SCORE)}),
    )
    check("matching embedded errors pass", ok, "\n".join(lines))

    other_message = dict(bad)
    other_message["message"] = "different failure"
    lines, ok = tool.compare_case(
        "fixture", worker_response({"refund": dict(other_message), "urgent": dict(NOUL), "severity": dict(SCORE)}),
        worker_response({"refund": dict(bad), "urgent": dict(NOUL), "severity": dict(SCORE)}),
    )
    check(
        "embedded error mismatch fails",
        not ok and "embedded error shape differs" in "\n".join(lines),
        "\n".join(lines),
    )

    # 5. Rejection agreement: matching 422s pass, mismatches fail.
    lines, ok = tool.compare_case("fixture", worker_response({}), ("422", "too long"))
    check(
        "worker answering a rejected request fails",
        not ok and "reference rejected the request: too long" in "\n".join(lines),
        "\n".join(lines),
    )

    rejected = worker_response({})
    rejected["worker rejected the request"] = {
        "status": 422,
        "body": {"invalid_questions": {"bad": "unsupported question type"}},
    }
    lines, ok = tool.compare_case(
        "fixture", rejected, ("422", {"invalid_questions": {"bad": "unsupported question type"}})
    )
    check(
        "matching 422 rejections pass",
        ok and "rejection agreement: PASS" in "\n".join(lines),
        "\n".join(lines),
    )

    wrong_body = dict(rejected)
    wrong_body["worker rejected the request"] = {
        "status": 422,
        "body": {"detail": "different"},
    }
    lines, ok = tool.compare_case(
        "fixture", wrong_body, ("422", {"invalid_questions": {"bad": "unsupported question type"}})
    )
    check(
        "mismatched rejection bodies fail",
        not ok and "rejection agreement: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    non_422 = dict(rejected)
    non_422["worker rejected the request"] = {"status": 500, "body": {"detail": "x"}}
    lines, ok = tool.compare_case("fixture", non_422, ("422", "too long"))
    check(
        "non-422 worker rejection fails",
        not ok and "rejection agreement: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    # 5b. The reference arm mirrors the upstream decode stage for media.
    class FakeRuntime:
        image_unavailable = None
        video_unavailable = None

        def __init__(self, unavailable=None):
            if unavailable == "image":
                self.image_unavailable = "the checkpoint has no vision tower (visual.* weights)"
                self.video_unavailable = self.image_unavailable
            elif unavailable == "video":
                self.video_unavailable = "video inputs need a Qwen3.5 backbone"
            self.image_calls = []
            self.video_calls = []
            self.prepared = []

        def load_images(self, images, strict=False):
            self.image_calls.append((list(images), strict))
            return [("image", value) for value in images]

        def load_videos(self, videos, strict=False):
            self.video_calls.append((list(videos), strict))
            return [("video", value) for value in videos]

        def prepare(self, state, questions, images=None, videos=None):
            self.prepared.append((images, videos))
            return SimpleNamespace(errors={})

        def run(self, prepared):
            return {}, 5

        def respond(self, prepared, probabilities, tokens):
            return {
                "model": "d3",
                "answers": {},
                "usage": {"input_tokens": tokens, "output_tokens": 0},
            }

    capable = FakeRuntime()
    images, videos = tool.decode_request_media(
        capable, {"images": ["data:image/png;base64,AAAA"], "videos": ["data:video/mp4;base64,AAAA"]}
    )
    check(
        "decode passes media through strict loading",
        images == [("image", "data:image/png;base64,AAAA")]
        and videos == [("video", "data:video/mp4;base64,AAAA")]
        and capable.image_calls == [(["data:image/png;base64,AAAA"], True)]
        and capable.video_calls == [(["data:video/mp4;base64,AAAA"], True)],
    )

    empty = FakeRuntime()
    images, videos = tool.decode_request_media(empty, {"images": [], "videos": None})
    check(
        "empty and absent media decode to nothing",
        images == [] and videos == [] and empty.image_calls == [] and empty.video_calls == [],
    )

    text_only = FakeRuntime(unavailable="image")
    try:
        tool.decode_request_media(text_only, {"images": ["data:image/png;base64,AAAA"]})
        check("text-only checkpoint refuses images", False, "no ValueError")
    except ValueError as exc:
        check(
            "text-only checkpoint refuses images with the upstream message",
            str(exc) == "This model reads text only; images are not supported.",
            str(exc),
        )

    no_video = FakeRuntime(unavailable="video")
    try:
        tool.decode_request_media(no_video, {"videos": ["data:video/mp4;base64,AAAA"]})
        check("video-incapable checkpoint refuses videos", False, "no ValueError")
    except ValueError as exc:
        check(
            "video-incapable checkpoint refuses videos with the upstream message",
            str(exc) == "This model does not read videos.",
            str(exc),
        )

    for field, bad in (("images", "data:image/png;base64,AAAA"), ("videos", 17)):
        try:
            tool.decode_request_media(capable, {field: bad})
            check(f"non-list {field} is refused", False, "no ValueError")
        except ValueError as exc:
            check(
                f"non-list {field} is refused with the upstream message",
                str(exc) == f"{field} must be a list of base64 data URLs",
                str(exc),
            )

    media = FakeRuntime()
    result = tool.reference_decide(
        media,
        {"state": "s", "questions": {"q": {"type": "noul"}}, "images": ["data:image/png;base64,AAAA"]},
    )
    check(
        "reference_decide prepares decoded images on the image branch",
        media.prepared == [([("image", "data:image/png;base64,AAAA")], None)]
        and result["usage"] == {"input_tokens": 5, "output_tokens": 0},
        str(media.prepared),
    )

    media = FakeRuntime()
    tool.reference_decide(
        media,
        {
            "state": "s",
            "questions": {"q": {"type": "noul"}},
            "images": ["data:image/png;base64,AAAA"],
            "videos": ["data:video/mp4;base64,AAAA"],
        },
    )
    check(
        "reference_decide prepares both media on the video branch",
        media.prepared
        == [
            (
                [("image", "data:image/png;base64,AAAA")],
                [("video", "data:video/mp4;base64,AAAA")],
            )
        ],
        str(media.prepared),
    )

    class RejectingLoad(FakeRuntime):
        def load_images(self, images, strict=False):
            raise ValueError("images[0]: each image must be at most 8,000,000 bytes")

    result = tool.reference_decide(
        RejectingLoad(),
        {"state": "s", "questions": {"q": {"type": "noul"}}, "images": ["data:image/png;base64,AAAA"]},
    )
    check(
        "reference_decide maps decode rejections to plain-detail 422",
        result == ("422", "images[0]: each image must be at most 8,000,000 bytes"),
        str(result),
    )

    # 6. The oracle export freezes the full stage-3 comparison surface.
    body = {"model": "d3", "state": "s", "questions": {"q": {"type": "noul"}}}

    class Prepared:
        sequences = {"q": [1, 2, 3]}

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "oracle.json"
        oracle = tool.export_oracle(
            body,
            Prepared(),
            {"q": [0.25, 0.75]},
            worker_response({"q": dict(NOUL)}),
            path,
        )
        reread = json.loads(path.read_text(encoding="utf-8"))
        check(
            "oracle export round-trips",
            oracle == reread
            and oracle["token_ids"] == {"q": [1, 2, 3]}
            and oracle["probabilities"] == {"q": [0.25, 0.75]}
            and oracle["answers"]["q"]["noul"] == 0.75
            and oracle["usage"] == {"input_tokens": 17, "output_tokens": 0},
            json.dumps(oracle)[:200],
        )

    # 7. The oracle is frozen only for a passing, non-rejected comparison.
    served = worker_response({"q": dict(NOUL)})
    check(
        "export allowed for a passing 200 case",
        tool.export_allowed(True, served, worker_response({"q": dict(NOUL)})),
    )
    check(
        "export refused for a failed comparison",
        not tool.export_allowed(False, served, worker_response({"q": dict(NOUL)})),
    )
    mutual_422 = worker_response({})
    mutual_422["worker rejected the request"] = {"status": 422, "body": {"detail": "too long"}}
    check(
        "export refused for an agreed 422 (nothing to freeze)",
        not tool.export_allowed(True, mutual_422, ("422", "too long")),
    )
    check(
        "export refused when only the worker rejected",
        not tool.export_allowed(True, mutual_422, worker_response({"q": dict(NOUL)})),
    )

    failed = 0
    for name, passed, detail in results:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
        if not passed:
            failed += 1
            print("       " + detail.strip().replace("\n", "\n       ")[:400])
    print(
        "compare_reference tests: ok"
        if not failed
        else f"compare_reference tests: {failed} FAILED"
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
