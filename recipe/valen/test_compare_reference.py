#!/usr/bin/env python3
"""Regression tests for compare_reference.py.

Checks the pure comparison helpers without the pinned Valen source, model
weights or a GPU: the reference confidence formula, data-URL decoding, the
reference record builders, and the PASS/FAIL verdicts of compare_case.

    python recipe/valen/test_compare_reference.py
"""

from __future__ import annotations

import base64
import hashlib
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare_reference as tool


def worker_response(probabilities, choice, confidence, input_tokens=17, compute_tokens=17):
    return {
        "model": "valen-preview-0923",
        "answers": {
            "move": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities,
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "internal_usage": {"compute_tokens": compute_tokens},
    }


REFERENCE = {
    "keys": {"move": ["up", "down"]},
    "probabilities": {"move": [0.25, 0.75]},
    "logical_tokens": 17,
    "compute_tokens": 17,
}


def main() -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, condition, detail))

    # 1. Valen's confidence formula, including the recipe's recorded numbers.
    check("single candidate is fully confident", tool.reference_confidence([1.0]) == 1.0)
    check("two-way split gives 0.5", tool.reference_confidence([0.25, 0.75]) == 0.5)
    check(
        "uniform probabilities give zero",
        tool.reference_confidence([1 / 3, 1 / 3, 1 / 3]) == 0.0,
    )
    check(
        "recipe fixture confidence matches",
        abs(
            tool.reference_confidence([0.4459854086288687, 0.5540145913711313])
            - 0.10802918274226259
        )
        < 1e-15,
    )

    # 2. Data-URL decoding.
    payload = b"png-bytes"
    encoded = base64.b64encode(payload).decode("ascii")
    data, fmt = tool.decode_image_data_url(f"data:image/png;base64,{encoded}")
    check("data URL decodes to bytes and format", data == payload and fmt == "PNG")
    for bad in ("http://remote/x.png", "data:image/gif;base64,AAAA", "not a url", 17):
        try:
            tool.decode_image_data_url(bad)
        except ValueError:
            pass
        else:
            check(f"rejects {bad!r}", False)
            bad = None
    check("rejects non-PNG/JPEG data URLs", True)

    # 3. Reference record builders.
    questions = {"move": {"type": "choice", "instructions": "Choose.",
                          "criteria": {"up": "Up", "down": "Down"}}}
    with tempfile.TemporaryDirectory() as directory:
        media = Path(directory)
        record = tool.reference_image_record(b"image-bytes", "PNG", questions, media)
        digest = hashlib.sha256(b"image-bytes").hexdigest()
        check(
            "image record keeps native messages and asset hash",
            record["request"]["state"]["messages"][0]["content"][0]["image_url"]["url"]
            == "reference.png"
            and record["assets"] == [{"path": "reference.png", "sha256": digest}]
            and (media / "reference.png").read_bytes() == b"image-bytes",
        )
    record = tool.reference_text_record("plain context", questions)
    check(
        "text record uses the compiler's plain-string state",
        record["request"]["state"] == "plain context"
        and record["request"]["questions"] is questions
        and "assets" not in record,
    )
    check(
        "reference questions map null labels to their keys",
        tool.reference_questions(
            {"questions": {"q": {"instructions": None,
                                 "criteria": {"a": None, "b": "B"}}}}
        )["q"]["criteria"] == {"a": "a", "b": "B"},
    )

    # 4. A matching worker response passes every verdict.
    lines, ok = tool.compare_case(
        "text", worker_response({"up": 0.25, "down": 0.75}, "down", 0.5), REFERENCE
    )
    report = "\n".join(lines)
    check(
        "exact match passes",
        ok
        and "decision agreement: PASS" in report
        and "probability tolerance <= 1e-06: PASS" in report
        and "token accounting: PASS" in report
        and "input_tokens: 17 == 17" in report,
        report,
    )

    # 5. A tiny delta stays inside the declared tolerance.
    lines, ok = tool.compare_case(
        "text",
        worker_response({"up": 0.25 + 1e-9, "down": 0.75 - 1e-9}, "down", 0.5),
        REFERENCE,
    )
    check("delta within tolerance passes", ok, "\n".join(lines))

    # 6. Each failure mode fails on its own.
    lines, ok = tool.compare_case(
        "text", worker_response({"up": 0.6, "down": 0.4}, "up", 0.2), REFERENCE
    )
    check(
        "probability and decision mismatch fails",
        not ok
        and "probability tolerance <= 1e-06: FAIL" in "\n".join(lines)
        and "decision agreement: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    lines, ok = tool.compare_case(
        "text", worker_response({"up": 0.25, "down": 0.75}, "down", 0.5, input_tokens=18),
        REFERENCE,
    )
    check(
        "token mismatch fails",
        not ok and "token accounting: FAIL" in "\n".join(lines)
        and "input_tokens: 18 != 17" in "\n".join(lines),
        "\n".join(lines),
    )

    lines, ok = tool.compare_case(
        "text", worker_response({"down": 0.75, "up": 0.25}, "down", 0.5), REFERENCE
    )
    check(
        "candidate order mismatch fails",
        not ok and "candidate order" in "\n".join(lines),
        "\n".join(lines),
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
