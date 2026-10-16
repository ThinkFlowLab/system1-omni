#!/usr/bin/env python3
"""Regression tests for compare_reference.py.

Checks the pure comparison helpers without the pinned Valen source, model
weights or a GPU: the reference confidence formula, data-URL decoding, the
reference record builders (including the noul/score record rules), and the
PASS/FAIL verdicts of compare_case.

    python recipe/valen/test_compare_reference.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import compare_reference as tool


def worker_response(answers, input_tokens=17, compute_tokens=17):
    return {
        "model": "valen-preview-0923",
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        "internal_usage": {"compute_tokens": compute_tokens},
    }


def choice_answer(probabilities, choice, confidence):
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": probabilities,
        "confidence": confidence,
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
        and "assets" not in record
    )

    # 3b. Reference question records reuse the worker's own record rules.
    def request_body(question):
        return {
            "model": "valen-preview-0923",
            "state": "The card was charged twice for one order.",
            "questions": {"q": question},
        }

    check(
        "reference questions map null labels to their keys",
        tool.reference_questions(request_body(
            {"type": "choice", "instructions": "Choose.",
             "criteria": {"a": None, "b": "B"}}))["q"]["criteria"] == {"a": "a", "b": "B"},
    )
    check(
        "noul without wording stays the reference's native noul record",
        tool.reference_questions(request_body(
            {"type": "noul", "instructions": "Was it duplicated?"}))["q"]
        == {"type": "noul", "instructions": "Was it duplicated?"},
    )
    custom = tool.reference_questions(request_body(
        {"type": "noul", "instructions": "Was it duplicated?",
         "criteria": {"true": "Duplicated.", "false": None}}))["q"]
    check(
        "custom noul wording becomes a true/false choice record",
        custom["type"] == "choice"
        and custom["criteria"]["true"] == "Duplicated."
        and custom["criteria"]["false"] == "False / 否：不满足问题中的条件。",
    )
    check(
        "score levels pass through as an ordered list",
        tool.reference_questions(request_body(
            {"type": "score", "instructions": "Grade.",
             "criteria": ["low", "high"]}))["q"]
        == {"type": "score", "instructions": "Grade.", "criteria": ["low", "high"]},
    )

    # 3c. The committed noul and score fixtures build valid records.
    noul_fixture = json.loads(
        (HERE / "example-noul-request.json").read_text(encoding="utf-8")
    )
    noul_records = tool.reference_questions(noul_fixture)
    check(
        "noul fixture builds one native and one worded record",
        noul_records["duplicate"] == {
            "type": "noul",
            "instructions": "Was the card charged twice for one order?",
        }
        and noul_records["authorized"]["type"] == "choice"
        and set(noul_records["authorized"]["criteria"]) == {"true", "false"},
    )
    score_fixture = json.loads(
        (HERE / "example-score-request.json").read_text(encoding="utf-8")
    )
    score_records = tool.reference_questions(score_fixture)
    check(
        "score fixture builds one three-level score record",
        score_records["severity"]["type"] == "score"
        and len(score_records["severity"]["criteria"]) == 3,
    )

    # 4. A matching worker response passes every verdict.
    lines, ok = tool.compare_case(
        "text",
        worker_response(
            {"move": choice_answer({"up": 0.25, "down": 0.75}, "down", 0.5)}
        ),
        REFERENCE,
        {"move": "choice"},
    )
    report = "\n".join(lines)
    check(
        "exact match passes",
        ok
        and "decision agreement: PASS" in report
        and "probability tolerance <= 1e-06: PASS" in report
        and "confidence tolerance <= 1e-06: PASS" in report
        and "token accounting: PASS" in report
        and "input_tokens: 17 == 17" in report,
        report,
    )

    # 5. A tiny delta stays inside the declared tolerance.
    lines, ok = tool.compare_case(
        "text",
        worker_response(
            {"move": choice_answer({"up": 0.25 + 1e-9, "down": 0.75 - 1e-9}, "down", 0.5)}
        ),
        REFERENCE,
        {"move": "choice"},
    )
    check("delta within tolerance passes", ok, "\n".join(lines))

    # 6. Each failure mode fails on its own.
    lines, ok = tool.compare_case(
        "text",
        worker_response({"move": choice_answer({"up": 0.6, "down": 0.4}, "up", 0.2)}),
        REFERENCE,
        {"move": "choice"},
    )
    check(
        "probability and decision mismatch fails",
        not ok
        and "probability tolerance <= 1e-06: FAIL" in "\n".join(lines)
        and "decision agreement: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    lines, ok = tool.compare_case(
        "text",
        worker_response({"move": choice_answer({"up": 0.25, "down": 0.75}, "down", 0.9)}),
        REFERENCE,
        {"move": "choice"},
    )
    check(
        "confidence drift beyond tolerance fails",
        not ok and "confidence tolerance <= 1e-06: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    lines, ok = tool.compare_case(
        "text",
        worker_response(
            {"move": choice_answer({"up": 0.25, "down": 0.75}, "down", 0.5)},
            input_tokens=18,
        ),
        REFERENCE,
        {"move": "choice"},
    )
    check(
        "token mismatch fails",
        not ok and "token accounting: FAIL" in "\n".join(lines)
        and "input_tokens: 18 != 17" in "\n".join(lines),
        "\n".join(lines),
    )

    lines, ok = tool.compare_case(
        "text",
        worker_response(
            {"move": choice_answer({"down": 0.75, "up": 0.25}, "down", 0.5)}
        ),
        REFERENCE,
        {"move": "choice"},
    )
    check(
        "candidate order mismatch fails",
        not ok and "candidate order" in "\n".join(lines),
        "\n".join(lines),
    )

    # 7. noul answers compare P(true) with tolerance.
    noul_reference = {
        "keys": {"ask": ["true", "false"]},
        "probabilities": {"ask": [0.75, 0.25]},
        # The pinned decoder answers native noul with P(true) only; custom
        # wording decodes as a choice pair instead.
        "answers": {"ask": {"type": "noul", "noul": 0.75}},
        "logical_tokens": 17,
        "compute_tokens": 17,
    }

    def noul_answer(p_true, confidence):
        return {
            "type": "noul",
            "noul": p_true,
            "probabilities": {"true": p_true, "false": 1.0 - p_true},
            "confidence": confidence,
        }

    lines, ok = tool.compare_case(
        "noul",
        worker_response({"ask": noul_answer(0.75, 0.5)}),
        noul_reference,
        {"ask": "noul"},
    )
    report = "\n".join(lines)
    check(
        "noul exact match passes with the answer tolerance",
        ok
        and "answer tolerance <= 1e-06: PASS" in report
        and "legend agreement" not in report,
        report,
    )

    lines, ok = tool.compare_case(
        "noul",
        worker_response({"ask": noul_answer(0.75 + 1e-4, 0.5)}),
        noul_reference,
        {"ask": "noul"},
    )
    check(
        "noul answer drift beyond tolerance fails",
        not ok and "answer tolerance <= 1e-06: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    # A noul request answered in choice shape must fail even when the
    # candidate probabilities, decision and tokens all match.
    drifted = {
        "type": "choice",
        "choice": "true",
        "probabilities": {"true": 0.75, "false": 0.25},
        "confidence": 0.5,
    }
    lines, ok = tool.compare_case(
        "noul", worker_response({"ask": drifted}), noul_reference, {"ask": "noul"}
    )
    check(
        "noul answered as choice fails",
        not ok
        and "decision agreement: FAIL" in "\n".join(lines)
        and "answer type 'choice' != requested 'noul'" in "\n".join(lines),
        "\n".join(lines),
    )

    choice_shaped = dict(noul_reference)
    choice_shaped["answers"] = {
        "ask": {"type": "choice", "probabilities": {"true": 0.75, "false": 0.25},
                "choice": "true", "confidence": 0.5}
    }
    lines, ok = tool.compare_case(
        "noul",
        worker_response({"ask": noul_answer(0.75, 0.5)}),
        choice_shaped,
        {"ask": "noul"},
    )
    check(
        "custom-wording noul compares against the choice-shaped decode",
        ok and "confidence delta: 0.0000000000e+00" in "\n".join(lines),
        "\n".join(lines),
    )

    # 8. score answers compare the expected level, legend and confidence.
    score_reference = {
        "keys": {"grade": ["0", "1", "2"]},
        "probabilities": {"grade": [0.7, 0.2, 0.1]},
        "answers": {
            "grade": {
                "type": "score",
                "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1},
                "score": 0.4,
                "legend": {"0": "No error", "1": "A minor error", "2": "A major error"},
                "confidence": 0.4,
            }
        },
        "logical_tokens": 17,
        "compute_tokens": 51,
    }

    def score_answer(score, legend, confidence):
        return {
            "type": "score",
            "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1},
            "score": score,
            "legend": legend,
            "confidence": confidence,
        }

    legend = {"0": "No error", "1": "A minor error", "2": "A major error"}
    lines, ok = tool.compare_case(
        "score",
        worker_response({"grade": score_answer(0.4, legend, 0.4)}, compute_tokens=51),
        score_reference,
        {"grade": "score"},
    )
    report = "\n".join(lines)
    check(
        "score exact match passes with legend agreement",
        ok
        and "answer tolerance <= 1e-06: PASS" in report
        and "legend agreement: PASS" in report,
        report,
    )

    wrong_legend = dict(legend, **{"2": "A catastrophic error"})
    lines, ok = tool.compare_case(
        "score",
        worker_response(
            {"grade": score_answer(0.4, wrong_legend, 0.4)}, compute_tokens=51
        ),
        score_reference,
        {"grade": "score"},
    )
    check(
        "score legend mismatch fails",
        not ok
        and "legend agreement: FAIL" in "\n".join(lines)
        and "score legend mismatch" in "\n".join(lines),
        "\n".join(lines),
    )

    lines, ok = tool.compare_case(
        "score",
        worker_response({"grade": score_answer(0.5, legend, 0.4)}, compute_tokens=51),
        score_reference,
        {"grade": "score"},
    )
    check(
        "score expected-level drift fails the answer tolerance",
        not ok and "answer tolerance <= 1e-06: FAIL" in "\n".join(lines),
        "\n".join(lines),
    )

    # A score request answered in choice shape must fail even when the
    # candidate probabilities, decision and tokens all match.
    drifted_score = {
        "type": "choice",
        "choice": "1",
        "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1},
        "confidence": 0.4,
    }
    lines, ok = tool.compare_case(
        "score",
        worker_response({"grade": drifted_score}, compute_tokens=51),
        score_reference,
        {"grade": "score"},
    )
    check(
        "score answered as choice fails",
        not ok
        and "decision agreement: FAIL" in "\n".join(lines)
        and "answer type 'choice' != requested 'score'" in "\n".join(lines),
        "\n".join(lines),
    )

    # A synthetic score reference without a reference confidence falls back
    # to the worker's score concentration formula (0.4 here), not the choice
    # formula (0.55) that would fail the confidence tolerance.
    synthetic_score = {
        "keys": {"grade": ["0", "1", "2"]},
        "probabilities": {"grade": [0.7, 0.2, 0.1]},
        "answers": {"grade": {"type": "score", "score": 0.4, "legend": legend}},
        "logical_tokens": 17,
        "compute_tokens": 51,
    }
    lines, ok = tool.compare_case(
        "score",
        worker_response({"grade": score_answer(0.4, legend, 0.4)}, compute_tokens=51),
        synthetic_score,
        {"grade": "score"},
    )
    check(
        "synthetic score without confidence uses the score formula",
        ok and "confidence tolerance <= 1e-06: PASS" in "\n".join(lines),
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
