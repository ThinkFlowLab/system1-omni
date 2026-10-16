#!/usr/bin/env python3
"""Compare the Valen worker pipeline against the pinned reference path.

Runs each request through two paths with the same pinned artifacts:

- reference: a Valen-native record (a plain-string text state, or a local
  image with a SHA-256 asset) compiled by the pinned ``Compiler``; ``choice``
  cases are decoded with a plain softmax plus Valen's confidence formula,
  while ``noul`` and ``score`` cases are decoded by the pinned reference
  ``valen.evaluation.inference.answer`` module itself;
- worker: the production ``frontend.valen.decide`` pipeline (protocol,
  preprocessing, executor, postprocessing).

Both paths share one loaded model instance, so the comparison isolates input
construction, candidate ordering, decision-head readout and probability
normalization instead of checkpoint loading. The declared criteria are exact
discrete agreement (candidate order, the requested answer type, the chosen
key, the score legend), equal token accounting and a tolerance of 1e-6
(``--tolerance``) on probabilities, confidences and the scalar answer fields
(``noul`` P(true), ``score`` expected level), which the two softmax
implementations never match bit-exactly. Requires the pinned Valen source,
the Preview checkpoint, the Qwen3.5-2B base and a working torch device; see
recipe/valen/reference.md.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
PROBABILITY_TOLERANCE = 1e-6

DEFAULT_TEXT_BODY = {
    "model": "valen-preview-0923",
    "state": "The card was charged twice for one order.",
    "questions": {
        "refund": {
            "type": "choice",
            "instructions": "Decide the refund action.",
            "criteria": {"refund": "Refund the duplicate charge", "wait": "Wait for review"},
        }
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valen-source", required=True, help="pinned Valen source checkout")
    parser.add_argument("--checkpoint", required=True, help="Valen-Preview-0923 directory")
    parser.add_argument("--base", required=True, help="Qwen3.5-2B base directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--case",
        choices=("image", "text", "noul", "score", "both", "all"),
        default="both",
    )
    parser.add_argument(
        "--fixture",
        default=str(HERE / "example-request.json"),
        help="image-case request body (default: the recipe fixture)",
    )
    parser.add_argument(
        "--text-body",
        default=None,
        help="text-case request body JSON file (default: the recipe text example)",
    )
    parser.add_argument(
        "--noul-body",
        default=str(HERE / "example-noul-request.json"),
        help="noul-case request body JSON file (default: the recipe noul example)",
    )
    parser.add_argument(
        "--score-body",
        default=str(HERE / "example-score-request.json"),
        help="score-case request body JSON file (default: the recipe score example)",
    )
    parser.add_argument("--tolerance", type=float, default=PROBABILITY_TOLERANCE)
    return parser.parse_args(argv)


def decode_image_data_url(url):
    """Split a PNG/JPEG data URL into (bytes, format)."""

    if not isinstance(url, str):
        raise ValueError("state.image must be a PNG/JPEG base64 data URL")
    prefix, separator, encoded = url.partition(",")
    expected = {"data:image/png;base64": "PNG", "data:image/jpeg;base64": "JPEG"}
    if not separator or prefix not in expected:
        raise ValueError("state.image must be a PNG/JPEG base64 data URL")
    return base64.b64decode(encoded, validate=True), expected[prefix]


def reference_questions(body, special_tokens=None):
    """Rebuild the Valen question mapping straight from the fixture body.

    The questions go through the worker's own ``parse_request`` +
    ``question_record`` path, so the reference record cannot drift from the
    one the worker compiles: custom ``noul`` wording becomes the same
    ``true``/``false`` choice record, and unspecified sides get the same
    fixed reference wording.
    """

    from models.valen.preprocess import question_record
    from models.valen.protocol import DEFAULT_SPECIAL_TOKENS, parse_request

    request = parse_request(
        body, special_tokens=special_tokens or DEFAULT_SPECIAL_TOKENS
    )
    return {question.name: question_record(question) for question in request.questions}


def expected_answer_types(body, special_tokens=None):
    """The requested public answer type per question, from the wire body.

    The worker's public ``type`` field mirrors the request's question type,
    so parity dispatches on the requested type: an answer that drifts to
    another shape (for example a ``noul`` answered as a plain choice) fails
    the comparison instead of silently skipping the noul/score checks.
    """

    from models.valen.protocol import DEFAULT_SPECIAL_TOKENS, parse_request

    request = parse_request(
        body, special_tokens=special_tokens or DEFAULT_SPECIAL_TOKENS
    )
    return {question.name: question.kind for question in request.questions}


def reference_image_record(image_bytes, image_format, questions, media_dir):
    """A Valen-native image record with its own materialized media copy."""

    media_dir = Path(media_dir)
    media_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if image_format == "PNG" else ".jpg"
    media_path = media_dir / f"reference{suffix}"
    media_path.write_bytes(image_bytes)
    return {
        "request": {
            "state": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": media_path.name}}
                        ],
                    }
                ]
            },
            "questions": questions,
        },
        "assets": [
            {"path": media_path.name, "sha256": hashlib.sha256(image_bytes).hexdigest()}
        ],
    }


def reference_text_record(text, questions):
    """A Valen-native text record: the compiler's plain-string state form."""

    return {"request": {"state": text, "questions": questions}}


def reference_confidence(probabilities):
    """Valen's original choice confidence formula."""

    count = len(probabilities)
    if count == 1:
        return 1.0
    return max(0.0, (max(probabilities) - 1 / count) / (1 - 1 / count))


def reference_score_confidence(probabilities):
    """Mirror the worker's score concentration formula (see postprocess.py)."""

    mode = max(range(len(probabilities)), key=probabilities.__getitem__)
    count = len(probabilities)
    distance = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
    uniform_mad = sum(abs(i - (count - 1) / 2) for i in range(count)) / count
    return max(0.0, 1.0 - distance / uniform_mad)


def compare_case(
    name, worker_response, reference, expected_types, tolerance=PROBABILITY_TOLERANCE
):
    """Compare one worker response against one reference result.

    ``expected_types`` maps each question id to its requested public answer
    type; every answer must keep that type before any per-type comparison
    runs. ``choice`` answers must agree on the chosen key; ``noul`` answers
    on P(true); ``score`` answers on the expected level and the level legend.
    All scalar comparisons — probabilities, confidences and the scalar answer
    fields — use the declared tolerance because the worker and the pinned
    reference normalize probabilities through different softmax
    implementations. Returns (report lines, passed).
    """

    lines = [f"case: {name}", "reference model: Valen", f"worker model: {worker_response['model']}"]
    problems = []
    legend_problems = []
    decision_exact = True
    legend_exact = True
    max_prob_delta = 0.0
    max_conf_delta = 0.0
    max_answer_delta = 0.0
    scalar_answers = False
    for qid, ref_keys in reference["keys"].items():
        answer = worker_response["answers"][qid]
        expected_type = expected_types[qid]
        if answer["type"] != expected_type:
            problems.append(
                f"{qid}: answer type {answer['type']!r} != requested {expected_type!r}"
            )
            decision_exact = False
            continue
        worker_keys = list(answer["probabilities"])
        if worker_keys != list(ref_keys):
            problems.append(f"{qid}: candidate order {worker_keys} != {list(ref_keys)}")
            decision_exact = False
            continue
        ref_probs = reference["probabilities"][qid]
        for key, worker_p, ref_p in zip(ref_keys, answer["probabilities"].values(), ref_probs):
            max_prob_delta = max(max_prob_delta, abs(worker_p - ref_p))
        # The pinned reference decode when the harness ran with the pinned
        # source; synthetic references may omit it and fall back to the
        # shared probabilities and formulas.
        ref_answer = reference.get("answers", {}).get(qid)
        if expected_type == "noul":
            scalar_answers = True
            if ref_answer is not None and "noul" in ref_answer:
                ref_p_true = ref_answer["noul"]
            elif ref_answer is not None:
                # Custom-wording noul compiles as a choice record, so the
                # pinned decoder answers it in choice shape.
                ref_p_true = ref_answer["probabilities"]["true"]
            else:
                ref_p_true = ref_probs[ref_keys.index("true")]
            max_answer_delta = max(max_answer_delta, abs(answer["noul"] - ref_p_true))
            if ref_answer is not None and "confidence" in ref_answer:
                ref_confidence = ref_answer["confidence"]
            else:
                # The reference noul answer carries no confidence; the worker
                # exposes Valen's choice formula evaluated on the pair.
                ref_confidence = reference_confidence(ref_probs)
            max_conf_delta = max(max_conf_delta, abs(answer["confidence"] - ref_confidence))
        elif expected_type == "score":
            scalar_answers = True
            ref_score = (
                ref_answer["score"]
                if ref_answer is not None
                else sum(index * p for index, p in enumerate(ref_probs))
            )
            max_answer_delta = max(max_answer_delta, abs(answer["score"] - ref_score))
            if ref_answer is not None and "legend" in ref_answer:
                if answer["legend"] != ref_answer["legend"]:
                    legend_problems.append(f"{qid}: score legend mismatch")
                    legend_exact = False
            if ref_answer is not None and "confidence" in ref_answer:
                ref_confidence = ref_answer["confidence"]
            else:
                # The reference score answer carries no confidence; the worker
                # exposes Valen's score concentration formula.
                ref_confidence = reference_score_confidence(ref_probs)
            max_conf_delta = max(max_conf_delta, abs(answer["confidence"] - ref_confidence))
        else:
            decision = ref_keys[max(range(len(ref_probs)), key=ref_probs.__getitem__)]
            if answer["choice"] != decision:
                problems.append(f"{qid}: worker chose {answer['choice']!r}, reference {decision!r}")
                decision_exact = False
            ref_confidence = reference_confidence(ref_probs)
            max_conf_delta = max(max_conf_delta, abs(answer["confidence"] - ref_confidence))

    usage = worker_response["usage"]
    compute = worker_response["internal_usage"]["compute_tokens"]
    tokens_exact = (
        usage["input_tokens"] == reference["logical_tokens"]
        and usage["output_tokens"] == 0
        and compute == reference["compute_tokens"]
    )
    lines.append(
        f"input_tokens: {usage['input_tokens']}"
        f" {'==' if usage['input_tokens'] == reference['logical_tokens'] else '!='}"
        f" {reference['logical_tokens']}"
    )
    lines.append(f"output_tokens: {usage['output_tokens']} == 0")
    lines.append(
        f"compute_tokens: {compute}"
        f" {'==' if compute == reference['compute_tokens'] else '!='}"
        f" {reference['compute_tokens']}"
    )
    lines.append(f"max probability delta: {max_prob_delta:.10e}")
    if scalar_answers:
        lines.append(f"max answer delta: {max_answer_delta:.10e}")
    lines.append(f"confidence delta: {max_conf_delta:.10e}")
    lines.append(
        "decision agreement: "
        + ("PASS" if decision_exact else "FAIL: " + "; ".join(problems))
    )
    prob_ok = max_prob_delta <= tolerance
    confidence_ok = max_conf_delta <= tolerance
    lines.append(
        f"probability tolerance <= {tolerance:g}: " + ("PASS" if prob_ok else "FAIL")
    )
    lines.append(
        f"confidence tolerance <= {tolerance:g}: "
        + ("PASS" if confidence_ok else "FAIL")
    )
    if scalar_answers:
        answer_ok = max_answer_delta <= tolerance
        lines.append(
            f"answer tolerance <= {tolerance:g}: " + ("PASS" if answer_ok else "FAIL")
        )
        if any("legend" in answer for answer in worker_response["answers"].values()):
            lines.append(
                "legend agreement: "
                + ("PASS" if legend_exact else "FAIL: " + "; ".join(legend_problems))
            )
    lines.append("token accounting: " + ("PASS" if tokens_exact else "FAIL"))
    passed = (
        decision_exact and legend_exact and prob_ok and confidence_ok and tokens_exact
    )
    if scalar_answers:
        passed = passed and max_answer_delta <= tolerance
    return lines, passed


def run_reference(model, base_processor, record, media_root, max_length, media_kwargs):
    """Compile a Valen-native record and decode it with the pinned modules."""

    import torch

    from valen.data.compiler import Compiler
    from valen.evaluation.inference import answer as reference_answer

    compiler = Compiler(
        base_processor,
        media_root=media_root,
        max_length=max_length,
        media_kwargs=media_kwargs,
    )
    state = compiler.compile(record)
    keys = {}
    probabilities = {}
    answers = {}
    with torch.no_grad():
        for question in state.questions:
            logits = model(question).float().cpu()
            probabilities[question.qid] = torch.softmax(logits, dim=-1).tolist()
            keys[question.qid] = list(question.keys)
            # The pinned decoder is the answer-field authority, so every
            # question is decoded by it as well as by the plain softmax.
            answers[question.qid] = reference_answer(question, logits)
    return {
        "keys": keys,
        "probabilities": probabilities,
        "answers": answers,
        "logical_tokens": state.logical_tokens,
        "compute_tokens": state.compute_tokens,
    }


def _verify_clean_checkout(source):
    """Reject a dirty pinned checkout before it can produce evidence.

    Every evidence-producing harness in this recipe imports the pinned
    valen package from this tree, so a local modification would make the
    records certify agreement with code nobody pinned; a clean tracked
    tree with no importable strays is part of what a record claims.
    """

    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    problems = []
    for line in status.stdout.splitlines():
        entry = line[3:]
        if line.startswith("??"):
            # Untracked non-code cannot change what gets imported; a stray
            # .py on the package path can.
            if entry.endswith(".py"):
                problems.append(f"untracked python file {entry}")
        else:
            problems.append(f"locally changed {entry}")
    if problems:
        raise SystemExit(
            "--valen-source must be a clean checkout of the pinned revision "
            "(commit or restore):\n  " + "\n  ".join(problems)
        )


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from frontend.valen import _verify_source_revision, decide
    from models.valen.engine import ValenExecutor
    from models.valen.preprocess import ValenProcessor

    source = Path(args.valen_source).resolve()
    _verify_source_revision(source)
    _verify_clean_checkout(source)
    sys.path.insert(0, str(source))

    checkpoint = Path(args.checkpoint).resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    max_length = int(config["max_length"])
    media_kwargs = config.get("media_kwargs") or {}
    processor = ValenProcessor(Path(args.base), max_length, config.get("media_kwargs"))
    executor = ValenExecutor(checkpoint, Path(args.base), args.device, args.dtype)

    cases = []
    if args.case in ("image", "both", "all"):
        cases.append(("image", json.loads(Path(args.fixture).read_text(encoding="utf-8"))))
    if args.case in ("text", "both", "all"):
        if args.text_body:
            cases.append(("text", json.loads(Path(args.text_body).read_text(encoding="utf-8"))))
        else:
            cases.append(("text", DEFAULT_TEXT_BODY))
    if args.case in ("noul", "all"):
        cases.append(("noul", json.loads(Path(args.noul_body).read_text(encoding="utf-8"))))
    if args.case in ("score", "all"):
        cases.append(("score", json.loads(Path(args.score_body).read_text(encoding="utf-8"))))

    failed = 0
    try:
        for name, body in cases:
            worker_response = decide(json.dumps(body).encode("utf-8"), processor, executor)
            questions = reference_questions(body, processor.special_tokens)
            expected = expected_answer_types(body, processor.special_tokens)
            with TemporaryDirectory(prefix="valen-parity-") as directory:
                media_root = Path(directory)
                if name == "image":
                    image_bytes, image_format = decode_image_data_url(body["state"]["image"])
                    record = reference_image_record(image_bytes, image_format, questions, media_root)
                else:
                    record = reference_text_record(body["state"], questions)
                reference = run_reference(
                    executor.model, processor.processor, record, media_root, max_length, media_kwargs
                )
            lines, ok = compare_case(
                name, worker_response, reference, expected, args.tolerance
            )
            print("\n".join(lines))
            print()
            if not ok:
                failed += 1
    finally:
        executor.close()
    total = len(cases)
    print("compare_reference: ok" if not failed else f"compare_reference: {failed} of {total} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
