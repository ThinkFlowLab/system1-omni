#!/usr/bin/env python3
"""Compare the Valen worker pipeline against the pinned reference path.

Runs each request through two paths with the same pinned artifacts:

- reference: a Valen-native record (a plain-string text state, or a local
  image with a SHA-256 asset) compiled by the pinned ``Compiler`` and decoded
  with a plain softmax plus Valen's confidence formula;
- worker: the production ``frontend.valen.decide`` pipeline (protocol,
  preprocessing, executor, postprocessing).

Both paths share one loaded model instance, so the comparison isolates input
construction, candidate ordering, decision-head readout and probability
normalization instead of checkpoint loading. The declared criteria are exact
decision agreement, equal token accounting and a probability tolerance of
1e-6 (``--tolerance``). Requires the pinned Valen source, the Preview
checkpoint, the Qwen3.5-2B base and a working torch device; see
recipe/valen/reference.md.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
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
    parser.add_argument("--case", choices=("image", "text", "both"), default="both")
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


def reference_questions(body):
    """Rebuild the Valen question mapping straight from the fixture body."""

    return {
        name: {
            "type": "choice",
            "instructions": question.get("instructions") or "",
            "criteria": {
                key: (label if label is not None else key)
                for key, label in question["criteria"].items()
            },
        }
        for name, question in body["questions"].items()
    }


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


def compare_case(name, worker_response, reference, tolerance=PROBABILITY_TOLERANCE):
    """Compare one worker response against one reference result.

    Returns (report lines, passed).
    """

    lines = [f"case: {name}", "reference model: Valen", f"worker model: {worker_response['model']}"]
    problems = []
    decision_exact = True
    max_prob_delta = 0.0
    max_conf_delta = 0.0
    for qid, ref_keys in reference["keys"].items():
        answer = worker_response["answers"][qid]
        worker_keys = list(answer["probabilities"])
        if worker_keys != list(ref_keys):
            problems.append(f"{qid}: candidate order {worker_keys} != {list(ref_keys)}")
            decision_exact = False
            continue
        ref_probs = reference["probabilities"][qid]
        for key, worker_p, ref_p in zip(ref_keys, answer["probabilities"].values(), ref_probs):
            max_prob_delta = max(max_prob_delta, abs(worker_p - ref_p))
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
    lines.append(f"confidence delta: {max_conf_delta:.10e}")
    lines.append(
        "decision agreement: "
        + ("PASS" if decision_exact else "FAIL: " + "; ".join(problems))
    )
    prob_ok = max_prob_delta <= tolerance
    lines.append(
        f"probability tolerance <= {tolerance:g}: " + ("PASS" if prob_ok else "FAIL")
    )
    lines.append("token accounting: " + ("PASS" if tokens_exact else "FAIL"))
    return lines, decision_exact and prob_ok and tokens_exact


def run_reference(model, base_processor, record, media_root, max_length, media_kwargs):
    """Compile a Valen-native record and decode it with a plain softmax."""

    import torch

    from valen.data.compiler import Compiler

    compiler = Compiler(
        base_processor,
        media_root=media_root,
        max_length=max_length,
        media_kwargs=media_kwargs,
    )
    state = compiler.compile(record)
    keys = {}
    probabilities = {}
    with torch.no_grad():
        for question in state.questions:
            logits = model(question).float().cpu()
            probabilities[question.qid] = torch.softmax(logits, dim=-1).tolist()
            keys[question.qid] = list(question.keys)
    return {
        "keys": keys,
        "probabilities": probabilities,
        "logical_tokens": state.logical_tokens,
        "compute_tokens": state.compute_tokens,
    }


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from frontend.valen import _verify_source_revision, decide
    from models.valen.engine import ValenExecutor
    from models.valen.preprocess import ValenProcessor

    source = Path(args.valen_source).resolve()
    _verify_source_revision(source)
    sys.path.insert(0, str(source))

    checkpoint = Path(args.checkpoint).resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    max_length = int(config["max_length"])
    media_kwargs = config.get("media_kwargs") or {}
    processor = ValenProcessor(Path(args.base), max_length, config.get("media_kwargs"))
    executor = ValenExecutor(checkpoint, Path(args.base), args.device, args.dtype)

    cases = []
    if args.case in ("image", "both"):
        cases.append(("image", json.loads(Path(args.fixture).read_text(encoding="utf-8"))))
    if args.case in ("text", "both"):
        if args.text_body:
            cases.append(("text", json.loads(Path(args.text_body).read_text(encoding="utf-8"))))
        else:
            cases.append(("text", DEFAULT_TEXT_BODY))

    failed = 0
    try:
        for name, body in cases:
            worker_response = decide(json.dumps(body).encode("utf-8"), processor, executor)
            questions = reference_questions(body)
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
            lines, ok = compare_case(name, worker_response, reference, args.tolerance)
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
