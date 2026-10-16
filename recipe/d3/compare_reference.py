#!/usr/bin/env python3
"""Compare the d3 worker pipeline against the pinned upstream server logic.

Runs each request through two paths that share one loaded pinned model, so
the comparison isolates this repository's admission and orchestration from
the checkpoint's own code:

- reference: the ``system_one`` body of the upstream ``d3_server.py``
  (decode_images/decode_videos with their empty short-circuits and
  capability gates, then the decide body: prepare with the media-aware
  branches, reject over-limit and invalid questions with 422, run,
  respond), driven directly on the loaded runtime;
- worker: the production ``frontend.d3.decide`` pipeline (protocol envelope,
  decode stage, prepared-error admission, run, respond, response finishing).

Both paths execute the same verified ``d3_runtime``, so the criteria are
exact decision agreement (including embedded ``invalid_model_output``
shapes and agreed 422 rejections, media decode errors among them), equal
token accounting, and a probability tolerance of 1e-6
(``--tolerance``). ``--export`` additionally freezes the oracle fixture —
token ids, probabilities and answers per question — for the future native
worker, and writes it only after a passing, non-rejected comparison.
Requires the pinned d3 package and a working device; see
recipe/d3/reference.md. GPU use is opt-in: run this tool by hand only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBABILITY_TOLERANCE = 1e-6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="pinned d3 package directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--verify", default="fast", choices=("fast", "full", "none"))
    parser.add_argument(
        "--fixture",
        default=str(HERE / "example-request.json"),
        help="request body (default: the recipe fixture)",
    )
    parser.add_argument("--tolerance", type=float, default=PROBABILITY_TOLERANCE)
    parser.add_argument(
        "--export",
        default=None,
        help="write the frozen oracle fixture (token ids, probabilities, answers); "
        "only a passing, non-rejected comparison is written",
    )
    return parser.parse_args(argv)


def _decode_images(model, images):
    """Upstream d3_server.py decode_images, verbatim."""

    if not isinstance(images, list):
        raise ValueError("images must be a list of base64 data URLs")
    if not images:
        return []
    if getattr(model, "image_unavailable", "unknown") is not None:
        raise ValueError("This model reads text only; images are not supported.")
    return model.load_images(images, strict=True)


def _decode_videos(model, videos):
    """Upstream d3_server.py decode_videos, verbatim."""

    if not isinstance(videos, list):
        raise ValueError("videos must be a list of base64 data URLs")
    if not videos:
        return []
    if getattr(model, "video_unavailable", "unknown") is not None:
        raise ValueError("This model does not read videos.")
    return model.load_videos(videos, strict=True)


def decode_request_media(model, body):
    """The media half of upstream system_one: images then videos."""

    images = _decode_images(model, body["images"]) if body.get("images") is not None else []
    videos = _decode_videos(model, body["videos"]) if body.get("videos") is not None else []
    return images, videos


def reference_decide(model, body):
    """The upstream d3_server.py system_one body, without FastAPI.

    Returns the 200 response dict, or a ("422", detail) tuple when the
    upstream server would have rejected the request before execution —
    including media rejections from the decode stage, which upstream maps
    to a plain-detail 422.
    """

    try:
        images, videos = decode_request_media(model, body)
        if videos:
            prepared = model.prepare(body.get("state"), body.get("questions"), images, videos)
        elif images:
            prepared = model.prepare(body.get("state"), body.get("questions"), images)
        else:
            prepared = model.prepare(body.get("state"), body.get("questions"))
    except ValueError as exc:
        return ("422", str(exc))
    over = [e for e in prepared.errors.values() if e["error"] == "max_length_exceeded"]
    if over:
        return ("422", over[0]["message"])
    invalid = {k: e["message"] for k, e in prepared.errors.items()}
    if invalid:
        return ("422", {"invalid_questions": invalid})
    probabilities, tokens = model.run(prepared)
    return model.respond(prepared, probabilities, tokens)


def _numeric_delta(answer, reference, problems, deltas):
    """Compare one answer pair; returns exact-match problems and max deltas."""

    if set(answer) != set(reference):
        problems.append(f"fields {sorted(answer)} != {sorted(reference)}")
        return
    if answer.get("type") != reference.get("type"):
        problems.append(f"type {answer.get('type')!r} != {reference.get('type')!r}")
    if "error" in reference:
        if answer.get("error") != reference.get("error") or answer.get(
            "message"
        ) != reference.get("message"):
            problems.append("embedded error shape differs")
        return
    if "choice" in reference and answer.get("choice") != reference["choice"]:
        problems.append(f"choice {answer.get('choice')!r} != {reference['choice']!r}")
    if "noul" in reference:
        deltas.append(abs(float(answer["noul"]) - float(reference["noul"])))
    if "score" in reference:
        deltas.append(abs(float(answer["score"]) - float(reference["score"])))
    if "legend" in reference and answer.get("legend") != reference.get("legend"):
        problems.append("legend differs")
    if "probabilities" in reference:
        if list(answer.get("probabilities") or ()) != list(reference["probabilities"]):
            problems.append("probability key order differs")
        else:
            deltas.extend(
                abs(float(a) - float(r))
                for a, r in zip(
                    answer["probabilities"].values(), reference["probabilities"].values()
                )
            )
    if "confidence" in reference:
        deltas.append(abs(float(answer["confidence"]) - float(reference["confidence"])))


def compare_case(name, worker_response, reference, tolerance=PROBABILITY_TOLERANCE):
    """Compare one worker response against one reference result.

    Returns (report lines, passed).
    """

    lines = [f"case: {name}", "reference path: d3_server system_one()", "worker model: d3"]
    problems = []
    deltas: list[float] = []
    if isinstance(reference, tuple):
        rejection = worker_response.get("worker rejected the request")
        if not isinstance(rejection, dict):
            problems.append(f"reference rejected the request: {reference[1]}")
        else:
            detail = reference[1]
            body = rejection.get("body")
            if isinstance(detail, dict):
                agreed = rejection.get("status") == 422 and body == detail
            else:
                # The worker deliberately wraps prepare-stage rejections in a
                # fixed prefix (src/frontend/d3.py decide); upstream system_one
                # surfaces the raw runtime message. Both forms agree here.
                agreed = rejection.get("status") == 422 and body in (
                    {"detail": detail},
                    {"detail": "request rejected by the pinned runtime: " + detail},
                )
            lines.append(f"worker rejection: {rejection.get('status')} {body}")
            lines.append("rejection agreement: " + ("PASS" if agreed else "FAIL"))
            return lines, bool(agreed)
    elif "worker rejected the request" in worker_response:
        problems.append(str(worker_response["worker rejected the request"]))
    else:
        if list(worker_response["answers"]) != list(reference["answers"]):
            problems.append("answer order differs")
        for qid, ref_answer in reference["answers"].items():
            _numeric_delta(worker_response["answers"][qid], ref_answer, problems, deltas)
        usage, ref_usage = worker_response["usage"], reference["usage"]
        lines.append(
            f"input_tokens: {usage.get('input_tokens')}"
            f" {'==' if usage.get('input_tokens') == ref_usage.get('input_tokens') else '!='}"
            f" {ref_usage.get('input_tokens')}"
        )
        lines.append(f"output_tokens: {usage.get('output_tokens')} == 0")
    max_delta = max(deltas, default=0.0)
    lines.append(f"max numeric delta: {max_delta:.10e}")
    lines.append("decision agreement: " + ("PASS" if not problems else "FAIL: " + "; ".join(problems[:4])))
    tolerance_ok = max_delta <= tolerance
    lines.append(f"numeric tolerance <= {tolerance:g}: " + ("PASS" if tolerance_ok else "FAIL"))
    tokens_exact = (
        not isinstance(reference, tuple)
        and worker_response["usage"] == reference["usage"]
        and worker_response["usage"].get("output_tokens") == 0
    )
    lines.append("token accounting: " + ("PASS" if tokens_exact else "FAIL"))
    return lines, not problems and tolerance_ok and tokens_exact


def export_allowed(ok, worker_response, reference):
    """The oracle freezes a served 200 response, never a rejection.

    A comparison can pass on an agreed 422 rejection, but that path has no
    answers or token ids to freeze — the worker side is a synthetic empty
    response with ``-1`` usage — so only a passing non-rejected case may
    overwrite the fixture.
    """

    return (
        ok
        and not isinstance(reference, tuple)
        and "worker rejected the request" not in worker_response
    )


def export_oracle(body, prepared, probabilities, worker_response, path):
    """Freeze the oracle fixture for the future native worker."""

    oracle = {
        "request": body,
        "token_ids": {q: list(prepared.sequences[q]) for q in sorted(prepared.sequences)},
        "probabilities": {q: list(values) for q, values in sorted(probabilities.items())},
        "answers": worker_response["answers"],
        "usage": worker_response["usage"],
    }
    Path(path).write_text(json.dumps(oracle, indent=1) + "\n", encoding="utf-8")
    return oracle


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from frontend.d3 import decide
    from models.d3.engine import D3Executor

    executor = D3Executor(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        verify=args.verify,
    )
    body = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    failed = 0
    try:
        raw = json.dumps(body).encode("utf-8")
        try:
            worker_response = decide(raw, executor)
        except Exception as exc:  # noqa: BLE001 - reported through the comparison
            status = getattr(exc, "status", 500)
            message = str(exc)
            rejection_body = getattr(exc, "body", None)
            if rejection_body is None:
                rejection_body = {"detail": message}
            worker_response = {
                "answers": {},
                "usage": {"input_tokens": -1, "output_tokens": -1},
                "worker rejected the request": {"status": status, "body": rejection_body},
            }
        reference = reference_decide(executor.model, body)
        lines, ok = compare_case(Path(args.fixture).name, worker_response, reference, args.tolerance)
        print("\n".join(lines))
        if args.export and export_allowed(ok, worker_response, reference):
            images, videos = decode_request_media(executor.model, body)
            prepared = executor.prepare(
                body.get("state"), dict(body["questions"]), images, videos
            )
            probabilities, _ = executor.run(prepared)
            export_oracle(body, prepared, probabilities, worker_response, args.export)
            print(f"oracle exported: {args.export}")
        elif args.export:
            print("oracle export skipped: no passing 200-response comparison to freeze")
        print()
        if not ok:
            failed += 1
    finally:
        executor.close()
    print("compare_reference: ok" if not failed else "compare_reference: FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
