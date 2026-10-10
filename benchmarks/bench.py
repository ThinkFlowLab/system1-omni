#!/usr/bin/env python3
"""Replay frozen System One requests against an already running GPU server."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import httpx


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_cases(path):
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("manifest must contain nonempty, unique request IDs")
    for case in cases:
        request = case["request"]
        if "model" in request:
            raise ValueError(
                "set the backend model alias with --model, not in the manifest"
            )
        if "state" not in request or not request.get("questions"):
            raise ValueError("each request needs state and questions")
        if set(case["expected"]) != set(request["questions"]):
            raise ValueError("expected labels must cover every question")
        for qid, question in request["questions"].items():
            kind, target = question["type"], case["expected"][qid]
            if kind == "choice":
                criteria = question["criteria"]
                if (
                    not isinstance(criteria, dict)
                    or not criteria
                    or target not in criteria
                ):
                    raise ValueError(
                        "Choice requires a criteria map and a matching label"
                    )
            elif kind == "noul":
                if not isinstance(target, bool):
                    raise ValueError("Noul target must be boolean")
            elif kind == "score":
                criteria = question["criteria"]
                if not isinstance(criteria, list) or len(criteria) < 2:
                    raise ValueError("Score requires at least two ordered criteria")
                if not number(target) or not 0 <= target <= len(criteria) - 1:
                    raise ValueError("Score target must be a finite level index")
            else:
                raise ValueError(f"unsupported question type: {kind}")
    return cases


def number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


WIRE_DECIMALS = 4  # workers round probabilities on the wire to four decimal places
SCORE_WIRE_DECIMALS = 2


def sum_allowance(count):
    """How far `count` wire probabilities may miss a normalized sum: each rounded value
    carries up to half a decimal step, so the allowance grows with the label count. A
    flat 1e-4 rejected four-value distributions summing to 0.9999 at the float boundary
    (PR #40, first GPU run)."""
    return count * 0.5 * 10**-WIRE_DECIMALS + 1e-9


def score_allowance(count):
    """Bound independent rounding of the scalar and its weighted probabilities."""
    return (
        0.5 * 10**-SCORE_WIRE_DECIMALS
        + 0.5 * 10**-WIRE_DECIMALS * count * (count - 1) / 2
        + 1e-9
    )


def zero_score_fit(answer, probabilities):
    """Recognize Decider's isolated Score response when no level has any fit."""
    level_fit = answer.get("level_fit")
    return (
        answer.get("type") == "score"
        and all(p == 0 for p in probabilities.values())
        and isinstance(level_fit, dict)
        and set(level_fit) == set(probabilities)
        and all(v == 0 and number(v) for v in level_fit.values())
        and all(
            answer.get(key) == 0 and number(answer[key])
            for key in ("score", "confidence", "x_p_max", "fit_mass")
        )
        and answer.get("certainty") == 1
        and number(answer["certainty"])
    )


def read_answers(case, payload):
    """Validate wire results and normalize probabilities by label, never by order."""
    answers = payload["answers"]
    if set(answers) != set(case["request"]["questions"]):
        raise ValueError("response question IDs do not match the request")
    normalized = {}
    for qid, question in case["request"]["questions"].items():
        answer = answers[qid]
        kind = question["type"]
        if kind == "noul":
            value = answer["noul"]
            if not number(value):
                raise ValueError("Noul probability must be finite")
            probabilities = {"false": 1 - value, "true": value}
        else:
            labels = (
                list(question["criteria"])
                if kind == "choice"
                else [str(i) for i in range(len(question["criteria"]))]
            )
            probabilities = answer["probabilities"]
            if set(probabilities) != set(labels):
                raise ValueError("response probability labels do not match criteria")
            value = answer[kind]
        if any(not number(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("probabilities must be finite and in [0, 1]")
        total = math.fsum(probabilities.values())
        if abs(total - 1) > sum_allowance(len(probabilities)) and not (
            kind == "score" and total == 0 and zero_score_fit(answer, probabilities)
        ):
            raise ValueError("probabilities must sum to one")
        if kind == "choice":
            if value not in probabilities or probabilities[value] != max(
                probabilities.values()
            ):
                raise ValueError("choice must have maximal probability")
        elif not number(value):
            raise ValueError("decision value must be finite")
        if kind == "score":
            if not 0 <= value <= len(probabilities) - 1:
                raise ValueError("score must be in the level range")
            expected = math.fsum(int(k) * p for k, p in probabilities.items())
            if not math.isclose(
                value, expected, rel_tol=0, abs_tol=score_allowance(len(probabilities))
            ):
                raise ValueError("score must equal the expected level")
        normalized[qid] = {"value": value, "probabilities": probabilities}
    return normalized


def quality(cases, records):
    values = {"choice_accuracy": [], "noul_accuracy": [], "score_mae": []}
    for case, record in zip(cases, records):
        if record["error"]:
            continue
        for qid, target in case["expected"].items():
            value = record["answers"][qid]["value"]
            kind = case["request"]["questions"][qid]["type"]
            if kind == "score":
                values["score_mae"].append(abs(value - target))
            else:
                prediction = value >= 0.5 if kind == "noul" else value
                values[kind + "_accuracy"].append(float(prediction == target))
    return {
        name: {"value": statistics.mean(v) if v else None, "count": len(v)}
        for name, v in values.items()
    }


def summarize(cases, records, elapsed):
    successful = [r for r in records if r["error"] is None]
    latencies = sorted(r["latency_ms"] for r in successful)
    decisions = sum(len(r["answers"]) for r in successful)
    errors = {}
    for record in records:
        if record["error"]:
            key = record["error_kind"]
            errors[key] = errors.get(key, 0) + 1
    return {
        "requests": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "errors": errors,
        "successful_decisions": decisions,
        "wall_seconds": elapsed,
        "requests_per_second": len(successful) / elapsed,
        "decisions_per_second": decisions / elapsed,
        "successful_latency_p50_ms": statistics.median(latencies)
        if latencies
        else None,
        "successful_latency_p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1]
        if latencies
        else None,
        "quality_on_successful_requests": quality(cases, records),
    }


async def request_one(client, endpoint, case, model, timeout):
    started = time.perf_counter()
    record = {
        "id": case["id"],
        "status": None,
        "error": None,
        "error_kind": None,
        "response": None,
        "answers": {},
    }
    try:
        async with asyncio.timeout(timeout):
            response = await client.post(
                endpoint, json={**case["request"], "model": model}
            )
        record["status"] = response.status_code
        record["response"] = response.text
        response.raise_for_status()
        record["answers"] = read_answers(case, response.json())
    except httpx.HTTPStatusError as exc:
        record.update(error=str(exc), error_kind=f"http_{record['status']}")
    except (TimeoutError, httpx.TimeoutException) as exc:
        record.update(
            error=str(exc) or "request deadline exceeded", error_kind="timeout"
        )
    except httpx.RequestError as exc:
        record.update(error=str(exc), error_kind="transport")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        record.update(error=str(exc), error_kind="invalid_response")
    record["latency_ms"] = (time.perf_counter() - started) * 1000
    return record


async def replay(client, endpoint, cases, model, concurrency, timeout):
    pending = iter(enumerate(cases))
    records = [None] * len(cases)

    async def worker():
        for index, case in pending:
            records[index] = await request_one(client, endpoint, case, model, timeout)

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return records, time.perf_counter() - started


async def run(args, cases):
    metadata = json.loads(args.metadata.read_text())
    for name in (
        "gpu",
        "gpu_ids",
        "driver",
        "cuda",
        "precision",
        "model_revision",
        "runtime_revision",
        "cache_policy",
        "cuda_evidence",
        "reservation",
    ):
        if not metadata.get(name):
            raise ValueError(f"metadata requires {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "config.json",
        {
            "manifest_sha256": digest(args.manifest),
            "runner_sha256": digest(Path(__file__)),
            "metadata": metadata,
            "endpoint": args.endpoint,
            "model": args.model,
            "phase": args.phase,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "timeout_seconds": args.timeout,
            "python": platform.python_version(),
            "httpx": httpx.__version__,
        },
    )
    (args.output / "requests.jsonl").write_bytes(args.manifest.read_bytes())
    headers = {}
    if token := os.environ.get("OMNI_JEV_TEST_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(
        headers=headers,
        trust_env=False,
        timeout=None,
        limits=httpx.Limits(
            max_connections=args.concurrency, max_keepalive_connections=args.concurrency
        ),
    ) as client:
        # The first real inference validates readiness; preparation is not timed as load.
        warmup = []
        for i in range(args.warmup):
            warmup.append(
                await request_one(
                    client,
                    args.endpoint,
                    cases[i % len(cases)],
                    args.model,
                    args.timeout,
                )
            )
            if warmup[-1]["error"]:
                write_json(args.output / "warmup.json", warmup)
                raise ValueError("readiness/warmup failed; see warmup.json")
        write_json(args.output / "warmup.json", warmup)
        records, elapsed = await replay(
            client, args.endpoint, cases, args.model, args.concurrency, args.timeout
        )
    (args.output / "responses.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
    )
    summary = summarize(cases, records, elapsed)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return int(summary["failed_requests"] > 0)


def compare(args):
    configs = [
        json.loads((p / "config.json").read_text())
        for p in (args.reference, args.candidate)
    ]
    for path, config in zip((args.reference, args.candidate), configs):
        if digest(path / "requests.jsonl") != config["manifest_sha256"]:
            raise ValueError("saved manifest checksum does not match config")
    if configs[0]["manifest_sha256"] != configs[1]["manifest_sha256"]:
        raise ValueError("cannot compare different request manifests")
    runs = [
        [json.loads(line) for line in (p / "responses.jsonl").read_text().splitlines()]
        for p in (args.reference, args.candidate)
    ]
    if [r["id"] for r in runs[0]] != [r["id"] for r in runs[1]]:
        raise ValueError("response IDs differ")
    drift, flips, score_drift, failed, compared = 0.0, 0, 0.0, 0, 0
    cases = load_cases(args.reference / "requests.jsonl")
    if [c["id"] for c in cases] != [r["id"] for r in runs[0]]:
        raise ValueError("response IDs do not cover the manifest")
    for case, ref, candidate in zip(cases, *runs):
        if ref["error"] or candidate["error"]:
            failed += 1
            continue
        for qid, question in case["request"]["questions"].items():
            a, b = ref["answers"][qid], candidate["answers"][qid]
            drift = max(
                drift,
                max(
                    abs(p - b["probabilities"][k])
                    for k, p in a["probabilities"].items()
                ),
            )
            kind = question["type"]
            if kind == "score":
                score_drift = max(score_drift, abs(a["value"] - b["value"]))
            elif kind == "noul":
                flips += (a["value"] >= 0.5) != (b["value"] >= 0.5)
            else:
                flips += a["value"] != b["value"]
            compared += 1
    passed = (
        failed == 0
        and compared > 0
        and drift <= args.max_probability_drift
        and score_drift <= args.max_score_drift
        and flips <= args.max_flips
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "compared_decisions": compared,
                "failed_request_pairs": failed,
                "decision_flips": flips,
                "max_probability_drift": drift,
                "max_score_drift": score_drift,
            },
            indent=2,
        )
    )
    return int(not passed)


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    runner = commands.add_parser("run")
    runner.add_argument("manifest", type=Path)
    runner.add_argument("--endpoint", required=True, help="full /v1/systemone URL")
    runner.add_argument(
        "--model", required=True, help="backend's alias for the pinned checkpoint"
    )
    runner.add_argument("--metadata", type=Path, required=True)
    runner.add_argument(
        "--output", type=Path, required=True, help="new directory; never overwritten"
    )
    runner.add_argument("--phase", choices=("feasibility", "measured"), required=True)
    runner.add_argument("--concurrency", type=positive, default=1)
    runner.add_argument("--warmup", type=positive, default=3)
    runner.add_argument("--timeout", type=positive, default=60)
    comparison = commands.add_parser("compare")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--max-probability-drift", type=float, required=True)
    comparison.add_argument("--max-score-drift", type=float, required=True)
    comparison.add_argument("--max-flips", type=int, required=True)
    args = parser.parse_args()
    try:
        if args.command == "compare":
            tolerances = (
                args.max_probability_drift,
                args.max_score_drift,
                args.max_flips,
            )
            if any(not number(v) or v < 0 for v in tolerances):
                raise ValueError("comparison tolerances must be finite and nonnegative")
            return compare(args)
        cases = load_cases(args.manifest)
        if args.command == "validate":
            print(f"Validated {len(cases)} requests; SHA256 {digest(args.manifest)}")
            return 0
        return asyncio.run(run(args, cases))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
