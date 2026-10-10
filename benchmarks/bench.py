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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def response_digest(text):
    if text is None:
        return None
    if not isinstance(text, str):
        raise ValueError("raw response must be text")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def sum_allowance(count):
    """How far `count` wire probabilities may miss a normalized sum: each rounded value
    carries up to half a decimal step, so the allowance grows with the label count. A
    flat 1e-4 rejected four-value distributions summing to 0.9999 at the float boundary
    (PR #40, first GPU run)."""
    return count * 0.5 * 10**-WIRE_DECIMALS + 1e-9


def read_answers(case, payload):
    """Validate wire results and normalize probabilities by label, never by order."""
    answers = payload["answers"]
    if not isinstance(answers, dict) or set(answers) != set(
        case["request"]["questions"]
    ):
        raise ValueError("response question IDs do not match the request")
    normalized = {}
    for qid, question in case["request"]["questions"].items():
        answer = answers[qid]
        kind = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != kind:
            raise ValueError(
                "response answer must be an object with the requested type"
            )
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
        if abs(total - 1) > sum_allowance(len(probabilities)):
            raise ValueError("probabilities must sum to one")
        if kind == "choice":
            if value not in probabilities or probabilities[value] != max(
                probabilities.values()
            ):
                raise ValueError("choice must have maximal probability")
        elif not number(value):
            raise ValueError("decision value must be finite")
        if kind == "score":
            expected = sum(int(k) * p for k, p in probabilities.items())
            if not math.isclose(value, expected, abs_tol=1e-4):
                raise ValueError("score must equal the expected level")
        normalized[qid] = {"value": value, "probabilities": probabilities}
    return normalized


def quality(cases, records):
    values = {"choice_accuracy": [], "noul_accuracy": [], "score_mae": []}
    by_id = {r["id"]: r for r in records}
    for case in cases:
        record = by_id.get(case["id"])
        if record is None or record["error"] is not None:
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


def summarize(cases, records, elapsed, attempted_ids=None):
    attempted = set(
        attempted_ids if attempted_ids is not None else (r["id"] for r in records)
    )
    completed = [r for r in records if r.get("state", "completed") == "completed"]
    successful = [r for r in completed if r["error"] is None]
    failed = [r for r in completed if r["error"] is not None]
    completed_ids = {r["id"] for r in completed}
    incomplete = attempted - completed_ids
    latencies = sorted(r["latency_ms"] for r in successful)
    failed_latencies = sorted(r["latency_ms"] for r in failed)
    decisions = sum(len(r["answers"]) for r in successful)
    counts = {
        "planned": 0,
        "attempted": 0,
        "unattempted": 0,
        "incomplete": 0,
        "failed": 0,
    }
    failed_ids = {r["id"] for r in failed}
    for case in cases:
        size = len(case["request"]["questions"])
        counts["planned"] += size
        counts["attempted" if case["id"] in attempted else "unattempted"] += size
        if case["id"] in incomplete:
            counts["incomplete"] += size
        if case["id"] in failed_ids:
            counts["failed"] += size
    errors = {}
    for record in failed:
        key = record["error_kind"]
        errors[key] = errors.get(key, 0) + 1
    return {
        "requests": len(records),
        "planned_requests": len(cases),
        "attempted_requests": len(attempted),
        "unattempted_requests": len(cases) - len(attempted),
        "incomplete_requests": len(incomplete),
        "completed_requests": len(completed),
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "errors": errors,
        **{name + "_decisions": count for name, count in counts.items()},
        "successful_decisions": decisions,
        "completed_decisions": decisions + counts["failed"],
        "wall_seconds": elapsed,
        "requests_per_second": len(successful) / elapsed if elapsed else None,
        "decisions_per_second": decisions / elapsed if elapsed else None,
        "evidence_complete": elapsed is not None and len(completed) == len(cases),
        "latency_basis": "post_to_body_json_and_schema_validation",
        "successful_latency_p50_ms": statistics.median(latencies)
        if latencies
        else None,
        "successful_latency_p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1]
        if latencies
        else None,
        "failed_latency_p50_ms": statistics.median(failed_latencies)
        if failed_latencies
        else None,
        "failed_latency_p95_ms": failed_latencies[
            math.ceil(0.95 * len(failed_latencies)) - 1
        ]
        if failed_latencies
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
        "state": "completed",
    }
    try:
        async with asyncio.timeout(timeout):
            response = await client.post(
                endpoint, json={**case["request"], "model": model}
            )
        record["status"] = response.status_code
        record["response"] = response.text
        response.raise_for_status()
        record["answers"] = read_answers(case, json.loads(record["response"]))
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
    except asyncio.CancelledError:
        record.update(
            state="incomplete",
            error="request cancelled before response validation completed",
            error_kind="interrupted",
        )
    duration = (time.perf_counter() - started) * 1000
    record["latency_ms"] = duration if record["state"] == "completed" else None
    if record["state"] == "incomplete":
        record["elapsed_until_abort_ms"] = duration
    record["response_sha256"] = response_digest(record["response"])
    return record


def measurement_state(output):
    state = {
        "started_at": None,
        "finished_at": None,
        "wall_seconds": None,
        "stop_reason": None,
        "attempted_ids": [],
        "completed_ids": [],
        "active_ids": [],
        "responses_sha256": None,
    }
    if output is not None:
        state["manifest_sha256"] = digest(output / "requests.jsonl")
        state["config_sha256"] = digest(output / "config.json")
    return state


async def replay(client, endpoint, cases, model, concurrency, timeout, output=None):
    pending = iter(enumerate(cases))
    records = [None] * len(cases)
    state = measurement_state(output)
    responses = None
    response_hash = hashlib.sha256()
    if output is not None:
        state["responses_sha256"] = response_hash.hexdigest()
        responses = (output / "responses.jsonl").open(
            "w", encoding="utf-8", newline=""
        )

    def save_state():
        if output is not None:
            write_json(output / "completion.json", state)

    async def worker():
        for index, case in pending:
            state["attempted_ids"].append(case["id"])
            state["active_ids"].append(case["id"])
            save_state()
            records[index] = await request_one(client, endpoint, case, model, timeout)
            record = records[index]
            if responses is not None:
                line = json.dumps(record, allow_nan=False) + "\n"
                responses.write(line)
                responses.flush()
                response_hash.update(line.encode("utf-8"))
                state["responses_sha256"] = response_hash.hexdigest()
            if record["state"] == "completed":
                state["completed_ids"].append(case["id"])
            state["active_ids"].remove(case["id"])
            save_state()
            if record["state"] == "incomplete":
                raise asyncio.CancelledError

    started = time.perf_counter()
    state["started_at"] = time.time()
    save_state()
    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks)
        if responses is not None:
            responses.close()
            responses = None
            path = output / "responses.jsonl"
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "".join(json.dumps(r, allow_nan=False) + "\n" for r in records),
                encoding="utf-8",
                newline="",
            )
            temporary.replace(path)
            state["responses_sha256"] = digest(path)
        state["stop_reason"] = "completed"
    except BaseException as exc:
        state["stop_reason"] = (
            "interrupted" if isinstance(exc, asyncio.CancelledError) else "error"
        )
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        state["wall_seconds"] = time.perf_counter() - started
        state["finished_at"] = time.time()
        if responses is not None:
            responses.close()
        save_state()
    return records, state["wall_seconds"]


async def run(args, cases):
    warmup_manifest = getattr(args, "warmup_manifest", None)
    warmup_cases = load_cases(warmup_manifest) if warmup_manifest else cases
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
            "warmup_manifest_sha256": digest(warmup_manifest)
            if warmup_manifest
            else None,
            "timeout_seconds": args.timeout,
            "latency_basis": "post_to_body_json_and_schema_validation",
            "python": platform.python_version(),
            "httpx": httpx.__version__,
        },
    )
    (args.output / "requests.jsonl").write_bytes(args.manifest.read_bytes())
    if warmup_manifest:
        (args.output / "warmup-requests.jsonl").write_bytes(
            warmup_manifest.read_bytes()
        )
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
                    warmup_cases[i % len(warmup_cases)],
                    args.model,
                    args.timeout,
                )
            )
            if warmup[-1]["error"] is not None:
                write_json(args.output / "warmup.json", warmup)
                state = measurement_state(args.output)
                state["stop_reason"] = (
                    "warmup_interrupted"
                    if warmup[-1]["state"] == "incomplete"
                    else "warmup_failed"
                )
                write_json(args.output / "completion.json", state)
                write_json(args.output / "summary.json", summarize_saved(args.output))
                raise ValueError("readiness/warmup failed; see warmup.json")
        write_json(args.output / "warmup.json", warmup)
        try:
            await replay(
                client,
                args.endpoint,
                cases,
                args.model,
                args.concurrency,
                args.timeout,
                args.output,
            )
        except asyncio.CancelledError:
            # Replay has saved the completed samples and the measured stop boundary.
            pass
    summary = summarize_saved(args.output)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return int(summary["failed_requests"] > 0 or not summary["evidence_complete"])


def summarize_saved(path):
    """Recompute from this runner's saved inputs, responses and timing evidence."""
    config = json.loads((path / "config.json").read_text())
    state = json.loads((path / "completion.json").read_text())
    if digest(path / "config.json") != state["config_sha256"]:
        raise ValueError("saved config checksum does not match completion")
    if config["latency_basis"] != "post_to_body_json_and_schema_validation":
        raise ValueError("unsupported saved latency basis")
    manifest_hash = digest(path / "requests.jsonl")
    if (
        manifest_hash != config["manifest_sha256"]
        or manifest_hash != state["manifest_sha256"]
    ):
        raise ValueError("saved manifest checksum does not match config/completion")
    if config["warmup_manifest_sha256"] is not None:
        if digest(path / "warmup-requests.jsonl") != config["warmup_manifest_sha256"]:
            raise ValueError("saved warmup manifest checksum does not match config")
    responses_path = path / "responses.jsonl"
    if state["started_at"] is None:
        if state["responses_sha256"] is not None or responses_path.exists():
            raise ValueError("unstarted measurement must not claim responses")
        records = []
    elif digest(responses_path) != state["responses_sha256"]:
        raise ValueError("saved responses checksum does not match completion")
    else:
        records = [json.loads(line) for line in responses_path.read_text().splitlines()]
    cases = load_cases(path / "requests.jsonl")
    by_id = {c["id"]: c for c in cases}
    ids = {}
    for name in ("attempted_ids", "completed_ids", "active_ids"):
        values = state[name]
        if len(values) != len(set(values)) or not set(values) <= set(by_id):
            raise ValueError(f"duplicate or unknown {name}")
        ids[name] = set(values)
    if not ids["completed_ids"] <= ids["attempted_ids"]:
        raise ValueError("completed IDs must have a saved start")
    if not ids["active_ids"] <= ids["attempted_ids"] - ids["completed_ids"]:
        raise ValueError("active IDs must be attempted and incomplete")
    record_ids = [r["id"] for r in records]
    if (
        len(record_ids) != len(set(record_ids))
        or not set(record_ids) <= ids["attempted_ids"]
    ):
        raise ValueError("duplicate, unknown or unattempted response ID")
    completed_ids = set()
    for record in records:
        if record["response_sha256"] != response_digest(record["response"]):
            raise ValueError("raw response checksum mismatch")
        if record["state"] == "incomplete":
            if (
                record["latency_ms"] is not None
                or record["error_kind"] != "interrupted"
                or not record["error"]
                or record["answers"]
            ):
                raise ValueError("incomplete response must not claim a latency/answer")
            if (
                not number(record["elapsed_until_abort_ms"])
                or record["elapsed_until_abort_ms"] < 0
            ):
                raise ValueError("invalid elapsed-until-abort")
            continue
        if (
            record["state"] != "completed"
            or not number(record["latency_ms"])
            or record["latency_ms"] < 0
        ):
            raise ValueError("invalid completed response or latency")
        completed_ids.add(record["id"])
        answers, error_kind = {}, None
        status = record["status"]
        if status is None:
            if record["response"] is not None or record["error_kind"] not in (
                "timeout", "transport"
            ):
                raise ValueError("response without status must be a client failure")
            error_kind = record["error_kind"]
        elif (
            not isinstance(status, int)
            or isinstance(status, bool)
            or not 100 <= status <= 599
        ):
            raise ValueError("invalid HTTP status")
        elif not 200 <= status < 300:
            error_kind = f"http_{status}"
        else:
            try:
                answers = read_answers(
                    by_id[record["id"]], json.loads(record["response"])
                )
            except (ValueError, KeyError, TypeError, AttributeError):
                error_kind = "invalid_response"
        if (
            record["answers"] != answers
            or record["error_kind"] != error_kind
            or (record["error"] is None) != (error_kind is None)
        ):
            raise ValueError("raw response and saved validation result disagree")
    if completed_ids != ids["completed_ids"]:
        raise ValueError("completed IDs do not match saved responses")
    elapsed = state["wall_seconds"]
    if state["started_at"] is None:
        if (
            state["finished_at"] is not None
            or elapsed is not None
            or any(ids.values())
            or state["stop_reason"] not in ("warmup_failed", "warmup_interrupted")
        ):
            raise ValueError("invalid unstarted measurement")
    elif not number(state["started_at"]):
        raise ValueError("invalid measured start boundary")
    elif state["finished_at"] is None:
        if elapsed is not None or state["stop_reason"] is not None:
            raise ValueError("unfinished measurement must not claim a terminal wall")
    else:
        if (
            not number(state["finished_at"])
            or state["finished_at"] < state["started_at"]
            or not number(elapsed)
            or elapsed <= 0
            or ids["active_ids"]
            or state["stop_reason"] not in ("completed", "interrupted", "error")
        ):
            raise ValueError("invalid measured completion boundary")
        if state["stop_reason"] == "completed" and completed_ids != set(by_id):
            raise ValueError("completed traversal is missing response IDs")
    summary = summarize(cases, records, elapsed, ids["attempted_ids"])
    summary["stop_reason"] = state["stop_reason"]
    summary["evidence_complete"] = (
        summary["evidence_complete"] and state["stop_reason"] == "completed"
    )
    return summary


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
    drift, flips, score_drift, failed, compared = 0.0, 0, 0.0, 0, 0
    cases = load_cases(args.reference / "requests.jsonl")
    ordered = []
    for records in runs:
        by_id = {r["id"]: r for r in records}
        if len(by_id) != len(records) or set(by_id) != {c["id"] for c in cases}:
            raise ValueError("response IDs do not uniquely cover the manifest")
        ordered.append([by_id[c["id"]] for c in cases])
    for case, ref, candidate in zip(cases, *ordered):
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
    runner.add_argument(
        "--warmup-manifest", type=Path, help="independent labelled warmup requests"
    )
    runner.add_argument("--timeout", type=positive, default=60)
    comparison = commands.add_parser("compare")
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--max-probability-drift", type=float, required=True)
    comparison.add_argument("--max-score-drift", type=float, required=True)
    comparison.add_argument("--max-flips", type=int, required=True)
    summary = commands.add_parser(
        "summary", help="recompute a saved run without HTTP requests"
    )
    summary.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "summary":
            print(json.dumps(summarize_saved(args.output), indent=2, allow_nan=False))
            return 0
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
