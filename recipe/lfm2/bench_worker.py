#!/usr/bin/env python3
"""Real-GPU benchmark of the public LFM2.5-350M ``Worker.systemone`` path.

Measures the serialized public worker call, not the internal ``Engine.score``
scorer (bench.py) nor the Rust frontend (compare_with_frontend.py). One FP16
weight set is shared by every candidate batch size; each timed call serializes the
request JSON and runs the real worker with no HTTP transport. The matrix is
question count 1/3/8 x short/long state x batch 1/8/all with 16 candidates per
question. Repeats must match the same-config warmup answer (1e-6), usage tokens
must equal the sum of per-question prompt tokens, and question ids must stay out
of the prompt. Raw samples append as JSONL beside a source-hash manifest.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H

QCOUNTS = [1, 3, 8]
CONTEXTS = ["short", "long"]
BATCHES = ["1", "8", "all"]
CANDIDATE_COUNT = 16
DISTRACTOR = "An old record was checked and filed."
SHORT_REPEATS = 16
LONG_REPEATS = 200
TOLERANCE = 1e-6
OOM_ERROR = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine-dir", default=str(H.default_engine_dir()))
    parser.add_argument("--model-id", default=H.MODEL_ID)
    parser.add_argument("--model-revision", default=H.MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "bench_worker_results.jsonl"))
    parser.add_argument("--summary", default=None)
    return parser.parse_args(argv)


def build_questions(count):
    questions = {}
    for index in range(count):
        criteria = {"code_%02d" % candidate:
                    "Candidate code %02d for request %02d" % (candidate, index)
                    for candidate in range(CANDIDATE_COUNT)}
        questions["q%02d" % index] = {
            "type": "choice", "instructions": "Request %02d: copy the exact assigned code." % index,
            "criteria": criteria}
    return questions


def batch_size(batch):
    return CANDIDATE_COUNT if batch == "all" else int(batch)


def measure(device, function):
    H.device_sync(device)
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = function()
    H.device_sync(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    allocated = int(torch.cuda.max_memory_allocated()) if cuda else None
    reserved = int(torch.cuda.max_memory_reserved()) if cuda else None
    return elapsed_ms, allocated, reserved, output


def compare_answers(reference, response, tolerance=TOLERANCE):
    """Return ``(all_equal, max_probability_diff)`` for two worker responses."""
    if set(reference["answers"]) != set(response["answers"]):
        return False, None
    maximum = 0.0
    for qid, expected in reference["answers"].items():
        actual = response["answers"][qid]
        if expected["type"] != actual["type"] or expected["choice"] != actual["choice"]:
            return False, None
        expected_p, actual_p = expected["probabilities"], actual["probabilities"]
        if set(expected_p) != set(actual_p):
            return False, None
        maximum = max(maximum, max(abs(float(expected_p[k]) - float(actual_p[k])) for k in expected_p))
        if abs(float(expected["confidence"]) - float(actual["confidence"])) > tolerance:
            return False, maximum
    return maximum <= tolerance, maximum


def invoke(worker, cell):
    payload = {"model": worker.model_alias, "state": cell["state"], "questions": cell["questions"]}
    response = worker.systemone(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return response


def main(argv=None):
    args = parse_args(argv)
    if args.warmup < 1 or args.samples < 1:
        raise SystemExit("--warmup and --samples must be positive")
    if Path(args.output).exists():
        raise SystemExit("%s exists; choose another --output" % args.output)

    worker_module = H.load_target_worker(args.engine_dir)
    engine_module = H.load_target_engine(args.engine_dir)
    model, tokenizer = H.load_shared_model(args.device, "float16", args.model_id, args.model_revision)
    engines, workers = {}, {}

    def get_engine(size):
        if size not in engines:
            engines[size] = engine_module.Engine(size, args.device, "float16",
                                                 model=model, tokenizer=tokenizer)
        return engines[size]

    def get_worker(batch):
        if batch not in workers:
            workers[batch] = worker_module.Worker(get_engine(batch_size(batch)))
        return workers[batch]

    probe = get_engine(1)
    cells = []
    for qcount in QCOUNTS:
        for context in CONTEXTS:
            repeats = SHORT_REPEATS if context == "short" else LONG_REPEATS
            state = ("Archive entries follow.\n" + (DISTRACTOR + "\n") * repeats
                     + "\nThe assigned code is code_14.")
            questions = build_questions(qcount)
            first = next(iter(questions.values()))
            text = probe.prompt(state, worker_module.build_schema(first["instructions"], first["criteria"]))
            expected = sum(len(probe.encode(probe.prompt(
                state, worker_module.build_schema(q["instructions"], q["criteria"]))))
                for q in questions.values())
            for batch in BATCHES:
                cells.append({"key": "%d|%s|%s" % (qcount, context, batch), "qcount": qcount,
                              "context": context, "state": state, "questions": questions,
                              "batch": batch, "candidate_batch_size": batch_size(batch),
                              "expected_input_tokens": expected,
                              "qid_in_prompt": any(qid in text for qid in questions),
                              "context_sha256": H.sha256_json(state),
                              "questions_sha256": H.sha256_json(questions)})

    references = {}
    for cell in cells:
        worker = get_worker(cell["batch"])
        for _ in range(args.warmup):
            _, _, _, response = measure(args.device, lambda: invoke(worker, cell))
            if cell["key"] not in references:
                references[cell["key"]] = response
            elif not compare_answers(references[cell["key"]], response)[0]:
                raise SystemExit("warmup output inconsistent for %s" % cell["key"])
    for cell in cells:
        cell["reference_answer"] = references[cell["key"]]

    source_paths = [Path(args.engine_dir) / "engine.py", Path(args.engine_dir) / "worker.py",
                    Path(__file__), Path(__file__).resolve().parent / "_harness.py"]
    reference = SimpleNamespace(root="", layout="worker-only")
    environment = H.collect_environment(args.device, reference, source_paths,
                                        args.model_id, args.model_revision)
    environment["recorded_at_utc"] = H.utc_now()
    protocol = {
        "matrix": "worker", "qcounts": QCOUNTS, "contexts": CONTEXTS, "batches": BATCHES,
        "candidate_count": CANDIDATE_COUNT, "warmup": args.warmup, "samples": args.samples,
        "timing": "request/response JSON encoding and Worker.systemone inside perf_counter; device sync "
                  "before/after; reset_peak_memory_stats before each sample; no HTTP transport",
        "consistency": "every warmup and measured output must match the same-config warmup "
                       "reference choice and probabilities within %g" % TOLERANCE,
        "usage": "usage.input_tokens must equal the sum of independently measured per-question "
                 "prompt tokens; question ids must not appear in the prompt",
        "caveat": "candidate batch size bounds simultaneous branches, not bytes; reserved memory "
                  "is allocator high-water, not a per-run bound",
    }
    fields = ("key", "qcount", "context", "batch", "candidate_batch_size", "expected_input_tokens",
              "qid_in_prompt", "context_sha256", "questions_sha256", "reference_answer")
    cell_metadata = [{name: cell[name] for name in fields} for cell in cells]
    manifest_id = H.sha256_json({"environment_id": H.environment_id(environment),
                                 "protocol": protocol, "cells": cell_metadata})
    H.atomic_write_json(Path(args.output).with_suffix(".manifest.json"),
                        {"manifest_id": manifest_id, "environment": environment,
                         "protocol": protocol, "cells": cell_metadata})

    records = []
    for sample in range(args.samples):
        offset = sample % len(cells)
        for cell in cells[offset:] + cells[:offset]:
            worker = get_worker(cell["batch"])
            record = {"manifest_id": manifest_id, "key": cell["key"], "qcount": cell["qcount"],
                      "context": cell["context"], "batch": cell["batch"],
                      "candidate_batch_size": cell["candidate_batch_size"], "sample": sample,
                      "status": "ok", "error": None, "latency_ms": None, "allocated_bytes": None,
                      "reserved_bytes": None, "input_tokens": None,
                      "expected_input_tokens": cell["expected_input_tokens"], "usage_linear_ok": None,
                      "answer_match": None, "max_probability_diff": None,
                      "qid_in_prompt": cell["qid_in_prompt"], "response_sha256": None}
            try:
                latency, allocated, reserved, response = measure(
                    args.device, lambda: invoke(worker, cell))
                equal, difference = compare_answers(references[cell["key"]], response)
                tokens = int(response["usage"]["input_tokens"])
                record.update({"latency_ms": latency, "allocated_bytes": allocated,
                               "reserved_bytes": reserved, "input_tokens": tokens,
                               "usage_linear_ok": tokens == cell["expected_input_tokens"],
                               "answer_match": equal, "max_probability_diff": difference,
                               "response_sha256": H.sha256_json(response)})
            except OOM_ERROR as error:
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                record.update({"status": "oom", "error": str(error)})
            H.append_jsonl(args.output, record)
            records.append(record)
            print("%s sample=%d %s" % (cell["key"], sample, record["status"]),
                  file=sys.stderr, flush=True)

    grouped = {}
    for record in records:
        grouped.setdefault(record["key"], []).append(record)
    summaries = []
    for key, group in sorted(grouped.items()):
        latencies = [r["latency_ms"] for r in group if r["latency_ms"] is not None]
        allocations = [r["allocated_bytes"] for r in group if r["allocated_bytes"] is not None]
        reservations = [r["reserved_bytes"] for r in group if r["reserved_bytes"] is not None]
        matches = [r["answer_match"] for r in group if r["answer_match"] is not None]
        linear = [r["usage_linear_ok"] for r in group if r["usage_linear_ok"] is not None]
        head = group[0]
        summaries.append({
            "key": key, "qcount": head["qcount"], "context": head["context"], "batch": head["batch"],
            "samples": len(latencies), "oom": any(r["status"] == "oom" for r in group),
            "answer_match_all": all(matches) if matches else None,
            "usage_linear_all": all(linear) if linear else None,
            "latency_ms": {"p50": H.percentile(latencies, 50), "p95": H.percentile(latencies, 95),
                           "mean": (sum(latencies) / len(latencies)) if latencies else None,
                           "min": min(latencies) if latencies else None,
                           "max": max(latencies) if latencies else None, "samples": latencies},
            "allocated_peak_bytes": max(allocations) if allocations else None,
            "reserved_peak_bytes": max(reservations) if reservations else None})
    complete = len(records) == len(cells) * args.samples
    clean = all(not s["oom"] and s["answer_match_all"] and s["usage_linear_all"] for s in summaries)
    passed = complete and bool(summaries) and clean and not any(cell["qid_in_prompt"] for cell in cells)
    report = {"kind": "lfm2-bench-worker", "manifest_id": manifest_id, "complete": complete,
              "passed": passed, "protocol": protocol, "environment": environment,
              "cells": cell_metadata, "variants": summaries, "jsonl": str(args.output)}
    summary_path = Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json")
    H.atomic_write_json(summary_path, {"manifest_id": manifest_id, "summary": report})
    print(json.dumps({"cells": len(cells), "records": len(records), "passed": passed}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
