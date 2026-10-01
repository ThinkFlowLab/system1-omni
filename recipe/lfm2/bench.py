#!/usr/bin/env python3
"""Reproducible GPU benchmark for LFM2.5-350M constrained candidate scoring.

Same CUDA model and environment for every variant: an uncached per-candidate
oracle, the pinned upstream reference ``Engine.constrained`` all-at-once fork,
and the in-repo engine at explicit candidate batch sizes 1/8/16/32/64/all.
The workflow matrix crosses candidate count 2/16/64/255 with short/long context
(actual token counts are recorded) and short/mixed candidate lengths. A separate
multi-field table can be added with ``--suites`` and is never mixed into the
public string-enum numbers. Results are appended as JSONL so an interrupted run
can resume; timings and memory peaks are recorded per sample, never synthesized.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _harness as H

SUITE_STRING_ENUM = "string_enum"
SUITE_MULTI_FIELD = "multi_field"
DISTRACTOR = "An old record was checked and filed."
SHORT_REPEATS = 16
LONG_REPEATS = 200
FULL_CANDIDATE_COUNTS = [2, 16, 64, 255]
FULL_FIELD_COUNTS = [3, 8, 16]
QUICK_CANDIDATE_COUNTS = [2, 16]
QUICK_FIELD_COUNTS = [3]
BATCH_VARIANTS = [1, 8, 16, 32, 64]
OOM_ERROR = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


class Cell:
    def __init__(self, suite, cell_id, schema, context, expected, candidate_count):
        self.suite = suite
        self.cell_id = cell_id
        self.schema = schema
        self.context = context
        self.expected = expected
        self.candidate_count = candidate_count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine-dir", default=str(H.default_engine_dir()))
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--model-id", default=H.MODEL_ID)
    parser.add_argument("--model-revision", default=H.MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--suites", default=SUITE_STRING_ENUM,
                        help="comma-separated suite names: string_enum,multi_field")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--quick", action="store_true",
                        help="small matrix; results are labelled quick, never full")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "bench_results.jsonl"))
    parser.add_argument("--summary", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def sha256_text(text):
    return H.sha256_json(text)


def build_context(repeats, chosen):
    return "Archive entries follow.\n" + (DISTRACTOR + "\n") * repeats \
        + "\nFinal assigned code: %s." % chosen


def candidate_words(count, style):
    words = []
    for index in range(count):
        if style == "mixed" and index % 2 == 1:
            words.append("long_code_value_%03d_with_extra_padding" % index)
        else:
            words.append("code_%03d" % index)
    return words


def string_enum_cell(candidate_count, context_kind, candidate_length):
    words = candidate_words(candidate_count, candidate_length)
    chosen = words[-2]
    repeats = SHORT_REPEATS if context_kind == "short" else LONG_REPEATS
    schema = {"type": "object",
              "properties": {"answer": {"type": "string",
                                        "description": "Copy the exact assigned code from the text.",
                                        "enum": words}},
              "required": ["answer"], "additionalProperties": False}
    cell_id = "string_enum|cc=%d|ctx=%s|clen=%s" % (candidate_count, context_kind, candidate_length)
    return Cell(SUITE_STRING_ENUM, cell_id, schema, build_context(repeats, chosen),
                {"answer": chosen}, candidate_count)


def multi_field_cell(field_count, context_kind):
    fields = {"flag_%02d" % index: {"type": "boolean",
              "description": "True if item %02d is enabled; false if disabled" % index}
              for index in range(field_count)}
    expected = {name: index % 2 == 0 for index, name in enumerate(fields)}
    lines = ["Item %02d is %s." % (index, "enabled" if index % 2 == 0 else "disabled")
             for index in range(field_count)]
    if context_kind == "long":
        lines = [DISTRACTOR] * LONG_REPEATS + lines
    cell_id = "multi_field|fields=%d|ctx=%s" % (field_count, context_kind)
    schema = {"type": "object", "properties": fields, "required": list(fields),
              "additionalProperties": False}
    return Cell(SUITE_MULTI_FIELD, cell_id, schema, "\n".join(lines), expected,
                H.candidate_count(schema))


def build_cells(suites, quick):
    cells = []
    if SUITE_STRING_ENUM in suites:
        counts = QUICK_CANDIDATE_COUNTS if quick else FULL_CANDIDATE_COUNTS
        contexts = ["short"] if quick else ["short", "long"]
        lengths = ["short"] if quick else ["short", "mixed"]
        for candidate_count in counts:
            for context_kind in contexts:
                for candidate_length in lengths:
                    cells.append(string_enum_cell(candidate_count, context_kind, candidate_length))
    if SUITE_MULTI_FIELD in suites:
        field_counts = QUICK_FIELD_COUNTS if quick else FULL_FIELD_COUNTS
        contexts = ["short"] if quick else ["short", "long"]
        for field_count in field_counts:
            for context_kind in contexts:
                cells.append(multi_field_cell(field_count, context_kind))
    return cells


def variant_names():
    return ["uncached_oracle", "reference_all"] \
        + ["engine_b%d" % size for size in BATCH_VARIANTS] + ["engine_all"]


def rotate(sequence, offset):
    offset = offset % len(sequence)
    return sequence[offset:] + sequence[:offset]


def resolve_summary(args):
    if args.summary:
        return Path(args.summary)
    return Path(args.output).with_suffix(".summary.json")


def invoke_variant(name, oracle_engine, reference_engine, engine_factory, context, schema):
    if name == "uncached_oracle":
        return H.oracle_scores(oracle_engine, context, schema)
    if name == "reference_all":
        return reference_engine.constrained(context, schema)
    if name == "engine_all":
        size = H.candidate_count(schema)
    else:
        size = int(name.split("_b", 1)[1])
    return engine_factory(size).score(context, schema)


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


def load_done_keys(path, manifest_id):
    done = {}
    if not Path(path).is_file():
        return done
    for record in H.read_jsonl(path):
        if record.get("manifest_id") != manifest_id:
            raise SystemExit("existing %s has a different manifest; use a new --output" % path)
        done[(record["suite"], record["cell"], record["variant"], record["sample"])] = True
    return done


def run_warmups(device, warmup, cells, oracle_engine, reference_engine, engine_factory):
    for index in range(warmup):
        for cell in rotate(cells, index):
            for variant in rotate(variant_names(), index):
                try:
                    invoke_variant(variant, oracle_engine, reference_engine,
                                   engine_factory, cell.context, cell.schema)
                except OOM_ERROR:
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    print("warmup OOM %s %s" % (cell.cell_id, variant),
                          file=sys.stderr, flush=True)


def run_samples(args, cells, oracle_engine, reference_engine, engine_factory,
                expected_selections, manifest_id, done):
    records = []
    for sample in range(args.samples):
        for cell in rotate(cells, sample):
            orders = rotate(variant_names(), sample)
            if sample % 2:
                orders = list(reversed(orders))
            for variant in orders:
                key = (cell.suite, cell.cell_id, variant, sample)
                if key in done:
                    continue
                record = {"suite": cell.suite, "cell": cell.cell_id, "variant": variant,
                          "sample": sample, "manifest_id": manifest_id,
                          "candidate_count": cell.candidate_count,
                          "status": "ok", "selection_ok": None,
                          "latency_ms": None, "allocated_bytes": None,
                          "reserved_bytes": None, "error": None}
                try:
                    latency, allocated, reserved, output = measure(
                        args.device,
                        lambda: invoke_variant(variant, oracle_engine, reference_engine,
                                               engine_factory, cell.context, cell.schema))
                    selection = H.selected_from(output["scores"])
                    record.update({"latency_ms": latency, "allocated_bytes": allocated,
                                   "reserved_bytes": reserved, "selection_ok":
                                   selection == expected_selections[cell.cell_id]})
                except OOM_ERROR as error:
                    if args.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    record.update({"status": "oom", "error": str(error)})
                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        if args.device.startswith("cuda"):
                            torch.cuda.empty_cache()
                        record.update({"status": "oom", "error": str(error)})
                    else:
                        raise
                H.append_jsonl(args.output, record)
                records.append(record)
                print("%s sample=%d %s %s %s" % (
                    cell.suite, sample, cell.cell_id, variant, record["status"]),
                    file=sys.stderr, flush=True)
    return records


def summarize_records(all_records):
    grouped = {}
    for record in all_records:
        key = (record["suite"], record["cell"], record["variant"])
        grouped.setdefault(key, []).append(record)
    summaries = []
    for (suite, cell, variant), records in sorted(grouped.items()):
        latencies = [record["latency_ms"] for record in records
                     if record["latency_ms"] is not None]
        allocations = [record["allocated_bytes"] for record in records
                       if record["allocated_bytes"] is not None]
        reservations = [record["reserved_bytes"] for record in records
                        if record["reserved_bytes"] is not None]
        selections = [record["selection_ok"] for record in records
                      if record["selection_ok"] is not None]
        oom = any(record["status"] == "oom" for record in records)
        summaries.append({
            "suite": suite, "cell": cell, "variant": variant,
            "samples": len(latencies),
            "oom": oom,
            "selection_ok_all": all(selections) if selections else None,
            "latency_ms": {"p50": H.percentile(latencies, 50), "p95": H.percentile(latencies, 95),
                           "mean": (sum(latencies) / len(latencies)) if latencies else None,
                           "min": min(latencies) if latencies else None,
                           "max": max(latencies) if latencies else None,
                           "samples": latencies},
            "allocated_peak_bytes": max(allocations) if allocations else None,
            "reserved_peak_bytes": max(reservations) if reservations else None,
        })
    return summaries


def main(argv=None):
    args = parse_args(argv)
    if args.warmup < 1 or args.samples < 1:
        raise SystemExit("--warmup and --samples must be positive")
    suites = [token.strip() for token in args.suites.split(",") if token.strip()]
    if not suites:
        raise SystemExit("--suites must not be empty")
    for suite in suites:
        if suite not in (SUITE_STRING_ENUM, SUITE_MULTI_FIELD):
            raise SystemExit("unknown suite: %s" % suite)
    if Path(args.output).exists() and not args.resume:
        raise SystemExit("%s exists; pass --resume or choose another --output" % args.output)

    reference = H.load_reference(args.reference_dir)
    engine_module = H.load_target_engine(args.engine_dir)
    model, tokenizer = H.load_shared_model(args.device, args.dtype, args.model_id, args.model_revision)
    reference_engine = H.build_reference_engine(reference, args.device, args.dtype, model, tokenizer)
    oracle_engine = engine_module.Engine(1, args.device, args.dtype, model=model, tokenizer=tokenizer)
    engine_cache = {}

    def engine_factory(size):
        if size not in engine_cache:
            engine_cache[size] = engine_module.Engine(size, args.device, args.dtype,
                                                      model=model, tokenizer=tokenizer)
        return engine_cache[size]

    cells = build_cells(suites, args.quick)
    expected_selections = {}
    cell_metadata = []
    for cell in cells:
        oracle = H.oracle_scores(oracle_engine, cell.context, cell.schema)
        expected_selections[cell.cell_id] = H.selected_from(oracle["scores"])
        cell_metadata.append({
            "suite": cell.suite, "cell": cell.cell_id,
            "candidate_count": cell.candidate_count,
            "prompt_tokens": oracle["prompt_tokens"],
            "context_sha256": sha256_text(cell.context),
            "candidate_words_sha256": H.sha256_json(cell.schema["properties"]),
            "expected_selection": expected_selections[cell.cell_id],
        })

    source_paths = [Path(args.engine_dir) / "engine.py", Path(args.engine_dir) / "worker.py",
                    Path(__file__), Path(__file__).resolve().parent / "_harness.py",
                    Path(reference.root) / "engine.py"]
    if reference.layout == "rlcd-package":
        source_paths.append(Path(reference.root) / "rlcd" / "engine.py")
    environment = H.collect_environment(args.device, reference, source_paths,
                                        args.model_id, args.model_revision)
    environment["recorded_at_utc"] = H.utc_now()
    environment["tf32_matmul"] = torch.backends.cuda.matmul.allow_tf32
    environment["tf32_cudnn"] = torch.backends.cudnn.allow_tf32
    environment["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    protocol = {
        "matrix": "quick" if args.quick else "full",
        "suites": suites, "candidate_variant_sizes": BATCH_VARIANTS,
        "warmup": args.warmup, "samples": args.samples,
        "timing": "device synchronize before/after perf_counter; "
                  "reset_peak_memory_stats before every measured sample",
        "order": "cells and variants rotated by sample index, reversed on odd samples",
        "caveat": "candidate batch size bounds branch count, not a fixed byte budget; "
                  "reserved memory reflects allocator high-water and is not a per-run bound",
    }
    manifest_id = H.sha256_json({"environment_id": H.environment_id(environment),
                                 "protocol": protocol, "cells": cell_metadata})
    manifest = {"manifest_id": manifest_id, "environment": environment,
                "protocol": protocol, "cells": cell_metadata}
    H.atomic_write_json(Path(args.output).with_suffix(".manifest.json"), manifest)

    if args.resume:
        done = load_done_keys(args.output, manifest_id)
    else:
        done = {}
    run_warmups(args.device, args.warmup, cells, oracle_engine, reference_engine, engine_factory)
    records = run_samples(args, cells, oracle_engine, reference_engine, engine_factory,
                          expected_selections, manifest_id, done)
    if args.resume:
        all_records = [record for record in H.read_jsonl(args.output)
                       if record.get("manifest_id") == manifest_id]
    else:
        all_records = records
    complete = len(all_records) == len(cells) * len(variant_names()) * args.samples
    successful = [record for record in all_records if record["status"] == "ok"]
    parity_passed = bool(successful) and all(record["selection_ok"] for record in successful)
    report = {"kind": "lfm2-bench", "complete": complete, "parity_passed": parity_passed, "manifest": manifest["manifest_id"],
              "protocol": manifest["protocol"], "environment": environment,
              "cells": cell_metadata, "variants": summarize_records(all_records),
              "jsonl": str(args.output)}
    H.atomic_write_json(resolve_summary(args), {"manifest": manifest, "summary": report})
    print(json.dumps({"matrix": manifest["protocol"]["matrix"],
                      "cases": len(cell_metadata), "records": len(all_records)},
                     indent=2))
    return 0 if complete and parity_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
