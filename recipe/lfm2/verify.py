#!/usr/bin/env python3
"""Real-GPU verification for the LFM2.5-350M candidate-scoring engine.

Loads one set of read-only weights, loads the pinned upstream RLCD ``engine.py``
with importlib, and checks that the in-repo engine matches an independent
uncached oracle on the pinned diagnostic schemas. Also checks hybrid cache
isolation, candidate-batch invariance, multi-question API independence, and
near-tie margins. Writes a JSON report and exits non-zero on any failed check.

This script is a check, not a benchmark: see bench.py for latency and memory.
It never writes a passing report unless every required check actually passed.
"""

import argparse
import copy
import json
import sys
import threading
import traceback
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _diagnostics as FIXTURES
import _harness as H

NEAR_TIE_PROBES = [
    ("route-ambiguous", FIXTURES.ROUTING,
     "The warehouse note says the route is either north east or north west; "
     "the record does not say which one was assigned."),
    ("service-ambiguous", FIXTURES.ROUTING,
     "The note names both express and express plus; the requested delivery service "
     "is not stated clearly."),
    ("insurance-ambiguous", FIXTURES.ROUTING,
     "Insurance may or may not have been requested; the note is silent."),
    ("sentiment-ambiguous", FIXTURES.SENTIMENT,
     "The review is neither clearly good nor clearly bad; it is mixed."),
]

API_STATE = "I was charged twice for my order. Please refund the duplicate charge immediately."
API_Q1 = {"type": "choice", "instructions": "Which department should handle this?",
          "criteria": {"billing": "Charges and refunds", "technical": "Software problems",
                       "shipping": "Delivery"}}
API_Q2 = {"type": "choice", "instructions": "How urgent is the request?",
          "criteria": {"low": "Can wait", "high": "Needs immediate action"}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine-dir", default=str(H.default_engine_dir()),
                        help="directory containing the in-repo engine.py and worker.py")
    parser.add_argument("--reference-dir", required=True,
                        help="pinned RLCD checkout with rlcd/engine.py or engine.py")
    parser.add_argument("--model-id", default=H.MODEL_ID)
    parser.add_argument("--model-revision", default=H.MODEL_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-sizes", default="1,8,16,32,64,all",
                        help="comma-separated candidate batch sizes, 'all' means one batch")
    parser.add_argument("--api-batch-size", type=int, default=8)
    parser.add_argument("--score-tolerance", type=float, default=0.15)
    parser.add_argument("--distribution-atol", type=float, default=0.015)
    parser.add_argument("--distribution-rtol", type=float, default=0.08)
    parser.add_argument("--expected-attention-layers", type=int, default=6)
    parser.add_argument("--expected-conv-layers", type=int, default=10)
    parser.add_argument("--near-tie-threshold", type=float, default=0.25)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "verify_report.json"))
    return parser.parse_args(argv)


def parse_batch_sizes(text):
    sizes = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token != "all":
            if int(token) < 1:
                raise ValueError("batch sizes must be positive")
        sizes.append(token)
    if not sizes:
        raise ValueError("--batch-sizes must not be empty")
    return sizes


def source_paths(args, reference):
    paths = [Path(args.engine_dir) / "engine.py",
             Path(args.engine_dir) / "worker.py",
             Path(__file__),
             Path(__file__).resolve().parent / "_harness.py",
             Path(__file__).resolve().parent / "_diagnostics.py",
             Path(__file__).resolve().parent / "bench.py"]
    if reference.layout == "rlcd-package":
        paths += [reference.root / "rlcd" / "engine.py", reference.root / "rlcd" / "tasks.py",
                  reference.root / "rlcd" / "stress_tasks.py"]
    else:
        paths += [reference.root / "engine.py"]
    paths += getattr(reference, "task_source_paths", [])
    return paths


def score_matrix(scores):
    return {"%s\u0000%s" % (name, option["value"]): option["log_likelihood"]
            for name, options in scores.items() for option in options}


def batch_pairwise(per_batch, tolerance):
    labels = list(per_batch)
    matrices = [score_matrix(per_batch[label]["engine_scores"]) for label in labels]
    maximum = max((max(matrix[key] for matrix in matrices)
                   - min(matrix[key] for matrix in matrices) for key in matrices[0]), default=0.0)
    selections_equal = all(per_batch[label]["selected"] == per_batch[labels[0]]["selected"]
                           for label in labels[1:])
    return {"max_pairwise_abs_error": maximum, "selections_equal": selections_equal,
            "labels": labels, "consistent": maximum < tolerance and selections_equal}


def run_score_parity(engine_module, model, tokenizer, reference_engine, args):
    prompt_engine = engine_module.Engine(1, args.device, args.dtype,
                                         model=model, tokenizer=tokenizer)
    engine_cache = {}

    def get_engine(size):
        if size not in engine_cache:
            engine_cache[size] = engine_module.Engine(size, args.device, args.dtype,
                                                      model=model, tokenizer=tokenizer)
        return engine_cache[size]

    cases = list(FIXTURES.CASES)
    if not args.skip_stress:
        cases += list(FIXTURES.STRESS_CASES)
    results = []
    batch_consistency = {}
    passed = True
    for case_id, schema, context, expected in cases:
        oracle = H.oracle_scores(prompt_engine, context, schema)
        prefix_equal = oracle["prefix_token_ids"] == reference_engine.encode(reference_engine.prompt(context, schema))
        passed = passed and prefix_equal
        reference_output = reference_engine.constrained(context, schema)
        reference_comparison = H.compare_scores(reference_output["scores"], oracle["scores"],
                                                schema, args.score_tolerance)
        passed = passed and reference_comparison["within_tolerance"] \
            and reference_comparison["selection_match"]
        count = H.candidate_count(schema)
        per_batch = {}
        for label in args.batch_sizes:
            size = count if label == "all" else int(label)
            observed_ids = []
            def record_inputs(_module, positional, keyword):
                ids = positional[0] if positional else keyword["input_ids"]
                observed_ids.append(ids.detach().cpu().tolist())
            hook = model.register_forward_pre_hook(record_inputs, with_kwargs=True)
            try:
                output = get_engine(size).score(context, schema)
            finally:
                hook.remove()
            expected_ids = [[oracle["prefix_token_ids"]]]
            branches = []
            for name, options in oracle["scores"].items():
                suffix = reference_engine.encode("  " + json.dumps(name, ensure_ascii=False) + ": ")
                branches.extend(suffix + option["token_ids"] for option in options)
            for start in range(0, len(branches), size):
                chunk = branches[start:start + size]
                width = max(map(len, chunk))
                expected_ids.append([ids + [tokenizer.pad_token_id] * (width - len(ids))
                                     for ids in chunk])
            token_ids_equal = observed_ids == expected_ids
            passed = passed and token_ids_equal
            comparison = H.compare_scores(output["scores"], oracle["scores"], schema,
                                          args.score_tolerance)
            per_batch[label] = {
                "candidate_batch_size": size,
                "token_ids_equal": token_ids_equal,
                "forward_input_ids_sha256": H.sha256_json(observed_ids),
                "max_abs_error": comparison["max_abs_error"],
                "within_tolerance": comparison["within_tolerance"],
                "selection_match": comparison["selection_match"],
                "selected": H.selected_from(output["scores"]),
                "margins": H.margins_from(oracle["scores"]),
                "engine_scores": output["scores"],
                "fields": comparison["fields"],
                "prompt_tokens": output.get("prompt_tokens"),
                "telemetry": output.get("telemetry"),
            }
            passed = passed and comparison["within_tolerance"] and comparison["selection_match"]
        consistency = batch_pairwise(per_batch, args.score_tolerance)
        batch_consistency[case_id] = consistency
        passed = passed and consistency["consistent"]
        gold_selected = per_batch[args.batch_sizes[-1]]["selected"]
        results.append({
            "id": case_id,
            "prefix_token_ids_equal": prefix_equal,
            "prefix_token_ids_sha256": H.sha256_json(oracle["prefix_token_ids"]),
            "candidate_count": count,
            "oracle": {"prompt_tokens": oracle["prompt_tokens"], "scores": oracle["scores"]},
            "reference": {
                "max_abs_error": reference_comparison["max_abs_error"],
                "within_tolerance": reference_comparison["within_tolerance"],
                "selection_match": reference_comparison["selection_match"],
                "selected": H.selected_from(reference_output["scores"]),
            },
            "batches": per_batch,
            "gold_diagnostic": {
                "expected": expected,
                "selected": gold_selected,
                "matches": gold_selected == expected,
                "note": "hand-authored diagnostic, not a population accuracy estimate",
            },
        })
    gold_matches = sum(1 for row in results if row["gold_diagnostic"]["matches"])
    return {
        "passed": passed,
        "score_tolerance": args.score_tolerance,
        "cases": len(results),
        "max_abs_error": max((batch["max_abs_error"] for row in results
                               for batch in row["batches"].values()), default=None),
        "batch_consistency": batch_consistency,
        "gold_diagnostic_matches": gold_matches,
        "gold_diagnostic_note": "individual hand-authored labels; not aggregate accuracy",
        "results": results,
    }


def run_cache_isolation(engine_module, engine, args):
    schema = FIXTURES.ROUTING
    context = "Route: north east. Service: express plus. Insurance requested."
    prefix = engine.encode(engine.prompt(context, schema))
    base = engine.model(engine.tensor([prefix]), use_cache=True,
                        logits_to_keep=1).past_key_values
    snapshot = copy.deepcopy(base)
    attention_layers = sum(1 for layer in base.layers if hasattr(layer, "keys"))
    conv_layers = sum(1 for layer in base.layers if hasattr(layer, "conv_states"))
    suffixes = []
    for name, candidates in list(H.field_specs(schema))[:2]:
        text = "  " + json.dumps(name, ensure_ascii=False) + ": " \
            + json.dumps(candidates[0], ensure_ascii=False) + "\n"
        suffixes.append(engine.encode(text))
    forked = engine_module.fork_cache(base, len(suffixes))
    width = max(len(suffix) for suffix in suffixes)
    pad = engine.tokenizer.pad_token_id
    ids = engine.tensor([suffix + [pad] * (width - len(suffix)) for suffix in suffixes])
    mask = engine.tensor([[1] * (len(prefix) + len(suffix)) + [0] * (width - len(suffix))
                          for suffix in suffixes])
    batched = engine.model(ids, attention_mask=mask, past_key_values=forked).logits
    distribution_ok = True
    argmax_ok = True
    per_suffix = []
    for index, suffix in enumerate(suffixes):
        full = engine.model(engine.tensor([prefix + suffix]),
                            use_cache=False).logits[0, len(prefix):]
        cached = batched[index, :len(suffix)]
        difference = (cached - full).abs().max().item()
        distribution_ok = distribution_ok and bool(torch.allclose(
            cached.float().softmax(-1), full.float().softmax(-1),
            atol=args.distribution_atol, rtol=args.distribution_rtol))
        argmax_ok = argmax_ok and bool(torch.equal(cached.argmax(-1), full.argmax(-1)))
        per_suffix.append({"suffix_tokens": len(suffix), "max_abs_logit_error": difference})
    unchanged = True
    for old, current in zip(snapshot.layers, base.layers):
        if hasattr(old, "keys"):
            unchanged = unchanged and torch.equal(old.keys, current.keys) \
                and torch.equal(old.values, current.values)
        elif hasattr(old, "conv_states"):
            unchanged = unchanged and torch.equal(old.conv_states[0], current.conv_states[0])
    check = {
        "attention_layers": attention_layers,
        "conv_layers": conv_layers,
        "expected_attention_layers": args.expected_attention_layers,
        "expected_conv_layers": args.expected_conv_layers,
        "distribution_within_atol_rtol": distribution_ok,
        "argmax_equal": argmax_ok,
        "shared_cache_unchanged": unchanged,
        "per_suffix": per_suffix,
        "atol": args.distribution_atol,
        "rtol": args.distribution_rtol,
    }
    check["passed"] = (attention_layers == args.expected_attention_layers
                       and conv_layers == args.expected_conv_layers
                       and distribution_ok and argmax_ok and unchanged)
    return check


def answers_equal(left, right, tolerance=1e-6):
    if left.get("type") != right.get("type") or left.get("choice") != right.get("choice"):
        return False
    if abs(float(left.get("confidence", 0.0)) - float(right.get("confidence", 0.0))) > tolerance:
        return False
    left_probabilities = left.get("probabilities", {})
    right_probabilities = right.get("probabilities", {})
    if not isinstance(left_probabilities, dict) or set(left_probabilities) != set(right_probabilities):
        return False
    return all(abs(float(left_probabilities[key]) - float(right_probabilities[key])) <= tolerance
               for key in left_probabilities)


def _http(url, body=None):
    request = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"} if body else {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=120) as response:
        return response.status, response.read()


def run_api_multi_question(worker_module, engine):
    worker = worker_module.Worker(engine)
    worker.warmup()

    def call(questions):
        payload = {"model": worker.model_alias, "state": API_STATE, "questions": questions}
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return worker.systemone(raw), raw

    alone1, _ = call({"q1": API_Q1})
    alone2, _ = call({"q2": API_Q2})
    combined, combined_raw = call({"q1": API_Q1, "q2": API_Q2})
    reordered, _ = call({"q2": API_Q2, "q1": API_Q1})
    renamed, _ = call({"renamed": API_Q1})

    q1_alone = alone1["answers"]["q1"]
    q1_combined = combined["answers"]["q1"]
    q1_reordered = reordered["answers"]["q1"]
    q1_renamed = renamed["answers"]["renamed"]
    q2_alone = alone2["answers"]["q2"]
    q2_combined = combined["answers"]["q2"]

    independent = True
    for candidate in (q1_combined, q1_reordered, q1_renamed):
        independent = independent and answers_equal(q1_alone, candidate)
    q2_stable = answers_equal(q2_alone, q2_combined)

    usage_alone1 = alone1["usage"]["input_tokens"]
    usage_alone2 = alone2["usage"]["input_tokens"]
    usage_additive = combined["usage"]["input_tokens"] == usage_alone1 + usage_alone2
    usage_reorder = reordered["usage"]["input_tokens"] == combined["usage"]["input_tokens"]
    usage_rename = renamed["usage"]["input_tokens"] == usage_alone1

    expected1 = H.input_token_count(
        engine, API_STATE, worker_module.build_schema(API_Q1["instructions"], API_Q1["criteria"]))
    expected2 = H.input_token_count(
        engine, API_STATE, worker_module.build_schema(API_Q2["instructions"], API_Q2["criteria"]))
    usage_exact = usage_alone1 == expected1 and usage_alone2 == expected2

    probabilities_ok = True
    confidence_ok = True
    for answer in (q1_alone, q2_alone):
        total = sum(float(value) for value in answer["probabilities"].values())
        probabilities_ok = probabilities_ok and abs(total - 1.0) < 1e-6
        confidence = float(answer["confidence"])
        confidence_ok = confidence_ok and 0.0 <= confidence <= 1.0

    server = worker_module.make_server(worker, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base_url = "http://%s:%d" % (host, port)
    http_health_ok = http_combined_ok = False
    try:
        status, body = _http(base_url + "/health")
        http_health_ok = status == 200 and json.loads(body) == worker.health()
        status, body = _http(base_url + "/v1/systemone", combined_raw)
        http_combined_ok = status == 200 and json.loads(body) == combined
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)

    passed = independent and q2_stable and usage_additive and usage_reorder \
        and usage_rename and usage_exact and probabilities_ok and confidence_ok \
        and http_health_ok and http_combined_ok
    return {
        "passed": passed,
        "q1_independent_of_addition_order_and_id": independent,
        "q2_stable": q2_stable,
        "usage_is_sum_of_input_tokens": usage_additive,
        "usage_matches_unpadded_inputs": usage_exact,
        "usage_reorder_unchanged": usage_reorder,
        "usage_rename_unchanged": usage_rename,
        "probabilities_sum_to_one": probabilities_ok,
        "confidence_in_unit_range": confidence_ok,
        "http_health_matches": http_health_ok,
        "http_combined_matches_direct": http_combined_ok,
        "q1_alone": q1_alone,
        "q1_renamed": q1_renamed,
        "q2_alone": q2_alone,
        "usage": {"q1_alone": usage_alone1, "q2_alone": usage_alone2,
                  "combined": combined["usage"]["input_tokens"]},
    }


def run_near_ties(prompt_engine, engine, args):
    probes = []
    minimum = None
    minimum_id = None
    passed = True
    original_size = engine.candidate_batch_size
    try:
        for probe_id, schema, context in NEAR_TIE_PROBES:
            oracle = H.oracle_scores(prompt_engine, context, schema)
            oracle_margins = H.margins_from(oracle["scores"])
            oracle_selected = H.selected_from(oracle["scores"])
            finite = [value for value in oracle_margins.values() if value is not None]
            smallest = min(finite) if finite else None
            if smallest is not None and (minimum is None or smallest < minimum):
                minimum, minimum_id = smallest, probe_id
            batches = {}
            for label in args.batch_sizes:
                engine.candidate_batch_size = H.candidate_count(schema) if label == "all" else int(label)
                output = engine.score(context, schema)
                comparison = H.compare_scores(output["scores"], oracle["scores"], schema,
                                              args.score_tolerance)
                passed = passed and comparison["within_tolerance"] and comparison["selection_match"]
                batches[label] = {
                    "engine_selected": H.selected_from(output["scores"]),
                    "engine_margins": H.margins_from(output["scores"]),
                    "engine_scores": output["scores"],
                    "selection_match": comparison["selection_match"],
                    "max_abs_error": comparison["max_abs_error"],
                }
            probes.append({"id": probe_id, "oracle_margins": oracle_margins,
                           "oracle_selected": oracle_selected, "oracle_scores": oracle["scores"],
                           "batches": batches})
    finally:
        engine.candidate_batch_size = original_size
    return {
        "passed": passed, "threshold_label_only": args.near_tie_threshold,
        "minimum_margin": minimum, "minimum_margin_probe": minimum_id,
        "near_tie_observed": minimum is not None and minimum < args.near_tie_threshold,
        "note": "margins and flips are checked at every requested batch size",
        "probes": probes,
    }


def write_report(args, environment, sections, error=None):
    checks = []
    for name, section in sections.items():
        if isinstance(section, dict) and "passed" in section:
            checks.append({"name": name, "passed": bool(section["passed"])})
    required = {"score_parity", "cache_isolation", "near_ties", "api_multi_question"}
    status = "pass" if required <= set(sections) and all(check["passed"] for check in checks) and error is None else "fail"
    report = {
        "kind": "lfm2-verify",
        "status": status,
        "started_at_utc": environment.get("started_at_utc"),
        "finished_at_utc": H.utc_now(),
        "config": {
            "engine_dir": str(args.engine_dir),
            "reference_dir": str(args.reference_dir),
            "device": args.device,
            "dtype": args.dtype,
            "batch_sizes": args.batch_sizes,
            "api_batch_size": args.api_batch_size,
            "score_tolerance": args.score_tolerance,
            "distribution_atol": args.distribution_atol,
            "distribution_rtol": args.distribution_rtol,
            "expected_attention_layers": args.expected_attention_layers,
            "expected_conv_layers": args.expected_conv_layers,
            "skip_stress": args.skip_stress,
        },
        "environment": environment,
        "environment_id": H.environment_id(environment),
        "checks": checks,
        "sections": sections,
    }
    if error is not None:
        report["error"] = error
    H.atomic_write_json(args.output, report)
    return report


def main(argv=None):
    args = parse_args(argv)
    args.batch_sizes = parse_batch_sizes(args.batch_sizes)
    environment = {"started_at_utc": None}
    sections = {}
    error = None
    try:
        reference = H.load_reference(args.reference_dir)
        engine_module = H.load_target_engine(args.engine_dir)
        worker_module = H.load_target_worker(args.engine_dir)
        model, tokenizer = H.load_shared_model(args.device, args.dtype,
                                               args.model_id, args.model_revision)
        shared = {"model": model, "tokenizer": tokenizer}
        prompt_engine = engine_module.Engine(1, args.device, args.dtype, **shared)
        api_engine = engine_module.Engine(args.api_batch_size, args.device, args.dtype, **shared)
        reference_engine = H.build_reference_engine(reference, args.device, args.dtype,
                                                    model, tokenizer)
        environment = H.collect_environment(args.device, reference, source_paths(args, reference),
                                            args.model_id, args.model_revision)
        environment["started_at_utc"] = H.utc_now()
        sections["score_parity"] = run_score_parity(engine_module, model, tokenizer,
                                                    reference_engine, args)
        sections["cache_isolation"] = run_cache_isolation(engine_module, prompt_engine, args)
        sections["near_ties"] = run_near_ties(prompt_engine, api_engine, args)
        sections["api_multi_question"] = run_api_multi_question(worker_module, api_engine)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc),
                 "traceback": traceback.format_exc()}
    report = write_report(args, environment, sections, error)
    passed = report["status"] == "pass"
    for check in report["checks"]:
        print("%s %s" % ("PASS" if check["passed"] else "FAIL", check["name"]))
    if error is not None:
        print("ERROR %s: %s" % (error["type"], error["message"]), file=sys.stderr)
    print("wrote %s (%s)" % (args.output, report["status"]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
