#!/usr/bin/env python3
"""Pinned full-checkpoint eager parity; no latency or task-quality claim."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

GATES = {"max_probability_drift": 0.02, "max_score_drift": 0.1, "discrete_margin": 0.05}
ROUNDING_HALF_UNIT = 0.00005
VALIDATION_PROTOCOL = {
    "version": 2,
    "original_gates": GATES,
    "schema": "exact response/usage/answer keys; finite non-bool numbers; model-specific ranges, pinned wire rounding and ordered identities",
    "auxiliary_method": "confidence, certainty and x_p_max must fit pinned formulas over probability rounding intervals; Score confidence includes every feasible modal index",
    "probability_rounding_half_unit": ROUNDING_HALF_UNIT,
    "score_rounding_half_unit": 0.005,
    "fit_mass_consistency_bound": "(number_of_levels + 1) * probability_rounding_half_unit",
    "fit_mass_reference_bound": "number_of_levels * max_probability_drift + 2 * (number_of_levels + 1) * probability_rounding_half_unit",
    "score_method": "two-decimal expectation consistency plus unchanged reference max_score_drift",
    "numerical_slack": 1e-12,
}
REFERENCE_REVISION = "50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f"
REFERENCE_FILES = {
    "model.py": "74e0ca564bb52f7c2bd74c23cb34b16e0004ceadf7a13816f223c68829b5e1c5",
    "prompt.py": "f21ae1016a11ca3500b5a8e2e8c3ef524434d196bbd4891199b96a33b3ad6314",
    "prompt_fast.py": "fbfcc3a381a89785174cb9d29083ed14d404c7273d123ba8fe2b69584649ef25",
    "systemone.py": "884e4b0bae36f5e09dbd689b4390e94320cbbd77f7b8d675e50490b1d7c61d34",
    "temperature.py": "b940c253e5626145e806195cf5b2bc0de61461de633c07b174a924d18ceeab88",
}


def verify_sources(directory, expected):
    verified = {}
    for name, checksum in expected.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != checksum:
            raise ValueError(f"reference checksum mismatch: {name}")
        verified[name] = actual
    return verified


def workloads():
    cases = []
    for n in (2, 3, 10, 11, 26, 255):
        cases.append((f"choice_{n}", {"state": "The customer asks for a refund for a damaged order.", "questions": {
            "category": {"type": "choice", "instructions": "Choose the matching category.", "criteria": {
                f"option_{i}": "Refund request" if i == n - 1 else f"Unrelated category {i}" for i in range(n)}}}}))
    for n in (2, 3, 10):
        cases.append((f"score_{n}", {"state": "The customer received a damaged item and requests help immediately.", "questions": {
            "urgency": {"type": "score", "instructions": "How urgent is the customer request?", "criteria": [f"{i}. Urgency level {i}" for i in range(n)]}}}))
    cases.extend([
        ("noul_true", {"state": "The customer requests a refund.", "questions": {"refund": {"type": "noul", "instructions": "The customer requests a refund."}}}),
        ("noul_false", {"state": "The customer asks when the package will arrive.", "questions": {"refund": {"type": "noul", "instructions": "The customer requests a refund."}}}),
        ("noul_descriptions", {"state": "The parcel is delivered.", "questions": {"delivered": {"type": "noul", "criteria": {"true": "The parcel is delivered.", "false": "The parcel is still in transit."}}}}),
        ("structured_unicode", {"state": {"客户": "订单损坏，请退款", "orders": [{"id": i, "status": "damaged" if i == 7 else "delivered"} for i in range(9)]}, "questions": {
            "z_last": {"type": "choice", "instructions": {"task": "Choose the status of orders[7]."}, "criteria": {"damaged": "Damaged", "delivered": "Delivered"}},
            "a_first": {"type": "score", "instructions": "Severity of the damage", "criteria": {"0": "Low", "1": "Medium", "2": "High"}},
            "m_middle": {"type": "noul", "instructions": "The customer requests a refund."}}}),
        ("mixed", {"state": "A customer requests a refund because a delivery was damaged.", "questions": {
            "route": {"type": "choice", "instructions": "Choose the team.", "criteria": {"billing": "Refund and billing", "delivery": "Shipping tracking", "sales": "New purchases"}},
            "refund": {"type": "noul", "instructions": "The customer requests a refund."},
            "severity": {"type": "score", "instructions": "Severity of the damage", "criteria": ["Minor", "Moderate", "Severe"]}}}),
        ("long_context", {"state": "The item is damaged. " * 700 + "The customer requests a refund.", "questions": {"route": {"type": "choice", "instructions": "Choose the team.", "criteria": {"billing": "Refund and billing", "shipping": "Delivery status"}}}}),
        ("empty", {"state": {}, "questions": {}}),
    ])
    # Repetition and reverse question order test request/state isolation.
    cases.append(("mixed_reordered", {"state": cases[13][1]["state"], "questions": dict(reversed(list(cases[13][1]["questions"].items())))}))
    cases.append(("choice_repeat", cases[0][1]))
    return cases


def number(value, low, high, where, digits=4):
    assert type(value) in (int, float) and (type(value) is int or math.isfinite(value)), (where, "finite number required", value)
    assert low <= value <= high, (where, "out of range", value, low, high)
    assert abs(value - round(value, digits)) <= 1e-12, (where, "wire rounding", value, digits)
    return value


def within(value, low, high, rounding, where):
    # Round-to-even at an interval endpoint and binary64 evaluation need only
    # this tiny arithmetic slack, not an additional empirical fidelity gate.
    assert low - rounding - 1e-12 <= value <= high + rounding + 1e-12, (where, "formula mismatch", value, low, high)


def validate_answer(answer, where):
    assert isinstance(answer, dict), (where, "answer must be an object")
    kind = answer.get("type")
    assert kind in ("choice", "score", "noul"), (where, "unknown answer type", kind)
    if kind == "noul":
        assert set(answer) == {"type", "noul"}, (where, "answer keys")
        number(answer["noul"], 0, 1, (where, "noul"))
        return
    expected = {"type", "confidence", "certainty", "x_p_max", "probabilities"}
    expected |= {"choice"} if kind == "choice" else {"score", "legend", "level_fit", "fit_mass"}
    assert set(answer) == expected, (where, "answer keys")
    probabilities = answer["probabilities"]
    assert isinstance(probabilities, dict) and all(isinstance(k, str) for k in probabilities), (where, "probability keys")
    n = len(probabilities)
    assert 2 <= n <= (255 if kind == "choice" else 10), (where, "option count", n)
    p = [number(v, 0, 1, (where, "probability", k)) for k, v in probabilities.items()]
    for field in ("confidence", "certainty", "x_p_max"):
        number(answer[field], 0, 1, (where, field))
    zero_mass = sum(p) == 0
    assert (kind == "score" and zero_mass) or abs(sum(p) - 1) <= n * ROUNDING_HALF_UNIT + 1e-12, (where, "probability mass")
    intervals = [(max(0, v - ROUNDING_HALF_UNIT), min(1, v + ROUNDING_HALF_UNIT)) for v in p]
    if zero_mass:
        # At most ten rounded probabilities cannot hide a normalized unit mass.
        # The pinned zero-fit path emits zero probabilities, certainty 1 and
        # uses a uniform fallback solely for its confidence calculation.
        intervals = [(0, 0)] * n
    maximum_low = max(lo for lo, _ in intervals)
    maximum_high = max(hi for _, hi in intervals)
    within(answer["x_p_max"], maximum_low, maximum_high, ROUNDING_HALF_UNIT, (where, "x_p_max"))
    entropy = lambda x: -x * math.log(x) if x > 0 else 0
    minimum_entropy = sum(min(entropy(lo), entropy(hi)) for lo, hi in intervals)
    maximum_entropy = sum(max(entropy(lo), entropy(hi), 1 / math.e if lo <= 1 / math.e <= hi else 0)
                          for lo, hi in intervals)
    within(answer["certainty"], max(0, 1 - maximum_entropy / math.log(n)),
           max(0, 1 - minimum_entropy / math.log(n)), ROUNDING_HALF_UNIT, (where, "certainty"))
    if kind == "choice":
        assert isinstance(answer["choice"], str) and answer["choice"] in probabilities, (where, "choice identity")
        selected = list(probabilities).index(answer["choice"])
        assert intervals[selected][1] >= maximum_low, (where, "choice is not a possible mode")
        within(answer["confidence"], max(0, (n * maximum_low - 1) / (n - 1)),
               min(1, max(0, (n * maximum_high - 1) / (n - 1))), ROUNDING_HALF_UNIT, (where, "confidence"))
        return
    keys = [str(i) for i in range(n)]
    assert list(probabilities) == keys, (where, "score probability keys/order")
    for field in ("legend", "level_fit"):
        assert isinstance(answer[field], dict) and list(answer[field]) == keys, (where, field, "keys/order")
    assert all(isinstance(v, str) for v in answer["legend"].values()), (where, "legend must be text")
    fits = [number(v, 0, 1, (where, "level_fit", k)) for k, v in answer["level_fit"].items()]
    mass = number(answer["fit_mass"], 0, n, (where, "fit_mass"))
    within(mass, sum(fits), sum(fits), (n + 1) * ROUNDING_HALF_UNIT, (where, "fit_mass"))
    if zero_mass:
        assert all(v == 0 for v in fits) and mass == 0, (where, "zero-fit output")
        within(answer["confidence"], 0, 0, ROUNDING_HALF_UNIT, (where, "confidence"))
    else:
        fit_low = [max(0, v - ROUNDING_HALF_UNIT) for v in fits]
        fit_high = [min(1, v + ROUNDING_HALF_UNIT) for v in fits]
        for i, (lo, hi) in enumerate(intervals):
            normalized_low = fit_low[i] / sum(fit_high)
            normalized_high = min(1, fit_high[i] / sum(fit_low)) if sum(fit_low) else 1
            assert hi + 1e-12 >= normalized_low and lo <= normalized_high + 1e-12, (where, "normalized level_fit", i)
        uniform_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
        feasible = []
        for modal in range(n):
            if intervals[modal][1] >= maximum_low:
                spread_low = sum(lo * abs(i - modal) for i, (lo, _) in enumerate(intervals))
                spread_high = sum(hi * abs(i - modal) for i, (_, hi) in enumerate(intervals))
                feasible.append((max(0, 1 - spread_high / uniform_spread), max(0, 1 - spread_low / uniform_spread)))
        assert any(lo - ROUNDING_HALF_UNIT - 1e-12 <= answer["confidence"] <= hi + ROUNDING_HALF_UNIT + 1e-12
                   for lo, hi in feasible), (where, "confidence formula mismatch", feasible)
    score = number(answer["score"], 0, n - 1, (where, "score"), digits=2)
    within(score, sum(i * lo for i, (lo, _) in enumerate(intervals)),
           sum(i * hi for i, (_, hi) in enumerate(intervals)), 0.005, (where, "score"))


def validate_response(response, label):
    assert isinstance(response, dict) and set(response) == {"model", "usage", "answers"}, (label, "response keys")
    assert response["model"] == "decider-2b-v11", (label, "model")
    usage = response["usage"]
    assert isinstance(usage, dict) and set(usage) == {"input_tokens", "output_tokens"}, (label, "usage keys")
    assert all(type(v) is int and v >= 0 for v in usage.values()) and usage["output_tokens"] == 0, (label, "usage values")
    assert isinstance(response["answers"], dict) and all(isinstance(k, str) for k in response["answers"]), (label, "answer identities")
    for key, answer in response["answers"].items():
        validate_answer(answer, (label, key))


def compare(reference, native):
    validate_response(reference, "reference")
    validate_response(native, "native")
    assert native["usage"] == reference["usage"]
    assert list(native["answers"]) == list(reference["answers"])
    worst_p, worst_score = 0.0, 0.0
    mismatches, low_margin = [], []
    for key, ref in reference["answers"].items():
        got = native["answers"][key]
        assert set(got) == set(ref), (key, set(got), set(ref))
        assert got["type"] == ref["type"]
        if ref["type"] == "noul":
            drift = abs(got["noul"] - ref["noul"])
        else:
            assert list(got["probabilities"]) == list(ref["probabilities"])
            drift = max(abs(got["probabilities"][k] - v) for k, v in ref["probabilities"].items())
            if ref["type"] == "choice":
                probabilities = sorted(ref["probabilities"].values(), reverse=True)
                if ref["choice"] != got["choice"]:
                    (mismatches if probabilities[0] - probabilities[1] >= GATES["discrete_margin"] else low_margin).append(key)
            else:
                assert got["legend"] == ref["legend"]
                worst_score = max(worst_score, abs(got["score"] - ref["score"]))
                drift = max(drift, max(abs(got["level_fit"][k] - v) for k, v in ref["level_fit"].items()))
                n = len(ref["level_fit"])
                mass_bound = n * GATES["max_probability_drift"] + 2 * (n + 1) * ROUNDING_HALF_UNIT
                assert abs(got["fit_mass"] - ref["fit_mass"]) <= mass_bound + 1e-12, ("fit_mass", key, mass_bound)
        worst_p = max(worst_p, drift)
    assert worst_p <= GATES["max_probability_drift"], ("probability", worst_p)
    assert worst_score <= GATES["max_score_drift"], ("score", worst_score)
    assert not mismatches, ("discrete", mismatches)
    return {"max_probability_drift": worst_p, "max_score_drift": worst_score, "low_margin_disagreements": low_margin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, help="maintained additional corpus; replaces default18 cases")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = verify_sources(args.reference / "decider", REFERENCE_FILES)
    cases = workloads() if args.cases is None else [(case["name"],case["request"]) for case in json.loads(args.cases.read_text())]
    assert cases and len({name for name,_ in cases}) == len(cases)
    manifest = {"reference_revision": REFERENCE_REVISION, "reference_source_sha256": sources, "gates": GATES, "validation_protocol": VALIDATION_PROTOCOL, "runs": 1, "cases": [{"name": name, "request": request} for name, request in cases]}
    # Freeze the manifest before executing either implementation.
    (args.output / "protocol.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    inputs = "".join(json.dumps(request, ensure_ascii=False) + "\n" for _, request in cases)
    (args.output / "requests.jsonl").write_text(inputs)
    with (args.output / "native.stderr.log").open("w") as stderr:
        native = subprocess.run([str(args.binary), str(args.model), str(args.library)], input=inputs, capture_output=False,
                                stdout=subprocess.PIPE, stderr=stderr, text=True, check=True)
    (args.output / "native.jsonl").write_text(native.stdout)
    native = [json.loads(line) for line in native.stdout.splitlines()]
    assert len(native) == len(cases)

    # Import only the pinned reference package, with no remote-code execution.
    package = types.ModuleType("decider")
    package.__path__ = [str(args.reference / "decider")]
    sys.modules["decider"] = package
    import torch
    from transformers import AutoTokenizer
    from decider import systemone as s1, prompt_fast, temperature as tt, prompt
    from decider.model import DecisionModel

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    scalar, temperatures = tt.from_config(json.loads((args.model / "decider_config.json").read_text()))[0]
    del scalar
    labels = prompt.label_table(tokenizer)[1]
    model = DecisionModel(str(args.model), dtype=torch.bfloat16, grad_ckpt=False).to("cuda").eval()
    model.lm.model.config.use_cache = False
    results = []
    with (args.output / "reference.jsonl").open("w") as records, torch.inference_mode():
        for (name, request), got in zip(cases, native):
            rendered = {key: s1.render_question(value) for key, value in request["questions"].items()}
            plans, index = s1.plan_rows(rendered, True)
            items, _ = prompt_fast.build_rows(tokenizer, s1.render_state(request["state"]),
                                              [[(plan["question"], plan["options"])] for plan in plans], max_ctx_tokens=32768)
            kinds = s1.row_types(rendered, index)
            assert len(got["rows"]) == len(items), name
            logits, probabilities = [], []
            max_logit_drift = 0.0
            for i, (item, kind) in enumerate(zip(items, kinds)):
                count = item["nopts"][0]
                row = got["rows"][i]
                assert row["ids"] == item["ids"], (name, i, "tokens")
                assert row["candidate_ids"] == labels[:count], (name, i, "candidate IDs")
                assert row["readout_position"] == item["slots"][0] == len(item["ids"]) - 1
                ids = torch.tensor([item["ids"]], device="cuda", dtype=torch.long)
                lg = model.slot_logits(ids, torch.ones_like(ids), torch.tensor([item["slots"][0]], device="cuda"),
                                       torch.tensor([0], device="cuda"), torch.tensor([count], device="cuda"))[0, :count]
                logits.append(lg.cpu().tolist())
                probabilities.append(tt.scaled_softmax(lg[None], [temperatures[kind]])[0].cpu().tolist())
                max_logit_drift = max(max_logit_drift, max(abs(a - b) for a, b in zip(logits[-1], got["logits"][i])))
            response = {"model": "decider-2b-v11", "answers": s1.assemble(rendered, index, probabilities),
                        "usage": {"input_tokens": s1.unique_tokens(items), "output_tokens": 0}}
            records.write(json.dumps({"name": name, "logits": logits, "probabilities": probabilities, "response": response}, ensure_ascii=False) + "\n")
            records.flush()
            result = {"name": name, "rows": len(items), "processed_tokens": sum(len(item["ids"]) for item in items),
                      "max_logit_drift": max_logit_drift, **compare(response, got["response"])}
            results.append(result)
            print(json.dumps(result), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    summary = {"passed": len(results), "cases": results, "gates": GATES, "validation_protocol": VALIDATION_PROTOCOL,
               "torch": torch.__version__, "transformers": __import__("transformers").__version__,
               "gpu": torch.cuda.get_device_name(), "native_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
               "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
