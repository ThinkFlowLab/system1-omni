#!/usr/bin/env python3
"""Sequential matched diagnostic-runner comparisons; raw output and frozen plan retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from verify_reference import GATES, VALIDATION_PROTOCOL, compare


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan["repetitions"] == 2
    parity = Path(plan["parity_output"])
    protocol = json.loads((parity / "protocol.json").read_text())
    reference = [json.loads(line) for line in (parity / "reference.jsonl").read_text().splitlines()]
    baseline = [json.loads(line) for line in (parity / "native.jsonl").read_text().splitlines()]
    cases = protocol["cases"]
    assert len(cases) == len(reference) == len(baseline)
    selected = [i for i, case in enumerate(cases) if case["name"] in plan["timed_cases"]]
    assert len(selected) == len(plan["timed_cases"])
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = {**plan, "gates": GATES, "validation_protocol": VALIDATION_PROTOCOL, "baseline_protocol": protocol,
              "input_hashes": {name: digest(parity / name) for name in ("protocol.json", "reference.jsonl", "native.jsonl")},
              "runner_sha256": digest(__file__), "comparison_sha256": digest(Path(__file__).with_name("verify_reference.py")), "library_sha256": digest(plan["library"]),
              "binary_hashes": {mode["name"]: digest(mode["binary"]) for mode in plan["modes"]},
              "timing": "wall roundtrip: request JSON encoding/write through diagnostic stdout JSON decode; includes CPU preparation, GPU synchronization and serialization; excludes startup and one workload warmup; concurrency=1"}
    (args.output / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n")
    summaries = []
    control_records = {}
    for mode in plan["modes"]:
        directory = args.output / mode["name"]
        directory.mkdir()
        env = dict(os.environ)
        for key in ("CUA_S1_GRAPH", "DECIDER_GRAPH", "DECIDER_PREFIX", "DECIDER_FIXED", "DECIDER_BATCH_MAX_ROWS", "DECIDER_BATCH_MAX_TOKENS"):
            env.pop(key, None)
        env.update(mode["env"])
        with (directory / "stderr.log").open("w") as stderr, (directory / "records.jsonl").open("w") as records:
            start = time.perf_counter()
            child = subprocess.Popen([mode["binary"], plan["model"], plan["library"]], env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True)
            def run(request):
                begun = time.perf_counter()
                child.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                child.stdin.flush()
                line = child.stdout.readline()
                if not line:
                    raise RuntimeError(f"{mode['name']} terminated: {child.poll()}; see stderr.log")
                value = json.loads(line)
                return value, (time.perf_counter() - begun) * 1000
            try:
                run({"state": "", "questions": {}})
                startup_ms = (time.perf_counter() - start) * 1000
                checks, samples = [], []
                for i, case in enumerate(cases):
                    got, ms = run(case["request"])
                    records.write(json.dumps({"phase": "parity", "name": case["name"], "elapsed_ms": ms, "record": got}, ensure_ascii=False) + "\n")
                    records.flush()
                    assert got["rows"] == baseline[i]["rows"], (mode["name"], case["name"], "prepared rows")
                    if mode.get("exact_control"):
                        control = control_records[mode["exact_control"]][case["name"]]
                        assert got["logits"] == control["logits"] and got["response"] == control["response"], (mode["name"], case["name"], "fixed/shared exact equality")
                    control_records.setdefault(mode["name"], {})[case["name"]] = got
                    checks.append({"name": case["name"], **compare(reference[i]["response"], got["response"])})
                for i in selected:
                    got, ms = run(cases[i]["request"])
                    records.write(json.dumps({"phase": "warmup", "name": cases[i]["name"], "elapsed_ms": ms, "record": got}, ensure_ascii=False) + "\n")
                for repetition in range(plan["repetitions"]):
                    for i in selected:
                        got, ms = run(cases[i]["request"])
                        sample = {"name": cases[i]["name"], "repetition": repetition, "elapsed_ms": ms}
                        samples.append(sample)
                        records.write(json.dumps({"phase": "measured", **sample, "record": got}, ensure_ascii=False) + "\n")
                        records.flush()
                        assert got["rows"] == baseline[i]["rows"], (mode["name"], cases[i]["name"], "measured rows")
                        if mode.get("exact_control"):
                            control = control_records[mode["exact_control"]][cases[i]["name"]]
                            assert got["logits"] == control["logits"] and got["response"] == control["response"], (mode["name"], cases[i]["name"], "measured fixed/shared equality")
                        compare(reference[i]["response"], got["response"])
                if mode.get("require_shared"):
                    assert got["prefix"]["shared_requests"] > 0 and got["prefix"]["saved_tokens"] > 0
                if mode.get("require_replays"):
                    assert got["graph"]["enabled"] and got["graph"]["captures"] > 0
                    assert got["graph"]["replays"] >= len(samples)
                    assert got["graph"]["fallbacks"] == 0
                memory = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()
                summary = {"name": mode["name"], "startup_to_empty_diagnostic_ms": startup_ms,
                           "checks": checks, "samples": samples, "device_used_mib_at_end": int(memory),
                           "warm_ms": {name: {"median": statistics.median(values), "min": min(values), "max": max(values)}
                                       for name in plan["timed_cases"]
                                       for values in [[s["elapsed_ms"] for s in samples if s["name"] == name]]}}
                summaries.append(summary)
                (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps(summary), flush=True)
            finally:
                child.stdin.close()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
    (args.output / "summary.json").write_text(json.dumps({"modes": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
