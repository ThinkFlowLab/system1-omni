#!/usr/bin/env python3
"""One sequential pass over a frozen JEMM corpus; no retries or timing claims."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

def finite(v):
    return type(v) in (float, int) and math.isfinite(v)


def compare(reference, actual):
    failures, rows = [], []
    if actual.get("model") != "JEMM" or reference.get("model") != "JEMM":
        failures.append("model identity")
    for k in ("input_tokens", "output_tokens"):
        a, r = actual.get("usage", {}).get(k), reference.get("usage", {}).get(k)
        if type(a) is not int or a < 0 or a != r:
            failures.append("usage." + k)
    latency = actual.get("usage", {}).get("latency_ms")
    if not finite(latency) or latency <= 0:
        failures.append("native latency must be finite and positive")
    ra, aa = reference["answers"], actual.get("answers", {})
    if list(ra) != list(aa):
        failures.append("question identity/order")
    for qid, r in ra.items():
        a = aa.get(qid, {})
        problem = []
        if a.get("type") != r["type"] or not set(r) <= set(a):
            problem.append("answer type/required fields")
        rp, ap = r["probabilities"], a.get("probabilities", {})
        if list(rp) != list(ap):
            problem.append("candidate identity/order")
        valid = all(finite(v) and 0 <= v <= 1 for v in list(rp.values()) + list(ap.values()))
        if not valid or not ap or abs(sum(ap.values()) - 1) > 1e-6:
            problem.append("finite normalized probabilities")
        drift = max((abs(v - ap[k]) for k, v in rp.items() if k in ap and finite(ap[k])), default=math.inf)
        if drift > 0.02 + 1e-12:
            problem.append("probability drift")
        ordered = sorted(rp, key=rp.get, reverse=True)
        margin = rp[ordered[0]] - rp[ordered[1]]
        winner = max(ap, key=ap.get) if valid and ap else None
        if margin >= 0.05 and winner != ordered[0]:
            problem.append("winner disagreement")
        if r["type"] == "choice" and a.get("choice") != winner:
            problem.append("choice disagrees with own probabilities")
        for k, gate in (("confidence", 0.02), ("noul", 0.02), ("expected_value", 0.1)):
            if k in r and (not finite(a.get(k)) or abs(r[k] - a[k]) > gate + 1e-12):
                problem.append(k + " drift")
        rows.append({"question": qid, "probability_drift": drift, "reference_margin": margin,
                     "low_margin": margin < 0.05, "failures": problem})
        failures.extend(qid + ": " + p for p in problem)
    return {"passed": not failures, "failures": failures, "questions": rows}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="running worker URL on a reserved GPU")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    requests = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    references = json.loads(args.reference.read_text())
    assert len(requests) == len(references) > 0
    assert [x["id"] for x in requests] == [x["id"] for x in references]
    url = args.url.rstrip("/")
    with urlopen(url + "/health", timeout=5) as response:
        assert json.load(response)["status"] == "READY"
    records, responses = [], []
    for item, reference in zip(requests, references):
        request = Request(url + "/v1/systemone",
                          data=json.dumps(item["request"], ensure_ascii=False).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=120) as response:
            actual = json.load(response)
        record = {"id": item["id"], **compare(reference["response"], actual)}
        records.append(record)
        responses.append({"id": item["id"], "response": actual})
        print(json.dumps(record), flush=True)
    with urlopen(url + "/health", timeout=5) as response:
        ready = json.load(response)["status"] == "READY"
    report = {"passed": ready and all(x["passed"] for x in records),
              "requests": len(requests), "questions": sum(len(x["questions"]) for x in records),
              "gates": {"probability": 0.02, "score": 0.1, "winner_margin": 0.05},
              "max_probability_drift": max(q["probability_drift"] for x in records for q in x["questions"]),
              "records": records, "responses": responses,
              "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
              "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "performance_measurement": False, "retries": 0}
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
