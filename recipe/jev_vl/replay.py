#!/usr/bin/env python3
"""Serial JEV-VL replay: HTTP + JSON decode timing, fixed 0.025 probability gate.

Request serialization and reference comparison are outside client_s. Warmup is
repeated before every pass, recorded, and excluded from the latency summary.
This client does not start a server, reset caches, or establish GPU execution.
"""

import argparse
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

TOLERANCE = 0.025
FIELDS = ("kind", "effective_kind", "options", "choice_index", "choice")


def invalid_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def finite_float(value):
    parsed = float(value)
    return parsed if math.isfinite(parsed) else invalid_constant(value)


def loads(raw):
    return json.loads(raw, parse_constant=invalid_constant, parse_float=finite_float)


def probabilities(body):
    values, options, index = body["probabilities"], body["options"], body["choice_index"]
    if not isinstance(options, list) or not isinstance(values, list) or not values or len(values) != len(options):
        raise ValueError("probability/option length mismatch")
    if any(type(p) not in (int, float) or not 0 <= p <= 1 or not math.isfinite(p) for p in values):
        raise ValueError("invalid probability")
    if abs(sum(values) - 1) > 1e-4:
        raise ValueError("probabilities do not sum to one")
    if type(index) is not int or index != max(range(len(values)), key=values.__getitem__):
        raise ValueError("choice_index differs from probability argmax")
    if body["choice"] != options[index]:
        raise ValueError("choice differs from the selected option")
    return values


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def post(opener, endpoint, payload):
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    request = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
    row = {"status": None, "ok": False, "max_diff": None}
    start = time.monotonic()
    try:
        try:
            response = opener.open(request, timeout=300)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            row["status"] = response.status
            row["raw_body"] = response.read().decode("utf-8")
            row["body"] = loads(row["raw_body"])
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException) as error:
        row["error"] = str(error)
    row["client_s"] = time.monotonic() - start
    return row


def compare(row, reference):
    try:
        if row["status"] != 200 or "error" in row:
            raise ValueError(row.get("error", f"HTTP {row['status']}"))
        body = row["body"]
        current, expected = probabilities(body), probabilities(reference)
        row["probabilities"] = current
        if len(current) != len(expected):
            raise ValueError("reference probability length mismatch")
        row["max_diff"] = max(abs(a - b) for a, b in zip(current, expected))
        if any(body[field] != reference[field] for field in FIELDS):
            raise ValueError("decision mismatch")
        if row["max_diff"] > TOLERANCE:
            raise ValueError("probability tolerance exceeded")
        row["ok"] = True
    except (KeyError, TypeError, ValueError, IndexError) as error:
        row["error"] = str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "manifest", "reference", "out"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--pattern", default="", help="manifest ID prefix (default: all)")
    parser.add_argument("--warmup", type=int, default=4, help="requests before each pass")
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()
    if args.warmup < 0 or args.passes < 1:
        parser.error("warmup must be nonnegative and passes must be positive")
    raw = Path(args.manifest).read_bytes()
    entries = [loads(line) for line in raw.splitlines() if line.strip()]
    ids = [entry["id"] for entry in entries]
    if not entries or len(ids) != len(set(ids)) or any(not re.fullmatch(r"[\w.-]+", key) for key in ids):
        raise ValueError("manifest must have nonempty, unique, safe IDs")
    entries = [entry for entry in entries if entry["id"].startswith(args.pattern)]
    if not entries:
        raise ValueError("pattern selects no requests")
    references, hashes = {}, {}
    for entry in entries:
        if not isinstance(entry["request"], dict):
            raise ValueError("request must be an object")
        content = (Path(args.reference) / f"{entry['id']}.json").read_bytes()
        ref = loads(content)
        if ref["id"] != entry["id"] or ref["status"] != 200:
            raise ValueError("reference must match the ID and have status 200")
        probabilities(ref["body"])
        references[entry["id"]] = ref["body"]
        hashes[entry["id"]] = hashlib.sha256(content).hexdigest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {"tolerance": TOLERANCE, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                           "reference_sha256": hashes, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    opener = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))
    rows = []
    with (out / "results.jsonl").open("w") as stream:
        for iteration in range(args.passes):
            sequence = [("warmup", entries[i % len(entries)]) for i in range(args.warmup)]
            sequence += [("measured", entry) for entry in entries]
            for phase, entry in sequence:
                row = post(opener, args.base.rstrip("/") + "/v1/systemone", entry["request"])
                row.update(id=entry["id"], phase=phase, pass_index=iteration)
                compare(row, references[entry["id"]])
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if phase == "warmup" and not row["ok"]:
                    break
            if phase == "warmup" and not row["ok"]:
                break
    measured = [row for row in rows if row["phase"] == "measured"]
    latencies = [row["client_s"] for row in measured if row["ok"]]
    diffs = [row["max_diff"] for row in measured if row["max_diff"] is not None]
    summary = {"ok": bool(measured) and all(row["ok"] for row in rows), "concurrency": 1,
               "measured": len(measured), "warmup": len(rows) - len(measured),
               "failures": sum(not row["ok"] for row in rows), "max_diff": max(diffs, default=None),
               "mean_s": statistics.fmean(latencies) if latencies else None,
               "median_s": statistics.median(latencies) if latencies else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, TypeError, ValueError) as error:
        sys.exit(f"replay failed: {error}")
