"""Frozen-plan HTTP matrix client; Python 3.11+, standard library only."""

import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import threading
import time
from urllib.parse import urlsplit, urlunsplit


def decode_json(data):
    def reject_constant(value):
        raise ValueError(f"non-JSON numeric constant: {value}")
    return json.loads(data, parse_constant=reject_constant)


def case_variants(case):
    return case["variants"] if "variants" in case else [case]


def pointer_parent(value, pointer):
    """Resolve a non-root RFC 6901 pointer; array indices must be canonical."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("ignore_paths entries must be non-root JSON Pointers")
    tokens = pointer[1:].split("/")
    for token in tokens:
        if any(i + 1 == len(token) or token[i + 1] not in "01"
               for i, char in enumerate(token) if char == "~"):
            raise ValueError("ignore_paths has an invalid JSON Pointer escape")
    tokens = [token.replace("~1", "/").replace("~0", "~") for token in tokens]
    parent = value
    for index, token in enumerate(tokens):
        if isinstance(parent, dict) and token in parent:
            key = token
        elif isinstance(parent, list) and token.isascii() and token.isdigit():
            # Bound the lexeme before integer conversion, including pathological inputs.
            if len(token) > len(str(len(parent))):
                raise ValueError("ignore_paths array index is out of range")
            key = int(token)
            if str(key) != token or key >= len(parent):
                raise ValueError("ignore_paths array index is invalid")
        else:
            raise ValueError("ignore_paths does not resolve to an existing field")
        if index == len(tokens) - 1:
            return parent, key
        parent = parent[key]


def validate_plan(plan):
    try:
        json.dumps(plan, allow_nan=False)
    except (ValueError, RecursionError) as exc:
        raise ValueError("plan must contain only finite JSON values within serialization limits") from exc
    for key in ("endpoint", "health_endpoint"):
        if key not in plan and key == "health_endpoint":
            continue
        parts = urlsplit(plan[key])
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password or parts.fragment:
            raise ValueError(f"{key} must be an HTTP(S) URL without credentials or fragment")
        _ = parts.port
    cases = plan["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    names = set()
    ignore_paths = plan.get("ignore_paths", [])
    if (not isinstance(ignore_paths, list)
            or any(not isinstance(path, str) for path in ignore_paths)
            or len(set(ignore_paths)) != len(ignore_paths)):
        raise ValueError("ignore_paths must be a list of unique JSON Pointers")
    for case in cases:
        name = case["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("case names must be nonempty and unique")
        names.add(name)
        variants = case_variants(case)
        if "variants" in case and any(key in case for key in ("request", "expected_response", "expected_response_text")):
            raise ValueError("variants replaces the case's request and expectation")
        if not isinstance(variants, list) or not variants:
            raise ValueError("variants must be a nonempty list")
        variant_names = set()
        for variant in variants:
            variant_name = variant["name"]
            if not isinstance(variant_name, str) or not variant_name or variant_name in variant_names:
                raise ValueError("variant names must be nonempty and unique within a case")
            variant_names.add(variant_name)
            if not isinstance(variant["request"], dict):
                raise ValueError("request must be a JSON object")
            if ("expected_response" in variant) == ("expected_response_text" in variant):
                raise ValueError("each variant needs exactly one expected response")
            for pointer in ignore_paths:
                if "expected_response_text" in variant:
                    raise ValueError("ignore_paths cannot be combined with an exact-byte oracle")
                parent, key = pointer_parent(variant["expected_response"], pointer)
                if isinstance(parent[key], (dict, list)):
                    raise ValueError("ignore_paths may exclude only primitive leaf values")
            if "expected_response_text" in variant and not isinstance(variant["expected_response_text"], str):
                raise ValueError("expected_response_text must be a string")
            if "expected_response_text" in variant:
                try:
                    variant["expected_response_text"].encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError("expected_response_text must be valid UTF-8 text") from exc
    count = plan["requests_per_case"]
    concurrency = plan["concurrency"]
    if type(count) is not int or count < 1 or not isinstance(concurrency, list) or not concurrency:
        raise ValueError("requests_per_case and concurrency must be positive")
    if any(type(c) is not int or c < 1 or c > count for c in concurrency) or len(set(concurrency)) != len(concurrency):
        raise ValueError("concurrency values must be unique positive integers <= requests_per_case")
    for name in ("repetitions", "warmup_per_case"):
        value = plan.get(name, 2)
        if type(value) is not int or value < 2:
            raise ValueError(f"{name} must be an integer >= 2")
    timeout = plan["timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not isinstance(plan["metadata"], dict):
        raise ValueError("metadata must be a pinned JSON object")
    for name in (
        "gpu", "gpu_ids", "driver", "cuda", "precision", "model_revision",
        "runtime_revision", "cache_policy", "cuda_evidence", "reservation", "host",
    ):
        value = plan["metadata"].get(name)
        if not value or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"metadata requires {name}")


def exact_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(exact_json(a, b) for a, b in zip(left, right))
    return left == right


def matches_json(actual, expected, ignore_paths):
    if not ignore_paths:
        return exact_json(actual, expected)
    actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
    try:
        for pointer in ignore_paths:
            actual_parent, actual_key = pointer_parent(actual, pointer)
            expected_parent, expected_key = pointer_parent(expected, pointer)
            if type(actual_parent[actual_key]) is not type(expected_parent[expected_key]):
                return False
            if isinstance(actual_parent[actual_key], float) and not math.isfinite(actual_parent[actual_key]):
                return False
            actual_parent[actual_key] = expected_parent[expected_key] = None
    except ValueError:
        return False
    return exact_json(actual, expected)


def exchange(endpoint, timeout, case=None, request_headers=None, ignore_paths=()):
    """One connection, no proxy, redirect, pooling or retry; retain raw bytes."""
    started = time.perf_counter()
    parts = urlsplit(endpoint)
    path = urlunsplit(("", "", parts.path or "/", parts.query, ""))
    connection_type = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parts.hostname, parts.port, timeout=timeout)
    expired = threading.Event()
    timer = None
    payload = bytearray()
    status = None
    headers = {}
    error = None
    decisions = 0
    connect_seconds = None
    try:
        connect_started = time.perf_counter()
        connection.connect()
        connect_seconds = time.perf_counter() - connect_started
        remaining = timeout - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("connection exceeded deadline")
        transport = connection.sock
        transport.settimeout(remaining)

        def expire():
            expired.set()
            try:
                transport.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(remaining, expire)
        timer.daemon = True
        timer.start()
        body = json.dumps(case["request"], allow_nan=False).encode() if case else None
        connection.request("POST" if case else "GET", path, body=body,
                           headers={**(request_headers or {}), **({"Content-Type": "application/json"} if case else {})})
        response = connection.getresponse()
        status = response.status
        headers = dict(response.getheaders())
        while chunk := response.read1(65536):
            payload.extend(chunk)
        if expired.is_set():
            raise TimeoutError("response exceeded deadline")
        if response.length not in (None, 0):
            raise http.client.IncompleteRead(bytes(payload), response.length)
        if status != 200:
            error = {"kind": "http", "message": f"HTTP {status}"}
        elif case:
            try:
                decoded = decode_json(bytes(payload))
            except (ValueError, UnicodeError, RecursionError):
                decoded = None
            if "expected_response_text" in case:
                matches = bytes(payload) == case["expected_response_text"].encode("utf-8")
            else:
                matches = matches_json(decoded, case["expected_response"], ignore_paths)
                # Invalid JSON must fail even when the oracle is JSON null.
                if decoded is None:
                    try:
                        decode_json(bytes(payload))
                    except (ValueError, UnicodeError, RecursionError):
                        matches = False
            if not matches:
                error = {"kind": "correctness", "message": "response differs from frozen expectation"}
            elif isinstance(decoded, dict) and isinstance(decoded.get("answers"), (dict, list)):
                decisions = len(decoded["answers"])
    except RecursionError:
        error = {"kind": "correctness", "message": "response validation exceeds recursion limit"}
    except (OSError, http.client.HTTPException) as exc:
        kind = "timeout" if expired.is_set() or isinstance(exc, TimeoutError) else "transport"
        error = {"kind": kind, "message": f"{type(exc).__name__}: {exc}"}
    finally:
        if timer:
            timer.cancel()
        connection.close()
    completed_at = time.perf_counter()
    return {
        "endpoint": endpoint, "status": status, "headers": headers,
        "raw_response_base64": base64.b64encode(payload).decode("ascii"),
        "raw_response_text": bytes(payload).decode("utf-8", errors="replace"),
        "latency_seconds": completed_at - started, "connect_seconds": connect_seconds,
        "completed_at_monotonic_seconds": completed_at,
        "successful_decisions": decisions, "error": error,
    }


def round_summary(records, elapsed, case, concurrency, repetition, planned):
    successful = [r for r in records if r["error"] is None]
    latencies = sorted(r["latency_seconds"] for r in successful)
    quantiles = {f"p{p}": latencies[math.ceil(p / 100 * len(latencies)) - 1] if latencies else None for p in (50, 95)}
    decisions = sum(r["successful_decisions"] for r in successful)
    complete = len(records) == planned and len(successful) == planned
    steady = {"available": False, "excluded_each_end": concurrency,
              "reason": "round incomplete" if not complete else "requires more than 2 * concurrency completions"}
    if complete and len(records) > 2 * concurrency:
        ordered = sorted(records, key=lambda r: r["completed_at_monotonic_seconds"])
        interior = ordered[concurrency:-concurrency]
        start = ordered[concurrency - 1]["completed_at_monotonic_seconds"]
        end = interior[-1]["completed_at_monotonic_seconds"]
        duration = end - start
        if duration > 0:
            count = sum(r["successful_decisions"] for r in interior)
            steady = {"available": True, "excluded_each_end": concurrency,
                      "requests": len(interior), "decisions": count,
                      "started_at_monotonic_seconds": start, "finished_at_monotonic_seconds": end,
                      "elapsed_seconds": duration, "requests_per_second": len(interior) / duration,
                      "decisions_per_second": count / duration}
        else:
            steady["reason"] = "completion window has no positive elapsed time"
    return {
        "case": case, "concurrency": concurrency, "repetition": repetition,
        "planned_requests": planned, "attempted_requests": len(records),
        "elapsed_seconds": elapsed, "successful_requests": len(successful),
        "successful_decisions": decisions,
        "successful_requests_per_second": len(successful) / elapsed,
        "successful_decisions_per_second": decisions / elapsed,
        "successful_latency_seconds": quantiles,
        "failures": dict(Counter(r["error"]["kind"] for r in records if r["error"])),
        "failure_latency_seconds": [r["latency_seconds"] for r in records if r["error"]],
        "complete": complete, "steady_state": steady,
        "throughput_scope": "complete_wave_including_ramp_and_drain",
    }


def aggregate_rounds(rounds, repetitions):
    groups = {}
    for result in rounds:
        groups.setdefault((result["case"], result["concurrency"]), []).append(result)

    def spread(values):
        if not values:
            return None
        low, median, high = min(values), statistics.median(values), max(values)
        return {"rounds": len(values), "min": low, "median": median, "max": high,
                "relative_spread": (high - low) / median if median else (0.0 if high == low else None)}

    aggregated = []
    for (case, concurrency), results in groups.items():
        complete = [r for r in results if r["complete"]]
        windows = [r["steady_state"] for r in complete if r["steady_state"]["available"]]
        metrics = {key: spread([r[key] for r in complete]) for key in (
            "successful_requests_per_second", "successful_decisions_per_second")}
        for p in ("p50", "p95"):
            metrics[p + "_seconds"] = spread([r["successful_latency_seconds"][p] for r in complete])
        for kind in ("requests", "decisions"):
            metrics["steady_state_" + kind + "_per_second"] = spread([w[kind + "_per_second"] for w in windows])
        aggregated.append({"case": case, "concurrency": concurrency,
                           "planned_rounds": repetitions, "attempted_rounds": len(results),
                           "complete_rounds": len(complete), "incomplete_rounds": len(results) - len(complete),
                           "metrics": metrics})
    return aggregated


def run(plan_path, output):
    snapshot = plan_path.read_bytes()
    plan = decode_json(snapshot)
    validate_plan(plan)
    request_headers = {}
    if token := os.environ.get("OMNI_JEV_TEST_TOKEN"):
        if any(ord(c) < 32 or ord(c) > 126 for c in token):
            raise ValueError("invalid OMNI_JEV_TEST_TOKEN: expected printable ASCII")
        request_headers["Authorization"] = f"Bearer {token}"
    output.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    (output / "plan.json").write_bytes(snapshot)
    repetitions, warmup = plan.get("repetitions", 2), plan.get("warmup_per_case", 2)
    config = {
        "plan_sha256": hashlib.sha256(snapshot).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan": plan, "timing": "client wall time including DNS/TCP/TLS, request serialization and response validation",
        "retries": 0, "redirects": False, "environment_proxy": False,
        "effective_budget": {"repetitions": repetitions, "warmup_per_case": warmup},
        "started_at": datetime.now(timezone.utc).isoformat(), "finished_at": None,
        "client": {"hostname": socket.gethostname(), "platform": platform.platform(),
                   "python_version": platform.python_version(), "cpu_count": os.cpu_count()},
        "connection_policy": "new_connection_per_request",
    }
    save("config.json", config)
    parts = urlsplit(plan["endpoint"])
    health_endpoint = plan.get("health_endpoint", urlunsplit((parts.scheme, parts.netloc, "/health", "", "")))
    health = exchange(health_endpoint, plan["timeout_seconds"], request_headers=request_headers)
    save("health.json", health)
    all_records = []
    rounds = []
    stopped = threading.Event()
    if health["error"]:
        stopped.set()
    lock = threading.Lock()
    with (output / "responses.jsonl").open("w") as sink:
        def request(case, phase, index, concurrency=1, repetition=None, variant_index=None):
            variants = case_variants(case)
            if variant_index is None:
                variant_index = index % len(variants)
            variant = variants[variant_index]
            record = exchange(plan["endpoint"], plan["timeout_seconds"], variant, request_headers,
                              plan.get("ignore_paths", []))
            record.update(case=case["name"], phase=phase, index=index,
                          concurrency=concurrency, repetition=repetition,
                          variant=variant["name"], variant_index=variant_index)
            with lock:
                if record["error"]:
                    stopped.set()
                all_records.append(record)
                sink.write(json.dumps(record, allow_nan=False) + "\n")
                sink.flush()
            return record

        for case in plan["cases"]:
            if stopped.is_set():
                break
            for variant_index in range(len(case_variants(case))):
                request(case, "readiness", 0, variant_index=variant_index)
                if stopped.is_set():
                    break
        for case in plan["cases"]:
            for variant_index in range(len(case_variants(case))):
                for index in range(warmup):
                    if stopped.is_set():
                        break
                    request(case, "warmup", index, variant_index=variant_index)

        def wave(case, phase, concurrency, count, repetition=None):
            next_index = 0
            completed = []

            def worker():
                nonlocal next_index
                while True:
                    with lock:
                        if stopped.is_set() or next_index == count:
                            return
                        index = next_index
                        next_index += 1
                    record = request(case, phase, index, concurrency, repetition)
                    with lock:
                        completed.append(record)

            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(worker) for _ in range(concurrency)]
                for future in futures:
                    future.result()
            return completed, time.perf_counter() - start

        for case in plan["cases"]:
            for concurrency in plan["concurrency"]:
                if stopped.is_set():
                    break
                wave(case, "feasibility", concurrency, min(concurrency, plan["requests_per_case"]))
                for repetition in range(1, repetitions + 1):
                    if stopped.is_set():
                        break
                    records, elapsed = wave(case, "measured", concurrency, plan["requests_per_case"], repetition)
                    rounds.append(round_summary(records, elapsed, case["name"], concurrency, repetition, plan["requests_per_case"]))
    failures = Counter(r["error"]["kind"] for r in [health, *all_records] if r["error"])
    summary = {
        "complete": not bool(failures), "failures": dict(failures),
        "inference_requests": len(all_records), "health_requests": 1,
        "rounds": rounds,
        "round_aggregates": aggregate_rounds(rounds, repetitions),
        "excluded_phases": ["health", "readiness", "warmup", "feasibility"],
    }
    save("summary.json", summary)
    config["finished_at"] = datetime.now(timezone.utc).isoformat()
    save("config.json", config)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run(args.plan, args.output)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
