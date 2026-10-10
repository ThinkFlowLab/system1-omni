#!/usr/bin/env python3
"""Exercise real native and frontend sockets, readiness, rejection and recovery."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


def request(base, path, body=None, content_type="application/json"):
    headers = {"Content-Type": content_type} if body is not None else {}
    req = urllib.request.Request(base + path, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as response:
        return response.code, response.read()


def wait_ready(base, child, log):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f"worker exited: {log.read_text()[-3000:]}")
        try:
            status, raw = request(base, "/health")
            if status == 200:
                return json.loads(raw)
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise TimeoutError("readiness timeout")


def concurrency_request(cases):
    if not cases:
        raise ValueError("empty HTTP corpus")
    for case in cases:
        if case["name"] == "mixed" and case["request"]["questions"]:
            return case["request"]
    return next((case["request"] for case in cases if case["request"]["questions"]), cases[0]["request"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--parity-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution", default="eager")
    parser.add_argument("--native-records", type=Path, help="verify_modes records.jsonl for the selected mode")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    worker_base, frontend_base = "http://127.0.0.1:18110", "http://127.0.0.1:18111"
    environment = os.environ.copy()
    environment.update(DECIDER_MODEL=str(args.model), DECIDER_CUDA_LIB=str(args.library), DECIDER_PORT="18110", CUA_S1_GRAPH="0")
    processes, logs, checks = [], [], []
    try:
        path = args.output / "worker.log"
        logs.append(path.open("w"))
        worker = subprocess.Popen([str(args.worker)], env=environment, stdout=logs[-1], stderr=subprocess.STDOUT)
        processes.append(worker)
        metadata = wait_ready(worker_base, worker, path)
        assert metadata["checkpoint_revision"] == "533964dae8be954c5b5e19fa4948e48408094c1e"
        assert metadata["execution"] == args.execution and metadata["dtype"] == "bfloat16"
        environment.update(OMNI_JEV_BIND="127.0.0.1:18111", OMNI_JEV_BACKEND_URL=worker_base)
        path = args.output / "frontend.log"
        logs.append(path.open("w"))
        frontend = subprocess.Popen([str(args.frontend)], env=environment, stdout=logs[-1], stderr=subprocess.STDOUT)
        processes.append(frontend)
        wait_ready(frontend_base, frontend, path)
        protocol = json.loads((args.parity_output / "protocol.json").read_text())
        native_records = [json.loads(line) for line in (args.parity_output / "native.jsonl").read_text().splitlines()]
        if args.native_records:
            native_records = [item["record"] for line in args.native_records.read_text().splitlines()
                              for item in [json.loads(line)] if item["phase"] == "parity"]
        assert len(native_records) == len(protocol["cases"])
        for case, native in zip(protocol["cases"], native_records):
            body = json.dumps(case["request"], ensure_ascii=False).encode()
            direct, proxy = request(worker_base, "/v1/systemone", body), request(frontend_base, "/v1/systemone", body)
            assert direct[0] == proxy[0] == 200
            assert direct[1] == proxy[1]
            assert json.loads(direct[1]) == native["response"]
            checks.append({"name": case["name"], "status": direct[0], "byte_parity": True, "response": json.loads(direct[1])})
        negative = [
            (b"{}", "text/plain", 415),
            (b"{invalid", "application/json", 422),
            (b'{"state":0,"state":1,"questions":{}}', "application/json", 422),
            (b'{"state":0,"questions":{"q":{"instructions":"q","criteria":["one"]}}}', "application/json", 422),
            (b'{"state":0,"questions":{},"independent":false}', "application/json", 422),
            (b'{"state":0,"questions":{"q":{"type":"image","instructions":"q","criteria":["a","b"]}}}', "application/json", 422),
        ]
        for i, (body, content_type, expected) in enumerate(negative):
            direct = request(worker_base, "/v1/systemone", body, content_type)
            proxy = request(frontend_base, "/v1/systemone", body, content_type)
            assert direct[0] == expected and proxy == direct, (i, direct[0], proxy[0], expected)
            assert isinstance(json.loads(direct[1])["detail"], str)
            checks.append({"name": f"invalid_{i}", "status": expected, "byte_parity": True})
        oversized = request(worker_base, "/v1/systemone", b" " * (8 * 1024 * 1024 + 1))
        assert oversized[0] == 413
        checks.append({"name": "body_limit", "status": 413})
        raw = json.dumps(concurrency_request(protocol["cases"])).encode()
        baseline = request(worker_base, "/v1/systemone", raw)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            concurrent_results = list(pool.map(lambda i: request(frontend_base if i % 2 else worker_base, "/v1/systemone", raw), range(12)))
        assert all(result == baseline for result in concurrent_results)
        checks.append({"name": "concurrent_direct_and_proxy", "requests": len(concurrent_results), "byte_parity": True})
        status, health = request(worker_base, "/health")
        assert status == 200 and json.loads(health)["status"] == "ready"
        checks.append({"name": "recovery_health", "status": status})
        (args.output / "summary.json").write_text(json.dumps({"passed": len(checks), "metadata": metadata, "checks": checks}, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"passed": len(checks), "concurrent_requests": len(concurrent_results)}), flush=True)
    finally:
        for child in reversed(processes):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in logs:
            handle.close()


if __name__ == "__main__":
    main()
