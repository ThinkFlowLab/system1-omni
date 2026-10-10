#!/usr/bin/env python3
"""Fault-injection check using an isolated real worker after its genuine warmup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from verify_http import request, wait_ready


def memory(pid):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        fields = line.split(",")
        if fields[0].strip() == str(pid):
            return int(fields[1].strip())
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "worker.log"
    environment = os.environ.copy()
    environment.update(DECIDER_MODEL=str(args.model), DECIDER_CUDA_LIB=str(args.wrapper), DECIDER_PORT="18112", CUA_S1_GRAPH="0")
    with log.open("w") as output:
        child = subprocess.Popen([str(args.worker)], env=environment, stdout=output, stderr=subprocess.STDOUT)
        try:
            base = "http://127.0.0.1:18112"
            health = wait_ready(base, child, log)
            assert health["status"] == "ready"
            before = memory(child.pid)
            raw = b'{"state":"The item is damaged.","questions":{"q":{"instructions":"Choose a category.","criteria":["refund","shipping"]}}}'
            status, body = request(base, "/v1/systemone", raw)
            assert status == 503 and json.loads(body)["detail"] == "model inference failed"
            status, body = request(base, "/health")
            assert status == 503 and json.loads(body)["status"] == "unavailable"
            for _ in range(2):
                status, body = request(base, "/v1/systemone", raw)
                assert status == 503 and json.loads(body)["detail"] == "model unavailable"
            after = memory(child.pid)
            # The checksum-pinned 3.76GB model allocation must actually be retired.
            assert before - after > 3000, (before, after)
            text = log.read_text()
            assert text.count("Decider test projection") == 2
            assert "Decider test projection 2" in text
            result = {"passed": True, "projection_calls": 2, "health_after": "unavailable", "response_status": 503,
                      "sampled_before_mib": before, "sampled_after_mib": after, "later_requests_refused": 2}
            (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            child.terminate()
            child.wait(timeout=30)


if __name__ == "__main__":
    main()
