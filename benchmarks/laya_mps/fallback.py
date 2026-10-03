"""Laya's fallback to the CPU on a GPU out-of-memory error, as the worker reports and serves it.

Starts the worker on MPS with PyTorch's MPS memory limit lowered to `--limit-gb`, so that the model
fits but a 64-question request near the window does not; then sends one short request, that large
one, and short requests again, and prints /health before and after. The limit that triggers it
depends on the Mac and the options: on a 16 GB M1 Pro, 3.5 GB without the options and 2.5 GB with
`--compile --weights fp16`. Too low a limit makes the worker exit at startup (`--require-device`).

    python benchmarks/laya_mps/fallback.py --limit-gb 3.5
    python benchmarks/laya_mps/fallback.py --limit-gb 2.5 --flags "--compile --weights fp16"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from bench_http import Client, body_for, wait_ready  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit-gb", type=float, required=True, help="MPS memory limit"
    )
    parser.add_argument("--flags", default="", help="frontend.laya_mps flags")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workloads", default=str(HERE / "workloads.jsonl"))
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()
    with open(args.workloads) as f:
        workloads = {w["id"]: w for w in map(json.loads, filter(str.strip, f))}
    probe = [
        args.python,
        "-c",
        "import torch; print(torch.mps.recommended_max_memory())",
    ]
    recommended = int(subprocess.run(probe, capture_output=True, text=True).stdout)
    ratio = args.limit_gb * 2**30 / recommended
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": f"{ratio:.4f}",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": f"{ratio * 0.9:.4f}",
    }
    command = [
        args.python,
        "-m",
        "frontend.laya_mps",
        "--device",
        "mps",
        "--require-device",
        "--port",
        str(args.port),
        "--log-level",
        "warning",
        *args.flags.split(),
    ]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    log = open(Path(args.out) / "fallback.log", "w")  # noqa: SIM115
    worker = subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO
    )
    try:
        url = f"http://127.0.0.1:{args.port}"
        ready_s = wait_ready(url, {"worker": worker}, 900)[0]
        client = Client(url)

        def health():
            h = json.loads(client.request("GET", "/health", retry=True)[2])
            return {k: h.get(k) for k in ("device", "device_mismatch", "weights_dtype")}

        def decide(workload, questions=None):
            body = body_for(
                {**workload, "questions": questions or workload["questions"]}, "english"
            )
            ms, status, _ = client.request("POST", "/v1/systemone", body, retry=True)
            return status, round(ms)

        print(
            f"MPS limit {args.limit_gb} GB (ratio {ratio:.3f}), ready after {ready_s:.0f} s"
        )
        print("health before:", health())
        print("short request (status, ms):", decide(workloads["W1"]))
        large = {f"q{i}": workloads["W3"]["questions"]["route"] for i in range(64)}
        started = time.monotonic()
        status, _ = decide(workloads["W3"], large)
        print(
            f"64 questions near the window: status {status} after {time.monotonic() - started:.1f} s"
        )
        print("health after:", health())
        print(
            "short requests after (status, ms):",
            [decide(workloads["W1"]) for _ in range(3)],
        )
    finally:
        worker.terminate()
        worker.wait(timeout=60)


if __name__ == "__main__":
    main()
