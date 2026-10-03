"""A checkpoint Laya loads while the worker serves: how long its first request takes.

Starts the worker on MPS with one checkpoint (`--model`), optionally the Rust frontend in front of
it, then asks for another checkpoint (`--late`), directly and, with `--frontend`, through the
frontend (a fresh worker for each). The first run also downloads the late checkpoint; run once
beforehand to leave the download out. One JSON line per attempt:

- `through`, `flags`, `late`: what was run;
- `first`: status and time of the late request (504 from the frontend, or a client timeout, can
  still be a load that succeeds in the worker);
- `prepared_after_s`: when the worker stopped preparing the checkpoint;
- `loaded`: whether the late checkpoint is resident afterwards; false means the load failed and the
  worker unloaded it;
- `next`: the same request again, sent only when `loaded`, otherwise null (it would load again);
- `models`: device, revision and warmup time of each resident checkpoint.

    python benchmarks/laya_mps/late_load.py --flags "--compile --weights fp16" \\
        --frontend target/release/omni-jev
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_http import Client, wait_ready  # noqa: E402
from paired import spawn, stop  # noqa: E402

QUESTION = {
    "q": {"type": "noul", "instructions": "Does the customer ask for a refund?"}
}


def attempt(args, through_frontend):
    port, front = args.port, args.port + 1
    log = (
        Path(args.out) / f"late_load_{'frontend' if through_frontend else 'direct'}.log"
    )
    procs = [spawn(args.flags, port, args.python, args.model, log)]
    try:
        url = f"http://127.0.0.1:{port}"
        wait_ready(url, {"worker": procs[0]}, args.ready_timeout)
        if through_frontend:
            env = {
                **os.environ,
                "OMNI_JEV_BIND": f"127.0.0.1:{front}",
                "OMNI_JEV_BACKEND_URL": url,
            }
            procs.append(
                subprocess.Popen(
                    [args.frontend],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            )
            wait_ready(f"http://127.0.0.1:{front}", {"frontend": procs[1]}, 60)
        client = Client(
            f"http://127.0.0.1:{front if through_frontend else port}",
            timeout=args.timeout,
        )
        body = json.dumps(
            {
                "model": args.late,
                "state": "Please refund the duplicate charge.",
                "questions": QUESTION,
            }
        ).encode()
        worker = Client(url)

        def health():
            return json.loads(worker.request("GET", "/health", retry=True)[2])

        resident = set(health()["models"])
        started = time.monotonic()
        try:
            first_ms, first_status, _ = client.request("POST", "/v1/systemone", body)
        except OSError as exc:
            first_ms, first_status = (time.monotonic() - started) * 1000, repr(exc)
        while health()["preparing"]:
            time.sleep(0.2)
        prepared_s = time.monotonic() - started
        # A slow load can end in the frontend's 504 or a client timeout and still succeed; a failed one is
        # unloaded, and a next request would only load it again.
        loaded = bool(set(health()["models"]) - resident)
        following = None
        if loaded:
            next_ms, next_status, _ = client.request(
                "POST", "/v1/systemone", body, retry=True
            )
            following = {"status": next_status, "ms": round(next_ms, 1)}
        final = health()
        return {
            "through": "frontend" if through_frontend else "direct",
            "flags": args.flags,
            "late": args.late,
            "first": {"status": first_status, "ms": round(first_ms)},
            "prepared_after_s": round(prepared_s, 1),
            "loaded": loaded,
            "next": following,
            "models": {
                name: {k: m.get(k) for k in ("device", "revision", "warmup_ms")}
                for name, m in final["models"].items()
            },
        }
    finally:
        stop(procs)


def parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flags", default="", help="frontend.laya_mps flags")
    parser.add_argument(
        "--model", default="english", help="the checkpoint loaded at startup"
    )
    parser.add_argument(
        "--late", default="multilingual", help="the checkpoint asked for later"
    )
    parser.add_argument(
        "--frontend", help="Rust frontend binary: also time the request through it"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="worker port; the frontend uses the next"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="seconds to wait for the late request; the first run also downloads the checkpoint",
    )
    return parser


def main():
    from laya.router import normalise_name

    cli = parser()
    args = cli.parse_args()
    if normalise_name(args.late) == normalise_name(args.model):
        cli.error(f"--late {args.late} is the checkpoint the worker starts with")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for through_frontend in (False, True) if args.frontend else (False,):
        print(json.dumps(attempt(args, through_frontend)), flush=True)


if __name__ == "__main__":
    main()
