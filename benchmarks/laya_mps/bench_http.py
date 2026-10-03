"""HTTP benchmark against a /v1/systemone worker (C3 = laya-serve, C4 = laya-serve behind the Rust frontend).

With --spawn the script starts the worker itself and times process start to the first successful
/health, then the first request per workload after that. Without it, it attaches to --url. With
--frontend as well, the worker listens on --backend-port and the Rust frontend binary is started on
--url in front of it; readiness is then the frontend's /health, which proxies the worker's. Each
workload runs at every --concurrency level; each client thread keeps one keep-alive connection.

    python benchmarks/laya_mps/bench_http.py --config C3 --run m1 --spawn .venv/bin/laya-serve
    python benchmarks/laya_mps/bench_http.py --config C4 --run m1 --url http://127.0.0.1:8080 \
        --frontend target/release/omni-jev --spawn .venv/bin/laya-serve
    python benchmarks/laya_mps/bench_http.py --config C3o --run m1 \
        --spawn .venv/bin/python -m frontend.laya_mps --device {device} --model {model} \
        --compile --weights fp16 --port {port}

The spawned command gets LAYA_HOST/LAYA_PORT/LAYA_DEVICE/LAYA_MODELS in its environment (what laya-serve
reads) and PYTHONPATH=src (for `-m frontend.laya_mps`). `{port}`, `{device}` and `{model}` in its arguments
are replaced by the port and by --device and --model, which is how `frontend.laya_mps` gets them: it takes
flags and does not read those variables.
"""

import argparse
import http.client
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from env import footprint_mb, header, read_workloads, refuse_if_noisy  # noqa: E402

CHECKPOINT = "convaiinnovations/laya"  # what laya-serve's "english" model resolves to (laya/router.py)


class Client:
    """One keep-alive connection. Not thread-safe: one per thread."""

    def __init__(self, url, token=None, timeout=120):
        parts = urlsplit(url)
        self.conn = http.client.HTTPConnection(
            parts.hostname, parts.port or 80, timeout=timeout
        )
        self.headers = {"Content-Type": "application/json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def request(self, method, path, body=None, retry=False):
        """Timed calls pass retry=False so a dropped connection shows up as an error, not a slow request.
        Untimed calls retry once: uvicorn closes keep-alive connections idle for 5 s. Any failure closes
        the connection, so the next call reconnects instead of failing on a half-finished exchange."""
        try:
            started = time.perf_counter()
            self.conn.request(method, path, body=body, headers=self.headers)
            response = self.conn.getresponse()
            data = response.read()
            return (time.perf_counter() - started) * 1000, response.status, data
        except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
            self.conn.close()
            if not retry:
                raise
            return self.request(method, path, body)
        except BaseException:
            self.conn.close()
            raise

    def close(self):
        self.conn.close()


def wait_ready(url, processes, timeout_s):
    """Poll /health every 50 ms; return seconds from now until it answers 200, and the body."""
    started = time.perf_counter()
    while time.perf_counter() - started < timeout_s:
        for name, process in processes.items():
            if process.poll() is not None:
                sys.exit(f"{name} exited with {process.returncode} before it was ready")
        client = Client(url)
        try:
            _, status, body = client.request("GET", "/health")
            if status == 200:
                return time.perf_counter() - started, json.loads(body)
        except (OSError, http.client.HTTPException):
            pass
        finally:
            client.close()
        time.sleep(0.05)
    sys.exit(f"worker not ready after {timeout_s} s")


def fetch_answers(client, body):
    """(answers, None) or (None, error record fields) for one untimed request."""
    try:
        _, status, data = client.request("POST", "/v1/systemone", body, retry=True)
    except (OSError, http.client.HTTPException) as exc:
        return None, {"status": 0, "detail": repr(exc)}
    if status != 200:
        return None, {"status": status, "detail": data[:200].decode(errors="replace")}
    try:
        return json.loads(data)["answers"], None
    except (ValueError, KeyError) as exc:
        return None, {"status": status, "detail": f"no answers in response: {exc!r}"}


def body_for(workload, model):
    return json.dumps(
        {"model": model, "state": workload["state"], "questions": workload["questions"]}
    ).encode()


def run_level(url, token, body, n, concurrency):
    """n requests split over `concurrency` threads. Returns [(thread, ms, status)], elapsed seconds."""
    per_thread = [n // concurrency + (i < n % concurrency) for i in range(concurrency)]
    results, lock = [], threading.Lock()
    # a thread that fails to connect breaks it
    barrier = threading.Barrier(concurrency + 1, timeout=120)

    def worker(index, count):
        client = Client(url, token)
        # connect outside the timed window
        client.request("POST", "/v1/systemone", body, retry=True)
        barrier.wait()
        mine = []
        for _ in range(count):
            try:
                ms, status, _ = client.request("POST", "/v1/systemone", body)
            except (OSError, http.client.HTTPException):
                ms, status = None, 0  # counted as an error, excluded from latency
            mine.append((index, ms, status))
        client.close()
        with lock:
            results.extend(mine)

    threads = [
        threading.Thread(target=worker, args=(i, c)) for i, c in enumerate(per_thread)
    ]
    for t in threads:
        t.start()
    barrier.wait()
    started = time.perf_counter()
    for t in threads:
        t.join()
    return results, time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="label, e.g. C3 or C4")
    parser.add_argument("--run", required=True, help="feasibility, m1, m2, ...")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model", default="english", help="model name the worker serves"
    )
    parser.add_argument(
        "--frontend",
        help="Rust frontend binary to start on --url in front of the spawned worker",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=8000,
        help="worker port when --frontend is used",
    )
    parser.add_argument(
        "--spawn",
        nargs=argparse.REMAINDER,
        help="start this worker command, then benchmark it",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="device for a spawned worker: LAYA_DEVICE and {device}",
    )
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--workloads", default=str(HERE / "workloads.jsonl"))
    parser.add_argument(
        "--only", nargs="*", help="bench workload ids to run (default: all)"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=300,
        help="timed requests per workload and concurrency level",
    )
    parser.add_argument(
        "--discard", type=int, default=20, help="warmup requests per workload"
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seed", type=int, default=0, help="workload order seed")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument(
        "--max-load",
        type=float,
        default=2.0,
        help="1-min load average allowed for measured runs",
    )
    args = parser.parse_args()
    token = os.environ.get("OMNI_JEV_TEST_TOKEN")
    if args.frontend and not args.spawn:
        parser.error(
            "--frontend needs --spawn: the frontend is started in front of a spawned worker"
        )
    if args.frontend and urlsplit(args.url).port == args.backend_port:
        parser.error("--url and --backend-port must differ when --frontend is used")

    problems = refuse_if_noisy(args.max_load, args.run != "feasibility")

    workloads = read_workloads(args.workloads).values()
    bench = [
        w
        for w in workloads
        if w["kind"] == "bench" and (not args.only or w["id"] in args.only)
    ]
    parity = [w for w in workloads if w["kind"] == "parity"]

    Path(args.out).mkdir(parents=True, exist_ok=True)
    processes = {}

    def memory():
        mem = footprint_mb(processes["worker"].pid) if "worker" in processes else {}
        if "frontend" in processes:
            mem["frontend_footprint_mb"] = footprint_mb(processes["frontend"].pid).get(
                "footprint_mb"
            )
        return mem

    out = Path(args.out) / f"http_{args.config}_{args.run}.jsonl"
    common = {"config": args.config, "run": args.run}
    try:
        if args.spawn:
            port = args.backend_port if args.frontend else urlsplit(args.url).port
            env = {
                **os.environ,
                "LAYA_HOST": "127.0.0.1",
                "LAYA_PORT": str(port),
                "LAYA_DEVICE": args.device,
                "LAYA_MODELS": args.model,
                "LAYA_PRELOAD": "1",
                "LAYA_LOG_LEVEL": "warning",
            }
            env["PYTHONPATH"] = str(REPO / "src") + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
            )
            placeholders = {
                "{port}": str(port),
                "{device}": args.device,
                "{model}": args.model,
            }
            command = list(args.spawn)
            for placeholder, value in placeholders.items():
                command = [arg.replace(placeholder, value) for arg in command]
            log_path = Path(args.out) / f"http_{args.config}_{args.run}.worker.log"
            spawn_log = open(log_path, "w")  # noqa: SIM115
            processes["worker"] = subprocess.Popen(
                command, env=env, stdout=spawn_log, stderr=subprocess.STDOUT, cwd=REPO
            )
        if args.frontend:
            parts = urlsplit(args.url)
            env = {
                **os.environ,
                "OMNI_JEV_BIND": f"{parts.hostname}:{parts.port}",
                "OMNI_JEV_BACKEND_URL": f"http://127.0.0.1:{args.backend_port}",
            }
            log_path = Path(args.out) / f"http_{args.config}_{args.run}.frontend.log"
            frontend_log = open(log_path, "w")  # noqa: SIM115
            processes["frontend"] = subprocess.Popen(
                [args.frontend], env=env, stdout=frontend_log, stderr=subprocess.STDOUT
            )
        ready_s, health = wait_ready(args.url, processes, args.ready_timeout)
        with open(out, "w") as f:

            def emit(record):
                f.write(json.dumps({**common, **record}) + "\n")

            # /health's device is what the worker reports; laya-serve 0.3.20 echoes LAYA_DEVICE.
            emit(
                header(
                    CHECKPOINT,
                    n=args.n,
                    discard=args.discard,
                    seed=args.seed,
                    url=args.url,
                    spawned=args.spawn,
                    frontend=args.frontend,
                    health=health,
                    device_actual=health.get("device"),
                    # The worker reports these in /health; laya-serve does not, so its values are assumed.
                    mps_amp_min_rows=health.get(
                        "mps_amp_min_rows",
                        int(os.environ.get("LAYA_MPS_AMP_MIN_ROWS", "5")),
                    ),
                    amp_dtype=health.get("autocast_dtype")
                    or (
                        "torch.float16"
                        if health.get("device") == "mps"
                        else "torch.float32"
                    ),
                    weights_dtype=health.get("weights_dtype") or "torch.float32",
                    dtype_source=None
                    if "weights_dtype" in health
                    else "assumed: laya 0.3.20 defaults",
                )
            )

            client = Client(args.url, token)
            started = time.perf_counter()
            first_ms, routing = {}, None
            for w in bench:
                ms, status, data = client.request(
                    "POST", "/v1/systemone", body_for(w, args.model), retry=True
                )
                if status != 200:
                    sys.exit(f"{w['id']}: status {status}: {data[:200]!r}")
                first_ms[w["id"]] = ms
                routing = json.loads(data).get("routing")
                for _ in range(args.discard - 1):
                    client.request(
                        "POST", "/v1/systemone", body_for(w, args.model), retry=True
                    )
            warmup_s = time.perf_counter() - started
            emit(
                {
                    "type": "phase",
                    "process_to_ready_s": round(ready_s, 3)
                    if "worker" in processes
                    else None,
                    "warmup_s": round(warmup_s, 3),
                    "first_ms": {k: round(v, 2) for k, v in first_ms.items()},
                    "routing": routing,
                    "noise": problems,
                    **memory(),
                }
            )

            order = bench[:]
            random.Random(args.seed).shuffle(order)
            for w in order:
                body = body_for(w, args.model)
                answers, error = fetch_answers(client, body)
                # the parity section of report.py reports the workload as missing
                if error:
                    emit({"type": "answers_error", "workload": w["id"], **error})
                else:
                    emit({"type": "answers", "workload": w["id"], "answers": answers})
                for concurrency in args.concurrency:
                    results, elapsed = run_level(
                        args.url, token, body, args.n, concurrency
                    )
                    ok = sum(s == 200 for _, _, s in results)
                    for i, (thread, ms, status) in enumerate(results):
                        emit(
                            {
                                "type": "req",
                                "workload": w["id"],
                                "concurrency": concurrency,
                                "i": i,
                                "thread": thread,
                                "wall_ms": None if ms is None else round(ms, 3),
                                "status": status,
                                "rows": len(w["questions"]),
                            }
                        )
                    emit(
                        {
                            "type": "throughput",
                            "workload": w["id"],
                            "concurrency": concurrency,
                            "n": len(results),
                            "errors": len(results) - ok,
                            "elapsed_s": round(elapsed, 3),
                            "rps": round(ok / elapsed, 2),
                        }
                    )

            for w in parity:
                answers, error = fetch_answers(client, body_for(w, args.model))
                if error:
                    emit({"type": "answers_error", "workload": w["id"], **error})
                else:
                    emit({"type": "answers", "workload": w["id"], "answers": answers})
            _, _, health_end = client.request("GET", "/health", retry=True)
            client.close()
            emit({"type": "end", "health": json.loads(health_end), **memory()})
    finally:
        for proc in processes.values():
            proc.terminate()
        for proc in processes.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    print(out)


if __name__ == "__main__":
    main()
