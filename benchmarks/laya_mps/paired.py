"""Paired comparison of two worker configurations: both run at once, and every request goes to A and to B
back to back, alternating which goes first, so background load that shifts both cancels out.

    python benchmarks/laya_mps/paired.py --run p1 --a "" --b "--compile --weights fp16"
    python benchmarks/laya_mps/paired.py --run f1 --a-url http://127.0.0.1:8000 --b-url http://127.0.0.1:8080
    python benchmarks/laya_mps/paired.py --run i1 --a "" --b "--compile --weights fp16" \
        --gap 2 --only W1 -n 30 --discard 2
    python benchmarks/laya_mps/paired.py --summarize benchmarks/laya_mps/results/paired_p1.jsonl

`--a`/`--b` are extra flags for `frontend.laya_mps`, which the script starts on MPS with the english
model; `--a-url`/`--b-url` compare two servers that are already running (e.g. a worker directly and
through the Rust frontend). The summary reports, per input, the median of the per-pair ratio B/A with a
95% bootstrap interval, and B's answers against A's. With `--gap`, every request follows that many
seconds of idle on a fresh connection, as an agent asking now and then would see it; without it, requests
are back to back.
"""

import argparse
import http.client
import json
import os
import random
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from bench_http import Client, body_for, fetch_answers, wait_ready  # noqa: E402
from env import footprint_mb, header, read_workloads, refuse_if_noisy  # noqa: E402

CHECKPOINT = "convaiinnovations/laya"


def spawn(flags, port, python, model, log_path, env=None):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), **(env or {})}
    command = [
        python,
        "-m",
        "frontend.laya_mps",
        "--device",
        "mps",
        "--model",
        model,
        "--port",
        str(port),
        "--log-level",
        "warning",
        *shlex.split(flags),
    ]
    log = open(log_path, "w")  # noqa: SIM115
    return subprocess.Popen(
        command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO
    )


def stop(processes):
    for p in processes:
        p.terminate()
    for p in processes:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


def run(args):
    problems = refuse_if_noisy(args.max_load, args.run != "feasibility")
    workloads = read_workloads(args.workloads).values()
    bench = [
        w
        for w in workloads
        if w["kind"] == "bench" and (not args.only or w["id"] in args.only)
    ]
    parity = [w for w in workloads if w["kind"] == "parity"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.a_url or args.b_url:
        if not (args.a_url and args.b_url) or args.a is not None or args.b is not None:
            sys.exit("give both --a-url and --b-url, without --a/--b")
        sides = {"A": args.a_url, "B": args.b_url}
        procs = {}
        urls = sides
    else:
        sides = {"A": (args.a or "", args.port_a), "B": (args.b or "", args.port_b)}
        procs = {}
        urls = {s: f"http://127.0.0.1:{port}" for s, (_, port) in sides.items()}
    try:
        if not (args.a_url or args.b_url):
            for s, (flags, port) in sides.items():
                procs[s] = spawn(
                    flags,
                    port,
                    args.python,
                    args.model,
                    out_dir / f"paired_{args.run}_{s}.log",
                )
        health = {
            s: wait_ready(
                urls[s], {s: procs[s]} if s in procs else {}, args.ready_timeout
            )[1]
            for s in sides
        }
        with open(out_dir / f"paired_{args.run}.jsonl", "w") as f:

            def emit(record):
                f.write(json.dumps({"run": args.run, **record}) + "\n")

            emit(
                header(
                    CHECKPOINT,
                    a=args.a_url or f"frontend.laya_mps {args.a or ''}".strip(),
                    b=args.b_url or f"frontend.laya_mps {args.b or ''}".strip(),
                    n=args.n,
                    discard=args.discard,
                    seed=args.seed,
                    gap_s=args.gap,
                    health=health,
                    noise=problems,
                )
            )
            clients = {s: Client(urls[s]) for s in sides}
            for w in bench + parity:
                body = body_for(w, args.model)
                for s in sides:
                    answers, error = fetch_answers(clients[s], body)
                    emit(
                        {
                            "type": "answers",
                            "side": s,
                            "workload": w["id"],
                            "answers": answers,
                            "error": error,
                        }
                    )
            order = bench[:]
            random.Random(args.seed).shuffle(order)
            for w in order:
                body = body_for(w, args.model)
                for i in range(args.discard + args.n):
                    first, second = ("A", "B") if i % 2 == 0 else ("B", "A")
                    ms = {}
                    for s in (first, second):
                        if args.gap:
                            # uvicorn drops keep-alive after 5 s idle
                            clients[s].close()
                            time.sleep(args.gap)
                        try:
                            t, status, _ = clients[s].request(
                                "POST", "/v1/systemone", body
                            )
                        except (OSError, http.client.HTTPException):
                            t, status = None, 0
                        ms[s] = t if status == 200 else None
                    if i >= args.discard:
                        emit(
                            {
                                "type": "pair",
                                "workload": w["id"],
                                "i": i - args.discard,
                                "first": first,
                                "a_ms": ms["A"],
                                "b_ms": ms["B"],
                                "rows": len(w["questions"]),
                            }
                        )
            end_health = {
                s: json.loads(clients[s].request("GET", "/health", retry=True)[2])
                for s in sides
            }
            emit(
                {
                    "type": "end",
                    "health": end_health,
                    "footprint_mb": {
                        s: footprint_mb(procs[s].pid).get("footprint_mb") for s in procs
                    },
                }
            )
    finally:
        stop(list(procs.values()))
    print(out_dir / f"paired_{args.run}.jsonl")


def median_interval(ratios, seed=0, resamples=2000):
    rng = random.Random(seed)
    meds = sorted(
        statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(resamples)
    )
    return (
        statistics.median(ratios),
        meds[int(0.025 * resamples)],
        meds[int(0.975 * resamples) - 1],
    )


def flat(answer):
    return answer.get("probabilities", {"p": answer.get("noul")})


def decision(answer):
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    return max(answer["probabilities"], key=answer["probabilities"].get)


def margin(answer):
    p = sorted(flat(answer).values(), reverse=True)
    return abs(p[0] - 0.5) if len(p) == 1 else p[0] - p[1]


def summarize(paths):
    for path in paths:
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        env = next(r for r in records if r["type"] == "env")
        gap = env.get("gap_s")
        pacing = f"{gap} s idle before each request" if gap else "back to back"
        print(
            f"## {env['run']}: A = `{env['a']}`, B = `{env['b']}`, {pacing}, "
            f"load at start {env['loadavg_1m']}\n"
        )
        print(
            "| input | pairs | A p50 ms | B p50 ms | median B/A | 95% interval |\n|---|---|---|---|---|---|"
        )
        pairs, failed = {}, 0
        for r in records:
            if r["type"] == "pair" and r["a_ms"] and r["b_ms"]:
                pairs.setdefault(r["workload"], []).append(r)
            elif r["type"] == "pair":
                failed += 1
        for wid in sorted(pairs):
            ps = pairs[wid]
            med, lo, hi = median_interval([p["b_ms"] / p["a_ms"] for p in ps])
            print(
                f"| {wid} | {len(ps)} | {statistics.median(p['a_ms'] for p in ps):.1f} | "
                f"{statistics.median(p['b_ms'] for p in ps):.1f} | {med:.3f} | {lo:.3f}–{hi:.3f} |"
            )
        answers = {}
        for r in records:
            if r["type"] == "answers":
                answers.setdefault(r["workload"], {})[r["side"]] = r
        worst, flips, errors = 0.0, [], []
        for wid, sides in answers.items():
            if any(sides.get(s, {"error": "missing"}).get("error") for s in "AB"):
                errors.append(wid)
                continue
            a_answers, b_answers = sides["A"]["answers"], sides["B"]["answers"]
            for q in sorted(a_answers.keys() | b_answers.keys()):
                a, b = a_answers.get(q), b_answers.get(q)
                if a is None or b is None or a.get("type") != b.get("type"):
                    errors.append(f"{wid}/{q}")
                    continue
                worst = max(
                    worst, max(abs(flat(a)[k] - flat(b).get(k, 0.0)) for k in flat(a))
                )
                if decision(a) != decision(b):
                    flips.append((wid, q, round(margin(a), 4)))
        print(
            f"\nB vs A answers: max |Δp| {worst:.4f}, flips {flips}, errors {errors}; failed pairs: {failed}"
        )
        end = next((r for r in records if r["type"] == "end"), None)
        if end is None:
            print(
                "**The run did not finish: no end record, so no device, recompile or memory check.**\n"
            )
            continue
        compile_state = {
            s: h.get("compile", {}).get("recompiled_after_ready")
            for s, h in end["health"].items()
        }
        devices = {s: h.get("device") for s, h in end["health"].items()}
        off_gpu = [
            s
            for s, h in end["health"].items()
            if h.get("device_mismatch")
            or (
                h.get("compile", {}).get("enabled")
                and not h["compile"].get("active", True)
            )
        ]
        print(
            f"recompiled after ready: {compile_state}; device at end: {devices}; footprint MB: {end['footprint_mb']}"
        )
        if off_gpu:
            print(
                f"**Side {', '.join(off_gpu)} left its device or compiled path during the run; the ratios above mix both.**"
            )
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summarize", nargs="+", metavar="JSONL")
    parser.add_argument("--run")
    parser.add_argument(
        "--a", help="extra frontend.laya_mps flags for side A (default: none)"
    )
    parser.add_argument(
        "--b",
        help="extra frontend.laya_mps flags for side B (default: --compile --weights fp16)",
    )
    parser.add_argument(
        "--a-url",
        help="instead of starting workers: an already running server for side A",
    )
    parser.add_argument("--b-url", help="... and for side B")
    parser.add_argument("--port-a", type=int, default=8000)
    parser.add_argument("--port-b", type=int, default=8001)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default="english")
    parser.add_argument("--workloads", default=str(HERE / "workloads.jsonl"))
    parser.add_argument("-n", type=int, default=300, help="timed pairs per input")
    parser.add_argument("--discard", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gap", type=float, default=0, help="seconds of idle before every request"
    )
    parser.add_argument("--only", nargs="*", help="bench workload ids (default: all)")
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--max-load", type=float, default=2.0)
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    elif args.run:
        if args.b is None and not args.b_url:
            args.b = "--compile --weights fp16"
        run(args)
    else:
        parser.error("give --run or --summarize")


if __name__ == "__main__":
    main()
