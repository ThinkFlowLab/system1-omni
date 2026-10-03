"""New input lengths: what the first request of a length costs, and how the worker's memory grows.

Starts one worker per flag set (all alive at once, like paired.py), warms each up, then sends every
side a one-question request at each of `--lengths` lengths it has not seen, twice in a row. The
first request's extra time is the first minus the second; memory is each worker's physical footprint
before and after. Lengths the warmup ran are skipped, and the walk stops at the model's window.

    python benchmarks/laya_mps/lengths.py --run l1 --a "" --b "--compile --weights fp16"
    python benchmarks/laya_mps/lengths.py --run l2 --a "" --b "--compile --weights fp16" \\
        --first-words 1 --step 1 --lengths 1000                     # every length up to the window
    python benchmarks/laya_mps/lengths.py --summarize benchmarks/laya_mps/results/lengths_l1.jsonl
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from bench_http import Client, body_for, wait_ready  # noqa: E402
from env import footprint_mb, header, noise_problems  # noqa: E402
from paired import CHECKPOINT, spawn, stop  # noqa: E402

from models.laya.engine import WARMUP_SHAPES  # noqa: E402


def measure(sides, request, first_words, step, count, seen, emit):
    """Two requests per side at each input length the workers have not run, `count` lengths at most.

    `request(side, words)` returns (ms, tokens). A length is known only after its first request, so a
    length already in `seen` costs that one request and is skipped; the walk stops at the window, where a
    longer state is truncated to a length already run."""
    seen, measured, longest, words = set(seen), [], 0, first_words
    stopped = "count"
    while len(measured) < count:
        order = sorted(sides, reverse=len(measured) % 2 == 1)
        first, tokens = request(order[0], words)
        if tokens <= longest:
            stopped = "window"
            break
        longest = tokens
        if tokens not in seen:
            for s in order:
                if s != order[0]:
                    first, _ = request(s, words)
                again, _ = request(s, words)
                emit(
                    {
                        "type": "length",
                        "side": s,
                        "words": words,
                        "tokens": tokens,
                        "first_ms": first,
                        "again_ms": again,
                    }
                )
            seen.add(tokens)
            measured.append((words, tokens))
        words += step
    emit(
        {
            "type": "walk",
            "lengths": len(measured),
            "stopped": stopped,
            "longest_tokens": longest,
        }
    )
    return measured


def run(args):
    problems = noise_problems(args.max_load)
    if problems and args.run != "feasibility":
        sys.exit("refusing a measured run: " + "; ".join(problems))
    with open(args.workloads) as f:
        workloads = {w["id"]: w for w in map(json.loads, filter(str.strip, f))}
    question = workloads["W1"]["questions"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sides = {"A": args.a} if args.b is None else {"A": args.a, "B": args.b}
    ports = {"A": args.port_a, "B": args.port_b}
    procs = {}
    path = out_dir / f"lengths_{args.run}.jsonl"
    try:
        for s, flags in sides.items():
            log = out_dir / f"lengths_{args.run}_{s}.log"
            procs[s] = spawn(flags, ports[s], args.python, args.model, log)
        urls = {s: f"http://127.0.0.1:{ports[s]}" for s in sides}
        for s in sides:
            wait_ready(urls[s], {s: procs[s]}, args.ready_timeout)
        clients = {s: Client(urls[s]) for s in sides}

        def request(s, words):
            state = {"state": " ".join(["invoice"] * words), "questions": question}
            ms, status, data = clients[s].request(
                "POST", "/v1/systemone", body_for(state, args.model), retry=True
            )
            if status != 200:
                sys.exit(f"{s}: status {status} at {words} words: {data[:200]!r}")
            return ms, json.loads(data)["usage"]["input_tokens"]

        with open(path, "w") as f:

            def emit(record):
                f.write(json.dumps({"run": args.run, **record}) + "\n")

            emit(
                header(
                    CHECKPOINT,
                    a=f"frontend.laya_mps {args.a}".strip(),
                    b=None if args.b is None else f"frontend.laya_mps {args.b}".strip(),
                    lengths=args.lengths,
                    noise=problems,
                )
            )
            # The lengths the warmup ran, read back from the workers: those shapes are warm already.
            seen = set()
            for words, questions in WARMUP_SHAPES:
                if len(questions) == 1:
                    state = {
                        "state": " ".join(["refund"] * words),
                        "questions": questions,
                    }
                    for s in sides:
                        reply = clients[s].request(
                            "POST",
                            "/v1/systemone",
                            body_for(state, args.model),
                            retry=True,
                        )
                        seen.add(json.loads(reply[2])["usage"]["input_tokens"])
            before = {s: footprint_mb(procs[s].pid).get("footprint_mb") for s in sides}
            measure(
                sorted(sides),
                request,
                args.first_words,
                args.step,
                args.lengths,
                seen,
                emit,
            )
            health = {
                s: json.loads(clients[s].request("GET", "/health", retry=True)[2])
                for s in sides
            }
            after = {s: footprint_mb(procs[s].pid).get("footprint_mb") for s in sides}
            emit(
                {
                    "type": "end",
                    "footprint_before_mb": before,
                    "footprint_after_mb": after,
                    "recompiled_after_ready": {
                        s: h.get("compile", {}).get("recompiled_after_ready")
                        for s, h in health.items()
                    },
                }
            )
    finally:
        stop(procs.values())
    print(path)


def summarize(paths):
    for path in paths:
        records = [json.loads(line) for line in Path(path).read_text().splitlines()]
        env = records[0]
        end = next((r for r in records if r["type"] == "end"), None)
        walk = next((r for r in records if r["type"] == "walk"), None)
        if walk is None:
            stopped = "stop reason not recorded" if end else "the run did not finish"
        elif walk["stopped"] == "window":
            stopped = f"stopped at the window, {walk['longest_tokens']} tokens"
        else:
            stopped = f"stopped after the {walk['lengths']} lengths asked for"
        print(f"## {env['run']}: load at start {env['loadavg_1m']}, {stopped}\n")
        print(
            "| side | flags | lengths | extra ms on first request, median (p90) "
            "| footprint before → after MB | recompiled after ready |"
        )
        print("|---|---|---|---|---|---|")
        for s in ("A", "B"):
            rows = [r for r in records if r["type"] == "length" and r["side"] == s]
            if not rows:
                continue
            extra = sorted(r["first_ms"] - r["again_ms"] for r in rows)
            p90 = extra[max(0, int(0.9 * len(extra)) - 1)]
            memory = "unfinished"
            recompiled = None
            if end:
                before = end["footprint_before_mb"].get(s)
                after = end["footprint_after_mb"].get(s)
                memory = f"{before} → {after}"
                recompiled = end["recompiled_after_ready"].get(s)
            print(
                f"| {s} | `{env[s.lower()]}` | {len(rows)} "
                f"| {statistics.median(extra):.1f} ({p90:.1f}) | {memory} | {recompiled} |"
            )
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summarize", nargs="+", metavar="JSONL")
    parser.add_argument("--run")
    parser.add_argument(
        "--a", default="", help="frontend.laya_mps flags for side A (default: none)"
    )
    parser.add_argument(
        "--b", help="flags for an optional side B; a single flag: --b=--compile"
    )
    parser.add_argument("--lengths", type=int, default=100, help="new lengths per side")
    parser.add_argument("--first-words", type=int, default=37)
    parser.add_argument("--step", type=int, default=4, help="words between two lengths")
    parser.add_argument("--port-a", type=int, default=8000)
    parser.add_argument("--port-b", type=int, default=8001)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default="english")
    parser.add_argument("--workloads", default=str(HERE / "workloads.jsonl"))
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--max-load", type=float, default=2.0)
    parser.add_argument("--out", default=str(HERE / "results"))
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    elif args.run:
        run(args)
    else:
        parser.error("give --run or --summarize")


if __name__ == "__main__":
    main()
