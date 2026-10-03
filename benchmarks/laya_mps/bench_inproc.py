"""In-process Laya benchmark (configs C1 = CPU, C2 = MPS).

Phases are timed separately: import, load, warmup, then warm requests. Every request is one line of
JSONL; report.py turns the file into tables. Run from the repository root or this directory:

    python benchmarks/laya_mps/bench_inproc.py --device mps --config C2 --run m1
"""

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

T_START = time.perf_counter()
warnings.filterwarnings("ignore")
import laya  # noqa: E402
import torch  # noqa: E402

T_IMPORT = time.perf_counter() - T_START

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import footprint_mb, header, noise_problems  # noqa: E402


def load_workloads(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def timed_call(agent, workload):
    started = time.perf_counter()
    result = agent.system_one(workload["state"], workload["questions"])
    sync(agent.device)
    return (time.perf_counter() - started) * 1000, result


def memory(device):
    mem = footprint_mb()
    if device.type == "mps":
        mem["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2**20)
    return mem


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", required=True, choices=["cpu", "mps"])
    parser.add_argument("--config", required=True, help="label, e.g. C1 or C2")
    parser.add_argument("--run", required=True, help="feasibility, m1, m2, ...")
    parser.add_argument("--checkpoint", default="convaiinnovations/laya")
    parser.add_argument("--workloads", default=str(HERE / "workloads.jsonl"))
    parser.add_argument("--only", nargs="*", help="bench workload ids to run (default: all)")
    parser.add_argument("-n", type=int, default=300, help="timed requests per workload")
    parser.add_argument("--discard", type=int, default=20, help="warmup requests per workload")
    parser.add_argument("--seed", type=int, default=0, help="workload order seed")
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--max-load", type=float, default=2.0, help="1-min load average allowed for measured runs")
    args = parser.parse_args()

    problems = noise_problems(args.max_load)
    if problems and args.run != "feasibility":
        sys.exit("refusing a measured run: " + "; ".join(problems))
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)

    workloads = load_workloads(args.workloads)
    bench = [w for w in workloads if w["kind"] == "bench" and (not args.only or w["id"] in args.only)]
    parity = [w for w in workloads if w["kind"] == "parity"]

    out = Path(args.out) / f"inproc_{args.config}_{args.device}_{args.run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    common = {"config": args.config, "device_requested": args.device, "run": args.run}

    with open(out, "w") as f:

        def emit(record):
            f.write(json.dumps({**common, **record}) + "\n")

        started = time.perf_counter()
        agent = laya.load(args.checkpoint, device=args.device)
        load_s = time.perf_counter() - started

        emit(
            header(
                args.checkpoint,
                n=args.n,
                discard=args.discard,
                seed=args.seed,
                device_actual=str(agent.device),
                weights_dtype=str(next(agent.model.parameters()).dtype),
                amp_dtype=str(agent.dtype),
                mps_amp_min_rows=getattr(agent, "mps_amp_min_rows", None),
            )
        )
        if agent.device.type != args.device:
            print(f"warning: asked for {args.device}, laya is on {agent.device}", file=sys.stderr)

        # Warmup: the first call per workload is kept apart, it is the first-shape cost.
        started = time.perf_counter()
        first_ms = {}
        for w in bench:
            first_ms[w["id"]], _ = timed_call(agent, w)
            for _ in range(args.discard - 1):
                timed_call(agent, w)
        warmup_s = time.perf_counter() - started
        emit(
            {
                "type": "phase",
                "import_s": round(T_IMPORT, 3),
                "load_s": round(load_s, 3),
                "warmup_s": round(warmup_s, 3),
                "first_ms": {k: round(v, 2) for k, v in first_ms.items()},
                "noise": problems,
                **memory(agent.device),
            }
        )

        order = bench[:]
        random.Random(args.seed).shuffle(order)
        for w in order:
            rows = len(w["questions"])
            for i in range(args.n):
                ms, result = timed_call(agent, w)
                emit(
                    {
                        "type": "req",
                        "workload": w["id"],
                        "i": i,
                        "wall_ms": round(ms, 3),
                        "tokens": result["usage"]["input_tokens"],
                        "rows": rows,
                    }
                )
                if i == 0:
                    emit({"type": "answers", "workload": w["id"], "answers": result["answers"]})

        for w in parity:
            _, result = timed_call(agent, w)
            emit({"type": "answers", "workload": w["id"], "answers": result["answers"]})

        emit({"type": "end", "total_s": round(time.perf_counter() - T_START, 1), **memory(agent.device)})
    print(out)


if __name__ == "__main__":
    main()
