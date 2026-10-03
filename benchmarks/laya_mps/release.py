"""Releasing PyTorch's MPS caches: how much memory it gives back, and what it costs afterwards.

In this process, with the worker's options: runs `--lengths` new one-question lengths twice each,
releases the caches with `torch.mps.empty_cache()`, then runs the same lengths again. Prints the
footprint before, after the lengths and after the release, and the extra time of a length's first
request before and after the release (a released length is cold again).

    python benchmarks/laya_mps/release.py --compile --weights fp16
"""

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from env import footprint_mb  # noqa: E402


def measure(run, footprint, release, words):
    """`run(words)` returns one request's ms; each length is run twice in a row, before and after."""

    def extra():
        return statistics.median(run(w) - run(w) for w in words)

    before = footprint()
    extra_before = extra()
    after_lengths = footprint()
    started = time.perf_counter()
    release()
    release_ms = (time.perf_counter() - started) * 1000
    after_release = footprint()
    return {
        "lengths": len(words),
        "footprint_mb": {
            "before": before,
            "after_lengths": after_lengths,
            "after_release": after_release,
        },
        "release_ms": round(release_ms, 1),
        "extra_ms_before_release": round(extra_before, 1),
        "extra_ms_after_release": round(extra(), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--compile", action="store_true", help="as frontend.laya_mps --compile"
    )
    parser.add_argument(
        "--weights",
        default="fp32",
        choices=["fp32", "fp16"],
        help="as the worker's flag",
    )
    parser.add_argument("--lengths", type=int, default=100)
    parser.add_argument("--first-words", type=int, default=37)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--checkpoint", default="convaiinnovations/laya")
    parser.add_argument("--workloads", default=str(HERE / "workloads.jsonl"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    import laya
    import torch

    from models.laya import engine, optimize

    with open(args.workloads) as f:
        question = json.loads(f.readline())["questions"]  # W1: one choice question
    agent = laya.load(args.checkpoint, device="mps")
    optimize.apply(agent, fp16=args.weights == "fp16", compile=args.compile)
    for words, questions in engine.WARMUP_SHAPES:
        for _ in range(engine.WARMUP_REPEATS):
            agent.system_one(" ".join(["refund"] * words), questions)

    def run(words):
        started = time.perf_counter()
        agent.system_one(" ".join(["invoice"] * words), question)
        torch.mps.synchronize()
        return (time.perf_counter() - started) * 1000

    words = [args.first_words + i * args.step for i in range(args.lengths)]
    result = measure(
        run, lambda: footprint_mb()["footprint_mb"], torch.mps.empty_cache, words
    )
    print(json.dumps({"compile": args.compile, "weights": args.weights, **result}))


if __name__ == "__main__":
    main()
