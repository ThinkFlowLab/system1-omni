"""Releasing PyTorch's MPS caches: how much memory it gives back, and what it costs afterwards.

In this process, prepared as the worker prepares a checkpoint (options, warmup, and an exit if the
model is not on MPS): walks `--lengths` new one-question lengths as lengths.py does (each run twice
in a row, the warmup's lengths skipped, stopping at the window), releases the caches with
`torch.mps.empty_cache()`, then runs the same lengths again. Prints the footprint before, after the
lengths and after the release, and the extra time of a length's first request before and after the
release (a released length is cold again).

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
import lengths  # noqa: E402
from env import footprint_mb, read_workloads, refuse_if_noisy  # noqa: E402


def measure(run, footprint, release, first_words, step, count, seen):
    """`run(words)` returns one request's ms and input tokens; `seen` holds the warm lengths."""
    records = []
    before = footprint()
    measured = lengths.measure(
        ["A"],
        lambda _, words: run(words),
        first_words,
        step,
        count,
        seen,
        records.append,
    )
    after_lengths = footprint()
    started = time.perf_counter()
    release()
    release_ms = (time.perf_counter() - started) * 1000
    after_release = footprint()
    after = [run(words)[0] - run(words)[0] for words, _ in measured]
    before_release = [
        r["first_ms"] - r["again_ms"] for r in records if r["type"] == "length"
    ]
    walk = records[-1]
    return {
        "lengths": len(measured),
        "stopped": walk["stopped"],
        "longest_tokens": walk["longest_tokens"],
        "footprint_mb": {
            "before": before,
            "after_lengths": after_lengths,
            "after_release": after_release,
        },
        "release_ms": round(release_ms, 1),
        "extra_ms_before_release": round(statistics.median(before_release), 1)
        if measured
        else None,
        "extra_ms_after_release": round(statistics.median(after), 1)
        if measured
        else None,
    }


def prepare(args):
    """The worker's startup, in this process; raises if the model is not on MPS."""
    from frontend.laya_mps import build_app, make_router

    router = make_router("mps", args.model)
    build_app(
        router,
        args.model,
        "mps",
        require_device=True,
        compile=args.compile,
        fp16=args.weights == "fp16",
    )
    return router


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
    parser.add_argument("--model", default="english")
    parser.add_argument("--workloads", default=str(HERE / "workloads.jsonl"))
    parser.add_argument("--max-load", type=float, default=2.0)
    parser.add_argument(
        "--feasibility",
        action="store_true",
        help="run on battery or under load anyway; the numbers are not measurements",
    )
    args = parser.parse_args()
    problems = refuse_if_noisy(args.max_load, not args.feasibility)
    warnings.filterwarnings("ignore")
    import torch

    question = read_workloads(args.workloads)["W1"]["questions"]
    try:
        router = prepare(args)
    except RuntimeError as exc:
        sys.exit(str(exc))

    def ask(state, questions):
        return router.predict(state, questions, model=args.model)["usage"][
            "input_tokens"
        ]

    def release():
        # buffers of work still running on the GPU cannot be released
        torch.mps.synchronize()
        torch.mps.empty_cache()

    def run(words):
        started = time.perf_counter()
        tokens = ask(" ".join(["invoice"] * words), question)
        return (time.perf_counter() - started) * 1000, tokens

    result = measure(
        run,
        lambda: footprint_mb()["footprint_mb"],
        release,
        args.first_words,
        args.step,
        args.lengths,
        lengths.warmup_lengths(ask),
    )
    print(
        json.dumps(
            {
                "compile": args.compile,
                "weights": args.weights,
                "noise": problems,
                **result,
            }
        )
    )


if __name__ == "__main__":
    main()
