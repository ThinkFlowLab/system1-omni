"""Where a Laya request's time goes on MPS. Wraps laya's stages on the loaded instance; laya
itself is not modified.

Three measurements:

1. sweep: one choice question, state length swept; fits wall = a + b * tokens. A large `a` relative
   to a short request means fixed per-request cost (dispatch, Python, sync) dominates.
2. stages: per request, time in encode (tokenize and build sequences), collate, host dispatch of the
   forward (the call returns once kernels are queued), waiting for the GPU after dispatch, copy back,
   and decode; GPU execution time from MPS events. Every stage runs synchronously in order, so the
   stages add up to the request; "other" is what the wrappers do not cover.
3. ops: torch.profiler CPU trace of the forward: operator calls per request and the top operators by
   self CPU time, i.e. the host cost of issuing the forward.

    python benchmarks/laya_mps/profile_mps.py --run feasibility
"""

import argparse
import json
import statistics
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import laya  # noqa: E402
import laya.agent  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from env import header, noise_problems  # noqa: E402

STAGES = ["encode", "collate", "dispatch", "gpu_wait", "copy_back", "decode", "other"]


class StageTimer:
    """Times laya's request stages by wrapping them on one Agent instance."""

    def __init__(self, agent):
        self.agent = agent
        self.current = None
        self.use_events = agent.device.type == "mps"
        self._install()

    def _add(self, stage, ms):
        if self.current is not None:
            self.current[stage] = self.current.get(stage, 0.0) + ms

    def _timed(self, stage, fn):
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self._add(stage, (time.perf_counter() - started) * 1000)

        return wrapper

    def _install(self):
        agent = self.agent
        agent._encode_state = self._timed("encode", agent._encode_state)
        agent._decode_answers = self._timed("decode", agent._decode_answers)
        laya.agent.collate_items = self._timed("collate", laya.agent.collate_items)
        infer = agent._infer

        def forward(b):
            # Replaces Agent._forward: same result, with dispatch, GPU wait and copy back split apart.
            start_event = end_event = None
            if self.use_events:
                start_event = torch.mps.Event(enable_timing=True)
                end_event = torch.mps.Event(enable_timing=True)
                start_event.record()
            t0 = time.perf_counter()
            logits, act = infer(b)
            if self.use_events:
                end_event.record()
            t1 = time.perf_counter()
            if agent.device.type == "mps":
                torch.mps.synchronize()
            t2 = time.perf_counter()
            out = logits.float().cpu().numpy(), torch.softmax(act.float(), -1).cpu().numpy()
            t3 = time.perf_counter()
            self._add("dispatch", (t1 - t0) * 1000)
            self._add("gpu_wait", (t2 - t1) * 1000)
            self._add("copy_back", (t3 - t2) * 1000)
            if self.use_events:
                self._add("gpu_exec", start_event.elapsed_time(end_event))
            return out

        agent._forward = forward

    def request(self, state, questions):
        self.current = {}
        started = time.perf_counter()
        result = self.agent.system_one(state, questions)
        if self.agent.device.type == "mps":
            torch.mps.synchronize()
        stages, self.current = self.current, None
        stages["wall"] = (time.perf_counter() - started) * 1000
        stages["other"] = stages["wall"] - sum(stages.get(s, 0.0) for s in STAGES if s != "other")
        return stages, result


def fit(xs, ys):
    """Least squares y = a + b x, with R^2."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, 1 - ss_res / ss_tot if ss_tot else 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--checkpoint", default="convaiinnovations/laya")
    parser.add_argument("--workloads", default=str(HERE / "workloads.jsonl"))
    parser.add_argument("--stage-workloads", nargs="+", default=["W1", "W3", "W5"])
    parser.add_argument("-n", type=int, default=100, help="timed requests per point")
    parser.add_argument("--discard", type=int, default=10)
    parser.add_argument("--sweep-words", type=int, nargs="+", default=[1, 8, 24, 56, 120, 250, 380])
    parser.add_argument("--ops-requests", type=int, default=20)
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--max-load", type=float, default=2.0)
    args = parser.parse_args()

    problems = noise_problems(args.max_load)
    if problems and args.run != "feasibility":
        sys.exit("refusing a measured run: " + "; ".join(problems))
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)

    with open(args.workloads) as f:
        workloads = {w["id"]: w for w in (json.loads(line) for line in f if line.strip())}
    route = workloads["W1"]["questions"]

    agent = laya.load(args.checkpoint, device=args.device)
    timer = StageTimer(agent)
    out = Path(args.out) / f"profile_{args.device}_{args.run}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    common = {"config": f"profile-{args.device}", "run": args.run}

    with open(out, "w") as f:

        def emit(record):
            f.write(json.dumps({**common, **record}) + "\n")

        emit(
            header(
                args.checkpoint,
                device_actual=str(agent.device),
                noise=problems,
                weights_dtype=str(next(agent.model.parameters()).dtype),
                amp_dtype=str(agent.dtype),
                mps_amp_min_rows=getattr(agent, "mps_amp_min_rows", None),
            )
        )

        print("## Length sweep (1 choice question, 5 options)\n")
        print("| words | tokens | p50 ms |\n|---|---|---|")
        points = []
        for words in args.sweep_words:
            state = " ".join(["delivery"] * words)
            for _ in range(args.discard):
                timer.request(state, route)
            walls, tokens = [], None
            for _ in range(args.n):
                stages, result = timer.request(state, route)
                walls.append(stages["wall"])
                tokens = result["usage"]["input_tokens"]
            p50 = statistics.median(walls)
            points.append((tokens, p50))
            emit(
                {
                    "type": "sweep",
                    "words": words,
                    "tokens": tokens,
                    "p50_ms": round(p50, 3),
                    "wall_ms": [round(w, 3) for w in walls],
                }
            )
            print(f"| {words} | {tokens} | {p50:.1f} |")
        a, b, r2 = fit([t for t, _ in points], [p for _, p in points])
        emit({"type": "fit", "a_ms": round(a, 3), "b_ms_per_token": round(b, 5), "r2": round(r2, 4)})
        print(f"\nwall ≈ {a:.1f} ms + {b:.3f} ms/token × tokens (R² {r2:.3f})")

        print("\n## Stages (median ms per request)\n")
        columns = ["wall", *STAGES, "gpu_exec"]
        print("| workload | tokens | rows | " + " | ".join(columns) + " |\n|" + "---|" * (len(columns) + 3))
        for wid in args.stage_workloads:
            w = workloads[wid]
            for _ in range(args.discard):
                timer.request(w["state"], w["questions"])
            per_stage, tokens = defaultdict(list), None
            for _ in range(args.n):
                stages, result = timer.request(w["state"], w["questions"])
                tokens = result["usage"]["input_tokens"]
                for k, v in stages.items():
                    per_stage[k].append(v)
            medians = {k: statistics.median(v) for k, v in per_stage.items()}
            emit(
                {
                    "type": "stages",
                    "workload": wid,
                    "tokens": tokens,
                    "rows": len(w["questions"]),
                    "median_ms": {k: round(v, 3) for k, v in medians.items()},
                    "samples": {k: [round(x, 3) for x in v] for k, v in per_stage.items()},
                }
            )
            cells = " | ".join(f"{medians[c]:.1f}" if c in medians else "" for c in columns)
            print(f"| {wid} | {tokens} | {len(w['questions'])} | {cells} |")

        print("\n## Host operators for W1 (torch.profiler, CPU)\n")
        w = workloads["W1"]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            for _ in range(args.ops_requests):
                timer.request(w["state"], w["questions"])
        events = [e for e in prof.key_averages() if e.key.startswith("aten::")]
        calls = sum(e.count for e in events) / args.ops_requests
        top = sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[:12]
        emit(
            {
                "type": "ops",
                "workload": "W1",
                "requests": args.ops_requests,
                "aten_calls_per_request": calls,
                "top": [
                    {
                        "op": e.key,
                        "calls_per_request": e.count / args.ops_requests,
                        "self_cpu_ms_per_request": e.self_cpu_time_total / 1000 / args.ops_requests,
                    }
                    for e in top
                ],
            }
        )
        print(f"aten calls per request: {calls:.0f}\n")
        print("| op | calls/request | self CPU ms/request |\n|---|---|---|")
        for e in top:
            print(
                f"| {e.key} | {e.count / args.ops_requests:.0f} | {e.self_cpu_time_total / 1000 / args.ops_requests:.2f} |"
            )
    print(f"\n{out}")


if __name__ == "__main__":
    main()
