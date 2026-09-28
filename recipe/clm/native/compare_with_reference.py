"""Compare omni-clm's decision with the CLM reference, over the same embeddings.

Both sides call the same OpenAI-compatible `/v1/embeddings` endpoint, so the vectors are
identical and the only difference left is the code between the vectors and the answer —
which is what this is meant to test. Unlike the CPU-side oracles, this needs a real
encoder, so the decisions are meaningful as model output as well.

    python recipe/clm/native/compare_with_reference.py \
        --checkpoint /root/autodl-tmp/work/clm-export \
        --bin /root/autodl-tmp/work/repo/target/release/clm-run \
        --emb-url http://127.0.0.1:8090/v1/embeddings \
        --pt /root/autodl-tmp/work/CLM_v0.1-8B.pt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests

CASES = [
    ("choice_two", "choice", {"billing": "Charges and refunds", "technical": "Software problems"}),
    ("choice_five", "choice", {"a": "alpha", "b": "beta", "c": "gamma", "d": "delta", "e": "epsilon"}),
    ("score_three", "score", ["Not urgent", "Needs attention soon", "Needs attention immediately"]),
    ("noul_stmt", "noul", None),
]

STATE = "I was charged twice for order 4411 and want the second charge refunded."


def request_for(name: str, kind: str, criteria) -> dict:
    q = {"type": kind, "instructions": f"Question {name}"}
    if criteria is not None:
        q["criteria"] = criteria
    return {"model": "clm-latest", "state": STATE, "questions": {name: q}}


def reference(emb_url: str, emb_model: str, pt: Path, req: dict, temperature: float) -> dict:
    """CLM's own heads and schema, applied to the same embeddings the Rust side gets."""
    from clm.engine import Engine
    from clm.embedder import Embedder

    embedder = Embedder(url=emb_url, model=emb_model)
    engine = Engine(embedder=embedder, checkpoint=str(pt))
    return engine.answer(req["state"], req["questions"], "clm-latest", temperature)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True, help="converted export dir")
    parser.add_argument("--bin", type=Path, required=True, help="clm-run")
    parser.add_argument("--emb-url", required=True)
    parser.add_argument("--emb-model", default="qwen3-8b")
    parser.add_argument("--pt", type=Path, required=True, help="CLM_v0.1-8B.pt")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=2e-3)
    args = parser.parse_args()

    requests_ = [(n, request_for(n, k, c)) for n, k, c in CASES]
    payload = "".join(json.dumps(r) + "\n" for _, r in requests_)
    proc = subprocess.run(
        [str(args.bin), str(args.checkpoint), "--emb-url", args.emb_url, "--model", args.emb_model],
        input=payload, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"clm-run exited {proc.returncode}")
    mine = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(mine) != len(requests_):
        raise SystemExit(f"clm-run answered {len(mine)} of {len(requests_)}")

    failures = 0
    for (name, req), got in zip(requests_, mine):
        want = reference(args.emb_url, args.emb_model, args.pt, req, args.temperature)
        mine_p = got["answers"][name]
        want_p = want["answers"][name]
        kind = req["questions"][name]["type"]

        if kind == "noul":
            got_v, want_v = mine_p["noul"], want_p["noul"]
            label = "noul"
            ok = abs(got_v - want_v) <= args.tolerance
            detail = f"{got_v:.6f} vs {want_v:.6f}"
        else:
            gk, wk = mine_p["probabilities"], want_p["probabilities"]
            if set(gk) != set(wk):
                print(f"FAIL {name}: keys differ {sorted(gk)} vs {sorted(wk)}")
                failures += 1
                continue
            worst = max(abs(gk[k] - wk[k]) for k in gk)
            same_pick = (
                mine_p.get("choice") == want_p.get("choice")
                if kind == "choice"
                else abs(mine_p["score"] - want_p["score"]) <= args.tolerance
            )
            ok = worst <= args.tolerance and same_pick
            label = "choice" if kind == "choice" else "score"
            detail = f"max|dp|={worst:.2e} " + (
                f"choice={mine_p['choice']}/{want_p['choice']}"
                if kind == "choice"
                else f"score={mine_p['score']:.6f}/{want_p['score']:.6f}"
            )
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name:12} {label:7} {detail}")

    print(f"\n{'all cases agree' if not failures else str(failures) + ' FAILED'} "
          f"(tolerance {args.tolerance})")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
