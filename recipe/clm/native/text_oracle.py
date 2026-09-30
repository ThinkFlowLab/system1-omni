"""The text the CLM heads see, from the reference implementation itself.

`omni-clm` reimplements `state_text`, `candidates` and `to_text`. A separator in the
wrong place does not fail loudly — it shifts every probability — so the Rust side is
checked byte-for-byte against this file, which is produced by importing `clm.schema`
rather than by transcribing it.

    python recipe/clm/native/text_oracle.py OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clm.schema import candidates, state_text, to_text

STATES = [
    "I was charged twice.",
    {"body": "Charged twice", "order": 4411, "urgent": True},
    {"ticket": {"id": 7, "tags": ["a", "b"]}, "note": None},
    [{"k": 1}, {"k": 2}],
    {"empty_obj": {}, "empty_arr": [], "n": 0.5},
    {"nested": {"deep": {"x": "y"}}},
]

QUESTIONS = [
    {"type": "choice", "instructions": "Which team?",
     "criteria": {"billing": "Charges and refunds", "tech": "Software problems"}},
    {"type": "choice", "instructions": "Pick", "criteria": {"a": "", "b": None}},
    {"type": "score", "instructions": "How urgent?", "criteria": ["Not urgent", "Soon", "Now"]},
    {"type": "noul", "instructions": "Does the customer ask for a refund?", "criteria": None},
    {"type": "noul", "instructions": "Refund?",
     "criteria": {"true": "Yes they do", "false": "No they do not"}},
    # An empty container is a description that renders to nothing, not a missing one, so
    # these exercise the difference between "absent" and "renders empty".
    {"type": "choice", "instructions": "Pick a bucket",
     "criteria": {"empty_obj": {}, "empty_list": []}},
    {"type": "noul", "instructions": "Is it so?", "criteria": {"true": {}, "false": []}},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    out = {"to_text": [to_text(s) for s in STATES], "cases": []}
    for state in STATES:
        for q in QUESTIONS:
            keys, texts = candidates(q)
            out["cases"].append({
                "state": state,
                "kind": q["type"],
                "instructions": q.get("instructions") or "",
                "state_text": state_text(state, q.get("instructions")),
                "keys": keys,
                "candidate_texts": texts,
            })
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"TEXT_ORACLE {args.output} to_text={len(out['to_text'])} cases={len(out['cases'])}", flush=True)


if __name__ == "__main__":
    main()
