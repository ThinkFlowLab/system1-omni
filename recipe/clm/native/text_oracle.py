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

from clm.schema import build_pairs, candidates, state_text, to_text

STATES = [
    "I was charged twice.",
    {"body": "Charged twice", "order": 4411, "urgent": True},
    {"ticket": {"id": 7, "tags": ["a", "b"]}, "note": None},
    [{"k": 1}, {"k": 2}],
    {"empty_obj": {}, "empty_arr": [], "n": 0.5},
    {"nested": {"deep": {"x": "y"}}},
    # `str(float)` switches to exponent form outside [1e-4, 1e16) and keeps a `.0` on an
    # integral float, so these pin the number spelling the reference produces.
    {"tiny": 1e-5, "smaller": 1e-7, "edge": 1e-4, "round": 1e15, "huge": 1e16, "neg": -1e-6},
    # 17 significant digits, where a parse that is not correctly rounded lands on
    # the neighbouring double and renders differently.
    {"seventeen": 7.8190461323667115, "inexact": 9007199254740993.0},
    # A JSON integer is an arbitrary-precision int in Python and keeps every digit; one
    # larger than a u64 cannot survive a detour through a double.
    {"big": 18446744073709551616, "huge": 340282366920938463463374607431768211456,
     "negzero": -0, "negbig": -18446744073709551616},
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
    # Numbers are spelled by `str(float)` in criteria too, not only in the state.
    {"type": "score", "instructions": "How much?", "criteria": [1e-5, 0.5, 1e16]},
]

# Whole request bodies, as the text a caller sends them in. `QUESTIONS` above is built
# from Python values, so it cannot say `18446744073709551616` and mean an integer: it is
# the literal text that carries those digits. These go through the request path instead,
# which is the path a server takes and the only one the digits survive.
RAW_REQUESTS = [
    # A `noul` reads its two descriptions in `false`/`true` order, not in the order they
    # were written, so a number has to be found by its key rather than by counting.
    '{"state": {"n": 1}, "questions": {"q": {"type": "noul", "instructions": "Is it so?",'
    ' "criteria": {"true": 1, "false": 2}}}}',
    # A statement that is a number rather than a string. The default `noul` candidates are
    # built from the statement, so it has to keep every digit there as well as in the
    # state text.
    '{"state": {}, "questions": {"q": {"type": "noul",'
    ' "instructions": 18446744073709551616, "criteria": null}}}',
    # The same digits in a state, in a `choice`'s descriptions and in a `score`'s levels,
    # with the two questions reaching the same state text.
    '{"state": {"big": 18446744073709551616, "seventeen": 7.8190461323667115},'
    ' "questions": {'
    '"a": {"type": "choice", "instructions": "Pick",'
    ' "criteria": {"x": 340282366920938463463374607431768211456, "y": 7.8190461323667115}},'
    ' "b": {"type": "score", "instructions": "How much?",'
    ' "criteria": [1e-5, 18446744073709551616]}}}',
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    out = {"to_text": [to_text(s) for s in STATES], "cases": [], "raw_cases": []}
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
    for line in RAW_REQUESTS:
        body = json.loads(line)
        pairs = build_pairs(body["state"], body["questions"])
        out["raw_cases"].append({
            "line": line,
            "questions": {qid: {"state_text": s, "keys": k, "candidate_texts": t}
                          for qid, (s, k, t) in pairs.items()},
        })
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"TEXT_ORACLE {args.output} to_text={len(out['to_text'])} "
          f"cases={len(out['cases'])} raw_cases={len(out['raw_cases'])}", flush=True)


if __name__ == "__main__":
    main()
