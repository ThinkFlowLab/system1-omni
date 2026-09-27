"""Build-time CPU oracle. Uses Laya 0.3.20; never imported by the Rust runtime."""

import argparse, json
from pathlib import Path
from laya.agent import Agent, _load_tokenizer
from laya.common import collate_items

p = argparse.ArgumentParser()
p.add_argument("checkpoint", type=Path)
p.add_argument("fixtures", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
cfg = json.loads((a.checkpoint / "rl_agent_config.json").read_text())
# Only tokenizer and official preprocessing are needed; do not allocate model weights.
agent = object.__new__(Agent)
agent.cfg = cfg
agent.tok = _load_tokenizer(str(a.checkpoint / "tokenizer"), cfg)
cases = json.loads(a.fixtures.read_text())
base = {"state": "Please refund the duplicate charge.", "model": "english"}
cases += [
    {
        "name": "structured",
        "request": {
            **base,
            "state": {"text": "你好, x:y", "nested": [1, False, None]},
            "questions": {
                "z": {
                    "type": "choice",
                    "instructions": {"ask": "Which?", "x": False},
                    "criteria": {
                        "last": False,
                        "first": 0,
                        "middle": {"text": "a,b:c"},
                    },
                }
            },
        },
    },
    {
        "name": "conversation_left",
        "request": {
            **base,
            "state": [
                {"role": "user", "content": "old " * 1000},
                {"role": "user", "content": "refund NOW"},
            ],
            "questions": {
                "a": {
                    "type": "noul",
                    "instructions": "Refund?",
                    "criteria": {"TRUE": "是", "False": "否"},
                    "labels": {"true": " YES ", "false": " NO "},
                }
            },
        },
    },
    {
        "name": "long_options",
        "request": {
            **base,
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "[MASK] " * 50 + "Choose",
                    "criteria": {str(i): "description " * 90 for i in range(40)},
                }
            },
        },
    },
    {"name": "empty", "request": {**base, "questions": {}}},
    {
        "name": "list_duplicates",
        "request": {
            **base,
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Pick",
                    "criteria": ["a", "b", "a"],
                }
            },
        },
    },
]
out = []
for case in cases:
    req = case["request"]
    qs = req["questions"]
    internal = {}
    for qid, q in qs.items():
        agent._check_question(qid, q)
        internal[qid] = agent._to_internal(q)
    items = agent._encode_state(req["state"], list(qs), internal)
    n = len(items)
    l0 = max((len(i["ids"]) for i in items), default=0)
    l = ((l0 + 15) // 16 * 16) if l0 <= 256 else ((l0 + 63) // 64 * 64)
    b = 1 << (n - 1).bit_length() if n else 0
    ids = [0] * (b * l)
    lens = [0] * b
    types = [0] * b
    for j, item in enumerate(items):
        ids[j * l : j * l + l0] = item["ids"] + [agent.tok.pad_token_id] * (
            l0 - len(item["ids"])
        )
        lens[j] = len(item["ids"])
        types[j] = item["qtype"]
    out.append(
        {
            "name": case["name"],
            "request": req,
            "expected": {
                "items": items,
                "b": b,
                "l": l,
                "input_ids": ids,
                "lens": lens,
                "qtypes": types,
                "usage": sum(lens),
            },
        }
    )
a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2))
print("packing oracle cases:", len(out))
