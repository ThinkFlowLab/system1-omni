"""Golden fixtures for the openjev/openjev letter-readout contract, from the pinned helper.

Runs openjev/openjev@ac97900's helper/shim.py (sha256 81a22f1b...) unmodified, with the serving guide's settings
(serve/SERVE.md: READOUT_T=0.85, READOUT_NOUL_T=1.829074, READOUT_NOUL_BIAS=0, READOUT_TARGETED=1,
READOUT_INSTR_STYLE=pyrepr), against a fake OpenAI client: no model, no GPU. The fake records every chat request
the helper makes and answers it with fixed candidate log-probabilities, so each case pins the prompt text, the
token ids the model would read (the chat template rendered by the pinned tokenizer), the candidate letter token
ids, the scores fed in and the helper's typed answer computed from them.

Usage: python tests/openjev/make_fixtures.py <openjev snapshot dir with helper/shim.py, tokenizer.json,
tokenizer_config.json, chat_template.jinja> tests/openjev/data
"""

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

SNAPSHOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SHIM_SHA256 = "81a22f1b1b8912a465059207ef9f60b7c6c16b4de6372305d867efbe38a1987a"
assert hashlib.sha256((SNAPSHOT / "helper/shim.py").read_bytes()).hexdigest() == SHIM_SHA256

# The serving guide's settings; every other knob stays at the file default.
os.environ.update(
    TOKENIZER=str(SNAPSHOT), READOUT_T="0.85", READOUT_NOUL_T="1.829074", READOUT_NOUL_BIAS="0",
    READOUT_TARGETED="1", READOUT_INSTR_STYLE="pyrepr",
)
for knob in ("READOUT_PERMS", "SHIM_COMPACT", "SHIM_COMPACT_CAP", "SHIM_LAYOUT", "SHIM_PAD", "SHIM_LOOP_BREAK"):
    os.environ.pop(knob, None)

from transformers import AutoTokenizer  # noqa: E402

TOKENIZER = AutoTokenizer.from_pretrained(str(SNAPSHOT))
CALLS: list[dict] = []
SCORES: list[list[float]] = []  # the log-probabilities the next calls return, one list per call, in call order


def _create(**kwargs):
    """A vLLM chat completion with logprob_token_ids: one top_logprobs entry per candidate id, matched by id."""
    content = kwargs["messages"][0]["content"]
    prompt = TOKENIZER.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = TOKENIZER.encode(prompt, add_special_tokens=False)
    allowed = kwargs["extra_body"]["logprob_token_ids"]
    scores = SCORES.pop(0)
    assert len(scores) == len(allowed)
    CALLS.append({"user_content": content, "prompt": prompt, "input_ids": ids, "candidate_token_ids": allowed,
                  "logprobs": scores})
    top = [types.SimpleNamespace(token=f"token_id:{i}", logprob=s) for i, s in zip(allowed, scores)]
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(logprobs=types.SimpleNamespace(content=[types.SimpleNamespace(top_logprobs=top)]))],
        usage=types.SimpleNamespace(prompt_tokens=len(ids)),
    )


class _OpenAI:
    def __init__(self, **_):
        self.base_url = "http://fake/v1"
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


sys.modules["openai"] = types.SimpleNamespace(OpenAI=_OpenAI)
spec = importlib.util.spec_from_file_location("shim", SNAPSHOT / "helper/shim.py")
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)
assert (shim.TEMP, shim.NOUL_T, shim.NOUL_BIAS, shim.TARGETED, shim.PERMS) == (0.85, 1.829074, 0.0, True, 1)


def scores_for(n: int, seed: int) -> list[float]:
    """Fixed, distinct log-probabilities for n candidates (a deterministic spread, not a model output)."""
    return [round(-0.37 * ((i * 7 + seed * 3) % (n + 3)) - 0.05 * i - 0.01 * seed, 6) for i in range(n)]


def run(name: str, state, questions: dict) -> dict:
    """One request through the helper's own question path (with_image, ANSWER[type]) in question order."""
    state_text = shim.with_image(state)
    answers, usage, calls = {}, 0, []
    for seed, (qid, q) in enumerate(questions.items()):
        n = {"choice": len(q.get("criteria") or {}), "score": len(q.get("criteria") or []), "noul": 2}[q["type"]]
        SCORES.append(scores_for(n, seed))
        CALLS.clear()
        answer, tokens = shim.ANSWER[q["type"]](state_text, q)
        answers[qid], usage = answer, usage + tokens
        calls.append({"question": qid, **CALLS[0]})
    return {"name": name, "request": {"model": "openjev", "state": state, "questions": questions},
            "calls": calls, "response": {"answers": answers, "usage": {"input_tokens": usage, "output_tokens": 0}}}


def bad(name: str, state, questions: dict) -> dict:
    """A request the helper rejects as a BadQuestion (422) before any model call."""
    try:
        for q in questions.values():
            shim.ANSWER[q["type"]](shim.with_image(state), q)
    except shim.BadQuestion as error:
        return {"name": name, "request": {"model": "openjev", "state": state, "questions": questions}, "error": str(error)}
    raise AssertionError(f"{name}: the helper accepted it")


TEAMS = {"billing": "charges, refunds, invoices", "shipping": "delivery and tracking", "technical": "bugs and login problems"}
ELEMENTS = {"page": "Google Flights", "elements": [{"index": 3, "tag": "input", "text": "Where from?"},
                                                  {"index": 4, "tag": "input", "text": "Where to?", "value": None}],
            "recent_actions": [{"action": "CLICK", "kind": "click", "page_changed": False}]}
cases = [
    run("serving guide example", "Customer message: I was charged twice for my order last week and nobody has replied.", {
        "route": {"type": "choice", "instructions": "Which team should handle this message?", "criteria": TEAMS},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {"type": "score", "instructions": "How urgent is this message?",
                    "criteria": ["can wait", "should be handled today", "needs an immediate reply"]},
    }),
    run("object state, pyrepr instructions, null and object descriptions", ELEMENTS, {
        "operation": {"type": "choice",
                      "instructions": {"goal": "Find one-way flights from Zürich to London", "rules": ["click before typing", None, True, 1.5]},
                      "criteria": {"CLICK": None, "TYPE_TEXT": {"element": "[3] input Where from?", "current_value": ""}, "DONE": "the task is complete"}},
        "done": {"type": "noul", "instructions": ["Is the task complete?", {"strict": False}],
                 "criteria": {"true": "the results show", "false": None}},
    }),
    run("single option, two score levels, quotes and newlines", 'He said "it\'s fine"\nthen left.', {
        "only": {"type": "choice", "instructions": "Pick it.", "criteria": {"it": "the only one"}},
        "two": {"type": "score", "instructions": {"ask": "Rate it", "scale": "it's 'quoted'"}, "criteria": ["low", "high"]},
    }),
    run("52 options, every letter", "A long list.", {
        "pick": {"type": "choice", "instructions": "Which one?", "criteria": {f"o{i}": f"option {i}" for i in range(52)}},
    }),
    run("unicode and empty strings", "Café — 東京 🚀", {
        "pick": {"type": "choice", "instructions": "Où ?", "criteria": {"Paris": "", "東京": "Tokyo", "": "empty key"}},
    }),
]
errors = [
    bad("choice without criteria", "x", {"q": {"type": "choice", "instructions": "x", "criteria": {}}}),
    bad("score with one level", "x", {"q": {"type": "score", "instructions": "x", "criteria": ["only"]}}),
    bad("missing instructions", "x", {"q": {"type": "noul"}}),
]

OUT.mkdir(parents=True, exist_ok=True)
meta = {"helper": "openjev/openjev@ac97900fd034fdd7e7e536f3d4c21b836cae0750 helper/shim.py", "helper_sha256": SHIM_SHA256,
        "settings": {"READOUT_T": 0.85, "READOUT_NOUL_T": 1.829074, "READOUT_NOUL_BIAS": 0.0, "READOUT_TARGETED": True,
                     "READOUT_INSTR_STYLE": "pyrepr"},
        "tokenizer_sha256": hashlib.sha256((SNAPSHOT / "tokenizer.json").read_bytes()).hexdigest(),
        "chat_template_sha256": hashlib.sha256((SNAPSHOT / "chat_template.jinja").read_bytes()).hexdigest(),
        "letter_token_ids": {letter: TOKENIZER.encode(letter, add_special_tokens=False)[0] for letter in shim.LETTERS}}
(OUT / "contract.json").write_text(json.dumps({"meta": meta, "cases": cases, "errors": errors}, ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
print(f"{len(cases)} cases, {sum(len(c['calls']) for c in cases)} readouts, {len(errors)} errors -> {OUT / 'contract.json'}")
