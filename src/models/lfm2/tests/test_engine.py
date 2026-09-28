"""Engine tests on a tiny randomly initialized LFM2 model; no weights are downloaded.

The tiny fixture exercises the real Lfm2ForCausalLM cache classes (hybrid KV +
short-convolution) on CPU. An independently written uncached forward path is the
oracle for candidate log-likelihoods and argmax selection.
"""

import copy
import json
import pathlib
import sys

import pytest
import torch
from transformers import Lfm2Config, Lfm2ForCausalLM

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from engine import Engine, fork_cache  # noqa: E402

CONTEXT = "Route: north west. Service: express plus. No insurance."
SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "description": "Named route",
                  "enum": ["north west", "south east", "north east"]},
        "express": {"type": "boolean", "description": "Express delivery requested"},
    },
    "required": ["route", "express"],
    "additionalProperties": False,
}


class StubTokenizer:
    """Deterministic character tokenizer; avoids downloading a tokenizer."""

    def __init__(self, vocab_size=128):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 + (ord(char) % (self.vocab_size - 1)) for char in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"<{message['role']}>{message['content']}" for message in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    config = Lfm2Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, conv_L_cache=3, block_multiple_of=1,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )
    return Lfm2ForCausalLM(config).eval(), StubTokenizer(128)


def make_engine(tiny, candidate_batch_size):
    model, tokenizer = tiny
    return Engine(candidate_batch_size=candidate_batch_size, device="cpu", dtype="float32",
                  model=model, tokenizer=tokenizer)


def oracle_scores(engine, context, schema):
    """Uncached, unbatched reference for every field candidate."""
    prefix = engine.encode(engine.prompt(context, schema))
    reference = {}
    for name, spec in schema["properties"].items():
        candidates = [True, False] if spec["type"] == "boolean" else spec["enum"]
        suffix = engine.encode("  " + json.dumps(name, ensure_ascii=False) + ": ")
        options = {}
        for candidate in candidates:
            value = engine.encode(json.dumps(candidate, ensure_ascii=False) + "\n")
            tokens = engine.tensor([prefix + suffix + value])
            start = len(prefix) + len(suffix)
            logits = engine.model(tokens, use_cache=False).logits[0, start - 1:start + len(value) - 1].float()
            score = logits.log_softmax(-1).gather(1, engine.tensor(value)[:, None]).sum().item()
            options[candidate] = score
        reference[name] = options
    return reference


@torch.inference_mode()
def test_candidate_batches_match_uncached_oracle(tiny):
    reference = oracle_scores(make_engine(tiny, 1), CONTEXT, SCHEMA)
    for batch_size in (1, 8, 1000):
        result = make_engine(tiny, batch_size).score(CONTEXT, SCHEMA)
        assert result["telemetry"]["candidate_batch_size"] == batch_size
        selected = json.loads(result["text"])
        for name, options in reference.items():
            scores = {entry["value"]: entry["log_likelihood"] for entry in result["scores"][name]}
            assert set(scores) == set(options)
            for value, expected in options.items():
                assert abs(scores[value] - expected) < 1e-3
            assert selected[name] == max(options, key=options.get)


@torch.inference_mode()
def test_hybrid_cache_fork_is_isolated(tiny):
    engine = make_engine(tiny, 8)
    prefix = engine.encode(engine.prompt(CONTEXT, SCHEMA))
    base = engine.model(engine.tensor([prefix]), use_cache=True).past_key_values
    snapshot = copy.deepcopy(base)
    fork = fork_cache(base, 2)
    assert sum(hasattr(layer, "keys") for layer in fork.layers) == 2
    assert sum(hasattr(layer, "conv_states") for layer in fork.layers) == 2

    suffix = engine.encode('  "route": "north west"\n')
    ids = engine.tensor([suffix, suffix])
    mask = engine.tensor([[1] * (len(prefix) + len(suffix))] * 2)
    engine.model(ids, past_key_values=fork, attention_mask=mask)

    for old, current in zip(snapshot.layers, base.layers):
        if hasattr(old, "keys"):
            assert torch.equal(old.keys, current.keys)
            assert torch.equal(old.values, current.values)
        else:
            assert torch.equal(old.conv_states[0], current.conv_states[0])
    assert any(
        hasattr(old, "keys") and old.keys.numel() and not torch.equal(old.keys, current.keys)
        for old, current in zip(snapshot.layers, fork.layers)
    )


def test_candidate_batch_size_must_be_positive_integer(tiny):
    for bad in (0, -3, 1.5, True):
        with pytest.raises(ValueError):
            make_engine(tiny, bad)
