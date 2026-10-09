"""Shared-prefill candidate scoring for LFM2.5-350M.

Prompt construction, hybrid KV/conv cache forking and full-sequence candidate
log-likelihood scoring are adapted from the RLCD reference engine
(notnotsamuel/LFM2.5-350M-RLCD, revision
deb589d803d141cabd158ef55f6617b128529f36, MIT, Copyright (c) 2026
notnotsamuel). The pinned base weights belong to LiquidAI/LFM2.5-350M at
revision 9e6c6ccf47cd318696e137d381a7ded8fe4df09f and keep their upstream
license. Only inference is performed; no weights are trained or modified.
"""

import copy
import json

import jsonschema
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2.5-350M"
REVISION = "9e6c6ccf47cd318696e137d381a7ded8fe4df09f"

__all__ = ["Engine", "fork_cache", "validate_schema", "MODEL_ID", "REVISION"]


def validate_schema(schema):
    """Accept only the flat, closed object subset the reference engine supports."""
    jsonschema.Draft202012Validator.check_schema(schema)
    if not isinstance(schema, dict):
        raise ValueError("Schema must be an object")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("Only closed, flat object schemas are supported")
    fields = schema.get("properties")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("Schema must declare at least one property")
    if set(schema.get("required", [])) != set(fields):
        raise ValueError("All fields must be required")
    if set(schema) - {"type", "properties", "required", "additionalProperties"}:
        raise ValueError("Unsupported object constraints")
    for spec in fields.values():
        if not isinstance(spec, dict) or set(spec) - {"type", "enum", "description"}:
            raise ValueError("Unsupported field constraints")
        if spec.get("type") == "boolean" and "enum" not in spec:
            continue
        values = spec.get("enum")
        if spec.get("type") != "string" or not isinstance(values, list) or not values:
            raise ValueError("Fields must be booleans or nonempty string enums")
        if any(type(value) is not str for value in values) or len(set(values)) != len(values):
            raise ValueError("String enums must be unique strings")


def fork_cache(cache, count):
    """Copy all cache state and repeat batch row zero `count` times.

    Generic batch_repeat_interleave does not cover LFM2 convolution layer
    objects, so we deep-copy and call reorder_cache, whose per-layer kernels
    index_select both KV tensors and convolution history. index_select allocates
    independent storage: no mutable views are shared with the source cache.
    """
    cloned = copy.deepcopy(cache)
    device = None
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        if keys is not None and keys.numel():
            device = keys.device
            break
    if device is None:
        device = torch.device("cpu")
    cloned.reorder_cache(torch.zeros(count, dtype=torch.long, device=device))
    return cloned


class Engine:
    """Owns the model and scores candidate sequences from one shared prefill."""

    def __init__(self, candidate_batch_size, device="cuda", dtype="float16",
                 model=None, tokenizer=None, model_id=MODEL_ID, revision=REVISION):
        if isinstance(candidate_batch_size, bool) or not isinstance(candidate_batch_size, int) \
                or candidate_batch_size < 1:
            raise ValueError("candidate_batch_size must be a positive integer")
        self.candidate_batch_size = candidate_batch_size
        self.device = device
        self.dtype = dtype
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, dtype=getattr(torch, dtype),
                attn_implementation="eager",
            ).to(device)
        model.eval()
        model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.model = model

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def prompt(self, context, schema):
        validate_schema(schema)
        messages = [
            {"role": "system", "content":
                "Extract the attributes from the text. Return only a JSON object matching this schema. "
                "Use the exact allowed values. No explanation or markdown.\n"
                + json.dumps(schema, ensure_ascii=False)},
            {"role": "user", "content": context},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) + "{\n"

    def tensor(self, tokens):
        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def score(self, context, schema):
        """Score every field candidate and return the best value per field.

        Each candidate batch starts from a fresh copy of the shared prefill
        cache. Scores are full-sequence FP32 log-softmax sums with no length
        normalization; padding and the field suffix are never scored.
        """
        validate_schema(schema)
        prefix = self.encode(self.prompt(context, schema))
        cache = self.model(self.tensor([prefix]), use_cache=True, logits_to_keep=1).past_key_values
        branches = []
        for name, spec in schema["properties"].items():
            candidates = [True, False] if spec["type"] == "boolean" else list(spec["enum"])
            suffix = self.encode("  " + json.dumps(name, ensure_ascii=False) + ": ")
            for candidate in candidates:
                value = self.encode(json.dumps(candidate, ensure_ascii=False) + "\n")
                branches.append((name, candidate, len(suffix), value, suffix + value))
        collected = []
        padded_tokens = 0
        for start in range(0, len(branches), self.candidate_batch_size):
            chunk = branches[start:start + self.candidate_batch_size]
            width = max(len(branch[4]) for branch in chunk)
            ids = self.tensor([branch[4] + [self.tokenizer.pad_token_id] * (width - len(branch[4]))
                               for branch in chunk])
            mask = self.tensor([[1] * (len(prefix) + len(branch[4])) + [0] * (width - len(branch[4]))
                                for branch in chunk])
            forked = fork_cache(cache, len(chunk))
            logits = self.model(ids, past_key_values=forked, attention_mask=mask, use_cache=True).logits
            for row, (_, _, suffix_len, value, _) in enumerate(chunk):
                logp = logits[row, suffix_len - 1:suffix_len + len(value) - 1].float().log_softmax(-1)
                targets = ids[row, suffix_len:suffix_len + len(value)]
                collected.append(logp.gather(1, targets[:, None]).sum())
            padded_tokens += len(chunk) * width
            del logp, logits, forked, ids, mask
        scores = torch.stack(collected).cpu().tolist() if collected else []
        selected, telemetry = {}, {}
        for name in schema["properties"]:
            options = [(branch[1], value) for branch, value in zip(branches, scores) if branch[0] == name]
            selected[name] = max(options, key=lambda option: option[1])[0]
            telemetry[name] = [{"value": value, "log_likelihood": score} for value, score in options]
        batches = (len(branches) + self.candidate_batch_size - 1) // self.candidate_batch_size
        return {
            "text": json.dumps(selected, ensure_ascii=False, allow_nan=False),
            "scores": telemetry,
            "prompt_tokens": len(prefix),
            "input_tokens": len(prefix) + sum(len(branch[4]) for branch in branches),
            "telemetry": {
                "branches": len(branches),
                "candidate_batch_size": self.candidate_batch_size,
                "branch_tokens_padded": padded_tokens,
                "forward_calls": (1 if branches else 0) + batches,
            },
        }
