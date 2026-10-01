"""Qwen3.5-4B with the Cua-S1 `text` adapter, loaded and scored as upstream
`cua_s1.four_b.FourBModel` does: an unmerged PEFT adapter, the chat template with its
generation prompt, full logits, and a float32 softmax over the option letters at the
last position. That keeps the probabilities bitwise identical to the reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .contract import LETTERS, Question, build_messages


class TextModel:
    def __init__(self, base: str, adapter: str, device: str, dtype: str) -> None:
        # PEFT only warns about keys it cannot place: refuse the multimodal adapter.
        config = json.loads((Path(adapter) / "adapter_config.json").read_text())
        if "linear_fc1" in config["target_modules"]:
            raise ValueError(
                f"{adapter} is the multimodal adapter; pass its text/ directory"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(
            base, dtype=getattr(torch, dtype), device_map=device
        )
        self.model = PeftModel.from_pretrained(model, adapter).eval()
        self.letter_ids = self.tokenizer.convert_tokens_to_ids(list(LETTERS))

    def encode(self, state: str, question: Question):
        text = self.tokenizer.apply_chat_template(
            build_messages(state, question), tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer(text, return_tensors="pt")

    @torch.no_grad()
    def score(self, inputs, n_options: int) -> list[float]:
        logits = self.model(**inputs.to(self.model.device)).logits[0, -1]
        return torch.softmax(
            logits[self.letter_ids[:n_options]].float(), dim=-1
        ).tolist()
