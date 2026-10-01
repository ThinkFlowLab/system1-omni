"""Direct Transformers/PEFT execution. No production dependency on cua_s1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protocol import InvalidRequest, Question, Request, answer, build_messages

REFERENCE_REVISION = "0e75660ce4c2edda519e0c795fa3ad98abf4e76f"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER_REVISION = "16818868b0cc7813808aae4e87b417657046ab79"
IDENTITY = f"cua-ai/cua-s1-4b-0.2@{ADAPTER_REVISION}:multimodal"
WEIGHTS_MANIFEST_SHA256 = (
    "9820bd232c5762f114e19680c0f8203d7e1faaf8a60c196cfe01964d6d8a6c09"
)
MAX_TOKENS = 4096


def letter_ids(tokenizer, count: int) -> list[int]:
    ids = []
    for index in range(count):
        encoded = tokenizer.encode(chr(65 + index), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError("each candidate letter must be a single token")
        ids.append(encoded[0])
    return ids


def validate_adapter_config(config: dict):
    targets = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "linear_fc1",
        "linear_fc2",
    }
    if (
        config.get("peft_type") != "LORA"
        or config.get("r") != 16
        or config.get("lora_alpha") != 32
        or set(config.get("target_modules", [])) != targets
        or config.get("base_model_name_or_path") != "Qwen/Qwen3.5-4B"
    ):
        raise ValueError("expected the pinned 0.2 multimodal LoRA adapter")


def parse_weights_manifest(raw: bytes) -> dict:
    """Accept only the manifest from the pinned upstream reference commit."""
    if hashlib.sha256(raw).hexdigest() != WEIGHTS_MANIFEST_SHA256:
        raise ValueError("upstream weights manifest checksum mismatch")
    return json.loads(raw)


def verify_weights(base: Path, adapter: Path):
    """Check local artifacts before assigning the pinned identity to responses."""
    lock = parse_weights_manifest((base.parent / "weights.lock.json").read_bytes())
    allowed = {base: set(), adapter: set()}
    for artifact in lock["artifacts"]:
        for name, expected in artifact["files"].items():
            if artifact["role"] == "adapter":
                if not name.startswith("multimodal/"):
                    continue
                path = adapter / name.removeprefix("multimodal/")
            else:
                path = base / name
            root = adapter if artifact["role"] == "adapter" else base
            allowed[root].add(path.relative_to(root).as_posix())
            if not path.is_file() or path.stat().st_size != expected["size"]:
                raise ValueError(f"missing or wrong-size pinned artifact: {path.name}")
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != expected["sha256"]:
                raise ValueError(f"checksum mismatch: {path.name}")
    for root, names in allowed.items():
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if path.is_file() and relative.parts[0] != ".cache":
                if relative.as_posix() not in names:
                    raise ValueError(
                        f"unlisted artifact may override pinned files: {relative}"
                    )


class MultimodalEngine:
    def __init__(
        self, base: str, adapter: str, device: str = "cuda", dtype: str = "bfloat16"
    ):
        import torch
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        base_path, adapter_path = Path(base), Path(adapter)
        verify_weights(base_path, adapter_path)
        validate_adapter_config(
            json.loads((adapter_path / "adapter_config.json").read_text())
        )
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            base,
            torch_dtype=getattr(torch, dtype),
            device_map=device,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        modules = [
            name
            for name, module in self.model.named_modules()
            if isinstance(module, LoraLayer)
        ]
        if len(modules) != 178 or not any(".visual." in name for name in modules):
            raise RuntimeError(
                "multimodal adapter did not attach to all 178 expected modules"
            )
        self.adapter_modules = len(modules)
        self.model.eval()
        self.dtype = dtype

    def prepare(self, image, question: Question):
        messages = build_messages(question)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        if inputs["input_ids"].shape[-1] > MAX_TOKENS:
            raise InvalidRequest(f"processed prompt exceeds {MAX_TOKENS} tokens")
        return inputs

    def score(self, inputs, question: Question) -> list[float]:
        import torch

        ids = letter_ids(self.tokenizer, len(question.keys))
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
        logits = output.logits[0, -1, :]
        return torch.softmax(
            logits[torch.tensor(ids, device=logits.device)].float(), dim=-1
        ).tolist()

    def predict(self, request: Request) -> dict:
        # Validate all processed lengths before executing any question.
        prepared = [self.prepare(request.image, q) for q in request.questions]
        answers = {
            q.name: answer(q, self.score(inputs, q))
            for q, inputs in zip(request.questions, prepared)
        }
        return {
            "model": IDENTITY,
            "answers": answers,
            "usage": {
                "input_tokens": sum(x["input_ids"].shape[-1] for x in prepared),
                "output_tokens": 0,
            },
        }

    def warmup(self):
        from PIL import Image

        q = Question(
            "warmup", ("continue", "cancel"), ("Continue", "Cancel"), "Continue"
        )
        self.predict(Request(Image.new("RGB", (224, 224), "white"), (q,)))
