"""Direct Transformers/PEFT execution. No production dependency on cua_s1."""

from __future__ import annotations

import copy
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
            if (
                path.is_file()
                and relative.parts[0] != ".cache"
                and relative.as_posix() not in names
            ):
                raise ValueError(
                    f"unlisted artifact may override pinned files: {relative}"
                )


class _RequestImageProcessor:
    """Reuse one image result; each processor call owns its mutable mapping."""

    def __init__(self, processor):
        self.processor = processor
        self.result = None

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, *args, **kwargs):
        if self.result is None:
            self.result = self.processor(*args, **kwargs)
        # BatchFeature.to and processor token expansion can mutate mappings.
        # Share only tensor values, which the pinned processor does not modify.
        return type(self.result)(dict(self.result))


class MultimodalEngine:
    def __init__(
        self,
        base: str,
        adapter: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        graph_config=None,
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
        self.graph_runtime = None
        if graph_config is not None:
            from .graph_runtime import GraphRuntime

            self.graph_runtime = GraphRuntime(self.model, graph_config)

    def prepare(self, image, question: Question):
        return self._prepare(self.processor, image, question)

    @staticmethod
    def _prepare(processor, image, question: Question):
        messages = build_messages(question)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        if inputs["input_ids"].shape[-1] > MAX_TOKENS:
            raise InvalidRequest(f"processed prompt exceeds {MAX_TOKENS} tokens")
        return inputs

    def prepare_reused(self, image, questions):
        """Tokenize each question normally, preprocessing this request's image once."""
        processor = copy.copy(self.processor)
        processor.image_processor = _RequestImageProcessor(
            self.processor.image_processor
        )
        return [self._prepare(processor, image, question) for question in questions]

    def encode_image(self, inputs):
        """Encode once through the vision modules with their active LoRA adapters."""
        import torch

        core = self.model.get_base_model().model
        with torch.no_grad():
            output = core.get_image_features(
                inputs["pixel_values"].to(self.model.device),
                inputs["image_grid_thw"].to(self.model.device),
                return_dict=True,
            )
            return torch.cat(output.pooler_output, dim=0)

    def score_reused(self, inputs, question: Question, features) -> list[float]:
        """Build fresh text embeddings and 3D positions around shared image features."""
        import torch

        ids = letter_ids(self.tokenizer, len(question.keys))
        # Keep prepared CPU inputs intact and avoid another image transfer.
        text_inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
            if key != "pixel_values"
        }
        core = self.model.get_base_model().model
        with torch.no_grad():
            input_ids = text_inputs.pop("input_ids")
            embeds = core.get_input_embeddings()(input_ids)
            image_embeds = features.to(embeds.device, embeds.dtype)
            image_mask, _ = core.get_placeholder_mask(
                input_ids, inputs_embeds=embeds, image_features=image_embeds
            )
            embeds = embeds.masked_scatter(image_mask, image_embeds)
            position_ids, _ = core.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=text_inputs.pop("mm_token_type_ids"),
                image_grid_thw=text_inputs.pop("image_grid_thw"),
                attention_mask=text_inputs.get("attention_mask"),
            )
            # Candidate scoring reads only the final position.
            graph_runtime = getattr(self, "graph_runtime", None)
            if graph_runtime is None:
                output = self.model(
                    inputs_embeds=embeds,
                    position_ids=position_ids,
                    logits_to_keep=1,
                    **text_inputs,
                )
                logits = output.logits[0, -1, :]
            else:
                logits = graph_runtime.forward(
                    {
                        "inputs_embeds": embeds,
                        "position_ids": position_ids,
                        **text_inputs,
                    }
                )
        return torch.softmax(
            logits[torch.tensor(ids, device=logits.device)].float(), dim=-1
        ).tolist()

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

    def predict_reference(self, request: Request) -> dict:
        """Original full-processor/full-model path retained for paired experiments."""
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

    def predict(self, request: Request) -> dict:
        if len(request.questions) == 1:
            return self.predict_reference(request)
        # No image encoder or language model runs until every prompt is valid.
        prepared = self.prepare_reused(request.image, request.questions)
        features = self.encode_image(prepared[0])
        answers = {
            q.name: answer(q, self.score_reused(inputs, q, features))
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
