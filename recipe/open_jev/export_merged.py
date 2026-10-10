"""Export a pinned Open-Jev text backbone and scalar head on CPU.

Use the reference environment documented in native.md. This preparation step
runs without a GPU. Open-Jev-27B-v1.1 needs about 110 GB of host RAM and 52 GB
of output storage; Open-Jev-9B needs about 20 GB of RAM and 16 GB of storage.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

# Base model -> (base revision, checkpoint revision, scalar head width).
CHECKPOINTS = {
    "Qwen/Qwen3.8-27B": (
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "28cf73067d5b337860bbef3c85b8b82ba8730956",
        5120,
    ),
    "Qwen/Qwen3.5-9B": (
        "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "47e966881e489511c0c7f5633a9e1960a676a551",
        4096,
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "model.json").read_text())
    pins = CHECKPOINTS.get(config["model_id"])
    if pins is None or config["revision"] != pins[0]:
        raise ValueError("expected Open-Jev-27B-v1.1's or Open-Jev-9B's pinned base")
    base_revision, checkpoint_revision, width = pins
    if args.out.exists():
        raise ValueError("output already exists; choose a new export directory")
    if not 1 <= args.max_length <= 16384:
        raise ValueError("max length must be within 1..=16384")
    temperature = json.loads((args.checkpoint / "temperature.json").read_text())["temperature"]
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    marker = "\x00OMNI_OPEN_JEV\x00"
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    if chat.count(marker) != 1:
        raise ValueError("expected a single-user text chat template")
    prefix, suffix = chat.split(marker)
    head = torch.load(args.checkpoint / "head.pt", map_location="cpu", weights_only=True)
    if head["weight"].shape != (1, width) or head["bias"].shape != (1,):
        raise ValueError(f"expected a {width}-wide trained scalar head")
    if not all(torch.isfinite(v).all() for v in head.values()):
        raise ValueError("non-finite scalar head")
    full = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map={"": "cpu"},
        attn_implementation="sdpa", local_files_only=True,
    )
    backbone = full.model.language_model
    del full
    backbone = PeftModel.from_pretrained(backbone, args.checkpoint / "adapter")
    backbone = backbone.merge_and_unload(safe_merge=True)
    backbone.save_pretrained(args.out, max_shard_size="5GB")
    tokenizer.save_pretrained(args.out)
    # Written last: the native worker refuses incomplete exports or plain base weights.
    (args.out / "open_jev_export.json").write_text(json.dumps({
        "format": "open-jev-text-merged/1",
        "model_id": config["model_id"], "base_revision": base_revision,
        "checkpoint_revision": checkpoint_revision, "temperature": temperature,
        "max_length": args.max_length, "chat_prefix": prefix, "chat_suffix": suffix,
        "head_weight": head["weight"].float().reshape(-1).tolist(),
        "head_bias": head["bias"].float().item(),
    }, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
