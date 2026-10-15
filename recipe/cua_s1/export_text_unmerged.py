"""Export BF16 language weights with the original FP32 text adapter kept separate."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def validate_adapter(config):
    targets = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    if config.get("r") != 16 or config.get("lora_alpha") != 32:
        raise ValueError("Cua-S1 0.2 text requires rank 16 and alpha 32")
    if set(config.get("target_modules", [])) != targets:
        raise ValueError("expected the Cua-S1 text adapter, not a multimodal adapter")
    if (
        config.get("bias", "none") != "none"
        or config.get("lora_bias")
        or config.get("use_dora")
        or config.get("use_rslora")
        or config.get("fan_in_fan_out")
        or config.get("modules_to_save")
        or config.get("rank_pattern")
        or config.get("alpha_pattern")
    ):
        raise ValueError("unsupported text adapter scaling or projection policy")


def export(base: Path, adapter: Path, out: Path):
    validate_adapter(json.loads((adapter / "adapter_config.json").read_text()))
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory must be empty")
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_file = adapter / "adapter_model.safetensors"
    with safe_open(adapter_file, framework="pt", device="cpu") as tensors:
        if len(tensors.keys()) != 256 or any(
            tensors.get_slice(name).get_dtype() != "F32" for name in tensors.keys()
        ):
            raise ValueError("expected the complete 256-tensor FP32 text adapter")
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map="cpu"
    )
    if model.config.hidden_size != 2560 or model.config.num_hidden_layers != 32:
        raise ValueError("expected the Qwen3.5-4B language model")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    shutil.copyfile(adapter_file, out / "adapter.safetensors")
    with (out / "adapter.safetensors").open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    (out / "cua_s1_export.json").write_text(
        json.dumps(
            {
                "format": "cua-s1-text-unmerged/1",
                "lora_scale": 2.0,
                "adapter_sha256": digest,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export(args.base, args.adapter, args.out)


if __name__ == "__main__":
    main()
