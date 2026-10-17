"""Export the pinned JEV vision tower for the shared Rust/CUDA backend (CPU only)."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(model, out):
    if out.exists():
        raise ValueError("output already exists; choose a new export directory")
    config = json.loads((model / "config.json").read_text())
    if (config.get("model_type") != "qwen3_5"
            or config.get("vision_config", {}).get("hidden_size") != 1152
            or config.get("vision_config", {}).get("out_hidden_size") != 5120
            or config.get("text_config", {}).get("hidden_size") != 5120):
        raise ValueError("expected the JEV-27B-VL Qwen3.8 backbone")
    processor = json.loads((model / "preprocessor_config.json").read_text())
    expected_processor = {
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast",
    }
    if processor != expected_processor:
        raise ValueError("unsupported image processor; verify resize and normalization first")

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    adapter = model / "adapter_vllm"
    with safe_open(adapter / "adapter_model.safetensors", framework="pt") as weights:
        for name in weights.keys():
            if not name.startswith(("base_model.model.model.language_model.",
                                    "base_model.model.lm_head.")):
                raise ValueError(f"unsupported non-language adapter tensor: {name}")

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    names = {name for name in index if name.startswith("model.visual.")}
    if not names:
        raise ValueError("checkpoint contains no vision tensors")
    tensors = {}
    for shard in sorted({index[name] for name in names}):
        path = (model / shard).resolve()
        if not path.is_relative_to(model.resolve()):
            raise ValueError("checkpoint shard escapes model directory")
        with safe_open(path, framework="pt") as weights:
            for name in sorted(names):
                if index[name] != shard:
                    continue
                tensor = weights.get_tensor(name)
                if tensor.dtype != torch.bfloat16 or not torch.isfinite(tensor).all():
                    raise ValueError(f"expected finite BF16 vision weights: {name}")
                tensors[name] = tensor.contiguous()

    out.mkdir(parents=True)
    save_file(tensors, out / "vision.safetensors")
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copyfile(model / name, out / name)
    manifest = {
        "format": "jev-vl-vision/1",
        "model_id": "autotrust/JEV-27B-VL",
        "pins": {
            "model_index_sha256": sha256(model / "model.safetensors.index.json"),
            "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
            "adapter_model_sha256": sha256(adapter / "adapter_model.safetensors"),
        },
        "files": {name: sha256(out / name) for name in
                  ("config.json", "preprocessor_config.json", "vision.safetensors")},
    }
    # The worker requires this manifest, so a partial export is never loadable.
    (out / "jev_vl_vision.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported {len(tensors)} vision tensors -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.model, args.out)


if __name__ == "__main__":
    main()
