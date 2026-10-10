"""Offline CPU export of pinned JEMM, one safetensors shard at a time.

No full Transformers model is instantiated. Raw inputs remain untouched. The
completion marker is written last; restart with the same arguments to verify
and reuse atomically completed shards under the same producer script/runtime.
Changed producers require a new export directory. Language shards and the FP32
adapter are hardlinked or copied byte for byte, never merged into BF16 weights.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_MODEL_ID = "Qwen/Qwen3.8-27B"
PINS = {
    "base_model_id": BASE_MODEL_ID,
    "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "checkpoint_revision": "76e3c209e8441fa658221c7ba2725bad2f811176",
    "source_revision": "6822fe0fd53c5e6670af6ba99fb2c857a661e532",
}
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
SYSTEM = ("Choose the best available candidate for the question using only the supplied state. "
          "Return exactly one candidate label.")
CALIBRATION = {"temperature": 1.3480874159655591, "mm_temperature": 1.3954832341582943,
               "threshold": 0.9872681877423998}
INVENTORY_PATH = Path(__file__).with_name("pinned_inventory.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def producer_runtime():
    versions = {}
    for name in ("safetensors", "transformers", "tokenizers"):
        try:
            versions[name + "_version"] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name + "_version"] = None
    return {"python_version": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(), **versions}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_tensors(path, tensors):
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata={"format": "pt"})
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_original(source, destination):
    """Publish an unchanged regular source file, using a hardlink where possible."""
    source = Path(source).resolve(strict=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def validate_artifacts(directory, inventory):
    """Verify every pinned file's bytes and tensor inventory without materializing weights."""
    hashes = {}
    for name, expected in inventory["files"].items():
        if Path(name).name != name:
            raise ValueError("inventory contains an unsafe filename")
        path = directory / name
        if not path.is_file():
            raise ValueError(f"missing pinned artifact: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch: {path}")
        hashes[name] = actual
    observed = {}
    for name in sorted({v["file"] for v in inventory["tensors"].values()}):
        with safe_open(str(directory / name), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in observed:
                    raise ValueError(f"duplicate tensor: {key}")
                tensor = reader.get_slice(key)
                observed[key] = {"shape": tensor.get_shape(), "dtype": tensor.get_dtype(), "file": name}
    if observed != inventory["tensors"]:
        missing = sorted(set(inventory["tensors"]) - set(observed))
        extra = sorted(set(observed) - set(inventory["tensors"]))
        mismatch = sorted(k for k in observed.keys() & inventory["tensors"].keys() if observed[k] != inventory["tensors"][k])
        raise ValueError(f"tensor inventory mismatch: missing={missing[:8]} extra={extra[:8]} shape/dtype/file={mismatch[:8]}")
    return hashes


def adapter_pairs(base_inventory, adapter_inventory, config):
    if config.get("peft_type") != "LORA" or config.get("base_model_name_or_path") != BASE_MODEL_ID:
        raise ValueError("adapter must target pinned Qwen3.8-27B")
    for key in ("use_dora", "use_rslora", "use_qalora", "fan_in_fan_out", "lora_bias"):
        if config.get(key):
            raise ValueError(f"unsupported adapter option: {key}")
    for key in ("rank_pattern", "alpha_pattern", "modules_to_save", "target_parameters", "layer_replication"):
        if config.get(key):
            raise ValueError(f"unsupported adapter option: {key}")
    if config.get("bias", "none") != "none":
        raise ValueError("adapter bias unsupported")
    rank, alpha = config["r"], config["lora_alpha"]
    if type(rank) is not int or rank < 1 or type(alpha) not in (int, float) or not 0 < alpha < float("inf"):
        raise ValueError("invalid LoRA rank/alpha")
    pairs = {}
    for name, spec in adapter_inventory.items():
        prefix = "base_model.model."
        if not name.startswith(prefix):
            raise ValueError(f"unexpected adapter prefix: {name}")
        target, letter = None, None
        for candidate in ("A", "B"):
            suffix = f".lora_{candidate}.weight"
            if name.endswith(suffix):
                target, letter = name[len(prefix):-len(suffix)] + ".weight", candidate
        if target is None or not target.startswith("model.language_model.layers."):
            raise ValueError(f"adapter targets unsupported module: {name}")
        if target not in base_inventory or spec["dtype"] != "F32":
            raise ValueError(f"adapter lacks matching base weight: {name}")
        pair = pairs.setdefault(target, {})
        if letter in pair:
            raise ValueError(f"duplicate LoRA component: {name}")
        pair[letter] = name
    for target, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"unpaired LoRA tensor: {target}")
        rows, cols = base_inventory[target]["shape"]
        if adapter_inventory[pair["A"]]["shape"] != [rank, cols] or adapter_inventory[pair["B"]]["shape"] != [rows, rank]:
            raise ValueError(f"LoRA shape mismatch: {target}")
    return pairs, alpha / rank


def export_checkpoint(base, adapter, out, provenance, *, tokenizer=None, inventory=None):
    """Public CLI uses the bundled pinned inventory. Tiny CPU tests inject a toy inventory."""
    base, adapter, out = Path(base).resolve(), Path(adapter).resolve(), Path(out).resolve()
    if out == base or out == adapter or base in out.parents or adapter in out.parents:
        raise ValueError("export directory must be separate from source checkpoints")
    if any(provenance.get(key) != value for key, value in PINS.items()):
        raise ValueError("download provenance does not match pinned revisions")
    manifest_path, progress_path = out / "jemm_export.json", out / "export_progress.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("format") != "jemm-native/2":
        raise ValueError("legacy export is unsupported; jemm-native/2 requires a new export directory")
    inventory = inventory or json.loads(INVENTORY_PATH.read_text())
    source_hashes = {"base": validate_artifacts(base, inventory["base"]), "adapter": validate_artifacts(adapter, inventory["adapter"])}
    index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
    expected_map = {key: value["file"] for key, value in inventory["base"]["tensors"].items()}
    if index != expected_map:
        raise ValueError("base index disagrees with pinned safetensors inventory")
    config = json.loads((base / "config.json").read_text())
    text = config["text_config"]
    if config.get("tie_word_embeddings", False) or text.get("tie_word_embeddings", False):
        raise ValueError("JEMM needs the original untied lm_head")
    if inventory["base"]["tensors"]["lm_head.weight"]["shape"] != [text["vocab_size"], text["hidden_size"]]:
        raise ValueError("untied head shape disagrees with text configuration")
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    pairs, scale = adapter_pairs(inventory["base"]["tensors"], inventory["adapter"]["tensors"], adapter_config)
    if inventory.get("provenance") == PINS and (adapter_config["r"] != 16 or len(pairs) != 496 or scale != 2.):
        raise ValueError("pinned JEMM requires rank16, 496 FP32 LoRA pairs and scale2")
    calibration = json.loads((adapter / "decision_config.json").read_text())
    if any(calibration.get(key) != value for key, value in CALIBRATION.items()):
        raise ValueError("decision calibration differs from pinned JEMM")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True)
    label_ids = []
    for label in LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1 or not isinstance(ids[0], int) or not 0 <= ids[0] < text["vocab_size"]:
            raise ValueError(f"label is not one valid vocabulary token: {label}")
        label_ids.append(ids[0])
    if len(set(label_ids)) != len(LABELS):
        raise ValueError("label token IDs must be unique")
    if inventory.get("provenance") == PINS and label_ids != list(range(32, 58)) + list(range(15, 21)):
        raise ValueError("tokenizer label IDs differ from the pinned JEMM readout")
    marker = "\x00SYSTEM1_JEMM_USER\x00"
    chat = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": marker}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False, preserve_thinking=False)
    if chat.count(marker) != 1:
        raise ValueError("chat template did not preserve a unique user marker")
    chat_prefix, chat_suffix = chat.split(marker)
    producer = {"exporter_sha256": sha256(Path(__file__)), "runtime": producer_runtime()}
    contract = {"format": "jemm-native/2", "lora_mode": "unmerged_fp32", "pins": PINS, "inventory_sha256": canonical_hash(inventory), "source_sha256": source_hashes,
                "label_token_ids": label_ids, "chat_prefix": chat_prefix, "chat_suffix": chat_suffix, "producer": producer}
    fingerprint = canonical_hash(contract)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("exporter_sha256") != producer["exporter_sha256"] or manifest.get("producer_runtime") != producer["runtime"]:
            raise ValueError("existing export belongs to different producer; preserve it and choose a new export directory")
        if manifest.get("export_fingerprint") != fingerprint:
            raise ValueError("existing export belongs to different source artifacts or contract")
        for name, expected in manifest["export_sha256"].items():
            if sha256(out / name) != expected:
                raise ValueError(f"completed export checksum mismatch: {name}")
        return manifest
    if out.exists() and not progress_path.exists():
        raise ValueError("output directory exists without resumable export provenance")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"fingerprint": fingerprint, "producer": producer, "completed": {}}
    if progress.get("producer") != producer:
        raise ValueError("partial export belongs to different producer; preserve it and choose a new export directory")
    if progress.get("fingerprint") != fingerprint:
        raise ValueError("partial export belongs to different source artifacts or contract")
    # Check all adapter values before creating output, including projections in later shards.
    adapter_file = adapter / "adapter_model.safetensors"
    with safe_open(str(adapter_file), framework="pt", device="cpu") as lora:
        for key in lora.keys():
            if not torch.isfinite(lora.get_tensor(key)).all():
                raise ValueError(f"nonfinite adapter tensor: {key}")
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(progress_path, progress)
        weight_map, total_bytes = {}, 0
        for number, shard in enumerate(sorted(set(index.values())), 1):
            completed = progress["completed"].get(shard)
            if completed is not None:
                for name, expected in completed["outputs"].items():
                    if sha256(out / name) != expected:
                        raise ValueError(f"partial export checksum mismatch: {name}")
                weight_map.update(completed["weight_map"])
                total_bytes += completed["language_bytes"]
                continue
            language, visual, selected_head = {}, {}, None
            with safe_open(str(base / shard), framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    if key == "lm_head.weight":
                        # Reading individual rows avoids materializing the 2.54 GB full head.
                        view = reader.get_slice(key)
                        selected_head = torch.cat([view[token:token + 1] for token in label_ids], dim=0).contiguous()
                    elif key.startswith("model.visual."):
                        visual[key] = reader.get_tensor(key)
                        if not torch.isfinite(visual[key]).all():
                            raise ValueError(f"nonfinite vision tensor: {key}")
                    elif key.startswith("model.language_model."):
                        value = reader.get_tensor(key)
                        if not torch.isfinite(value).all():
                            raise ValueError(f"nonfinite base tensor: {key}")
                        language[key] = inventory["base"]["tensors"][key]
                        del value
                    elif not key.startswith("mtp."):
                        raise ValueError(f"unexpected base tensor outside validated native scope: {key}")
                outputs, mapping, language_bytes = {}, {}, 0
                if language:
                    filename = shard
                    atomic_original(base / shard, out / filename)
                    outputs[filename] = sha256(out / filename)
                    if outputs[filename] != source_hashes["base"][shard]:
                        raise ValueError(f"original language shard checksum mismatch: {shard}")
                    mapping = {key: filename for key in language}
                    language_bytes = sum(math.prod(spec["shape"]) * 2 for spec in language.values())
                if visual:
                    filename = "vision.safetensors"
                    if any(filename in x["outputs"] for x in progress["completed"].values()):
                        raise ValueError("vision spans multiple shards; bundled inventory expected one")
                    atomic_tensors(out / filename, visual)
                    outputs[filename] = sha256(out / filename)
                if selected_head is not None:
                    if not torch.isfinite(selected_head).all():
                        raise ValueError("nonfinite selected head")
                    atomic_tensors(out / "jemm_lm_head.safetensors", {"weight": selected_head})
                    outputs["jemm_lm_head.safetensors"] = sha256(out / "jemm_lm_head.safetensors")
            progress["completed"][shard] = {"source_sha256": source_hashes["base"][shard], "outputs": outputs,
                                              "weight_map": mapping, "language_bytes": language_bytes}
            atomic_json(progress_path, progress)
            weight_map.update(mapping)
            total_bytes += language_bytes
            del language, visual, selected_head
            gc.collect()
            print(f"completed {number}/{len(set(index.values()))}: {shard}", flush=True)
    atomic_json(out / "model.safetensors.index.json", {"metadata": {"total_size": total_bytes}, "weight_map": weight_map})
    for name in inventory["base"]["files"]:
        if not name.endswith(".safetensors") and name != "model.safetensors.index.json":
            temporary = out / (name + ".tmp")
            shutil.copyfile(base / name, temporary)
            os.replace(temporary, out / name)
    for name in ("adapter_config.json", "decision_config.json", "adapter_model.safetensors"):
        atomic_original(adapter / name, out / name)
    exports = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and not p.name.endswith(".tmp") and p.name != "export_progress.json"}
    if exports["adapter_model.safetensors"] != source_hashes["adapter"]["adapter_model.safetensors"]:
        raise ValueError("original FP32 adapter checksum mismatch")
    if "jemm_lm_head.safetensors" not in exports or "vision.safetensors" not in exports:
        raise ValueError("export lacks selected head or vision tensors")
    manifest = {"format": "jemm-native/2", "model_id": "JEMM", **PINS, **CALIBRATION,
                "max_tokens": 8192, "max_mm_tokens": 3072, "chat_prefix": chat_prefix, "chat_suffix": chat_suffix,
                "enable_thinking": False, "preserve_thinking": False, "labels": list(LABELS), "label_token_ids": label_ids,
                "label_head_file": "jemm_lm_head.safetensors", "label_head_tensor": "weight", "label_head_dtype": "BF16",
                "label_head_shape": [32, text["hidden_size"]], "untied_lm_head": True, "vision_file": "vision.safetensors",
                "lora_mode": "unmerged_fp32", "lora_file": "adapter_model.safetensors",
                "lora_rank": adapter_config["r"], "lora_pairs": len(pairs), "lora_scale": scale,
                "lora_sha256": source_hashes["adapter"]["adapter_model.safetensors"], "source_sha256": source_hashes,
                "upstream_source_sha256": inventory.get("upstream_source_sha256", {}), "export_sha256": exports,
                "inventory_sha256": canonical_hash(inventory), "exporter_sha256": producer["exporter_sha256"],
                "producer_runtime": producer["runtime"],
                "download_provenance": provenance, "export_fingerprint": fingerprint,
                "unindexed_tensors": [k for k in index if k.startswith("mtp.") or k == "lm_head.weight"]}
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True, help="JSON with the four pinned revision/model fields and download provenance")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    export_checkpoint(args.base, args.adapter, args.out, json.loads(args.provenance.read_text()))


if __name__ == "__main__":
    main()
