"""Export the merged JEV-27B-VL text backbone plus the verbalizer label head, on CPU.

Streaming merge, low RAM: the script never materializes the whole model. It walks
the base checkpoint shard by shard, adds the trained vLLM-LoRA deltas
(``scale * (lora_B @ lora_A)``, ``scale = lora_alpha / r``) computed in float32 and
rounded once back to bfloat16, and rewrites only the ``model.language_model.*``
tensors under their original names, so the native backend's prefix autodetection
loads the output unchanged. ``lm_head.weight`` is merged the same way and the rows
the System-1 verbalizer readout needs (the union of the trained 24 head slots and
the tokenizer's single-token option labels) land in ``label_head.safetensors``
(float32). ``jev_vl_export.json`` — written last — carries the decision semantics:
per-kind temperatures, the 24-slot bias/ranges, the exported label list, and the
checkpoint pins. System-1 uses a raw prompt without a chat template.

See recipe/jev_vl/README.md for the pinned export environment.
No PEFT and no GPU are needed.
"""

import argparse
import hashlib
import json
import shutil
import string
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

EXPORT_FORMAT = "jev-vl-text-merged/1"
MODEL_ID = "autotrust/JEV-27B-VL"
PROTOCOL = "jev27-bare-v1"
MAX_OPTIONS = 256
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_revision(model: Path) -> str | None:
    meta = model / ".cache/huggingface/download/config.json.metadata"
    try:
        return meta.read_text().splitlines()[0].strip()
    except OSError:
        return None


def load_adapter(adapter: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    assert cfg["peft_type"] == "LORA" and cfg["inference_mode"] and cfg["lora_dropout"] == 0.0
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as f:
        for key in f.keys():
            assert key.startswith("base_model.model.") and key.endswith(".weight"), key
            base = key[len("base_model.model."):]
            name, kind = base.rsplit(".lora_", 1)
            name += ".weight"
            pairs.setdefault(name, {})[kind.split(".")[0].lower()] = f.get_tensor(key)
    out = {}
    for name, ab in pairs.items():
        assert set(ab) == {"a", "b"}, f"{name}: adapter pair incomplete"
        a, b = ab["a"].float(), ab["b"].float()
        assert a.shape[0] == cfg["r"] and b.shape[1] == cfg["r"], f"{name}: rank mismatch"
        out[name] = (a, b)
    return out, scale


def single_token_labels(tok, context: str) -> list[tuple[str, int]]:
    """serve_decide.py's option-label scan, verbatim."""
    out = []
    for lab in list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase
                                               for b in string.ascii_uppercase]:
        t = tok.encode(lab, add_special_tokens=False)
        if len(t) == 1 and t[0] in tok.encode(context.format(lab), add_special_tokens=False):
            out.append((lab, t[0]))
        if len(out) == MAX_OPTIONS:
            break
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=16384)
    args = parser.parse_args()
    model, out = args.model, args.out
    if out.exists():
        raise ValueError("output already exists; choose a new export directory")
    if not 1 <= args.max_length <= 32768:
        raise ValueError("max length must be within 1..=32768")
    adapter = model / "adapter_vllm"
    head_cfg = json.loads((adapter / "decision_head.json").read_text())
    temperatures = json.loads((model / "calibration.json").read_text())["per_kind"]
    assert sorted(head_cfg["slots"]["ranges"]) == ["choice", "noul", "score"]
    assert head_cfg["slots"]["ranges"]["noul"] == [0, 2]
    assert head_cfg["slots"]["ranges"]["choice"] == [8, 24]
    assert len(head_cfg["verbalizer_ids"]) == len(head_cfg["bias"]) == 24
    deltas, scale = load_adapter(adapter)
    assert "lm_head.weight" in deltas, "adapter must train lm_head"
    lm_delta = deltas.pop("lm_head.weight")

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    labels = single_token_labels(tokenizer, "x\n{}) y")
    assert len(labels) == MAX_OPTIONS, f"only {len(labels)} single-token labels"
    lo, hi = head_cfg["slots"]["ranges"]["choice"]
    assert [t for _, t in labels[: hi - lo]] == head_cfg["verbalizer_ids"][lo:hi], \
        "first labels must be the trained A-P head"
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    files = sorted(set(weight_map.values()))
    out.mkdir(parents=True)
    consumed: set[str] = set()
    new_map: dict[str, list[str]] = {}
    print(f"merging with LoRA scale={scale} from {adapter}", flush=True)
    for shard in files:
        merged: dict[str, torch.Tensor] = {}
        with safe_open(str(model / shard), framework="pt") as f:
            for name in sorted(f.keys()):
                if not name.startswith("model.language_model."):
                    continue
                w = f.get_tensor(name)
                if name in deltas:
                    a, b = deltas[name]
                    w = (w.float() + scale * (b @ a)).to(torch.bfloat16)
                    consumed.add(name)
                merged[name] = w.contiguous()
        if merged:
            save_file(merged, str(out / shard))
            for name in merged:
                new_map.setdefault(shard, []).append(name)
            print(f"  {shard}: {len(merged)} language tensors", flush=True)
    missing = set(deltas) - consumed
    assert not missing, f"adapter tensors without a base tensor: {sorted(missing)[:4]}"
    (out / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 0, "jev_vl_export": EXPORT_FORMAT},
        "weight_map": {name: shard for shard, names in sorted(new_map.items()) for name in sorted(names)},
    }) + "\n")

    # The merged label rows for the verbalizer readout (float32).
    head_ids = list(dict.fromkeys(head_cfg["verbalizer_ids"] + [t for _, t in labels]))
    lm_shard = weight_map["lm_head.weight"]
    with safe_open(str(model / lm_shard), framework="pt") as f:
        lm_head = f.get_tensor("lm_head.weight")
    a, b = lm_delta
    idx = torch.tensor(head_ids)
    rows = lm_head[idx].float() + scale * (b[idx] @ a)
    assert torch.isfinite(rows).all(), "non-finite merged label rows"
    save_file({"rows": rows.contiguous(),
               "ids": idx.to(torch.int64)}, str(out / "label_head.safetensors"))

    for name in TOKENIZER_FILES:
        src = model / name
        if src.exists():
            shutil.copy2(src, out / name)
    shutil.copy2(model / "config.json", out / "config.json")

    manifest = {
        "format": EXPORT_FORMAT,
        "model_id": MODEL_ID,
        "protocol": PROTOCOL,
        "hf_revision": hf_revision(model),
        "pins": {
            "model_index_sha256": sha256(model / "model.safetensors.index.json"),
            "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
            "adapter_model_sha256": sha256(adapter / "adapter_model.safetensors"),
            "decision_head_sha256": sha256(adapter / "decision_head.json"),
            "calibration_sha256": sha256(model / "calibration.json"),
        },
        "merge": {"lora_scale": scale, "delta_dtype": "float32", "backbone_dtype": "bfloat16"},
        "temperatures": temperatures,
        "verbalizer_ids": head_cfg["verbalizer_ids"],
        "verbalizer_bias": head_cfg["bias"],
        "slots": head_cfg["slots"],
        "labels": [l for l, _ in labels],
        "label_ids": [t for _, t in labels],
        "label_head": {"file": "label_head.safetensors", "ids": head_ids, "dtype": "float32"},
        "max_length": args.max_length,
    }
    # Written last: the native worker refuses incomplete exports or plain base weights.
    (out / "jev_vl_export.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    print(f"exported {len(new_map)} shards + {len(head_ids)} label rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
