"""Export a CLM head checkpoint to safetensors, and record the conversion oracle.

A CLM checkpoint is a ``torch.save`` dict (``state_head``/``action_head`` state dicts,
``logit_scale``, ``cfg``), so it is a pickle and no non-Python reader can open it. This
writes the tensors to safetensors with the head name as a prefix, keeps the scalar and
config entries in the safetensors metadata, and emits the FP32/FP16/BF16 hash of every
tensor so a reader can be checked without comparing floats directly.

    python recipe/clm/native/export_weights.py CLM_v0.1-8B.pt OUT_DIR
    python recipe/clm/native/export_weights.py CLM_v0.1-8B.pt OUT_DIR --oracle oracle.json

Writes ``model.safetensors`` and, unless ``--no-oracle``, ``oracle.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

HEADS = ("state_head", "action_head")


def tensors(ckpt: dict) -> dict[str, torch.Tensor]:
    """Every parameter, prefixed by its head, in a stable order."""
    out: dict[str, torch.Tensor] = {}
    for head in HEADS:
        state = ckpt.get(head)
        if not isinstance(state, dict):
            raise SystemExit(f"checkpoint has no {head!r} state dict")
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise SystemExit(f"{head}.{name} is {type(value).__name__}, not a tensor")
            out[f"{head}.{name}"] = value.detach().to(torch.float32).contiguous()
    return out


def oracle(weights: dict[str, torch.Tensor]) -> list[dict]:
    rows = []
    for name, x in weights.items():
        row: dict = {"name": name, "shape": list(x.shape), "source_dtype": str(x.dtype)}
        for key, dtype in (("f32", torch.float32), ("f16", torch.float16), ("bf16", torch.bfloat16)):
            y = x.to(torch.float32).to(dtype).contiguous()
            row[key] = hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest()
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path, help="CLM_v0.1-8B.pt")
    parser.add_argument("output", type=Path, help="directory to write into")
    parser.add_argument("--oracle", type=Path, help="where to write the conversion oracle")
    parser.add_argument("--no-oracle", action="store_true", help="skip the oracle")
    args = parser.parse_args()

    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("cfg") or {})
    weights = tensors(ckpt)
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format": "clm-heads",
        "logit_scale": repr(float(ckpt["logit_scale"])),
        "hidden_size": str(int(ckpt.get("hidden_size") or cfg.get("hidden_size"))),
        "projection_dim": str(int(ckpt.get("projection_dim") or cfg.get("projection_dim"))),
        "cfg": json.dumps(cfg, sort_keys=True),
    }
    path = args.output / "model.safetensors"
    save_file(weights, str(path), metadata=metadata)

    params = sum(v.numel() for v in weights.values())
    print(f"WROTE {path} tensors={len(weights)} params={params}", flush=True)

    if not args.no_oracle:
        where = args.oracle or (args.output / "oracle.json")
        where.write_text(json.dumps(oracle(weights), indent=2) + "\n")
        print(f"ORACLE {where} rows={len(weights)}", flush=True)


if __name__ == "__main__":
    main()
