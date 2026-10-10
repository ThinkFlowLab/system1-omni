"""Reference decisions for the CLM heads, for the Rust port to be checked against.

Reads the exported safetensors (not the .pt) so both sides load byte-identical weights,
computes a decision for each fixed case with plain NumPy, and writes the result as JSON.

    python recipe/clm/native/head_oracle.py EXPORT_DIR OUT.json

The embeddings here are synthesised from the case name, so the numbers are meaningless
as model output -- the point is that two implementations of the same arithmetic agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

HIDDEN = 4096


def embedding(text: str, dim: int) -> np.ndarray:
    """A stable unit vector per text, so the cases are reproducible without an encoder."""
    out: list[float] = []
    counter = 0
    while len(out) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        for i in range(0, len(digest) - 3, 4):
            if len(out) == dim:
                break
            out.append(int.from_bytes(digest[i:i + 4], "big") / 2**31 - 1.0)
        counter += 1
    x = np.asarray(out, dtype=np.float32)
    return x / np.linalg.norm(x)


def load_head(f, prefix: str, width: int, proj: int, hidden: int, blocks: int) -> dict:
    def get(name: str) -> np.ndarray:
        return np.asarray(f.get_tensor(f"{prefix}.{name}"), dtype=np.float32)

    head = {
        "inp_w": get("inp.weight"), "inp_b": get("inp.bias"),
        "out_w": get("out.weight"), "out_b": get("out.bias"),
    }
    for i in range(blocks):
        head[f"hidden{i}_w"] = get(f"hidden.{i}.weight")
        head[f"hidden{i}_b"] = get(f"hidden.{i}.bias")
        head[f"norm{i}_w"] = get(f"norms.{i}.weight")
        head[f"norm{i}_b"] = get(f"norms.{i}.bias")
    return head


def gelu(x: np.ndarray) -> np.ndarray:
    # The erf form, which is what torch.nn.GELU() uses by default.
    from math import sqrt

    return 0.5 * x * (1.0 + _erf(x / sqrt(2.0)))


def _erf(x: np.ndarray) -> np.ndarray:
    """Vectorised erf via the same A&S 7.1.26 form the Rust side uses."""
    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.0614054 * t - 1.45315203) * t) + 1.42141374) * t - 0.284496736) * t
               + 0.2548296) * t * np.exp(-x * x)
    return sign * y


def layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def project(head: dict, cfg: dict, x: np.ndarray) -> np.ndarray:
    h = gelu(x @ head["inp_w"].T + head["inp_b"])
    blocks = cfg["depth"] - 2
    for i in range(blocks):
        z = h @ head[f"hidden{i}_w"].T + head[f"hidden{i}_b"]
        if cfg["layernorm"]:
            z = layernorm(z, head[f"norm{i}_w"], head[f"norm{i}_b"])
        z = gelu(z)
        h = h + z if cfg["residual"] else z
    out = h @ head["out_w"].T + head["out_b"]
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


def confidence(probs: np.ndarray) -> float:
    if probs.size < 2:
        return 1.0
    j = int(np.argmax(probs))
    rest = np.delete(probs, j).mean()
    return float(min(1.0, max(0.0, probs[j] - rest)))


CASES = [
    ("choice_binary", "choice", ["billing", "technical"]),
    ("choice_five", "choice", ["a", "b", "c", "d", "e"]),
    ("score_three", "score", ["0", "1", "2"]),
    ("noul", "noul", ["false", "true"]),
    ("choice_single", "choice", ["only"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("export", type=Path, help="directory holding model.safetensors")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with safe_open(args.export / "model.safetensors", framework="np") as f:
        md = f.metadata()
        cfg = json.loads(md["cfg"])
        hidden = int(md["hidden_size"])
        proj = int(md["projection_dim"])
        width = cfg["width"]
        blocks = cfg["depth"] - 2
        # The reference caps this: heads.py does exp(logit_scale).clamp(max=100.0),
        # and the published checkpoint's 4.6132 exponentiates to 100.82, so the cap
        # binds. Without it every probability is about 0.8 % off.
        scale = min(float(np.exp(np.float32(md["logit_scale"]))), 100.0)
        state = load_head(f, "state_head", width, proj, hidden, blocks)
        action = load_head(f, "action_head", width, proj, hidden, blocks)

    rows = []
    for name, kind, keys in CASES:
        state_vec = embedding(f"state::{name}", hidden)
        cand_vecs = np.stack([embedding(f"cand::{name}::{k}", hidden) for k in keys])
        zs = project(state, cfg, state_vec[None, :])[0]
        zc = project(action, cfg, cand_vecs)
        cos = zc @ zs
        temperature = 1.0 if name != "choice_five" else 2.5
        probs = softmax((np.float32(scale) * cos / np.float32(temperature)).astype(np.float32))
        row = {
            "name": name, "kind": kind, "keys": keys, "temperature": temperature,
            "probabilities": [float(p) for p in probs],
        }
        if kind == "choice":
            row["choice"] = keys[int(np.argmax(probs))]
            row["confidence"] = confidence(probs)
        elif kind == "score":
            row["score"] = float(sum(i * float(p) for i, p in enumerate(probs)))
            row["confidence"] = confidence(probs)
        else:
            row["noul"] = float(probs[keys.index("true")])
        rows.append(row)

    args.output.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    print(f"ORACLE {args.output} cases={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
