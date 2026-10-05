"""CPU reference hashes for each tensor's FP32, FP16 and BF16 conversions."""

import argparse, hashlib, json
from pathlib import Path
import torch
from safetensors import safe_open

p = argparse.ArgumentParser()
p.add_argument("checkpoint", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
torch.set_num_threads(4)
rows = []
with safe_open(a.checkpoint / "model.safetensors", framework="pt", device="cpu") as f:
    for name in f.keys():
        x = f.get_tensor(name)
        row = {"name": name, "shape": list(x.shape), "source_dtype": str(x.dtype)}
        for key, dtype in [
            ("f32", torch.float32),
            ("f16", torch.float16),
            ("bf16", torch.bfloat16),
        ]:
            y = x.to(torch.float32).to(dtype).contiguous()
            row[key] = hashlib.sha256(y.view(torch.uint8).numpy().tobytes()).hexdigest()
        rows.append(row)
a.output.write_text(json.dumps(rows, indent=2))
print("WEIGHT_ORACLE", len(rows), flush=True)
