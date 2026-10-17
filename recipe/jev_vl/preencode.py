"""Pre-encode manifest images into the (url-hash keyed) adapted-embedding assets.

Runs the official HF vision tower (same model the vLLM reference path used) on the
GPU once per unique image in a manifest, and stores per-image
``<out>/<sha256(url)>/{emb.safetensors,grid.json}`` with the adapted rows
(``model.model.visual(pixel_values, grid_thw).pooler_output``, shape
[n = prod(grid)/merge^2, 5120], bfloat16) plus the patch grid. The L2 cache holds
these assets in prepared-image mode. Online mode uses its own native vision tower.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path


def load_vision(model_dir: Path) -> "torch.nn.Module":
    """Hand-assemble the vision tower: no accelerate, no full-model device map."""
    import safetensors
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    vis = Qwen3_5VisionModel(cfg.vision_config)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    names = {n for n in index if n.startswith("model.visual.")}
    tensors = {}
    for f in sorted({index[n] for n in names}):
        with safetensors.safe_open(str(model_dir / f), framework="pt") as h:
            for n in h.keys():
                if n in names:
                    tensors[n[len("model.visual."):]] = h.get_tensor(n)
    missing, unexpected = vis.load_state_dict(tensors, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return vis.to(torch.bfloat16).to("cuda").eval()


def images_of(manifest: Path) -> list[str]:
    urls = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)["request"]
        state = r.get("state", "")
        if isinstance(state, list):
            for p in state:
                if not isinstance(p, dict):
                    continue
                # Match contract::parts: untyped image_url fields are text,
                # and the image shorthand takes precedence over typed parts.
                if "image" in p:
                    url = p["image"]
                elif p.get("type") == "image_url":
                    image_url = p.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else None
                else:
                    continue
                if not isinstance(url, str) or not url:
                    raise ValueError("image part must contain a nonempty URL string")
                urls.append(url)
    return sorted(set(urls))


def decode_url(url: str):
    from PIL import Image

    if not url.startswith("data:"):
        raise ValueError("recipe supports data: URIs (manifest uses them)")
    header, payload = url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("data URI without base64 encoding")
    import base64

    return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    urls = images_of(args.manifest)
    print(f"{len(urls)} unique images", flush=True)
    if not urls:
        return
    import torch
    from safetensors.torch import save_file
    from transformers import AutoProcessor

    model_dir = args.model
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = load_vision(model_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    for url in urls:
        key = hashlib.sha256(url.encode()).hexdigest()
        dest = args.out / key
        if (dest / "emb.safetensors").is_file() and (dest / "grid.json").is_file():
            print(f"{key}: cached", flush=True)
            continue
        im = decode_url(url)
        inputs = processor(images=[im], return_tensors="pt")
        grid = inputs["image_grid_thw"][0].tolist()
        pixel_values = inputs["pixel_values"].to("cuda", dtype=torch.bfloat16)
        grid_thw = torch.tensor([grid], device="cuda")
        with torch.no_grad():
            emb = model(pixel_values, grid_thw=grid_thw).pooler_output
        emb = emb.detach().cpu().to(torch.bfloat16)
        n = torch.prod(torch.tensor(grid)).item() // model.spatial_merge_size**2
        assert emb.shape == (n, 5120), f"{key}: {emb.shape} vs n={n}"
        assert torch.isfinite(emb.float()).all(), f"{key}: non-finite"
        dest.mkdir(parents=True, exist_ok=True)
        # Publish metadata last so an interrupted write is retried on the next run.
        (dest / "grid.json").unlink(missing_ok=True)
        save_file({"rows": emb.contiguous()}, str(dest / "emb.safetensors"))
        grid_tmp = dest / "grid.json.tmp"
        grid_tmp.write_text(json.dumps({
            "url_sha256": key, "grid_thw": grid, "n_tokens": n, "dtype": "bfloat16",
            "shape": [int(emb.shape[0]), int(emb.shape[1])],
            "model_index_sha256": hashlib.sha256((model_dir / "model.safetensors.index.json")
                                                 .read_bytes()).hexdigest(),
        }) + "\n")
        grid_tmp.replace(dest / "grid.json")
        print(f"{key}: grid={grid} n={n} rows={tuple(emb.shape)}", flush=True)


if __name__ == "__main__":
    main()
