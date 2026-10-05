"""Verify export integrity and multimodal tensor relations; optionally compare runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from export_multimodal_reference import SCHEMA, sha256, tensor_info
from safetensors.torch import load_file

from models.cua_s1.multimodal.protocol import answer, decode_request, parse_request

REQUIRED = {
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "pixel_values",
    "image_grid_thw",
    "image_features",
    "image_token_indices",
    "inputs_embeds",
    "position_ids",
    "rope_deltas",
    "last_hidden_state",
    "candidate_token_ids",
    "candidate_logits",
    "candidate_probabilities",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_files(folder, manifest):
    for name, info in manifest["files"].items():
        path = folder / name
        require(
            path.resolve().is_relative_to(folder.resolve()), f"unsafe file path: {name}"
        )
        require(path.is_file(), f"missing file: {name}")
        require(
            path.stat().st_size == info["size"]
            and sha256(path.read_bytes()) == info["sha256"],
            f"file checksum: {name}",
        )

    actual = {
        path.relative_to(folder).as_posix()
        for path in folder.rglob("*")
        if path.is_file() and path != folder / "manifest.json"
    }
    require(actual == set(manifest["files"]), "file inventory mismatch")


def check_tensors(tensors, entry, configs):
    require(
        set(tensors) == REQUIRED,
        f"missing tensors or unknown keys: {set(tensors) ^ REQUIRED}",
    )
    require(
        {name: tensor_info(value) for name, value in tensors.items()}
        == entry["tensors"],
        "tensor fingerprint mismatch",
    )
    ids = tensors["input_ids"]
    sequence = ids.shape[-1]
    hidden = configs["base"]["text_config"]["hidden_size"]
    vision = configs["base"]["vision_config"]
    merge = vision["spatial_merge_size"]
    patch = vision["patch_size"]
    temporal = vision["temporal_patch_size"]
    grid = tensors["image_grid_thw"]
    require(
        grid.shape == (1, 3) and grid[0, 0].item() == 1, "expected one still-image grid"
    )
    require(
        grid[0, 1].item() % merge == grid[0, 2].item() % merge == 0,
        "grid merge alignment",
    )
    image_tokens = grid.prod().item() // merge**2
    expected_indices = (ids[0] == configs["base"]["image_token_id"]).nonzero().flatten()
    shapes = {
        "input_ids": (1, sequence),
        "attention_mask": (1, sequence),
        "mm_token_type_ids": (1, sequence),
        "pixel_values": (grid.prod().item(), 3 * temporal * patch**2),
        "image_features": (image_tokens, hidden),
        "image_token_indices": (image_tokens,),
        "inputs_embeds": (1, sequence, hidden),
        "position_ids": (3, 1, sequence),
        "rope_deltas": (1, 1),
        "last_hidden_state": (1, hidden),
        "candidate_token_ids": (len(entry["option_keys"]),),
        "candidate_logits": (len(entry["option_keys"]),),
        "candidate_probabilities": (len(entry["option_keys"]),),
    }
    for name, shape in shapes.items():
        value = tensors[name]
        require(tuple(value.shape) == shape, f"shape mismatch: {name}")
        require(bool(torch.isfinite(value).all()), f"nonfinite tensor: {name}")
    for name in [
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "image_grid_thw",
        "image_token_indices",
        "position_ids",
        "rope_deltas",
        "candidate_token_ids",
    ]:
        require(tensors[name].dtype == torch.int64, f"expected int64: {name}")
    for name in [
        "image_features",
        "inputs_embeds",
        "last_hidden_state",
        "candidate_logits",
    ]:
        require(tensors[name].dtype == torch.bfloat16, f"expected bfloat16: {name}")
    require(
        tensors["pixel_values"].dtype
        == tensors["candidate_probabilities"].dtype
        == torch.float32,
        "expected float32 pixels/probabilities",
    )
    require(bool((tensors["attention_mask"] == 1).all()), "expected unpadded prompt")
    require(
        torch.equal(expected_indices, tensors["image_token_indices"]),
        "image token indices mismatch",
    )
    require(
        torch.equal(
            expected_indices, (tensors["mm_token_type_ids"][0] == 1).nonzero().flatten()
        ),
        "image token types mismatch",
    )
    require(
        torch.equal(
            tensors["inputs_embeds"][0, expected_indices], tensors["image_features"]
        ),
        "image insertion mismatch",
    )
    require(
        tensors["rope_deltas"].item()
        == tensors["position_ids"].max().item() + 1 - sequence,
        "rope delta mismatch",
    )
    require(
        torch.equal(
            tensors["candidate_token_ids"],
            torch.arange(32, 32 + len(entry["option_keys"])),
        ),
        "candidate token ordering",
    )
    require(
        torch.allclose(
            torch.softmax(tensors["candidate_logits"].float(), dim=-1),
            tensors["candidate_probabilities"],
            atol=1e-7,
            rtol=0,
        ),
        "readout mismatch",
    )


def verify(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    require(manifest["schema"] == SCHEMA, "unsupported schema")
    check_files(folder, manifest)
    configs = {
        name: json.loads((folder / f"configs/{name}.json").read_text())
        for name in ["base", "processor", "adapter"]
    }
    seen_images = {}
    count = 0
    for entry in manifest["questions"]:
        require(entry["tensors_file"] in manifest["files"], "unlisted tensor file")
        require(
            entry["request"] in manifest["files"]
            and entry["image"] in manifest["files"],
            "unlisted input file",
        )
        tensors = load_file(str(folder / entry["tensors_file"]))
        check_tensors(tensors, entry, configs)
        request = parse_request(
            decode_request((folder / entry["request"]).read_bytes())
        )
        question = next(q for q in request.questions if q.name == entry["question"])
        require(
            list(request.image.size) == entry["image_size_wh"], "image size mismatch"
        )
        require(list(question.keys) == entry["option_keys"], "option ordering mismatch")
        require(
            answer(question, tensors["candidate_probabilities"].tolist())
            == entry["answer"],
            "answer mismatch",
        )
        require(entry["ordinary_score_equal"] is True, "ordinary score check missing")
        image = manifest["files"][entry["image"]]["sha256"]
        features = {
            name: entry["tensors"][name]
            for name in ["pixel_values", "image_grid_thw", "image_features"]
        }
        require(
            image not in seen_images or seen_images[image] == features,
            "same-image vision mismatch",
        )
        seen_images[image] = features
        count += len(tensors)
    require(len(manifest["questions"]) == 8, "expected eight questions")
    return manifest, {
        "questions": 8,
        "tensors": count,
        "files": len(manifest["files"]),
        "integrity_and_relations": "pass",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    manifest, summary = verify(args.bundle)
    if args.compare:
        other, _ = verify(args.compare)
        require(
            manifest == other,
            "exports differ (environment, metadata, files or tensors)",
        )
        summary["independent_export_equality"] = "pass"
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
