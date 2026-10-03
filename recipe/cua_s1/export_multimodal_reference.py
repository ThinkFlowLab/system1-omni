"""Export the #12 reference's actual multimodal forward inputs and readout."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from models.cua_s1.multimodal.model import (
    ADAPTER_REVISION,
    BASE_REVISION,
    REFERENCE_REVISION,
    WEIGHTS_MANIFEST_SHA256,
    MultimodalEngine,
    letter_ids,
)
from models.cua_s1.multimodal.protocol import (
    answer,
    build_messages,
    decode_request,
    parse_request,
)

SCHEMA = "cua-s1-multimodal-reference-v1"
PACKAGES = {
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "transformers": "5.17.0",
    "peft": "0.21.0",
    "accelerate": "1.15.0",
    "Pillow": "11.3.0",
    "safetensors": "0.8.0",
    "huggingface-hub": "1.32.0",
    "tokenizers": "0.23.2",
    "numpy": "2.5.3",
}


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def make_cases(folder):
    folder.mkdir(parents=True)
    cases = []
    for name, size, fmt in [
        ("small", (320, 240), "PNG"),
        ("wide", (640, 320), "PNG"),
        ("portrait", (320, 640), "PNG"),
        ("jpeg", (640, 480), "JPEG"),
        ("single-option", (320, 240), "PNG"),
        ("26-options", (256, 256), "PNG"),
        ("two-questions", (320, 240), "PNG"),
    ]:
        image = Image.new("RGB", size, "#f4f6f8")
        draw = ImageDraw.Draw(image)
        width, height = size
        draw.rectangle(
            (16, 16, width - 16, height - 16), fill="white", outline="#8899aa"
        )
        draw.text((24, 24), "Account settings", fill="black")
        draw.text((24, 48), "Display name: Alice", fill="black")
        draw.rectangle((24, height // 2, width // 2, height // 2 + 32), fill="#1460b4")
        draw.text((28, height // 2 + 8), "Save", fill="white")
        draw.text((width // 2 + 16, height // 2 + 8), "Cancel", fill="black")
        image_path = folder / (name + (".jpg" if fmt == "JPEG" else ".png"))
        image.save(image_path, format=fmt)
        criteria = {"save": "Click Save", "cancel": "Click Cancel", "wait": "Wait"}
        if name == "single-option":
            criteria = {"save": "Click Save"}
        elif name == "26-options":
            criteria = {f"option-{i}": f"Choose action {i}" for i in range(26)}
        questions = {
            "next": {
                "type": "choice",
                "instructions": "Save the changed display name.",
                "criteria": criteria,
            }
        }
        if name == "two-questions":
            questions["second"] = {
                "type": "choice",
                "instructions": {"goal": "保存名称"},
                "criteria": {"continue": {"label": "Save"}, "cancel": None},
            }
        mime = "jpeg" if fmt == "JPEG" else "png"
        request = {
            "model": "cua-s1-4b-0.2",
            "state": {
                "image": f"data:image/{mime};base64,"
                + base64.b64encode(image_path.read_bytes()).decode()
            },
            "questions": questions,
        }
        write_json(folder / f"{name}.json", request)
        cases.append({"name": name, "image": image_path.name, "request": request})
    return cases


def tensor_info(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "sha256": sha256(value.view(torch.uint8).numpy().tobytes()),
    }


def save_tensors(path, tensors):
    from safetensors.torch import save_file

    # Clone individually: safetensors refuses shared storage, even for equal inputs.
    tensors = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in tensors.items()
    }
    save_file(tensors, str(path), metadata={"schema": SCHEMA})
    return {name: tensor_info(value) for name, value in tensors.items()}


def capture(engine, inputs, question):
    import torch

    tensors = {}

    def keep(name, value):
        if name in tensors:
            raise RuntimeError(f"expected one forward per question: duplicate {name}")
        tensors[name] = value.detach().cpu().contiguous().clone()

    def vision_hook(module, args, output):
        keep("image_features", output.pooler_output)

    def language_pre_hook(module, args, kwargs):
        keep("inputs_embeds", kwargs["inputs_embeds"])
        keep("position_ids", kwargs["position_ids"])

    def language_hook(module, args, output):
        keep("last_hidden_state", output.last_hidden_state[:, -1, :])

    core = engine.model.get_base_model().model
    hooks = [
        core.visual.register_forward_hook(vision_hook),
        core.language_model.register_forward_pre_hook(
            language_pre_hook, with_kwargs=True
        ),
        core.language_model.register_forward_hook(language_hook),
    ]
    try:
        with torch.no_grad():
            output = engine.model(
                **{
                    name: value.to(engine.model.device)
                    for name, value in inputs.items()
                }
            )
            ids = torch.tensor(
                letter_ids(engine.tokenizer, len(question.keys)),
                device=output.logits.device,
            )
            logits = output.logits[0, -1, ids]
            keep("candidate_token_ids", ids)
            keep("candidate_logits", logits)
            keep("candidate_probabilities", torch.softmax(logits.float(), dim=-1))
    finally:
        for handle in hooks:
            handle.remove()
    for name, value in inputs.items():
        keep(name, value)
    keep("rope_deltas", core.rope_deltas)
    indices = (
        (inputs["input_ids"][0] == engine.model.config.image_token_id)
        .nonzero()
        .flatten()
    )
    keep("image_token_indices", indices)
    if not torch.equal(tensors["inputs_embeds"][0, indices], tensors["image_features"]):
        raise RuntimeError("image feature insertion differs from language input")
    return tensors


def environment():
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5

    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    for name, expected in PACKAGES.items():
        if packages[name].split("+")[0] != expected:
            raise ValueError(f"{name} must be {expected}, got {packages[name]}")
    if any(
        importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")
    ):
        raise ValueError(
            "reference export requires the PyTorch DeltaNet path, without FLA"
        )
    return {
        "python": platform.python_version(),
        "torch_build": str(torch.__version__),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "transformers_source_sha256": sha256(
            Path(modeling_qwen3_5.__file__).read_bytes()
        ),
        "dtype": "bfloat16",
        "adapter_merged": False,
        "tf32": False,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def export(weights, output):
    if output.exists():
        raise FileExistsError(f"output must be a new directory: {output}")
    # Must be set before Torch initializes CUDA, including model construction.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {
        "schema": SCHEMA,
        "reference_revision": REFERENCE_REVISION,
        "base_revision": BASE_REVISION,
        "adapter_revision": ADAPTER_REVISION,
        "weights_manifest_sha256": WEIGHTS_MANIFEST_SHA256,
        "environment": environment(),
        "files": {},
        "questions": [],
    }
    engine = MultimodalEngine(
        str(weights / "Qwen3.5-4B"), str(weights / "cua-s1-4b-0.2/multimodal")
    )
    core = engine.model.get_base_model().model
    report["execution"] = {
        "visual_attention": core.visual.config._attn_implementation,
        "text_attention": core.language_model.config._attn_implementation,
        "processor_class": type(engine.processor).__name__,
        "image_processor_class": type(engine.processor.image_processor).__name__,
    }
    output.mkdir(parents=True)
    cases = make_cases(output / "inputs")
    (output / "tensors").mkdir()
    (output / "configs").mkdir()
    for label, path in [
        ("base", weights / "Qwen3.5-4B/config.json"),
        ("processor", weights / "Qwen3.5-4B/preprocessor_config.json"),
        ("adapter", weights / "cua-s1-4b-0.2/multimodal/adapter_config.json"),
    ]:
        (output / "configs" / f"{label}.json").write_bytes(path.read_bytes())
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"] = {
        name: sha256((root / name).read_bytes())
        for name in [
            "recipe/cua_s1/export_multimodal_reference.py",
            "src/models/cua_s1/multimodal/model.py",
            "src/models/cua_s1/multimodal/protocol.py",
        ]
    }
    for case in cases:
        request_path = output / "inputs" / f"{case['name']}.json"
        request = parse_request(decode_request(request_path.read_bytes()))
        for index, question in enumerate(request.questions):
            inputs = engine.prepare(request.image, question)
            tensors = capture(engine, inputs, question)
            probabilities = tensors["candidate_probabilities"].tolist()
            if probabilities != engine.score(inputs, question):
                raise RuntimeError("hooked readout differs from ordinary #12 score")
            relative = f"tensors/{case['name']}-{index}.safetensors"
            entry = {
                "case": case["name"],
                "question": question.name,
                "request": request_path.relative_to(output).as_posix(),
                "image": f"inputs/{case['image']}",
                "image_size_wh": list(request.image.size),
                "option_keys": list(question.keys),
                "prompt": engine.processor.apply_chat_template(
                    build_messages(question), tokenize=False, add_generation_prompt=True
                ),
                "tensors_file": relative,
                "tensors": save_tensors(output / relative, tensors),
                "answer": answer(question, probabilities),
                "ordinary_score_equal": True,
            }
            report["questions"].append(entry)
            print(f"exported {case['name']}/{question.name}", flush=True)
    for path in sorted(output.rglob("*")):
        if path.is_file():
            report["files"][path.relative_to(output).as_posix()] = {
                "sha256": sha256(path.read_bytes()),
                "size": path.stat().st_size,
            }
    # A manifest is written only after every forward and ordinary-score check passed.
    write_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.weights, args.output)


if __name__ == "__main__":
    main()
