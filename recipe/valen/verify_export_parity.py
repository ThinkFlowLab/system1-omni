#!/usr/bin/env python3
"""Prove the exported Valen artifact set answers like the original checkpoint.

Two model stacks are loaded serially — only one lives on the GPU at a time,
each in its own subprocess so the ~6.6 GiB of VRAM is fully released between
loads:

- original: the pinned reference stack exactly as the worker builds it
  (``ValenExecutor`` -> ``valen.modeling.model.build_model`` +
  ``load_checkpoint``: pinned base, PEFT-wrapped language model with the
  FP32 LoRA applied unmerged, trained vision-top blocks, merger and head
  from ``checkpoint.pt``);
- exported: the consumption path the export manifest documents —
  ``Qwen3_5ForConditionalGeneration.from_pretrained(weights/valen-export,
  dtype=bf16, eager).model`` for the backbone, the language LoRA re-applied
  through the same peft build the pinned reference uses (every language
  Linear wrapped at the manifest-declared rank and alpha, the 372 FP32
  tensors of ``lora.safetensors`` loaded by key, adapters kept FP32), and
  the decision head rebuilt from ``head.safetensors`` as the pinned
  ``DecisionHead`` (2048 -> 256, FP32, no bias), all through the pinned
  ``Valen`` class so the forward path is the reference's, not a second
  invention.

Both stacks answer the recipe's existing text and image ``choice`` fixtures
(``compare_reference.py``'s text example and ``example-request.json``,
compiled by the pinned ``Compiler`` through the worker's own record rules).
The gate requires exact answer agreement and a probability delta <= 1e-6
(the tolerance the reference serving already declared).

Finally it fills the manifest's ``parity`` slot in
``weights/valen-export/valen_export.json`` with the verdict (status, device,
cases, agreement, max probability delta, evidence pointer) — a reproducible
command output, never a hand edit — and re-validates the artifact set.

Usage (from the repository root, on the GPU host):

    .venv/bin/python recipe/valen/verify_export_parity.py \
      --valen-source weights/Valen --checkpoint weights/Valen-Preview-0923 \
      --base weights/Qwen3.5-2B --export weights/valen-export --device cuda

Exit status 0 means the gate passed; 1 means it failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare_reference as reference  # noqa: E402  (the recipe's fixture helpers)
import export_checkpoint as exporter  # noqa: E402

PROBABILITY_TOLERANCE = 1e-6
STAGE_SENTINEL = "VERIFY-PARITY-STAGE-RESULT "
PARITY_CASES = ("text", "image")


# --- the gate's verdict: one case answered by both stacks ---


def compare_case(case, original, exported, tolerance=PROBABILITY_TOLERANCE):
    """Compare one fixture answered by the original and exported stacks.

    Each side carries ``questions`` (per qid: candidate ``keys`` in order and
    ``probabilities``), the compiler's ``logical_tokens`` and
    ``compute_tokens``. The gate requires the same candidate order, exact
    answer (argmax) agreement, equal token accounting and probability deltas
    within ``tolerance``.
    """

    problems = []
    exact = True
    max_delta = 0.0
    original_questions = original["questions"]
    exported_questions = exported["questions"]
    for qid, record in original_questions.items():
        if qid not in exported_questions:
            problems.append(f"{qid}: missing from the exported stack")
            exact = False
            continue
        other = exported_questions[qid]
        keys = list(record["keys"])
        if list(other["keys"]) != keys:
            problems.append(
                f"{qid}: candidate order {list(other['keys'])} != {list(keys)}"
            )
            exact = False
            continue
        for key, p_original, p_exported in zip(
            keys, record["probabilities"], other["probabilities"]
        ):
            max_delta = max(max_delta, abs(p_original - p_exported))
        chosen = max(range(len(keys)), key=record["probabilities"].__getitem__)
        other_chosen = max(range(len(keys)), key=other["probabilities"].__getitem__)
        if chosen != other_chosen:
            problems.append(
                f"{qid}: original chose {keys[chosen]!r}, exported {keys[other_chosen]!r}"
            )
            exact = False
    for qid in sorted(set(exported_questions) - set(original_questions)):
        problems.append(f"{qid}: not asked of the original stack")
        exact = False
    tokens_agree = (
        original["logical_tokens"] == exported["logical_tokens"]
        and original["compute_tokens"] == exported["compute_tokens"]
    )
    if not tokens_agree:
        problems.append(
            "token accounting: "
            f"({original['logical_tokens']}, {original['compute_tokens']}) != "
            f"({exported['logical_tokens']}, {exported['compute_tokens']})"
        )
    tolerance_pass = max_delta <= tolerance
    if not tolerance_pass:
        problems.append(
            f"probability delta {max_delta:.6e} exceeds tolerance {tolerance:g}"
        )
    return {
        "case": case,
        "exact_answer_agreement": exact,
        "max_probability_delta": max_delta,
        "probability_tolerance_pass": tolerance_pass,
        "tokens_agree": tokens_agree,
        "logical_tokens": int(original["logical_tokens"]),
        "compute_tokens": int(original["compute_tokens"]),
        "passed": exact and tokens_agree and tolerance_pass,
        "problems": problems,
    }


# --- the manifest parity slot ---


def build_parity_record(
    *,
    status,
    device,
    hardware,
    cases,
    exact_answer_agreement,
    max_probability_delta,
    tolerance,
    evidence,
):
    """Assemble the record written into the manifest's ``parity`` slot."""

    if status not in ("passed", "failed"):
        raise ValueError(f"parity status must be 'passed' or 'failed', got {status!r}")
    return {
        "status": status,
        "device": device,
        "hardware": hardware,
        "tolerance": tolerance,
        "cases": cases,
        "exact_answer_agreement": bool(exact_answer_agreement),
        "max_probability_delta": max_probability_delta,
        "evidence": evidence,
    }


def fill_parity_slot(manifest_path, record, *, force=False):
    """Write ``record`` into the manifest's ``parity`` slot.

    Refuses to silently replace a different existing record unless ``force``
    is set, keeps the manifest's documented formatting (sorted keys, indent
    2, trailing newline) and returns the updated manifest.
    """

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = manifest.get("parity")
    if existing is not None and existing != record and not force:
        raise ValueError(
            "parity slot already holds a different record (pass force=True to replace)"
        )
    manifest["parity"] = record
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


# --- the two model stacks ---


def load_original_stack(checkpoint, base, device):
    """The reference stack exactly as the worker's executor builds it."""

    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from models.valen.engine import ValenExecutor

    executor = ValenExecutor(Path(checkpoint), Path(base), device, "bf16")
    return executor


def load_exported_stack(export_dir, manifest, device):
    """The exported artifact set loaded through its documented consumption
    path: the ``from_pretrained`` backbone, the language LoRA re-applied
    through the same peft build the pinned reference uses, and the head
    rebuilt from ``head.safetensors`` into the pinned ``Valen`` class."""

    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open
    from transformers import Qwen3_5ForConditionalGeneration

    from valen.modeling.model import Valen

    container, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(export_dir),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        output_loading_info=True,
    )
    if any(
        loading_info.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Incomplete exported model load: {loading_info}")
    backbone = container.model
    # Only container.model is served; drop the vocabulary/generation wrapper.
    del container
    backbone.requires_grad_(False)

    # The pinned build path's own adaptation: every language Linear wrapped
    # at the manifest-declared rank and alpha, adapters kept fp32 at runtime.
    lora = manifest["lora"]
    targets = [
        name
        for name, module in backbone.language_model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if len(targets) != lora["targets"]:
        raise ValueError(
            f"backbone enumerates {len(targets)} language linears, the manifest "
            f"declares {lora['targets']}"
        )
    backbone.language_model = get_peft_model(
        backbone.language_model,
        LoraConfig(
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=0.0,
            target_modules=targets,
            bias="none",
        ),
    )
    model = Valen(backbone, manifest["head"]["projection_dim"])
    with safe_open(Path(export_dir) / lora["file"], framework="pt") as handle:
        adapters = {key: handle.get_tensor(key) for key in sorted(handle.keys())}
    expected = {name for name, _ in model.named_parameters() if "lora_" in name}
    if set(adapters) != expected:
        raise ValueError(
            "lora.safetensors keys do not match the wrapped model's adapters: "
            f"file-only {sorted(set(adapters) - expected)[:2]}, "
            f"model-only {sorted(expected - set(adapters))[:2]}"
        )
    model.load_state_dict(adapters, strict=False)
    with safe_open(Path(export_dir) / exporter.HEAD_FILE, framework="pt") as handle:
        head_state = {
            key[len(exporter.HEAD_PREFIX):]: handle.get_tensor(key)
            for key in sorted(handle.keys())
        }
    model.head.load_state_dict(head_state)
    model.to(device)
    model.eval()
    return model


# --- answering the fixtures through one loaded stack ---


def choice_fixtures():
    """The recipe's existing text and image ``choice`` fixtures."""

    return (
        ("text", reference.DEFAULT_TEXT_BODY, "compare_reference.py DEFAULT_TEXT_BODY"),
        (
            "image",
            json.loads((HERE / "example-request.json").read_text(encoding="utf-8")),
            "recipe/valen/example-request.json",
        ),
    )


def run_fixtures(model, processor, max_length, media_kwargs):
    """Answer every fixture with one loaded stack; return per-case results."""

    import torch

    from valen.data.compiler import Compiler

    cases = {}
    with torch.no_grad():
        for name, body, fixture in choice_fixtures():
            questions = reference.reference_questions(body, processor.special_tokens)
            with TemporaryDirectory(prefix="valen-parity-") as directory:
                media_root = Path(directory)
                if name == "image":
                    image_bytes, image_format = reference.decode_image_data_url(
                        body["state"]["image"]
                    )
                    record = reference.reference_image_record(
                        image_bytes, image_format, questions, media_root
                    )
                else:
                    record = reference.reference_text_record(body["state"], questions)
                compiler = Compiler(
                    processor.processor,
                    media_root=media_root,
                    max_length=max_length,
                    media_kwargs=media_kwargs,
                )
                state = compiler.compile(record)
                answered = {}
                for question in state.questions:
                    logits = model(question).float().cpu()
                    answered[question.qid] = {
                        "keys": list(question.keys),
                        "probabilities": torch.softmax(logits, dim=-1).tolist(),
                    }
                cases[name] = {
                    "fixture": fixture,
                    "questions": answered,
                    "logical_tokens": int(state.logical_tokens),
                    "compute_tokens": int(state.compute_tokens),
                }
    return cases


def stage_environment(args):
    """Verify the pinned source and build the shared compile-side state."""

    sys.path.insert(0, str(HERE.parents[1] / "src"))
    from frontend.valen import _verify_source_revision

    source = Path(args.valen_source).resolve()
    _verify_source_revision(source)
    reference._verify_clean_checkout(source)
    sys.path.insert(0, str(source))

    from models.valen.preprocess import ValenProcessor

    config = json.loads(
        (Path(args.checkpoint) / "config.json").read_text(encoding="utf-8")
    )
    processor = ValenProcessor(
        Path(args.base), int(config["max_length"]), config.get("media_kwargs")
    )
    return processor, int(config["max_length"]), config.get("media_kwargs") or {}


def run_stage(args):
    """Load one stack in this (sub)process, answer the fixtures, emit JSON."""

    import torch

    torch.set_grad_enabled(False)
    processor, max_length, media_kwargs = stage_environment(args)

    if args.stage == "original":
        executor = load_original_stack(args.checkpoint, args.base, args.device)
        model = executor.model
    else:
        manifest = json.loads(
            (Path(args.export) / exporter.MANIFEST_FILE).read_text(encoding="utf-8")
        )
        model = load_exported_stack(Path(args.export), manifest, args.device)

    device = str(next(model.parameters()).device)
    hardware = None
    if device.startswith("cuda"):
        index = torch.device(device).index or 0
        capability = torch.cuda.get_device_capability(index)
        _, total = torch.cuda.mem_get_info(index)
        hardware = (
            f"{torch.cuda.get_device_name(index)} "
            f"(sm_{capability[0]}{capability[1]}, {total / (1024 ** 3):.2f} GiB)"
        )
    try:
        cases = run_fixtures(model, processor, max_length, media_kwargs)
    finally:
        if args.stage == "original":
            executor.close()
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(
        STAGE_SENTINEL
        + json.dumps(
            {
                "stage": args.stage,
                "device": device,
                "hardware": hardware,
                "torch": torch.__version__,
                "cases": cases,
            }
        )
    )
    return 0


def run_stage_subprocess(args, stage):
    """Run one stack in its own subprocess so VRAM is fully released."""

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--valen-source", args.valen_source,
        "--checkpoint", args.checkpoint,
        "--base", args.base,
        "--export", args.export,
        "--device", args.device,
        "--stage", stage,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} stage failed with exit {result.returncode}:\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-4000:]}"
        )
    payload = None
    for line in result.stdout.splitlines():
        if line.startswith(STAGE_SENTINEL):
            payload = json.loads(line[len(STAGE_SENTINEL):])
    if payload is None:
        raise RuntimeError(f"{stage} stage produced no result: {result.stdout[-2000:]}")
    return payload


# --- reporting ---


def format_case_report(verdict, tolerance):
    lines = [f"case: {verdict['case']}"]
    lines.append(f"logical tokens: {verdict['logical_tokens']}")
    lines.append(f"compute tokens: {verdict['compute_tokens']}")
    lines.append(f"max probability delta: {verdict['max_probability_delta']:.10e}")
    lines.append(
        "answer agreement: " + ("PASS" if verdict["exact_answer_agreement"] else "FAIL")
    )
    lines.append(
        f"probability tolerance <= {tolerance:g}: "
        + ("PASS" if verdict["probability_tolerance_pass"] else "FAIL")
    )
    lines.append("token accounting: " + ("PASS" if verdict["tokens_agree"] else "FAIL"))
    for problem in verdict["problems"]:
        lines.append(f"problem: {problem}")
    return lines


def evidence_pointer():
    return (
        "recipe/valen/reference.md (export parity block); reproduce with "
        "recipe/valen/verify_export_parity.py --valen-source $VALEN_SRC "
        "--checkpoint $VALEN_CKPT --base $VALEN_BASE --export <dir> --device cuda"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--valen-source", required=True, help="pinned Valen source checkout")
    parser.add_argument("--checkpoint", required=True, help="Valen-Preview-0923 directory")
    parser.add_argument("--base", required=True, help="Qwen3.5-2B base directory")
    parser.add_argument("--export", required=True, type=Path, help="exported artifact directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=PROBABILITY_TOLERANCE)
    parser.add_argument("--stage", choices=("original", "export"), default=None, help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true", help="replace an existing parity record")
    return parser.parse_args(argv)


def verify_sources_match_manifest(args, manifest):
    """The parity evidence only holds for the artifacts the export recorded.

    Verifies the checkpoint sha, the base revision and — because 578 tensors
    pass through the export byte by byte — the base weights shard digests
    themselves against the manifest's recorded ``sources.base.weights_sha256``.
    """

    checkpoint_sha = exporter.sha256_file(Path(args.checkpoint) / exporter.CHECKPOINT_FILE)
    recorded = manifest["sources"]["checkpoint"]["sha256"]
    if recorded != checkpoint_sha:
        raise ValueError(
            f"the export was built from checkpoint sha256 {recorded}, but "
            f"--checkpoint hashes {checkpoint_sha}"
        )
    base_manifest = json.loads(
        (Path(args.base) / exporter.BASE_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    revision = manifest["sources"]["base"]["revision"]
    if base_manifest.get("revision") != revision:
        raise ValueError(
            f"the export was built from base revision {revision}, but --base "
            f"pins {base_manifest.get('revision')}"
        )
    weights_file = exporter.base_weights_file(Path(args.base))
    shard_sums = {weights_file.name: exporter.sha256_file(weights_file)}
    recorded_sums = manifest["sources"]["base"]["weights_sha256"]
    if recorded_sums != shard_sums:
        raise ValueError(
            f"the export was built from base weights sha256 {recorded_sums}, "
            f"but --base hashes {shard_sums}"
        )


def main(argv=None):
    args = parse_args(argv)
    if args.stage is not None:
        return run_stage(args)

    export_dir = Path(args.export)
    manifest = exporter.validate_export(export_dir)
    verify_sources_match_manifest(args, manifest)

    original = run_stage_subprocess(args, "original")
    exported = run_stage_subprocess(args, "export")
    verdicts = [
        compare_case(name, original["cases"][name], exported["cases"][name], args.tolerance)
        for name in PARITY_CASES
    ]

    status = "passed" if all(verdict["passed"] for verdict in verdicts) else "failed"
    record = build_parity_record(
        status=status,
        device=original["device"],
        hardware=original["hardware"],
        cases=[
            {
                "case": verdict["case"],
                "fixture": original["cases"][verdict["case"]]["fixture"],
                "exact_answer_agreement": verdict["exact_answer_agreement"],
                "max_probability_delta": verdict["max_probability_delta"],
                "logical_tokens": verdict["logical_tokens"],
                "compute_tokens": verdict["compute_tokens"],
                "passed": verdict["passed"],
                "problems": verdict["problems"],
            }
            for verdict in verdicts
        ],
        exact_answer_agreement=all(v["exact_answer_agreement"] for v in verdicts),
        max_probability_delta=max(v["max_probability_delta"] for v in verdicts),
        tolerance=args.tolerance,
        evidence=evidence_pointer(),
    )
    fill_parity_slot(export_dir / exporter.MANIFEST_FILE, record, force=args.force)
    exporter.validate_export(export_dir)

    for verdict in verdicts:
        print("\n".join(format_case_report(verdict, args.tolerance)))
        print()
    print(f"device: {record['device']} ({record['hardware']})")
    print(f"parity slot filled: {export_dir / exporter.MANIFEST_FILE}")
    print("verify_export_parity: ok" if status == "passed" else "verify_export_parity: FAILED")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
