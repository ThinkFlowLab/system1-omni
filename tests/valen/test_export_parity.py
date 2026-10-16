"""Parity-gate tests for recipe/valen/verify_export_parity.py.

CPU-only checks of the export-parity tool's observable contract: the
original-vs-exported case comparison verdicts and the manifest parity-slot
filling. The full parity gate against the real pinned artifacts is
env-gated on VALEN_CKPT / VALEN_BASE / VALEN_SRC plus CUDA.

PYTHONPATH=src python -m pytest tests/valen -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RECIPE_VALEN = HERE.parents[1] / "recipe" / "valen"
sys.path.insert(0, str(RECIPE_VALEN))

import compare_reference as reference  # noqa: E402
import verify_export_parity as tool  # noqa: E402


# --- the original-vs-exported case comparison (the gate's verdict) ---

def case_side(probs, keys=("refund", "wait"), logical=48, compute=48):
    return {
        "questions": {"refund": {"keys": list(keys), "probabilities": probs}},
        "logical_tokens": logical,
        "compute_tokens": compute,
    }


def test_compare_case_identical_stacks_pass():
    original = case_side([0.5159631, 0.4840369])
    verdict = tool.compare_case("text", original, case_side([0.5159631, 0.4840369]))
    assert verdict["exact_answer_agreement"] is True
    assert verdict["max_probability_delta"] == 0.0
    assert verdict["probability_tolerance_pass"] is True
    assert verdict["tokens_agree"] is True
    assert verdict["passed"] is True
    assert verdict["problems"] == []


def test_compare_case_delta_within_tolerance_passes():
    original = case_side([0.5159631, 0.4840369])
    drifted = case_side([0.5159631 + 5e-7, 0.4840369 - 5e-7])
    verdict = tool.compare_case("text", original, drifted)
    assert verdict["passed"] is True
    assert verdict["max_probability_delta"] == pytest.approx(5e-7)
    assert verdict["exact_answer_agreement"] is True


def test_compare_case_delta_beyond_tolerance_fails():
    original = case_side([0.5159631, 0.4840369])
    drifted = case_side([0.5159631 + 2e-6, 0.4840369 - 2e-6])
    verdict = tool.compare_case("text", original, drifted)
    assert verdict["exact_answer_agreement"] is True
    assert verdict["probability_tolerance_pass"] is False
    assert verdict["passed"] is False
    assert any("tolerance" in problem for problem in verdict["problems"])


def test_compare_case_flipped_answer_fails():
    original = case_side([0.5159631, 0.4840369])
    flipped = case_side([0.4840369, 0.5159631])
    verdict = tool.compare_case("text", original, flipped)
    assert verdict["exact_answer_agreement"] is False
    assert verdict["passed"] is False


def test_compare_case_candidate_order_mismatch_fails():
    original = case_side([0.5159631, 0.4840369])
    reordered = case_side([0.4840369, 0.5159631], keys=("wait", "refund"))
    verdict = tool.compare_case("text", original, reordered)
    assert verdict["passed"] is False
    assert any("candidate order" in problem for problem in verdict["problems"])


def test_compare_case_missing_question_fails():
    original = case_side([0.5159631, 0.4840369])
    emptied = case_side([0.5159631, 0.4840369])
    emptied["questions"] = {}
    verdict = tool.compare_case("text", original, emptied)
    assert verdict["passed"] is False


def test_compare_case_token_accounting_must_agree():
    original = case_side([0.5159631, 0.4840369])
    retokenized = case_side([0.5159631, 0.4840369], compute=49)
    verdict = tool.compare_case("text", original, retokenized)
    assert verdict["tokens_agree"] is False
    assert verdict["passed"] is False


# --- the manifest parity slot ---

def test_build_parity_record_covers_the_documented_slot_shape():
    record = tool.build_parity_record(
        status="passed",
        device="cuda:0",
        hardware="NVIDIA GeForce RTX 5060 Laptop GPU (sm_120)",
        cases=[{"case": "text", "passed": True}],
        exact_answer_agreement=True,
        max_probability_delta=0.0,
        tolerance=1e-6,
        evidence="recipe/valen/reference.md export-parity block",
    )
    # Every key the manifest's parity_slot_shape documents must be present.
    assert set(record) == {
        "status", "device", "hardware", "tolerance", "cases",
        "exact_answer_agreement", "max_probability_delta", "evidence",
    }
    assert record["status"] == "passed"
    assert record["max_probability_delta"] == 0.0
    assert record["tolerance"] == 1e-6


def test_build_parity_record_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        tool.build_parity_record(
            status="maybe",
            device="cuda:0",
            hardware="gpu",
            cases=[],
            exact_answer_agreement=True,
            max_probability_delta=0.0,
            tolerance=1e-6,
            evidence="",
        )


def minimal_manifest(path, parity=None):
    manifest = {
        "format": "valen-checkpoint-export/2",
        "parity": parity,
        "parity_slot_shape": {"status": "passed | failed"},
    }
    Path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parity_record(status="passed", delta=1e-8):
    return tool.build_parity_record(
        status=status,
        device="cuda:0",
        hardware="gpu",
        cases=[{"case": "text", "passed": status == "passed"}],
        exact_answer_agreement=status == "passed",
        max_probability_delta=delta,
        tolerance=1e-6,
        evidence="recipe/valen/reference.md",
    )


def test_fill_parity_slot_replaces_null(tmp_path):
    manifest_path = tmp_path / "valen_export.json"
    minimal_manifest(manifest_path)
    record = parity_record()
    updated = tool.fill_parity_slot(manifest_path, record)
    assert updated["parity"] == record
    reread = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert reread["parity"] == record
    # The documented formatting (sorted, indent 2, trailing newline) is kept.
    assert manifest_path.read_text(encoding="utf-8").endswith("}\n")


def test_fill_parity_slot_is_idempotent_for_the_same_record(tmp_path):
    manifest_path = tmp_path / "valen_export.json"
    minimal_manifest(manifest_path)
    record = parity_record()
    tool.fill_parity_slot(manifest_path, record)
    before = manifest_path.read_text(encoding="utf-8")
    tool.fill_parity_slot(manifest_path, record)
    assert manifest_path.read_text(encoding="utf-8") == before


def test_fill_parity_slot_refuses_to_silently_overwrite(tmp_path):
    manifest_path = tmp_path / "valen_export.json"
    minimal_manifest(manifest_path)
    tool.fill_parity_slot(manifest_path, parity_record(delta=1e-8))
    with pytest.raises(ValueError, match="parity slot already holds"):
        tool.fill_parity_slot(manifest_path, parity_record(delta=5e-3))
    # The earlier record survives the refusal.
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["parity"][
        "max_probability_delta"
    ] == 1e-8


# --- the source-pin cross-check before any model is loaded ---

def pin_args(checkpoint_dir, base_dir):
    import argparse

    return argparse.Namespace(checkpoint=str(checkpoint_dir), base=str(base_dir))


def pin_fixture(tmp_path):
    """A tiny checkpoint dir + single-shard base and a manifest recording
    their true digests, mirroring the real pinned layout."""

    import hashlib

    from export_checkpoint import sha256_file

    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    checkpoint_file = checkpoint_dir / "checkpoint.pt"
    checkpoint_file.write_bytes(b"checkpoint-bytes")
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    shard = base_dir / "model.safetensors-00001-of-00001.safetensors"
    shard.write_bytes(b"shard-bytes")
    (base_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shard.name}}), encoding="utf-8"
    )
    (base_dir / "valen_manifest.json").write_text(
        json.dumps({"revision": "badbee"}), encoding="utf-8"
    )
    manifest = {
        "sources": {
            "checkpoint": {"sha256": sha256_file(checkpoint_file)},
            "base": {
                "revision": "badbee",
                "weights_sha256": {
                    shard.name: hashlib.sha256(b"shard-bytes").hexdigest()
                },
            },
        }
    }
    return pin_args(checkpoint_dir, base_dir), manifest


def test_verify_sources_match_manifest_accepts_matching_pins(tmp_path):
    args, manifest = pin_fixture(tmp_path)
    tool.verify_sources_match_manifest(args, manifest)


def test_verify_sources_rejects_checkpoint_digest_drift(tmp_path):
    import pytest

    args, manifest = pin_fixture(tmp_path)
    manifest["sources"]["checkpoint"]["sha256"] = "00" * 32
    with pytest.raises(ValueError, match="checkpoint sha256"):
        tool.verify_sources_match_manifest(args, manifest)


def test_verify_sources_rejects_base_revision_drift(tmp_path):
    import pytest

    args, manifest = pin_fixture(tmp_path)
    manifest["sources"]["base"]["revision"] = "deadbee"
    with pytest.raises(ValueError, match="base revision"):
        tool.verify_sources_match_manifest(args, manifest)


def test_verify_sources_rejects_tampered_base_shard_digest(tmp_path):
    """The base weights themselves must be re-hashed against the manifest's
    recorded per-shard digests: 578 tensors pass through the export byte by
    byte, so a silently swapped base would poison the parity verdict."""

    import pytest

    args, manifest = pin_fixture(tmp_path)
    manifest["sources"]["base"]["weights_sha256"][
        "model.safetensors-00001-of-00001.safetensors"
    ] = "00" * 32
    with pytest.raises(ValueError, match="base weights sha256"):
        tool.verify_sources_match_manifest(args, manifest)


def test_verify_sources_rejects_swapped_base_shard_bytes(tmp_path):
    import pytest

    args, manifest = pin_fixture(tmp_path)
    shard = tmp_path / "base" / "model.safetensors-00001-of-00001.safetensors"
    shard.write_bytes(b"different-shard-bytes")
    with pytest.raises(ValueError, match="base weights sha256"):
        tool.verify_sources_match_manifest(args, manifest)


# --- the pinned-source checkout must be clean before evidence is produced ---

def fresh_checkout(path):
    """An empty git repository standing in for the pinned source tree."""

    import subprocess

    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def test_clean_checkout_passes(tmp_path):
    repo = fresh_checkout(tmp_path / "valen-src")
    # Untracked non-code cannot change what gets imported, so it is allowed.
    (repo / "notes.txt").write_text("editor droppings\n", encoding="utf-8")
    reference._verify_clean_checkout(repo)


def test_untracked_python_file_is_rejected(tmp_path):
    """A stray .py on the package path can shadow or extend the pinned code
    both stacks import, so it must fail the gate before evidence exists."""

    import pytest

    repo = fresh_checkout(tmp_path / "valen-src")
    (repo / "stray.py").write_text("# not pinned\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="stray.py"):
        reference._verify_clean_checkout(repo)


def test_tracked_modification_is_rejected(tmp_path):
    """Both stacks import from the same tree, so a local modification would
    make the two agree while neither ran the pinned code."""

    import pytest
    import subprocess

    repo = fresh_checkout(tmp_path / "valen-src")
    model = repo / "model.py"
    model.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "model.py"], check=True)
    model.write_text("locally modified\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="model.py"):
        reference._verify_clean_checkout(repo)


# --- the full parity gate against the real pinned artifacts ---
# Set VALEN_CKPT, VALEN_BASE and VALEN_SRC (the recipe reference.md artifact
# exports) on a CUDA host to run: it exports a fresh artifact set, then runs
# the two serial model loads (original checkpoint stack, exported stack) and
# fills the manifest parity slot. The unmerged export is expected to pass the
# declared gate; the test asserts the verdict and its internal consistency.

def test_real_artifact_export_parity_gate(tmp_path):
    import subprocess
    import torch

    env_vars = ("VALEN_CKPT", "VALEN_BASE", "VALEN_SRC")
    if not all(os.environ.get(name) for name in env_vars):
        pytest.skip("set VALEN_CKPT, VALEN_BASE and VALEN_SRC to run")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    import export_checkpoint as exporter

    export_dir = tmp_path / "export"
    exporter.export(
        Path(os.environ["VALEN_CKPT"]), Path(os.environ["VALEN_BASE"]), export_dir
    )

    result = subprocess.run(
        [
            sys.executable,
            str(RECIPE_VALEN / "verify_export_parity.py"),
            "--valen-source", os.environ["VALEN_SRC"],
            "--checkpoint", os.environ["VALEN_CKPT"],
            "--base", os.environ["VALEN_BASE"],
            "--export", str(export_dir),
            "--device", "cuda",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]

    manifest = exporter.validate_export(export_dir)
    parity = manifest["parity"]
    assert parity["status"] == "passed"
    assert parity["device"].startswith("cuda")
    assert parity["tolerance"] == 1e-6
    assert {case["case"] for case in parity["cases"]} == {"text", "image"}
    assert parity["exact_answer_agreement"] is True
    assert all(case["exact_answer_agreement"] for case in parity["cases"])
    assert parity["max_probability_delta"] == max(
        case["max_probability_delta"] for case in parity["cases"]
    )
    assert parity["max_probability_delta"] <= parity["tolerance"]
    for case in parity["cases"]:
        assert case["passed"] is True
        assert case["problems"] == []
