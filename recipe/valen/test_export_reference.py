#!/usr/bin/env python3
"""Regression wiring for the Valen checkpoint export.

Chains the export contract tests in ``tests/valen/test_export.py`` — the
CPU-only battery plus the env-gated real-artifact determinism gate — and
validates an exported artifact set through the export tool's own
``validate_export``, without duplicating a single assertion: every check
lives in the test module or the tool, and this script only orchestrates.

    python recipe/valen/test_export_reference.py

Without the pinned artifacts it still runs the CPU battery and prints a
clear skip line for each artifact-gated phase, exiting 0. With
``VALEN_CKPT`` and ``VALEN_BASE`` set (see [reference.md](reference.md),
"Pinned artifacts") the determinism gate runs too, exporting the real
checkpoint twice and comparing every byte. With ``VALEN_EXPORT`` set — or
a ``weights/valen-export`` directory next to the repository root — the
finished artifact set is validated against its manifest. See
[export.md](export.md) for the export itself.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "tests" / "valen"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import export_checkpoint as tool  # noqa: E402
import test_export as contract  # noqa: E402


FIXTURE_PARAMS = ("tmp_path", "monkeypatch")


def contract_tests():
    """The export contract battery from tests/valen/test_export.py, in module
    order; each recognized fixture parameter gets a fresh instance per
    test (``tmp_path`` a fresh directory, ``monkeypatch`` a pytest
    ``MonkeyPatch`` undone right after)."""

    tests = []
    for name in sorted(dir(contract)):
        if not name.startswith("test_"):
            continue
        function = getattr(contract, name)
        params = tuple(
            param for param in FIXTURE_PARAMS if param in function.__code__.co_varnames
        )
        tests.append((name, function, params))
    return tests


def run_contract_battery(results):
    """Run every test in tests/valen/test_export.py directly.

    A test that raises pytest's ``Skipped`` outcome (the real-artifact gate
    without ``VALEN_CKPT``/``VALEN_BASE``) is reported as a skip with the
    test's own reason, never as a failure.
    """

    import pytest

    for name, function, params in contract_tests():
        try:
            with tempfile.TemporaryDirectory() as tmp:
                monkeypatch = pytest.MonkeyPatch()
                kwargs = {}
                if "tmp_path" in params:
                    kwargs["tmp_path"] = Path(tmp)
                if "monkeypatch" in params:
                    kwargs["monkeypatch"] = monkeypatch
                try:
                    function(**kwargs)
                finally:
                    monkeypatch.undo()
        except BaseException as error:  # noqa: B036 - classified below
            if type(error).__name__ == "Skipped":
                record(results, name, "skip", str(error))
            else:
                record(results, name, False, f"{type(error).__name__}: {error}")
        else:
            record(results, name, True)


def validate_artifact_set(results, export_dir):
    """Validate an exported directory through the tool's own cross-check."""

    try:
        manifest = tool.validate_export(Path(export_dir))
    except Exception as error:
        record(results, f"validate {export_dir}", False, str(error))
        return
    counts = {}
    for entry in manifest["tensors"]:
        counts[entry["component"]] = counts.get(entry["component"], 0) + 1
    parity = manifest["parity"]
    lora = manifest["lora"]
    summary = (
        f"{manifest['format']}, {len(manifest['tensors'])} tensors "
        + ", ".join(f"{counts[c]} {c}" for c in sorted(counts))
        + f", {len(manifest['downcasts'])} fp32->bf16 downcasts"
        + f", lora unmerged r={lora['rank']} alpha={lora['alpha']}"
        + f" ({lora['keys']} fp32 keys / {lora['targets']} targets)"
        + f", parity slot {'filled' if parity is not None else 'not recorded'}"
    )
    record(results, f"validate {export_dir}", True, summary)


def record(results, name, outcome, detail=""):
    results.append((name, outcome, detail))


def main() -> None:
    results: list[tuple[str, object, str]] = []

    run_contract_battery(results)

    export_dir = os.environ.get("VALEN_EXPORT")
    if export_dir:
        validate_artifact_set(results, export_dir)
    elif (REPO_ROOT / "weights" / "valen-export").is_dir():
        validate_artifact_set(results, REPO_ROOT / "weights" / "valen-export")
    else:
        record(
            results,
            "validate exported artifact set",
            "skip",
            "set VALEN_EXPORT to an exported directory (the export's default "
            "output is weights/valen-export) to validate it against its manifest",
        )

    failed = skipped = 0
    for name, outcome, detail in results:
        if outcome is True:
            print(f"  ok   {name}" + (f" — {detail}" if detail else ""))
        elif outcome == "skip":
            skipped += 1
            print(f"  skip {name} — {detail}")
        else:
            failed += 1
            print(f"  FAIL {name}")
            print("       " + detail.strip().replace("\n", "\n       ")[:400])
    verdict = "export_reference tests: ok" if not failed else (
        f"export_reference tests: {failed} FAILED"
    )
    print(verdict + (f" ({skipped} skipped)" if skipped else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
