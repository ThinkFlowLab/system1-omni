#!/usr/bin/env python3
"""Tier-1 contract check for CUDA backend libraries. No GPU, no nvcc required.

Discovers every ``src/backends/cuda/<name>/<name>.backend.json`` manifest and
checks the parts of ``contract.md`` that do not need hardware:

  * the manifest schema,
  * that every declared source file exists,
  * that the build script's declared output and architectures match the manifest,
  * that the ABI version is consistent across backends,
  * that a ``validated`` backend declares a tolerance and a reference entrypoint,
  * that a kernel requiring a newer compute capability than the build declares is
    flagged.

It does not compile CUDA and does not prove numerics. Tier 2 (``--gpu`` in CI)
runs the reference entrypoint on a self-hosted GPU runner.

Usage:
    python3 src/backends/cuda/check_contract.py [--repo-root PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from build_script import parse_build_script

MANIFEST_SUFFIX = ".backend.json"
BACKENDS_DIR = os.path.join("src", "backends", "cuda")
STATUSES = ("planned", "experimental", "validated")
# Every key check_manifest reads directly. The missing-key report and the guard
# that stops after it are both derived from this tuple.
REQUIRED_KEYS = ("name", "abi_version", "status", "sources", "build")
CONTRACT_ABI_VERSION = 1



class Issue(object):
    """One contract violation or warning for a backend."""

    def __init__(self, level, backend, message):
        self.level = level  # "error" or "warning"
        self.backend = backend
        self.message = message

    def __str__(self):
        return "[%s] %s: %s" % (self.level, self.backend, self.message)

    def as_dict(self):
        return {"level": self.level, "backend": self.backend, "message": self.message}



def load_manifests(repo_root):
    """Return (manifests, issues) for every ``*.backend.json`` under backends/cuda."""
    issues = []
    manifests = []
    root = os.path.join(repo_root, BACKENDS_DIR)
    if not os.path.isdir(root):
        issues.append(Issue("error", "-", "%s does not exist" % BACKENDS_DIR))
        return manifests, issues

    # A backend may be a subdirectory named after itself, or the cuda directory
    # itself when its files sit directly under it (`kernels/`, `tools/`). The
    # layout is the model author's call; only the manifest's contents are fixed.
    found = []
    for entry in sorted(os.listdir(root)):
        directory = os.path.join(root, entry)
        if not os.path.isdir(directory) or entry.startswith("."):
            continue
        expected = os.path.join(directory, entry + MANIFEST_SUFFIX)
        if os.path.isfile(expected):
            found.append((entry, directory, expected))
    for filename in sorted(os.listdir(root)):
        if filename.endswith(MANIFEST_SUFFIX) and os.path.isfile(os.path.join(root, filename)):
            found.append((filename[: -len(MANIFEST_SUFFIX)], root,
                          os.path.join(root, filename)))

    # A directory holding `.cu` files with no manifest either way is a backend
    # someone forgot to declare. Directories that belong to a declared backend
    # (its `kernels/`, `tools/`) or hold a manifest of their own are not.
    declared_roots = {directory for _entry, directory, _path in found}
    for entry in sorted(os.listdir(root)):
        directory = os.path.join(root, entry)
        if not os.path.isdir(directory) or entry.startswith("."):
            continue
        if any(directory == root_of or directory.startswith(root_of + os.sep)
               for root_of in declared_roots):
            continue
        contents = [f for f in os.listdir(directory) if not f.startswith(".")]
        present = sorted(f for f in contents if f.endswith(MANIFEST_SUFFIX))
        if present:
            # A manifest is here but not under the name discovery looks for, so
            # nothing loaded it. Continuing silently meant a typo in the filename
            # bypassed every check for that backend: exit 0, `checked: []`, no
            # errors and no warnings.
            issues.append(Issue(
                "error", entry,
                "has %s but not %s, so it is never discovered; rename it or fix the "
                "directory name" % (", ".join(present), entry + MANIFEST_SUFFIX)))
            continue
        kernel_like = [f for f in contents if f.endswith((".cu", ".cuh"))]
        if kernel_like:
            issues.append(Issue(
                "error", entry,
                "has kernel sources (%s) but no %s manifest or parent backend manifest"
                % (", ".join(sorted(kernel_like)[:3]), entry + MANIFEST_SUFFIX)))

    for entry, directory, expected in found:
        try:
            with open(expected, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except ValueError as error:
            issues.append(Issue("error", entry, "%s is not valid JSON: %s"
                                % (os.path.basename(expected), error)))
            continue
        if not isinstance(manifest, dict):
            issues.append(Issue("error", entry, "%s must contain a JSON object"
                                % os.path.basename(expected)))
            continue
        manifest["_directory"] = directory
        manifest["_entry"] = entry
        manifests.append(manifest)
    return manifests, issues


def check_manifest(manifest, issues, repo_root=None):
    backend = manifest.get("_entry", "?")
    directory = manifest.get("_directory", "")
    if repo_root is None:
        # Best effort when called directly: four levels up from
        # <repo>/src/backends/cuda/<name>. Callers that know the root should pass
        # it, because the flat layout makes this guess wrong.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(directory or ".")))))

    # These were two hand-written lists that had to agree, and they drifted:
    # `name` was reported as missing and then read anyway, so a manifest missing
    # only `name` raised KeyError out of check_manifest and aborted the checks of
    # every other backend, in both the text and --json paths. Deriving the guard
    # from the same constant is what stops that recurring.
    missing = [key for key in REQUIRED_KEYS if key not in manifest]
    for key in missing:
        issues.append(Issue("error", backend, "manifest is missing required key %r" % key))
    if missing:
        return

    name = manifest["name"]
    if not isinstance(name, str) or not name:
        issues.append(Issue("error", backend, "name must be a non-empty string"))
    elif name != backend:
        issues.append(Issue("error", backend,
                            "name %r does not match directory name %r" % (name, backend)))

    abi = manifest["abi_version"]
    if not isinstance(abi, int) or abi < CONTRACT_ABI_VERSION:
        issues.append(Issue("error", backend,
                            "abi_version must be an integer >= %d, got %r"
                            % (CONTRACT_ABI_VERSION, abi)))

    status = manifest["status"]
    if status not in STATUSES:
        issues.append(Issue("error", backend,
                            "status must be one of %s, got %r" % (", ".join(STATUSES), status)))
        return

    sources = manifest["sources"]
    if not isinstance(sources, list) or not all(isinstance(item, str) for item in sources):
        issues.append(Issue("error", backend, "sources must be a list of strings"))
    else:
        for source in sources:
            if os.path.isabs(source) or ".." in source.split("/"):
                issues.append(Issue("error", backend,
                                    "source %r must be relative to the backend directory" % source))
                continue
            if not os.path.isfile(os.path.join(directory, source)):
                issues.append(Issue("error", backend, "declared source %r does not exist" % source))

    build = manifest["build"]
    if not isinstance(build, dict):
        issues.append(Issue("error", backend, "build must be an object"))
        return

    for key in ("script", "output"):
        if not isinstance(build.get(key), str) or not build.get(key):
            issues.append(Issue("error", backend, "build.%s must be a non-empty string" % key))
    architectures = build.get("architectures")
    if not isinstance(architectures, list) or not architectures \
            or not all(isinstance(item, int) and 50 <= item <= 200 for item in architectures):
        issues.append(Issue("error", backend,
                            "build.architectures must be a non-empty list of compute "
                            "capabilities, e.g. [89, 90]"))
        architectures = None
    default_arch = build.get("default_arch")
    if default_arch is not None and architectures is not None and default_arch not in architectures:
        issues.append(Issue("error", backend,
                            "build.default_arch %r is not listed in build.architectures %r"
                            % (default_arch, architectures)))

    # What the kernels require is currently stated only in comments and READMEs
    # ("tensor-core kernels need sm_80 or newer"). Declaring it makes the claim
    # checkable here rather than discoverable as a build failure on someone
    # else's GPU.
    min_capability = build.get("min_capability")
    if min_capability is not None:
        if not isinstance(min_capability, int):
            issues.append(Issue("error", backend,
                                "build.min_capability must be an integer compute capability, "
                                "e.g. 80"))
        elif architectures is not None:
            too_low = [item for item in architectures if item < min_capability]
            if too_low:
                issues.append(Issue("error", backend,
                                    "build.architectures includes %r, below build.min_capability "
                                    "%d; the kernels would not build for that target"
                                    % (too_low, min_capability)))

    # A backend can serve more than one model. Kev and Cua-S1 share the same
    # Qwen3.5 backbone, so listing consumers is what makes reuse visible instead
    # of a private arrangement between two PRs.
    models = manifest.get("models")
    if models is not None:
        if not isinstance(models, list) or not all(isinstance(item, str) for item in models):
            issues.append(Issue("error", backend, "models must be a list of strings"))
        else:
            for model in models:
                if not model.endswith("/") or os.path.isabs(model) or ".." in model.split("/"):
                    issues.append(Issue("error", backend,
                                        "models entry %r must be a repository-relative directory "
                                        "path ending in '/'" % model))
                elif not os.path.isdir(os.path.join(repo_root, model)):
                    issues.append(Issue("warning", backend,
                                        "models entry %r does not exist yet; the consumer engine "
                                        "is not in the tree" % model))

    if status == "planned":
        return

    # A non-planned backend must ship the build script it names.
    script_path = ""
    if isinstance(build.get("script"), str):
        script_path = os.path.join(directory, build["script"])
        if not os.path.isfile(script_path):
            issues.append(Issue("error", backend,
                                "build.script %r does not exist" % build["script"]))
        else:
            _check_build_script(backend, script_path, build, architectures, issues)
            # The compile job runs the script directly, so the execute bit must
            # be set in git, not only in a local working copy: a fresh clone is
            # what CI checks out.
            if os.name == "posix" and not os.access(script_path, os.X_OK):
                issues.append(Issue("error", backend,
                                    "build.script %r is not executable in git; the compile job "
                                    "runs it as ./%s. Fix with: git update-index --chmod=+x %s"
                                    % (build["script"], build["script"], script_path)))

    numerics = manifest.get("numerics")
    reference = manifest.get("reference")
    if status == "validated":
        if not isinstance(numerics, dict) or not isinstance(numerics.get("tolerance"), dict) \
                or not numerics["tolerance"]:
            issues.append(Issue("error", backend,
                                "status is validated but numerics.tolerance is missing; a "
                                "parity claim needs a tolerance declared before comparison"))
        if not isinstance(reference, dict) or not reference.get("entrypoint"):
            issues.append(Issue("error", backend,
                                "status is validated but reference.entrypoint is missing"))
    if isinstance(reference, dict) and reference.get("entrypoint"):
        entrypoint = reference["entrypoint"]
        # Type first. `os.path.isabs` and `split` raise TypeError on a list or a
        # number, and that exception escaped check_manifest: the CLI printed no
        # structured report at all, not even under --json, and every later
        # backend went unchecked.
        if not isinstance(entrypoint, str):
            issues.append(Issue("error", backend,
                                "reference.entrypoint must be a string, got %s"
                                % type(entrypoint).__name__))
            entrypoint = ""
        elif os.path.isabs(entrypoint) or ".." in entrypoint.split("/"):
            issues.append(Issue("error", backend,
                                "reference.entrypoint must be repository-relative"))
        elif entrypoint:
            manifest["_repo_relative_entrypoint"] = entrypoint
            _check_reference_entrypoint(backend, entrypoint, issues, repo_root)
    if isinstance(numerics, dict) and isinstance(numerics.get("tolerance"), dict):
        for key, value in numerics["tolerance"].items():
            if key == "note":
                continue
            if not isinstance(value, (int, float)):
                issues.append(Issue("error", backend,
                                    "numerics.tolerance.%s must be a number or a note, got %r"
                                    % (key, value)))


def _check_reference_entrypoint(backend, entrypoint, issues, repo_root):
    """A declared reference that is not there cannot be run by Tier 2."""
    if not os.path.isfile(os.path.join(repo_root, entrypoint)):
        issues.append(Issue("warning", backend,
                            "reference.entrypoint %r does not exist yet; the Tier-2 GPU job "
                            "cannot run parity for this backend" % entrypoint))


def _check_build_script(backend, script_path, build, architectures, issues):
    try:
        with open(script_path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        issues.append(Issue("error", backend, "cannot read %s: %s" % (build.get("script"), error)))
        return

    parsed = parse_build_script(text)

    declared_output = build.get("output")
    if parsed["output"] and declared_output and parsed["output"] != declared_output:
        issues.append(Issue("error", backend,
                            "build.output %r does not match the %r the build script writes"
                            % (declared_output, parsed["output"])))

    if architectures is None:
        return

    # What nvcc is actually told to build, from the -gencode flags. This is the
    # authority: a variable that never reaches one of those flags does not make a
    # target reachable. A script that assigns `arch=${2:-89}` and then writes
    # `-gencode arch=compute_90,code=sm_90` builds sm_90 only, so a manifest
    # declaring sm_89 is false even though an `arch=` variable says 89.
    gencode = parsed.get("gencode_architectures") or []
    by_variable = parsed["architectures"]
    if gencode and by_variable:
        unused = [item for item in by_variable if item not in gencode]
        if unused:
            issues.append(Issue("error", backend,
                                "build script assigns %r but only passes %r to nvcc; the "
                                "assigned value never reaches a -gencode flag, so targets "
                                "read from variables alone cannot be claimed"
                                % (by_variable, gencode)))
    script_arch = gencode or by_variable or parsed["literal_architectures"]
    if not script_arch:
        issues.append(Issue("warning", backend,
                            "build script declares no compute capability; the manifest claims "
                            "%r but CI cannot confirm the script honours it" % (architectures,)))
        return
    unbuildable = [item for item in architectures if item not in script_arch]
    if unbuildable:
        issues.append(Issue("error", backend,
                            "build.architectures claims %r but the build script only reaches %r "
                            "(missing %r)" % (architectures, script_arch, unbuildable)))
    extra = [item for item in script_arch if item not in architectures]
    if extra:
        issues.append(Issue("warning", backend,
                            "build script also targets %r, which build.architectures omits"
                            % (extra,)))


def check_abi_consistency(manifests, issues):
    """A loader cannot know two ABI versions at once, so they must agree."""
    versions = {}
    for manifest in manifests:
        backend = manifest.get("_entry", "?")
        version = manifest.get("abi_version")
        if isinstance(version, int):
            versions.setdefault(version, []).append(backend)
    if len(versions) > 1:
        detail = ", ".join("%d (%s)" % (version, ", ".join(sorted(names)))
                           for version, names in sorted(versions.items()))
        issues.append(Issue("error", "cuda",
                            "backends declare different abi_version values: %s; a process that "
                            "loads two of them cannot check one version" % detail))


def run(repo_root):
    manifests, issues = load_manifests(repo_root)
    for manifest in manifests:
        check_manifest(manifest, issues, repo_root)
    check_abi_consistency(manifests, issues)
    return manifests, issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", default=".",
                        help="repository root to check (default: current directory)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    args = parser.parse_args(argv)

    repo_root = os.path.abspath(args.repo_root)
    manifests, issues = run(repo_root)
    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue for issue in issues if issue.level == "warning"]

    if args.json:
        json.dump({
            "checked": sorted(m.get("_entry", "?") for m in manifests),
            "issues": [issue.as_dict() for issue in issues],
            "errors": len(errors),
            "warnings": len(warnings),
        }, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for issue in issues:
            print(issue)
        for manifest in manifests:
            print("checked %s: status=%s abi=%s"
                  % (manifest.get("_entry"), manifest.get("status"), manifest.get("abi_version")))
        print("%d backend(s), %d error(s), %d warning(s)"
              % (len(manifests), len(errors), len(warnings)))
        if not manifests:
            print("no backends declared; add src/backends/cuda/<name>/<name>.backend.json")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
