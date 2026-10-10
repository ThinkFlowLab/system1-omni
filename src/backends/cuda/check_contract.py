#!/usr/bin/env python3
"""Tier-1 contract check for CUDA backend libraries. No GPU, no nvcc required.

Discovers every ``*.backend.json`` under ``src/backends/cuda/`` -- either
``<name>/<name>.backend.json`` or ``<name>.backend.json`` beside it -- and checks
the parts of ``contract.md`` that do not need hardware:

  * the manifest schema,
  * that every declared source exists and stays inside the backend directory,
  * that the build script writes the declared library and covers the declared
    architectures, none of them below ``build.min_capability``,
  * that ``abi_version`` matches the ``<PREFIX>_ABI_VERSION`` macro the declared
    sources define, where they define one,
  * that a ``validated`` backend declares a tolerance and a reference entrypoint.

It does not compile CUDA and does not prove numerics. Those are the later tiers
described in ``contract.md``: Tier 2 compiles each backend with nvcc, and Tier 3
runs the reference entrypoint on a self-hosted GPU runner. Neither is wired up
here, and no flag in this script reaches them.

Usage:
    python3 src/backends/cuda/check_contract.py [--repo-root PATH] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from build_script import parse_build_script

MANIFEST_SUFFIX = ".backend.json"
BACKENDS_DIR = os.path.join("src", "backends", "cuda")
STATUSES = ("planned", "experimental", "validated")
# Every key check_manifest reads directly. The missing-key report and the guard
# that stops after it are both derived from this tuple.
REQUIRED_KEYS = ("name", "abi_version", "status", "sources", "build")
# The oldest library ABI this checker understands -- a floor on `abi_version`,
# not a version of this document. See check_abi_version.
MINIMUM_ABI_VERSION = 1
# `<PREFIX>_ABI_VERSION N` in a backend header. #19 uses `CS1_ABI_VERSION`,
# bumped whenever the C interface changes.
_ABI_MACRO = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w*ABI_VERSION)[ \t]+(\d+)[ \t]*$", re.M)



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

    loaded = {}
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
        loaded[expected] = manifest
        manifests.append(manifest)

    # A directory holding `.cu` files with no manifest either way is a backend
    # someone forgot to declare; declaration is what makes it checkable. A
    # directory that belongs to a declared backend is not undeclared: a
    # `<name>/` backend owns everything beneath its own directory, and a flat
    # backend -- one whose manifest sits directly in src/backends/cuda/ -- owns
    # the subdirectories its `sources` name, which is how Laya's manifest covers
    # its `kernels/` and `tools/`.
    #
    # A flat backend does not own the cuda directory itself. Reading the root as
    # owned exempted every sibling subdirectory with it, so `qwen3_5/` beside a
    # flat `laya.backend.json` was never reported -- the failure this check
    # exists to catch.
    owned = set()
    for _entry, directory, expected in found:
        if directory != root:
            owned.add(directory)
            continue
        manifest = loaded.get(expected)
        sources = manifest.get("sources") if manifest else None
        if not isinstance(sources, list):
            # Nothing states which subdirectories this backend owns: it did not
            # parse, and that is already reported. Owning the whole directory
            # keeps this to the one real error instead of adding a misleading
            # undeclared-backend error for each of its own subdirectories.
            owned.add(root)
            continue
        for source in sources:
            if not isinstance(source, str) or "/" not in source:
                continue
            head = source.split("/")[0]
            if head and head not in (".", ".."):
                owned.add(os.path.join(root, head))

    for entry in sorted(os.listdir(root)):
        directory = os.path.join(root, entry)
        if not os.path.isdir(directory) or entry.startswith("."):
            continue
        if any(directory == owner or directory.startswith(owner + os.sep)
               for owner in owned):
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

    check_abi_version(manifest, issues)

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
                                    "build.script %r is not executable on the checked-out tree "
                                    "(CI takes that bit from the committed one); the compile job "
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


def _declared_abi(manifest):
    """Every ``#define *ABI_VERSION N`` in the backend's declared sources.

    Read as text, like the build script: the checker must not need a compiler,
    and the macro is a fact the source already states.
    """
    directory = manifest.get("_directory", "")
    sources = manifest.get("sources")
    # Guard the container, not just the elements. A manifest with `"sources": 42`
    # or `true` is reported by the schema check but does not stop it, so reaching
    # here and iterating raises TypeError out of check_manifest -- which loses the
    # whole report, including --json, and every later backend. Same failure the
    # string check on reference.entrypoint exists to prevent.
    if not isinstance(sources, list):
        return []
    found = []
    for source in sources:
        if not isinstance(source, str):
            continue
        path = os.path.join(directory, source)
        if not os.path.isfile(path):
            continue        # a missing source is already reported separately
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue
        for match in _ABI_MACRO.finditer(text):
            found.append((os.path.basename(source), match.group(1), int(match.group(2))))
    return found


def check_abi_version(manifest, issues):
    """``abi_version`` is the library's own, and its header is the authority.

    This is the one manifest field whose source of truth lives in the code, and
    the checker could not see it before: nothing read the header, so the only
    guard against a stale manifest was the example in contract.md. #19's
    ``ops.h`` says ``CS1_ABI_VERSION 4``; a manifest claiming 1 would have been
    accepted.

    Two libraries may legitimately differ, so there is deliberately no
    cross-backend sameness requirement: what has to hold is that each manifest
    agrees with its own header.
    """
    backend = manifest.get("_entry", "?")
    abi = manifest.get("abi_version")
    if not isinstance(abi, int) or abi < MINIMUM_ABI_VERSION:
        issues.append(Issue("error", backend,
                            "abi_version must be an integer >= %d, got %r"
                            % (MINIMUM_ABI_VERSION, abi)))
        return

    found = _declared_abi(manifest)
    if not found:
        issues.append(Issue("warning", backend,
                            "no #define *ABI_VERSION in the declared sources, so abi_version "
                            "%d cannot be checked against the library" % abi))
        return

    versions = {version for _file, _macro, version in found}
    if len(versions) > 1:
        issues.append(Issue("error", backend,
                            "the declared sources define more than one ABI version: %s"
                            % ", ".join("%s %s=%d" % entry for entry in sorted(found))))
        return

    header_version = versions.pop()
    if header_version != abi:
        file_name, macro, _value = found[0]
        issues.append(Issue("error", backend,
                            "abi_version is %d but %s defines %s %d; the manifest and the "
                            "library it describes have to agree"
                            % (abi, file_name, macro, header_version)))


def run(repo_root):
    manifests, issues = load_manifests(repo_root)
    for manifest in manifests:
        check_manifest(manifest, issues, repo_root)
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
            print("no backends declared; add src/backends/cuda/<name>.backend.json "
                  "or src/backends/cuda/<name>/<name>.backend.json")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
