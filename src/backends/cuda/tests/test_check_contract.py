#!/usr/bin/env python3
"""Tests for the Tier-1 CUDA backend contract check.

No GPU, no nvcc, no CUDA toolkit. Run from the repository root:

    python3 -m unittest discover -s src/backends/cuda/tests -v
    python3 src/backends/cuda/tests/test_check_contract.py

The build-script case uses the real text of the ``qwen3_5/build.sh`` added by
PR #19, because a parser validated only against invented samples proves nothing
about the script it has to read.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import check_contract  # noqa: E402  (path is set above)

# The argument handling of src/backends/cuda/qwen3_5/build.sh in PR #19: `$1` is
# the output directory and `$2` defaults through CUDA_COMPUTE_CAP to 89.
QWEN3_5_BUILD_SCRIPT = """#!/usr/bin/env bash
# Build libqwen3_5_cuda.so from the kernels in this directory.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: build.sh <output dir> [compute capability]}
arch=${2:-${CUDA_COMPUTE_CAP:-89}}
mkdir -p "$out"
"$nvcc" -O3 -std=c++17 -gencode "arch=compute_${arch},code=[sm_${arch},compute_${arch}]" \\
    -shared -Xcompiler -fPIC -I"$here" "$here"/*.cu \\
    "${link[@]}" -o "$out/libqwen3_5_cuda.so"
echo "built $out/libqwen3_5_cuda.so for sm_${arch}"
"""

KERNELS_CU = """
#include "ops.h"
extern "C" int fake_kernel(const void* x, int n);
"""


def manifest(**overrides):
    """A minimal schema-complete manifest, matching PR #19's qwen3_5 backend.

    ``build.architectures`` is ``[89]`` because that is the whole truth about
    this build script: one invocation produces machine code for one compute
    capability, and PTX for newer parts to compile at load time.
    """
    value = {
        "name": "qwen3_5",
        "abi_version": 1,
        "status": "validated",
        "sources": ["ops.h", "kernels.cu"],
        "build": {
            "script": "build.sh",
            "output": "libqwen3_5_cuda.so",
            "default_arch": 89,
            "architectures": [89],
        },
        "numerics": {
            "precision": "bfloat16",
            "accumulation": "float32",
            "tolerance": {"max_abs": 0.039},
        },
        "reference": {"entrypoint": "recipe/cua_s1/check_native.py"},
    }
    value.update(overrides)
    return value


class CheckManifestTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="omni-contract-")
        self.directory = os.path.join(self.root, "src", "backends", "cuda", "qwen3_5")
        os.makedirs(self.directory)
        with open(os.path.join(self.directory, "kernels.cu"), "w", encoding="utf-8") as handle:
            handle.write(KERNELS_CU)
        with open(os.path.join(self.directory, "ops.h"), "w", encoding="utf-8") as handle:
            handle.write("#pragma once\n#define TEST_ABI_VERSION 1\n")
        with open(os.path.join(self.directory, "build.sh"), "w", encoding="utf-8") as handle:
            handle.write(QWEN3_5_BUILD_SCRIPT)
        os.chmod(os.path.join(self.directory, "build.sh"), 0o755)
        # The validated manifest declares this reference, so the fixture ships it.
        recipe = os.path.join(self.root, "recipe", "cua_s1")
        os.makedirs(recipe)
        with open(os.path.join(recipe, "check_native.py"), "w", encoding="utf-8") as handle:
            handle.write("import sys\nsys.exit(0)\n")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write_manifest(self, value):
        path = os.path.join(self.directory, "qwen3_5.backend.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(value, handle)

    def check(self):
        manifests, issues = check_contract.run(self.root)
        return manifests, [issue for issue in issues if issue.level == "error"], \
            [issue for issue in issues if issue.level == "warning"]

    def messages(self, issues):
        return " | ".join(issue.message for issue in issues)

    def test_a_complete_manifest_passes(self):
        self.write_manifest(manifest())
        manifests, errors, warnings = self.check()
        self.assertEqual(len(manifests), 1)
        self.assertEqual(errors, [], self.messages(errors))
        self.assertEqual(warnings, [], self.messages(warnings))

    def test_kernels_without_a_manifest_are_an_error(self):
        manifests, errors, _ = self.check()
        self.assertEqual(manifests, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("no qwen3_5.backend.json manifest", errors[0].message)

    def test_every_required_key_is_reported_and_then_guarded(self):
        """Removing any required key must produce a report, never an exception.

        The guard and the report were two hand-written lists, and `name` was in
        one but not the other: a manifest missing only `name` raised KeyError out
        of check_manifest, which also aborted the checks of every other backend
        and broke --json. Iterating over the keys rather than testing one of them
        is what covers the whole class.
        """
        for key in check_contract.REQUIRED_KEYS:
            value = manifest()
            del value[key]
            self.write_manifest(value)
            try:
                _, errors, _ = self.check()
            except KeyError as error:
                self.fail("removing %r raised %s instead of reporting it" % (key, error))
            self.assertIn("missing required key %r" % key, self.messages(errors))

    def test_a_manifest_missing_name_does_not_stop_other_backends(self):
        # The failure mode the maintainer reproduced: one bad manifest must not
        # hide the results for the rest.
        value = manifest()
        del value["name"]
        self.write_manifest(value)
        other = os.path.join(self.root, "src", "backends", "cuda", "other")
        os.makedirs(other)
        with open(os.path.join(other, "k.cu"), "w", encoding="utf-8") as handle:
            handle.write("// kernel\n")
        with open(os.path.join(other, "other.backend.json"), "w", encoding="utf-8") as handle:
            json.dump({"name": "other", "abi_version": 1, "status": "planned",
                       "sources": ["k.cu"],
                       "build": {"script": "build.sh", "output": "libother.so",
                                 "default_arch": 89, "architectures": [89]}}, handle)
        manifests, issues = check_contract.run(self.root)
        # A manifest missing `name` is still discovered and still listed; it
        # is the entry key that identifies it, which is why the checker
        # reports against the directory name rather than the missing field.
        self.assertEqual(sorted(m["_entry"] for m in manifests), ["other", "qwen3_5"])
        self.assertTrue(any("missing required key 'name'" in i.message for i in issues))

    def test_missing_required_key_is_an_error(self):
        value = manifest()
        del value["sources"]
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("missing required key 'sources'", self.messages(errors))

    def test_declared_source_that_does_not_exist_is_an_error(self):
        self.write_manifest(manifest(sources=["ops.h", "gdn_prefill.cu"]))
        _, errors, _ = self.check()
        self.assertIn("declared source 'gdn_prefill.cu' does not exist", self.messages(errors))

    def test_source_escaping_the_backend_directory_is_an_error(self):
        self.write_manifest(manifest(sources=["ops.h", "../../../etc/passwd"]))
        _, errors, _ = self.check()
        self.assertIn("must be relative to the backend directory", self.messages(errors))

    def test_output_name_must_match_the_build_script(self):
        value = manifest()
        value["build"]["output"] = "libwrong.so"
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("does not match the 'libqwen3_5_cuda.so' the build script writes",
                      self.messages(errors))

    def test_architecture_the_script_cannot_reach_is_an_error(self):
        # The build script defaults one architecture and offers no way to reach
        # another, so a manifest claiming [89, 90] is over-claiming.
        value = manifest()
        value["build"]["architectures"] = [89, 90]
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("build.architectures claims [89, 90] but the build script only reaches [89]",
                      self.messages(errors))

    def test_two_architectures_in_one_script_are_accepted(self):
        # The architectures have to be visible in the script. A `for arch in ...`
        # loop alone is not read as a declaration, so the script names them.
        script = """#!/usr/bin/env bash
set -euo pipefail
out=${1:?usage: build.sh <output dir>}
arch=89
"$nvcc" -gencode "arch=compute_${arch},code=sm_${arch}" -shared \\
    -o "$out/libmulti_cuda_${arch}.so" ./*.cu
arch=90
"$nvcc" -gencode "arch=compute_${arch},code=sm_${arch}" -shared \\
    -o "$out/libmulti_cuda_${arch}.so" ./*.cu
"""
        with open(os.path.join(self.directory, "build.sh"), "w", encoding="utf-8") as handle:
            handle.write(script)
        value = manifest()
        value["build"]["architectures"] = [89, 90]
        # The script's last assignment is arch=90, so that is the library a run
        # produces; the manifest has to name the same file.
        value["build"]["output"] = "libmulti_cuda_90.so"
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertEqual(errors, [], self.messages(errors))

    def test_warns_when_a_script_declares_no_compute_capability(self):
        with open(os.path.join(self.directory, "build.sh"), "w", encoding="utf-8") as handle:
            handle.write('#!/usr/bin/env bash\nnvcc -shared -o libqwen3_5_cuda.so ./*.cu\n')
        self.write_manifest(manifest())
        _, errors, warnings = self.check()
        self.assertEqual(errors, [], self.messages(errors))
        self.assertIn("declares no compute capability", self.messages(warnings))

    def test_min_capability_covered_by_the_architectures_passes(self):
        # The real #19 case: kernels documented as sm_80-and-later, built sm_89.
        value = manifest()
        value["build"]["min_capability"] = 80
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertEqual(errors, [], self.messages(errors))

    def test_an_architecture_below_the_kernel_requirement_is_an_error(self):
        value = manifest()
        value["build"]["min_capability"] = 90
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("below build.min_capability 90", self.messages(errors))

    def test_a_non_integer_min_capability_is_an_error(self):
        value = manifest()
        value["build"]["min_capability"] = "80"
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("must be an integer compute capability", self.messages(errors))

    def test_listing_a_consumer_engine_that_exists_passes(self):
        # Kev and Cua-S1 share the Qwen3.5 backbone, so one backend can serve
        # both without either model owning a private copy of the kernels.
        os.makedirs(os.path.join(self.root, "src", "models", "kev"))
        value = manifest(models=["src/models/cua_s1/", "src/models/kev/"])
        self.write_manifest(value)
        _, errors, warnings = self.check()
        self.assertEqual(errors, [], self.messages(errors))
        self.assertIn("src/models/cua_s1/", self.messages(warnings))

    def test_a_consumer_path_that_is_not_a_directory_is_an_error(self):
        value = manifest(models=["src/models/kev"])
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("must be a repository-relative directory path", self.messages(errors))

    def test_a_non_list_models_field_is_an_error(self):
        value = manifest(models="src/models/kev/")
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("models must be a list of strings", self.messages(errors))

    def test_min_capability_is_optional(self):
        # Omitting it must not fail: not every backend states a requirement yet.
        self.write_manifest(manifest())
        _, errors, _ = self.check()
        self.assertEqual(errors, [], self.messages(errors))

    def test_default_arch_outside_the_list_is_an_error(self):
        value = manifest()
        value["build"]["default_arch"] = 75
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("is not listed in build.architectures", self.messages(errors))

    def test_validated_without_a_tolerance_is_an_error(self):
        value = manifest()
        del value["numerics"]["tolerance"]
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("status is validated but numerics.tolerance is missing",
                      self.messages(errors))

    def test_validated_without_a_reference_entrypoint_is_an_error(self):
        value = manifest()
        del value["reference"]
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("status is validated but reference.entrypoint is missing",
                      self.messages(errors))

    def test_experimental_needs_neither_tolerance_nor_entrypoint(self):
        self.write_manifest(manifest(status="experimental", numerics={}, reference={}))
        _, errors, warnings = self.check()
        self.assertEqual(errors, [], self.messages(errors))
        self.assertEqual(warnings, [], self.messages(warnings))

    def test_planned_backend_is_not_required_to_build(self):
        # `planned` means the directory is declared but nothing is built yet, so
        # a build script that does not exist is fine. The schema still applies.
        self.write_manifest(manifest(status="planned", sources=["ops.h"], build={
            "script": "build.sh", "output": "libqwen3_5_cuda.so",
            "default_arch": 89, "architectures": [89],
        }))
        _, errors, _ = self.check()
        self.assertEqual(errors, [], self.messages(errors))

    def test_a_non_executable_build_script_is_an_error(self):
        # The compile job runs `./build.sh`, so the execute bit is part of the
        # contract, not a local detail.
        os.chmod(os.path.join(self.directory, "build.sh"), 0o644)
        self.write_manifest(manifest())
        _, errors, _ = self.check()
        self.assertIn("is not executable", self.messages(errors))

    def test_a_non_string_reference_entrypoint_is_reported_not_raised(self):
        # A list here reached os.path.isabs, whose TypeError escaped
        # check_manifest: no report at all, not even --json, and every later
        # backend went unchecked.
        self.write_manifest(manifest(reference={"entrypoint": ["reference.py"]}))
        _, errors, _ = self.check()
        self.assertIn("reference.entrypoint must be a string, got list",
                      self.messages(errors))

    def test_an_empty_reference_entrypoint_is_reported(self):
        self.write_manifest(manifest(reference={"entrypoint": ""}))
        _, errors, _ = self.check()
        self.assertIn("status is validated but reference.entrypoint is missing",
                      self.messages(errors))

    def test_an_assigned_architecture_never_passed_to_nvcc_is_an_error(self):
        # The variable says 89; the compiler is told 90. Reading the variable
        # alone accepted a manifest declaring sm_89 that the script cannot build.
        script = ("#!/usr/bin/env bash\n"
                  "out=${1:?usage}\n"
                  "arch=${2:-89}\n"
                  'nvcc -gencode "arch=compute_90,code=sm_90" '
                  '-shared -o "$out/libqwen3_5_cuda.so" ./*.cu\n')
        with open(os.path.join(self.directory, "build.sh"), "w", encoding="utf-8") as handle:
            handle.write(script)
        os.chmod(os.path.join(self.directory, "build.sh"), 0o755)
        self.write_manifest(manifest())
        _, errors, _ = self.check()
        self.assertIn("never reaches a -gencode flag", self.messages(errors))

    def test_a_misnamed_manifest_is_an_error_not_an_exemption(self):
        # Discovery looks for <dir>/<dir>.backend.json. A typo there used to be
        # exempted silently: exit 0, checked: [], no errors.
        self.write_manifest(manifest())
        os.rename(os.path.join(self.directory, "qwen3_5.backend.json"),
                  os.path.join(self.directory, "typo.backend.json"))
        manifests, issues = check_contract.run(self.root)
        self.assertEqual(manifests, [])
        messages = " | ".join(issue.message for issue in issues)
        self.assertIn("has typo.backend.json but not qwen3_5.backend.json", messages)
        self.assertIn("never discovered", messages)

    def test_no_field_of_the_wrong_type_crashes_the_checker(self):
        """Every manifest field, given a wrong type, must produce a report.

        This is the class the review kept finding one instance at a time: a field
        that is checked for existence or for being a list somewhere, then read
        somewhere else without the same guard. `sources: 42` reached an iteration
        in the ABI scan and raised, which cost the whole report including --json
        and every later backend -- exactly what the string check on
        reference.entrypoint was added to prevent. Iterating the fields, rather
        than testing them one at a time, is what covers the class.
        """
        wrong_values = [None, 42, True, "text", [], {}, [1, 2], ["a", 3]]
        for key in check_contract.REQUIRED_KEYS:
            for value in wrong_values:
                candidate = manifest()
                candidate[key] = value
                self.write_manifest(candidate)
                try:
                    check_contract.run(self.root)
                except Exception as error:            # noqa: BLE001 - the point
                    self.fail("%s = %r raised %s: %s"
                              % (key, value, type(error).__name__, error))

    def test_build_subfields_of_the_wrong_type_do_not_crash(self):
        for key in ("script", "output", "architectures", "default_arch", "min_capability"):
            for value in (None, 42, True, "text", [], {}, [1, 2], ["a", 3]):
                candidate = manifest()
                candidate["build"][key] = value
                self.write_manifest(candidate)
                try:
                    check_contract.run(self.root)
                except Exception as error:            # noqa: BLE001 - the point
                    self.fail("build.%s = %r raised %s: %s"
                              % (key, value, type(error).__name__, error))

    def test_optional_fields_of_the_wrong_type_do_not_crash(self):
        for key, value in (("models", 42), ("models", True), ("models", "text"),
                           ("models", {}), ("models", [1, 2]),
                           ("numerics", 42), ("numerics", "text"),
                           ("reference", 42), ("reference", "text"),
                           ("reference", {"entrypoint": 42}),
                           ("reference", {"entrypoint": True}),
                           ("numerics", {"tolerance": 42}),
                           ("numerics", {"tolerance": "text"})):
            candidate = manifest()
            candidate[key] = value
            self.write_manifest(candidate)
            try:
                check_contract.run(self.root)
            except Exception as error:                # noqa: BLE001 - the point
                self.fail("%s = %r raised %s: %s"
                          % (key, value, type(error).__name__, error))

    def test_unknown_status_is_an_error(self):
        self.write_manifest(manifest(status="done"))
        _, errors, _ = self.check()
        self.assertIn("status must be one of", self.messages(errors))

    def test_name_must_match_the_directory(self):
        self.write_manifest(manifest(name="laya"))
        _, errors, _ = self.check()
        self.assertIn("does not match directory name 'qwen3_5'", self.messages(errors))

    def test_non_numeric_tolerance_is_an_error(self):
        value = manifest()
        value["numerics"]["tolerance"] = {"max_abs": "small"}
        self.write_manifest(value)
        _, errors, _ = self.check()
        self.assertIn("numerics.tolerance.max_abs must be a number", self.messages(errors))

    def test_reference_entrypoint_that_is_not_there_warns(self):
        # The fixture ships the reference; a declared-but-missing one must warn,
        # because Tier 2 would have nothing to run.
        os.remove(os.path.join(self.root, "recipe", "cua_s1", "check_native.py"))
        self.write_manifest(manifest())
        _, errors, warnings = self.check()
        self.assertEqual(errors, [], self.messages(errors))
        self.assertIn("reference.entrypoint", self.messages(warnings))

    def test_absolute_reference_entrypoint_is_an_error(self):
        self.write_manifest(manifest(reference={"entrypoint": "/tmp/check.py"}))
        _, errors, _ = self.check()
        self.assertIn("must be repository-relative", self.messages(errors))

    def test_invalid_json_is_reported_not_raised(self):
        with open(os.path.join(self.directory, "qwen3_5.backend.json"), "w",
                  encoding="utf-8") as handle:
            handle.write("{not json")
        manifests, errors, _ = self.check()
        self.assertEqual(manifests, [])
        self.assertIn("is not valid JSON", self.messages(errors))


class AbiVersionTest(unittest.TestCase):
    """`abi_version` is the library's own, taken from its header.

    Read as text, so the checker needs no compiler. What has to hold is that a
    manifest agrees with the header of the library it describes. Two libraries
    may differ from each other, so there is deliberately no cross-backend
    sameness rule: the old code required one, which would have forced unrelated
    model engines onto a shared interface version.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="omni-abi-")
        self.directory = os.path.join(self.root, "src", "backends", "cuda")
        os.makedirs(self.directory)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def add_backend(self, name, abi_version, macro=None, extra_source=None):
        directory = os.path.join(self.directory, name)
        os.makedirs(directory)
        header = "#pragma once\n"
        if macro is not None:
            header += "#define %s_ABI_VERSION %d\n" % (name.upper(), macro)
        with open(os.path.join(directory, "ops.h"), "w", encoding="utf-8") as handle:
            handle.write(header)
        sources = ["ops.h"]
        if extra_source is not None:
            with open(os.path.join(directory, "extra.h"), "w", encoding="utf-8") as handle:
                handle.write("#define OTHER_ABI_VERSION %d\n" % extra_source)
            sources.append("extra.h")
        with open(os.path.join(directory, "build.sh"), "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\nout=${1:?}\narch=${2:-89}\n"
                         'nvcc -o "$out/lib%s.so" ./*.cu\n' % name)
        os.chmod(os.path.join(directory, "build.sh"), 0o755)
        value = {
            "name": name,
            "abi_version": abi_version,
            "status": "planned",
            "sources": sources,
            "build": {"script": "build.sh", "output": "lib%s.so" % name,
                      "default_arch": 89, "architectures": [89]},
        }
        with open(os.path.join(directory, name + ".backend.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(value, handle)

    def check(self, root=None):
        _, issues = check_contract.run(root or self.root)
        errors = [i.message for i in issues if i.level == "error"]
        warnings = [i.message for i in issues if i.level == "warning"]
        return errors, warnings

    def test_a_manifest_agreeing_with_its_header_passes(self):
        self.add_backend("qwen3_5", 4, macro=4)
        errors, warnings = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_a_manifest_disagreeing_with_its_header_is_an_error(self):
        # The case nothing caught before: #19's ops.h says 4, and a manifest
        # claiming 1 was accepted because no check read the header.
        self.add_backend("qwen3_5", 1, macro=4)
        errors, _ = self.check()
        self.assertEqual(len(errors), 1)
        self.assertIn("abi_version is 1 but ops.h defines QWEN3_5_ABI_VERSION 4", errors[0])

    def test_sources_without_an_abi_macro_warn(self):
        self.add_backend("qwen3_5", 1, macro=None)
        errors, warnings = self.check()
        self.assertEqual(errors, [])
        self.assertIn("cannot be checked against the library", warnings[0])

    def test_two_versions_in_one_backends_sources_are_an_error(self):
        self.add_backend("qwen3_5", 4, macro=4, extra_source=7)
        errors, _ = self.check()
        self.assertEqual(len(errors), 1)
        self.assertIn("more than one ABI version", errors[0])

    def test_abi_version_below_the_contract_is_an_error(self):
        self.add_backend("qwen3_5", 0, macro=0)
        errors, _ = self.check()
        self.assertIn("abi_version must be an integer >= 1", " | ".join(errors))

    def test_two_backends_may_declare_different_versions(self):
        # Deliberately allowed now. Requiring them to match would force unrelated
        # model engines onto one interface version, which the repository layout
        # explicitly does not ask for.
        self.add_backend("qwen3_5", 4, macro=4)
        self.add_backend("laya", 2, macro=2)
        errors, warnings = self.check()
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_a_planned_backend_is_still_checked(self):
        self.add_backend("qwen3_5", 9, macro=4)
        errors, _ = self.check()
        self.assertIn("abi_version is 9", " | ".join(errors))


class RepositoryStateTest(unittest.TestCase):
    """The checker must pass on a repository that satisfies the contract.

    The fixture mirrors the qwen3_5 backend as PR #19 defines it, including a
    build script with the same argument handling, so this exercises discovery,
    script parsing, schema checks and the ABI rule together.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="omni-repo-")
        self.directory = os.path.join(self.root, "src", "backends", "cuda", "qwen3_5")
        os.makedirs(self.directory)
        with open(os.path.join(self.directory, "kernels.cu"), "w", encoding="utf-8") as handle:
            handle.write(KERNELS_CU)
        with open(os.path.join(self.directory, "ops.h"), "w", encoding="utf-8") as handle:
            handle.write("#pragma once\n#define TEST_ABI_VERSION 1\n")
        with open(os.path.join(self.directory, "build.sh"), "w", encoding="utf-8") as handle:
            handle.write(QWEN3_5_BUILD_SCRIPT)
        os.chmod(os.path.join(self.directory, "build.sh"), 0o755)
        recipe = os.path.join(self.root, "recipe", "cua_s1")
        os.makedirs(recipe)
        with open(os.path.join(recipe, "check_native.py"), "w", encoding="utf-8") as handle:
            handle.write("import sys\nsys.exit(0)\n")
        with open(os.path.join(self.directory, "qwen3_5.backend.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(manifest(), handle)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_contract_satisfying_repository_has_no_findings(self):
        manifests, issues = check_contract.run(self.root)
        self.assertEqual([issue.message for issue in issues], [])
        self.assertEqual([m["name"] for m in manifests], ["qwen3_5"])

    def test_the_repository_under_test_is_discovered_by_the_script(self):
        """Running the module as CI does must succeed on this fixture."""
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = check_contract.main(["--repo-root", self.root, "--json"])
        self.assertEqual(exit_code, 0)
        self.assertIn('"errors": 0', buffer.getvalue())


class FlatLayoutTest(unittest.TestCase):
    """A backend whose files sit directly under src/backends/cuda/.

    Laya's kernels and tools are `src/backends/cuda/kernels/` and
    `src/backends/cuda/tools/` rather than a `<name>/` subdirectory, so the
    manifest sits beside them as `laya.backend.json`. The layout is the model
    author's call; only the manifest's contents are fixed.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="omni-flat-")
        self.cuda = os.path.join(self.root, "src", "backends", "cuda")
        os.makedirs(os.path.join(self.cuda, "kernels"))
        os.makedirs(os.path.join(self.cuda, "tools"))
        for path in ("kernels/runtime.cu", "kernels/model_ops.cu"):
            with open(os.path.join(self.cuda, path), "w", encoding="utf-8") as handle:
                handle.write("// kernel\n")
        with open(os.path.join(self.cuda, "ops.h"), "w", encoding="utf-8") as handle:
            handle.write("#pragma once\n#define LAYA_ABI_VERSION 1\n")
        with open(os.path.join(self.cuda, "build.sh"), "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\n"
                         "out=${1:?usage}\n"
                         "arch=${2:-90}\n"
                         'nvcc -gencode "arch=compute_${arch},code=sm_${arch}" '
                         '-shared -o "$out/liblaya_cuda.so" ./*.cu\n')
        os.chmod(os.path.join(self.cuda, "build.sh"), 0o755)
        os.makedirs(os.path.join(self.root, "recipe", "laya"))
        with open(os.path.join(self.root, "recipe", "laya", "check.py"), "w",
                  encoding="utf-8") as handle:
            handle.write("import sys\nsys.exit(0)\n")
        manifest = {
            "name": "laya",
            "abi_version": 1,
            "status": "experimental",
            "sources": ["ops.h", "kernels/runtime.cu", "kernels/model_ops.cu"],
            "build": {
                "script": "build.sh",
                "output": "liblaya_cuda.so",
                "default_arch": 90,
                "architectures": [90],
                "min_capability": 90,
            },
        }
        with open(os.path.join(self.cuda, "laya.backend.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_flat_backend_is_discovered(self):
        manifests, issues = check_contract.run(self.root)
        self.assertEqual([m["name"] for m in manifests], ["laya"])
        self.assertEqual([i.message for i in issues], [])

    def test_its_sources_are_resolved_against_the_cuda_directory(self):
        manifests, _ = check_contract.run(self.root)
        self.assertEqual(manifests[0]["_directory"],
                         os.path.join(self.root, "src", "backends", "cuda"))

    def test_a_validated_flat_backend_finds_its_reference(self):
        # The repo root must not be inferred by walking up from the backend
        # directory: for the flat layout that lands one level too high.
        path = os.path.join(self.cuda, "laya.backend.json")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["status"] = "validated"
        manifest["numerics"] = {"tolerance": {"max_abs": 0.002}}
        manifest["reference"] = {"entrypoint": "recipe/laya/check.py"}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        _, issues = check_contract.run(self.root)
        self.assertEqual([i.message for i in issues], [],
                         "an existing reference must not be reported missing")

    def test_an_undeclared_sibling_backend_is_still_reported(self):
        # A flat manifest owns the subdirectories its `sources` name, not the
        # cuda directory itself. Reading the root as owned exempted every
        # sibling along with it, so a backend sitting beside Laya with no
        # manifest of its own passed the check that exists to find it.
        sibling = os.path.join(self.cuda, "qwen3_5")
        os.makedirs(sibling)
        with open(os.path.join(sibling, "attention.cu"), "w", encoding="utf-8") as handle:
            handle.write("// kernel\n")
        _, issues = check_contract.run(self.root)
        self.assertEqual([i.backend for i in issues], ["qwen3_5"])
        self.assertIn("no qwen3_5.backend.json manifest", issues[0].message)

    def test_the_flat_backends_own_subdirectories_are_not_reported(self):
        # The same rule has to keep covering `kernels/`, which the manifest does
        # name. Without this, a fix could exempt nothing and hand the backend
        # its own sources back as an undeclared backend.
        _, issues = check_contract.run(self.root)
        self.assertEqual([i.message for i in issues], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
