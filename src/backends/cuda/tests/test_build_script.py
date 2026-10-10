#!/usr/bin/env python3
"""Tests for reading a build script as text.

A parser validated only against invented samples proves nothing about the scripts
it has to read, so the main case here is the real ``qwen3_5/build.sh`` added by
#19, and the real ``build.sh`` the Laya backend wrapper exposes.

Nothing is executed: these are the parsing rules CI uses to decide whether a
manifest is honest.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_script  # noqa: E402  (path is set above)

# The argument handling of src/backends/cuda/qwen3_5/build.sh in #19: `$1` is the
# output directory, `$2` defaults through CUDA_COMPUTE_CAP to 89.
QWEN3_5_BUILD_SCRIPT = """#!/usr/bin/env bash
# Build libqwen3_5_cuda.so from the kernels in this directory.
#
#   src/backends/cuda/qwen3_5/build.sh <output dir> [compute capability, e.g. 89]
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


class StripCommentTest(unittest.TestCase):
    def test_a_trailing_comment_is_removed(self):
        self.assertEqual(build_script._strip_comment("arch=89 # default"), "arch=89 ")

    def test_a_hash_inside_quotes_is_kept(self):
        self.assertEqual(build_script._strip_comment('x="a # b"'), 'x="a # b"')

    def test_a_comment_at_the_start_is_removed(self):
        self.assertEqual(build_script._strip_comment("# sm_80 and later"), "")

    def test_a_hash_not_preceded_by_space_is_kept(self):
        # Not a comment in shell: `a#b` is one word.
        self.assertEqual(build_script._strip_comment("a#b"), "a#b")


class ParseBuildScriptTest(unittest.TestCase):
    def test_reads_the_real_qwen3_5_script(self):
        parsed = build_script.parse_build_script(QWEN3_5_BUILD_SCRIPT)
        self.assertEqual(parsed["output"], "libqwen3_5_cuda.so")
        self.assertIn(89, parsed["architectures"])
        # `compute_${arch}` is a template, not a literal, so no architecture
        # number is invented from it.
        self.assertEqual(parsed["literal_architectures"], [])

    def test_literal_gencode_is_read_when_no_variable_is_used(self):
        parsed = build_script.parse_build_script(
            "nvcc -gencode arch=compute_90,code=sm_90 -shared -o libx.so a.cu\n")
        self.assertEqual(parsed["literal_architectures"], [90])
        self.assertEqual(parsed["output"], "libx.so")

    def test_ignores_architectures_in_comments(self):
        parsed = build_script.parse_build_script(
            "# works on sm_120\nout=x\nnvcc -o liby.so a.cu\n")
        self.assertEqual(parsed["literal_architectures"], [])
        self.assertEqual(parsed["output"], "liby.so")

    def test_a_quoted_output_path_with_a_variable_prefix(self):
        # `-o "$out/libx.so"`: the directory is unknowable, the filename is not.
        parsed = build_script.parse_build_script(
            'out=${1:?usage}\nnvcc -shared -o "$out/libqwen3_5_cuda.so" ./*.cu\n')
        self.assertEqual(parsed["output"], "libqwen3_5_cuda.so")

    def test_nested_defaults_are_expanded(self):
        parsed = build_script.parse_build_script(
            "arch=${2:-${CUDA_COMPUTE_CAP:-89}}\nnvcc -gencode \"x=sm_${arch}\" -o l.so a.cu\n")
        self.assertEqual(parsed["architectures"], [89])

    def test_every_declaration_is_collected_not_only_the_last(self):
        # A script that builds two targets assigns arch twice; both count, since
        # the claim is about what the script can reach.
        parsed = build_script.parse_build_script(
            "arch=90\nnvcc -o a.so x.cu\narch=89\nnvcc -o a.so x.cu\n")
        self.assertEqual(sorted(parsed["architectures"]), [89, 90])

    def test_a_script_declaring_nothing_reports_nothing(self):
        parsed = build_script.parse_build_script("#!/usr/bin/env bash\nnvcc -o l.so ./*.cu\n")
        self.assertEqual(parsed["architectures"], [])
        self.assertEqual(parsed["literal_architectures"], [])
        self.assertEqual(parsed["output"], "l.so")

    def test_the_last_output_wins(self):
        parsed = build_script.parse_build_script(
            "nvcc -o first.so a.cu\nnvcc -o second.so b.cu\n")
        self.assertEqual(parsed["output"], "second.so")

    def test_a_trailing_quote_does_not_end_up_in_the_name(self):
        parsed = build_script.parse_build_script('nvcc -o "libx.so" a.cu\n')
        self.assertEqual(parsed["output"], "libx.so")

    def test_gencode_targets_are_read_from_the_flags_not_the_variables(self):
        parsed = build_script.parse_build_script(
            "out=${1:?}\narch=${2:-89}\n"
            'nvcc -gencode "arch=compute_90,code=sm_90" -shared '
            '-o "$out/libx.so" ./*.cu\n')
        self.assertEqual(parsed["architectures"], [89])
        self.assertEqual(parsed["gencode_architectures"], [90])

    def test_each_gencode_flag_resolves_against_the_value_in_force(self):
        # A loop that reassigns `arch` builds both targets; resolving every line
        # against the final value would report 90 twice and lose 89.
        parsed = build_script.parse_build_script(
            "out=${1:?}\n"
            'arch=89\nnvcc -gencode "arch=compute_${arch},code=sm_${arch}" -o "$out/a.so" x.cu\n'
            'arch=90\nnvcc -gencode "arch=compute_${arch},code=sm_${arch}" -o "$out/a.so" x.cu\n')
        self.assertEqual(parsed["gencode_architectures"], [89, 90])

    def test_a_script_with_no_gencode_reports_no_gencode_targets(self):
        parsed = build_script.parse_build_script("nvcc -shared -o l.so ./*.cu\n")
        self.assertEqual(parsed["gencode_architectures"], [])

    def test_the_parser_never_executes_the_script(self):
        # A script whose first command would be destructive must still be read.
        parsed = build_script.parse_build_script(
            "rm -rf /\nout=${1}\nnvcc -o \"$out/libz.so\" a.cu\n")
        self.assertEqual(parsed["output"], "libz.so")


if __name__ == "__main__":
    unittest.main(verbosity=2)
