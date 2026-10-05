"""Check wrapper arguments and directory ownership without CUDA or TileLang."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "src/backends/cuda/build.sh"
FAKE_PYTHON = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys

step = "import" if sys.argv[1] == "-c" else Path(sys.argv[1]).stem
if os.environ.get("FAIL_STEP") == step:
    sys.exit(7)
if step == "build" and not os.environ.get("OMIT_LIBRARY"):
    stage = Path(sys.argv[2])
    (stage / "liblaya_cuda.so").write_bytes(b"test library")
    (stage / "build-manifest.json").write_text('{"arch":"sm_90a"}')
'''


class BuildEntryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "output dir"
        self.python = self.root / "fake python"
        self.python.write_text(FAKE_PYTHON)
        self.python.chmod(0o755)
        self.env = dict(os.environ, PYTHON=str(self.python), TMPDIR=str(self.root))
        for name in ("BUILD_STAGE", "KEEP_STAGE", "CUDA_COMPUTE_CAP",
                     "FAIL_STEP", "OMIT_LIBRARY"):
            self.env.pop(name, None)

    def run_build(self, *args, **env):
        return subprocess.run(
            ["bash", str(SCRIPT), *map(str, args)],
            env=dict(self.env, **env), text=True, capture_output=True,
        )

    def staging_path(self, result):
        line = next(line for line in result.stdout.splitlines()
                    if line.startswith("build.sh: staging in "))
        return Path(line.removeprefix("build.sh: staging in "))

    def assert_outputs(self):
        self.assertEqual((self.out / "liblaya_cuda.so").read_bytes(), b"test library")
        self.assertEqual((self.out / "build-manifest.json").read_text(), '{"arch":"sm_90a"}')

    def test_success_copies_outputs_and_removes_automatic_stage(self):
        result = self.run_build(self.out, "90")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_outputs()
        self.assertFalse(self.staging_path(result).exists())

    def test_custom_stage_is_preserved_on_success_and_failure(self):
        for step in ("", "export", "build"):
            with self.subTest(step=step):
                stage = self.root / ("custom " + (step or "success"))
                stage.mkdir()
                sentinel = stage / "existing.txt"
                sentinel.write_text("keep")
                result = self.run_build(self.out, BUILD_STAGE=str(stage), FAIL_STEP=step)
                self.assertEqual(result.returncode, 7 if step else 0, result.stderr)
                self.assertTrue(sentinel.exists(), result.stdout)
                self.assertEqual(sentinel.read_text(), "keep")

    def test_automatic_stage_is_removed_on_failure(self):
        for step in ("export", "build"):
            with self.subTest(step=step):
                result = self.run_build(self.out, FAIL_STEP=step)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertFalse(self.staging_path(result).exists())
                self.assertFalse(self.out.exists())

    def test_keep_stage_preserves_automatic_directory(self):
        result = self.run_build(self.out, KEEP_STAGE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.staging_path(result).is_dir())
        self.assert_outputs()

    def test_output_can_be_the_custom_stage(self):
        result = self.run_build(self.out, BUILD_STAGE=str(self.out))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_outputs()

    def test_invalid_architecture_is_rejected_before_staging(self):
        result = self.run_build(self.out, "89")
        self.assertEqual(result.returncode, 2)
        self.assertIn("sm_90a", result.stderr)
        self.assertFalse(self.out.exists())
        self.assertEqual(list(self.root.glob("laya-cuda.*")), [])

    def test_missing_python_or_tilelang_is_reported(self):
        for env in ({"PYTHON": str(self.root / "missing")}, {"FAIL_STEP": "import"}):
            with self.subTest(env=env):
                result = self.run_build(self.out, **env)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.out.exists())

    def test_missing_library_is_an_error(self):
        result = self.run_build(self.out, OMIT_LIBRARY="1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("did not produce", result.stderr)
        self.assertFalse(self.staging_path(result).exists())
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
