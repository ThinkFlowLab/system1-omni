import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

with mock.patch.object(sys, "path", [str(Path(__file__).parent), *sys.path]):
    spec = importlib.util.spec_from_file_location("decider_modes", Path(__file__).with_name("verify_modes.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class ModeProtocol(unittest.TestCase):
    def test_revised_comparison_method_is_frozen_before_mode_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parity = root / "parity"
            parity.mkdir()
            (parity / "protocol.json").write_text(json.dumps({"cases": []}))
            for name in ("reference.jsonl", "native.jsonl"):
                (parity / name).write_text("")
            binary = root / "binary"
            binary.write_bytes(b"test artifact, never executed")
            plan = root / "plan.json"
            plan.write_text(json.dumps({"repetitions": 2, "parity_output": str(parity), "timed_cases": [],
                "library": str(binary), "model": "model", "modes": [{"name": "control", "binary": str(binary), "env": {}}]}))
            output = root / "output"

            def launch(*args, **kwargs):
                protocol = json.loads((output / "protocol.json").read_text())
                self.assertEqual(protocol["validation_protocol"]["version"], 2)
                self.assertEqual(protocol["gates"], protocol["validation_protocol"]["original_gates"])
                self.assertEqual(protocol["baseline_protocol"], {"cases": []})
                raise RuntimeError("stopped before execution")

            with mock.patch("sys.argv", ["verify_modes.py", "--plan", str(plan), "--output", str(output)]), \
                    mock.patch.object(module.subprocess, "Popen", side_effect=launch):
                with self.assertRaisesRegex(RuntimeError, "stopped before execution"):
                    module.main()


if __name__ == "__main__":
    unittest.main()
