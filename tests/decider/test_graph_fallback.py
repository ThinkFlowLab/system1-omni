import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

# The CLI's sibling imports use its script directory in normal execution.
with mock.patch.object(sys, "path", [str(Path(__file__).parent), *sys.path]):
    spec = importlib.util.spec_from_file_location("decider_graph_fallback", Path(__file__).with_name("verify_graph_fallback.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


class GraphFallbackLaunch(unittest.TestCase):
    def test_launch_resets_modes_and_freezes_protocol_before_failed_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parity = root / "parity"
            parity.mkdir()
            (parity / "protocol.json").write_text(json.dumps({"cases": []}))
            (parity / "reference.jsonl").write_text("")
            output = root / "output"
            argv = ["verify_graph_fallback.py", "--model", "model", "--binary", "binary",
                    "--wrapper", "wrapper", "--parity-output", str(parity), "--output", str(output)]
            inherited = {"DECIDER_GRAPH": "0", "DECIDER_PREFIX": "auto", "DECIDER_FIXED": "1",
                         "DECIDER_BATCH_MAX_ROWS": "1", "DECIDER_BATCH_MAX_TOKENS": "1", "CUA_S1_GRAPH": "0"}

            def launch(*args, **kwargs):
                expected = {"DECIDER_GRAPH": "1", "DECIDER_PREFIX": "0", "DECIDER_FIXED": "0",
                            "DECIDER_BATCH_MAX_ROWS": "4", "DECIDER_BATCH_MAX_TOKENS": "4096", "CUA_S1_GRAPH": "1"}
                self.assertEqual({key: kwargs["env"][key] for key in expected}, expected)
                protocol = json.loads((output / "protocol.json").read_text())
                self.assertEqual(protocol["env"], expected)
                self.assertEqual(protocol["validation_protocol"]["version"], 2)
                raise subprocess.CalledProcessError(1, args[0])

            with mock.patch.dict(os.environ, inherited, clear=True), mock.patch("sys.argv", argv), \
                    mock.patch.object(module.subprocess, "run", side_effect=launch):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.main()
            self.assertTrue((output / "protocol.json").is_file())


if __name__ == "__main__":
    unittest.main()
