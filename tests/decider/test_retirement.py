import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "decider_retirement", Path(__file__).with_name("verify_retirement.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WorkerLaunch(unittest.TestCase):
    def run_cli(self, batched, inherited):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = ["verify_retirement.py", "--worker", "worker", "--model", "model",
                    "--wrapper", "wrapper.so", "--output", str(output)]
            if batched:
                args.append("--batched")

            def launch(*args, **kwargs):
                # Simulate external GPU evidence so this host-only test can
                # inspect the actual CLI's worker launch and request boundary.
                rows = 2 if batched else 1
                (output / "worker.log").write_text(
                    f"Decider test projection 1 rows 1\n"
                    f"Decider test projection 2 rows {rows}\n"
                )
                return child

            child = mock.Mock(pid=123)
            replies = [(503, b'{"detail":"model inference failed"}'),
                       (503, b'{"status":"unavailable"}'),
                       (503, b'{"detail":"model unavailable"}'),
                       (503, b'{"detail":"model unavailable"}')]
            with mock.patch.dict(os.environ, inherited, clear=True), \
                    mock.patch.object(module.subprocess, "Popen", side_effect=launch) as popen, \
                    mock.patch.object(module, "wait_ready", return_value={"status": "ready"}), \
                    mock.patch.object(module, "memory", side_effect=[4000, 0]), \
                    mock.patch.object(module, "request", side_effect=replies) as request, \
                    mock.patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            child.terminate.assert_called_once()
            child.wait.assert_called_once_with(timeout=30)
            raw = json.loads(request.call_args_list[0].args[2])
            self.assertEqual(raw["questions"]["q"].get("type", "choice"),
                             "score" if batched else "choice")
            self.assertEqual(len(raw["questions"]["q"]["criteria"]), 2)
            return popen.call_args.kwargs["env"]

    def assert_mode(self, environment, rows):
        self.assertEqual(environment["DECIDER_BATCH_MAX_ROWS"], str(rows))
        self.assertEqual(environment["DECIDER_BATCH_MAX_TOKENS"], "4096")
        for name in ("CUA_S1_GRAPH", "DECIDER_GRAPH", "DECIDER_PREFIX", "DECIDER_FIXED"):
            self.assertEqual(environment[name], "0", name)

    def test_batched_launch_without_exported_limits(self):
        self.assert_mode(self.run_cli(True, {}), 2)

    def test_batched_launch_overrides_inherited_limits_and_modes(self):
        inherited = {"DECIDER_BATCH_MAX_ROWS": "1", "DECIDER_BATCH_MAX_TOKENS": "1",
                     "CUA_S1_GRAPH": "1", "DECIDER_GRAPH": "1",
                     "DECIDER_PREFIX": "auto", "DECIDER_FIXED": "1"}
        self.assert_mode(self.run_cli(True, inherited), 2)

    def test_unbatched_launch_overrides_inherited_limits_and_modes(self):
        inherited = {"DECIDER_BATCH_MAX_ROWS": "4", "DECIDER_BATCH_MAX_TOKENS": "1",
                     "CUA_S1_GRAPH": "1", "DECIDER_GRAPH": "1",
                     "DECIDER_PREFIX": "1", "DECIDER_FIXED": "1"}
        self.assert_mode(self.run_cli(False, inherited), 1)


if __name__ == "__main__":
    unittest.main()
