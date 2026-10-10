"""Exercise the portable CLI against local HTTP, without model or GPU execution."""

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import threading
import unittest

RUNNER = Path(__file__).resolve().parents[2] / "recipe/jev_vl/replay.py"
BODY = {"kind": "noul", "effective_kind": "noul", "options": ["false", "true"],
        "probabilities": [0.2, 0.8], "choice_index": 1, "choice": "true"}


class ReplayCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.jsonl"
        self.request = {"kind": "noul", "state": "Synthetic test.", "question": "Ready?"}
        self.entries = [{"id": "txt-1", "request": self.request}]
        self.manifest.write_text(json.dumps(self.entries[0]) + "\n")
        self.reference = self.root / "reference"
        self.reference.mkdir()
        self.write_reference()
        self.calls = []
        self.status, self.response = 200, json.dumps(BODY)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.calls.append((self.path, payload))
                body = owner.response.encode()
                self.send_response(owner.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                if owner.status == 302:
                    self.send_header("Location", "/redirected")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def write_reference(self, body=BODY, status=200):
        (self.reference / "txt-1.json").write_text(
            json.dumps({"id": "txt-1", "status": status, "body": body})
        )

    def run_cli(self, name="run", *extra):
        return subprocess.run(
            [sys.executable, str(RUNNER), "--base", self.base,
             "--manifest", str(self.manifest), "--reference", str(self.reference),
             "--out", str(self.root / name), *extra],
            capture_output=True, text=True, timeout=15,
        )

    def read_results(self, name="run"):
        out = self.root / name
        return (json.loads((out / "summary.json").read_text()),
                [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()])

    def test_replays_unchanged_requests_and_excludes_per_pass_warmup(self):
        self.manifest.write_text("\n".join(json.dumps(row) for row in self.entries + [
            {"id": "img-1", "request": self.request}]) + "\n")
        result = self.run_cli("run", "--pattern", "txt-", "--warmup", "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        summary, rows = self.read_results()
        self.assertEqual((summary["measured"], summary["warmup"]), (2, 6))
        self.assertEqual(self.calls, [("/v1/systemone", self.request)] * 8)
        self.assertEqual(summary["mean_s"], statistics.fmean(
            row["client_s"] for row in rows if row["phase"] == "measured"))
        self.assertTrue(all(row["max_diff"] == 0 and row["probabilities"] == [0.2, 0.8] for row in rows))
        self.assertTrue(all(json.loads(row["raw_body"]) == BODY for row in rows))

    def test_http_and_malformed_probabilities_fail_with_raw_evidence(self):
        altered = []
        for field, value in [("probabilities", []), ("probabilities", [1.0]),
                             ("probabilities", [0.1, 0.9]), ("kind", "score"),
                             ("choice_index", 0), ("choice", "false")]:
            body = copy.deepcopy(BODY)
            body[field] = value
            altered.append((200, json.dumps(body)))
        alternate = BODY | {"options": ["a", "b", "c"], "probabilities": [0.1, 0.1, 0.8],
                            "choice_index": 2, "choice": "c"}
        altered += [(200, json.dumps(alternate)), (503, json.dumps(BODY)), (302, "{}"),
                    (200, "not json"), (200, "null")]
        for bad in ("NaN", "Infinity", "1e400"):
            altered.append((200, json.dumps(BODY).replace("0.2", bad)))
        for index, (self.status, self.response) in enumerate(altered):
            with self.subTest(status=self.status, response=self.response):
                result = self.run_cli(f"bad-{index}", "--warmup", "0", "--passes", "1")
                self.assertNotEqual(result.returncode, 0)
                summary, rows = self.read_results(f"bad-{index}")
                self.assertFalse(summary["ok"])
                self.assertEqual(summary["failures"], 1)
                self.assertEqual(rows[0]["raw_body"], self.response)
                self.assertGreater(rows[0]["client_s"], 0)
        self.assertEqual(len(self.calls), len(altered))  # A redirect is never followed.

    def test_warmup_failure_stops_before_measured_requests(self):
        self.status = 500
        result = self.run_cli()
        self.assertNotEqual(result.returncode, 0)
        summary, rows = self.read_results()
        self.assertEqual((summary["measured"], summary["warmup"]), (0, 1))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(rows[0]["phase"], "warmup")

    def test_probability_drift_within_fixed_tolerance_passes(self):
        self.response = json.dumps(BODY | {"probabilities": [0.22, 0.78]})
        result = self.run_cli("run", "--warmup", "0", "--passes", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        summary, _ = self.read_results()
        self.assertAlmostEqual(summary["max_diff"], 0.02)

    def test_rejects_empty_selection_empty_input_and_bad_reference_before_http(self):
        self.assertNotEqual(self.run_cli("zero-passes", "--passes", "0").returncode, 0)
        self.assertNotEqual(self.run_cli("negative-warmup", "--warmup", "-1").returncode, 0)
        result = self.run_cli("empty-selection", "--pattern", "missing-")
        self.assertNotEqual(result.returncode, 0)
        self.manifest.write_text("")
        self.assertNotEqual(self.run_cli("empty-input").returncode, 0)
        self.manifest.write_text(json.dumps(self.entries[0]) + "\n")
        for index, body in enumerate([BODY | {"probabilities": [float("nan"), 0.8]},
                                      BODY | {"probabilities": [1.0]}]):
            self.write_reference(body)
            self.assertNotEqual(self.run_cli(f"reference-{index}").returncode, 0)
        self.write_reference(status=500)
        self.assertNotEqual(self.run_cli("reference-status").returncode, 0)
        self.assertEqual(self.calls, [])

    def test_refuses_to_overwrite_existing_results(self):
        out = self.root / "run"
        out.mkdir()
        marker = out / "summary.json"
        marker.write_text("preserve me")
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertEqual(marker.read_text(), "preserve me")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
