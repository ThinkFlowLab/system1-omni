import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_bench import CASES, response_for

from benchmarks import bench, weekly

GATES = {
    "max_accuracy_drop": 0,
    "max_score_mae_increase": 0.02,
    "max_latency_increase": 0.15,
    "max_probability_drift": 0.02,
    "max_score_drift": 0.02,
    "max_flips": 0,
}


class WeeklyTests(unittest.TestCase):
    def test_reference_fidelity_failure_preserves_complete_measurements(self):
        self.make_run(self.root / "reference")
        self.make_run(self.root / "native", wrong=True)
        report = {
            "models": {
                name: {"status": "baseline_recorded", "measurements_complete": True}
                for name in ("reference", "native")
            }
        }
        suite = {
            "gates": GATES,
            "models": [
                {"id": "reference"},
                {"id": "native", "reference_model": "reference"},
            ],
        }
        weekly.check_references(suite, self.root, report)
        self.assertEqual(report["models"]["native"]["status"], "failed")
        self.assertTrue(report["models"]["native"]["measurements_complete"])
        self.assertFalse(report["models"]["native"]["reference_parity"]["passed"])
        self.assertTrue((self.root / "native/reference-parity.json").exists())

    def test_matching_reference_is_saved_without_marking_baseline_as_approved(self):
        self.make_run(self.root / "reference")
        self.make_run(self.root / "native")
        report = {
            "models": {
                name: {"status": "baseline_recorded", "measurements_complete": True}
                for name in ("reference", "native")
            }
        }
        weekly.check_references(
            {
                "gates": GATES,
                "models": [
                    {"id": "reference"},
                    {"id": "native", "reference_model": "reference"},
                ],
            },
            self.root,
            report,
        )
        self.assertTrue(report["models"]["native"]["reference_parity"]["passed"])
        self.assertEqual(report["models"]["native"]["status"], "baseline_recorded")

    def test_runtime_checkout_is_passed_to_child_after_scheduler_separator(self):
        runtime = self.root / "runtime"
        executable = runtime / "target/release/fake-worker"
        executable.parent.mkdir(parents=True)
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        suite = {
            "gates": GATES,
            "models": [
                {
                    "id": "test",
                    "manifest": str(self.manifest),
                    "model": "test",
                    "health": "http://localhost/health",
                    "endpoint": "http://localhost/v1/systemone",
                    "commands": [["target/release/fake-worker"]],
                    "metadata": {k: "mock-only" for k in weekly.CONTROL_KEYS[:7]},
                }
            ],
        }
        path = self.root / "suite.json"
        bench.write_json(path, suite)
        args = [
            "weekly.py",
            "run",
            "--repo",
            str(runtime),
            "--suite",
            str(path),
            "--output",
            str(self.root / "output"),
            "--gpu-ids",
            "0",
        ]
        with (
            patch.object(weekly, "ROOT", weekly.ROOT),
            patch.object(sys, "argv", args),
            patch.object(
                weekly.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run,
        ):
            self.assertEqual(weekly.main(), 0)
        command = run.call_args.args[0]
        self.assertNotIn("--repo", command[: command.index("--")])
        self.assertEqual(command[command.index("--repo") + 1], str(runtime))

    def test_decider_two_decimal_score_uses_only_its_declared_rounding_bound(self):
        case = {
            "request": {
                "questions": {"q": {"type": "score", "criteria": ["a", "b", "c", "d"]}}
            }
        }
        payload = {
            "answers": {
                "q": {
                    "score": 0.2,
                    "probabilities": {"0": 0.8114, "1": 0.179, "2": 0.006, "3": 0.0035},
                }
            }
        }
        bench.read_answers(case, payload, score_decimals=2)
        with self.assertRaises(ValueError):
            bench.read_answers(case, payload)
        payload["answers"]["q"]["score"] = 0.22
        with self.assertRaises(ValueError):
            bench.read_answers(case, payload, score_decimals=2)

    def test_observed_laya_score_rounding_is_valid_but_wrong_scores_fail(self):
        case = {
            "request": {
                "questions": {"q": {"type": "score", "criteria": ["a", "b", "c", "d"]}}
            }
        }
        for score, probabilities in (
            (0.9745, {"0": 0.448, "1": 0.1809, "2": 0.3198, "3": 0.0513}),
            (0.931, {"0": 0.1311, "1": 0.8271, "2": 0.0213, "3": 0.0204}),
            (2.003, {"0": 0.0337, "1": 0.0974, "2": 0.701, "3": 0.1678}),
        ):
            payload = {
                "answers": {"q": {"score": score, "probabilities": probabilities}}
            }
            bench.read_answers(case, payload)
            payload["answers"]["q"]["score"] += 0.001
            with self.assertRaises(ValueError):
                bench.read_answers(case, payload)

    def test_text_rendering_preserves_every_choice_record_and_label(self):
        root = weekly.ROOT / "benchmarks/e2e"
        original = bench.load_cases(root / "choice.jsonl")
        rendered = bench.load_cases(root / "choice-text.jsonl")
        self.assertEqual(len(original), 139)
        self.assertEqual([c["id"] for c in original], [c["id"] for c in rendered])
        for before, after in zip(original, rendered):
            self.assertEqual(before["expected"], after["expected"])
            self.assertEqual(
                before["request"]["questions"], after["request"]["questions"]
            )
            state = before["request"]["state"]
            if isinstance(state, str):
                self.assertEqual(state, after["request"]["state"])
            else:
                self.assertEqual(state, json.loads(after["request"]["state"]))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "cases.jsonl"
        self.manifest.write_text("".join(json.dumps(c) + "\n" for c in CASES))

    def make_run(self, directory, latency=10, wrong=False):
        for index in (1, 2):
            run = directory / f"measured-{index}"
            run.mkdir(parents=True)
            (run / "requests.jsonl").write_bytes(self.manifest.read_bytes())
            metadata = {k: "fixed" for k in weekly.CONTROL_KEYS}
            config = {
                "manifest_sha256": bench.digest(self.manifest),
                "metadata": metadata,
                "model": "test",
                "concurrency": 1,
                "warmup": 3,
                "timeout_seconds": 60,
                "python": "test",
                "httpx": "test",
                "protocol": "systemone",
            }
            bench.write_json(run / "config.json", config)
            records = []
            for case in CASES:
                payload = response_for(case)
                if wrong and case["id"] == "route":
                    payload["answers"]["route"] = {
                        "choice": "technical",
                        "probabilities": {"billing": 0.1, "technical": 0.9},
                    }
                records.append(
                    {
                        "id": case["id"],
                        "error": None,
                        "answers": bench.read_answers(case, payload),
                        "latency_ms": latency,
                    }
                )
            (run / "responses.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
            bench.write_json(run / "summary.json", bench.summarize(CASES, records, 1))

    def test_regression_checks_accuracy_latency_and_parity(self):
        old, good, slow, wrong = [
            self.root / name for name in ("old", "good", "slow", "wrong")
        ]
        self.make_run(old)
        self.make_run(good, 11)
        self.make_run(slow, 12)
        self.make_run(wrong, wrong=True)
        self.assertTrue(weekly.regression(old, good, GATES)["passed"])
        self.assertIn(
            "successful_latency_p95_ms regression",
            weekly.regression(old, slow, GATES)["errors"],
        )
        result = weekly.regression(old, wrong, GATES)
        self.assertIn("choice_accuracy regression", result["errors"])
        self.assertFalse(result["parity"][0]["passed"])

    def test_probability_drift_fails_even_without_accuracy_drop(self):
        old, new = self.root / "old", self.root / "new"
        self.make_run(old)
        self.make_run(new)
        path = new / "measured-2" / "responses.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["answers"]["route"]["probabilities"] = {
            "billing": 0.9,
            "technical": 0.1,
        }
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertFalse(weekly.regression(old, new, GATES)["passed"])

    def test_incompatible_controls_and_incomplete_pass_fail(self):
        old, new = self.root / "old", self.root / "new"
        self.make_run(old)
        self.make_run(new)
        path = new / "measured-2" / "config.json"
        config = json.loads(path.read_text())
        config["metadata"]["precision"] = "different"
        bench.write_json(path, config)
        with self.assertRaisesRegex(ValueError, "controls differ"):
            weekly.regression(old, new, GATES)
        path = old / "measured-2" / "summary.json"
        summary = json.loads(path.read_text())
        summary["failed_requests"] = 1
        bench.write_json(path, summary)
        with self.assertRaisesRegex(ValueError, "error-free"):
            weekly.aggregate([old / "measured-1", old / "measured-2"])

    def test_preflight_rejects_bad_gates_even_if_every_model_is_unavailable(self):
        suite = {
            "gates": GATES | {"max_latency_increase": float("nan")},
            "models": [{"id": "missing", "unavailable": "no weights"}],
        }
        path = self.root / "suite.json"
        bench.write_json(path, {**suite, "gates": GATES})
        self.assertEqual(len(weekly.load_suite(path)["models"]), 1)
        path.write_text(json.dumps(suite))
        with self.assertRaisesRegex(ValueError, "invalid gate"):
            weekly.load_suite(path)

    def test_jev_vl_adapter_validates_label_mapping_and_argmax(self):
        case = CASES[0]
        request = bench.wire_request(case, "autotrust/JEV-27B-VL", "jev-vl")
        self.assertEqual(
            request["options"],
            ["billing: Payments and refunds", "technical: Software faults"],
        )
        payload = {
            "kind": "choice",
            "effective_kind": "choice",
            "options": request["options"],
            "probabilities": [0.8, 0.2],
            "choice_index": 0,
            "choice": request["options"][0],
        }
        self.assertEqual(
            bench.wire_answers(case, payload, request["model"], "jev-vl")["route"][
                "value"
            ],
            "billing",
        )
        for change in (
            {"probabilities": [0.2, 0.8]},
            {"options": list(reversed(request["options"]))},
            {"choice_index": True},
            {"probabilities": [0.8]},
        ):
            with self.assertRaises(ValueError):
                bench.wire_answers(case, payload | change, request["model"], "jev-vl")

    def test_jev_vl_images_are_media_parts_and_structured_text_stays_text(self):
        for state, expected in (
            (
                {"image": "data:image/png;base64,test"},
                [{"image": "data:image/png;base64,test"}],
            ),
            ({"ticket": "refund"}, {"ticket": "refund"}),
        ):
            case = {**CASES[0], "request": {**CASES[0]["request"], "state": state}}
            self.assertEqual(
                bench.wire_request(case, "autotrust/JEV-27B-VL", "jev-vl")["state"],
                expected,
            )


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_passes_exclude_feasibility_and_cleanup_services(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            model = {
                "id": "test",
                "health": f"http://127.0.0.1:{port}/health",
                "endpoint": f"http://127.0.0.1:{port}/v1/systemone",
                "manifest": "benchmarks/smoke.jsonl",
                "model": "test",
                "commands": [
                    [
                        sys.executable,
                        str(Path(__file__).with_name("weekly_server.py")),
                        str(port),
                    ]
                ],
                "metadata": {k: "mock-only" for k in weekly.CONTROL_KEYS},
            }
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "mock-only"}):
                result = await weekly.run_model(
                    model,
                    {"startup_timeout": 10, "warmup": 1},
                    root / "run",
                    None,
                    {"sha": "test", "gpu_uuids": ["mock-only"], "harness_sha256": {}},
                )
            self.assertEqual(result["status"], "baseline_recorded", result)
            self.assertEqual(result["metrics"]["choice_accuracy"]["count_per_pass"], 2)
            for phase in ("feasibility", "measured-1", "measured-2"):
                run = root / "run" / phase
                self.assertEqual(
                    len((run / "responses.jsonl").read_text().splitlines()), 4
                )
                self.assertEqual(len(json.loads((run / "warmup.json").read_text())), 1)
            for pid in json.loads((root / "run" / "startup.json").read_text())[
                "process_ids"
            ]:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    async def test_startup_failure_preserves_logs_and_cleans_up_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            model = {
                "id": "test",
                "health": "http://127.0.0.1:1/health",
                "manifest": "benchmarks/smoke.jsonl",
                "commands": [
                    [
                        sys.executable,
                        "-c",
                        "print('startup failure'); raise SystemExit(3)",
                    ]
                ],
            }
            result = await weekly.run_model(
                model, {"startup_timeout": 2}, output, None, {"sha": "test"}
            )
            self.assertEqual(result["status"], "failed")
            self.assertIn("service exited", result["error"])
            self.assertIn("startup failure", (output / "service-0.log").read_text())


if __name__ == "__main__":
    unittest.main()
