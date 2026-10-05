import argparse
import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx

from benchmarks import bench

CASES = bench.load_cases(Path(bench.__file__).with_name("smoke.jsonl"))


def response_for(case):
    answers = {}
    for qid, question in case["request"]["questions"].items():
        kind = question["type"]
        target = case["expected"][qid]
        if kind == "noul":
            answers[qid] = {"noul": float(target)}
        else:
            labels = (
                list(question["criteria"])
                if kind == "choice"
                else [str(i) for i in range(len(question["criteria"]))]
            )
            answers[qid] = {
                kind: target,
                "probabilities": {k: float(k == str(target)) for k in labels},
            }
    return {"answers": answers}


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_order_and_question_counts(self):
        active, peak = 0, 0

        async def handler(request):
            nonlocal active, peak
            self.assertEqual(str(request.url), "http://localhost/v1/systemone")
            body = json.loads(request.content)
            self.assertEqual(body.pop("model"), "english")
            case = next(c for c in CASES if c["request"] == body)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01 if case["id"] == "route" else 0.001)
            active -= 1
            return httpx.Response(200, json=response_for(case))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            records, elapsed = await bench.replay(
                client, "http://localhost/v1/systemone", CASES, "english", 2, 1
            )
        self.assertEqual(peak, 2)
        self.assertEqual([r["id"] for r in records], [c["id"] for c in CASES])
        summary = bench.summarize(CASES, records, elapsed)
        self.assertEqual(summary["successful_requests"], 4)
        self.assertEqual(summary["successful_decisions"], 6)
        self.assertEqual(
            summary["quality_on_successful_requests"]["choice_accuracy"],
            {"value": 1, "count": 2},
        )

    async def test_failures_preserved_and_excluded_from_quality(self):
        responses = iter(
            [
                httpx.Response(503, text="busy"),
                httpx.Response(200, json={}),
                httpx.Response(200, text="not json"),
                httpx.Response(200, json=response_for(CASES[3])),
            ]
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: next(responses))
        ) as client:
            records, elapsed = await bench.replay(
                client, "http://localhost/v1/systemone", CASES, "english", 1, 1
            )
        summary = bench.summarize(CASES, records, elapsed)
        self.assertEqual(summary["errors"], {"http_503": 1, "invalid_response": 2})
        self.assertEqual(summary["successful_decisions"], 3)
        self.assertEqual(records[0]["response"], "busy")
        self.assertEqual(
            summary["quality_on_successful_requests"]["choice_accuracy"]["count"], 1
        )

    async def test_run_saves_warmup_and_rejects_overwrite(self):
        calls = []

        def handler(request):
            body = json.loads(request.content)
            body.pop("model")
            calls.append(body)
            case = next(c for c in CASES if c["request"] == body)
            return httpx.Response(200, json=response_for(case))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                k: "mock-only"
                for k in (
                    "gpu",
                    "gpu_ids",
                    "driver",
                    "cuda",
                    "precision",
                    "model_revision",
                    "runtime_revision",
                    "cache_policy",
                    "cuda_evidence",
                    "reservation",
                )
            }
            bench.write_json(root / "metadata.json", metadata)
            args = argparse.Namespace(
                metadata=root / "metadata.json",
                output=root / "run",
                manifest=Path(bench.__file__).with_name("smoke.jsonl"),
                endpoint="http://localhost/v1/systemone",
                model="english",
                phase="feasibility",
                concurrency=2,
                warmup=2,
                timeout=1,
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            with (
                patch.object(bench.httpx, "AsyncClient", return_value=client),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(await bench.run(args, CASES), 0)
            self.assertEqual(len(calls), 6)
            summary = json.loads((args.output / "summary.json").read_text())
            self.assertEqual(summary["requests"], 4)
            self.assertEqual(
                len(json.loads((args.output / "warmup.json").read_text())), 2
            )
            self.assertEqual(
                (args.output / "requests.jsonl").read_bytes(),
                args.manifest.read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                await bench.run(args, CASES)

    async def test_total_deadline(self):
        async def handler(request):
            await asyncio.sleep(1)
            return httpx.Response(200)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            record = await bench.request_one(
                client, "http://localhost/v1/systemone", CASES[0], "english", 0.01
            )
        self.assertEqual(record["error_kind"], "timeout")


class ValidationTests(unittest.TestCase):
    def test_all_primitives_and_invalid_distributions(self):
        for case in CASES:
            bench.read_answers(case, response_for(case))
        for probabilities in (
            {"billing": 0.2, "technical": 0.2},
            {"billing": float("nan"), "technical": 0},
            {"wrong": 1, "technical": 0},
        ):
            payload = response_for(CASES[0])
            payload["answers"]["route"]["probabilities"] = probabilities
            with self.assertRaises(ValueError):
                bench.read_answers(CASES[0], payload)
        payload = response_for(CASES[2])
        payload["answers"]["urgency"]["score"] = 0
        with self.assertRaises(ValueError):
            bench.read_answers(CASES[2], payload)

    def test_rounded_probabilities_at_the_sum_boundary(self):
        """PR #40's first GPU run: the worker rounds probabilities to four decimals, and
        four rounded values legitimately miss 1 by a full rounding step — a flat 1e-4
        tolerance rejected both vectors below at the floating-point boundary."""
        case = {
            "id": "mmlu-test-10312",
            "request": {
                "questions": {"q": {"type": "choice", "criteria": ["A", "B", "C", "D"]}}
            },
            "expected": {},
        }
        for probabilities in (
            {"A": 0.1099, "B": 0.1211, "C": 0.5589, "D": 0.2100},
            {"A": 0.1130, "B": 0.5135, "C": 0.3151, "D": 0.0583},
        ):
            answer = {
                "choice": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
            }
            bench.read_answers(case, {"answers": {"q": answer}})
        off = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.09}
        with self.assertRaises(ValueError):
            bench.read_answers(
                case, {"answers": {"q": {"choice": "A", "probabilities": off}}}
            )

    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text((json.dumps(CASES[0]) + "\n") * 2)
            with self.assertRaises(ValueError):
                bench.load_cases(path)

    def test_parity_and_missing_records(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, candidate = Path(directory) / "ref", Path(directory) / "candidate"
            records = [
                {
                    "id": c["id"],
                    "error": None,
                    "answers": bench.read_answers(c, response_for(c)),
                }
                for c in CASES
            ]
            for path in (ref, candidate):
                path.mkdir()
                (path / "requests.jsonl").write_text(
                    "".join(json.dumps(c) + "\n" for c in CASES)
                )
                bench.write_json(
                    path / "config.json",
                    {"manifest_sha256": bench.digest(path / "requests.jsonl")},
                )
                (path / "responses.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in records)
                )
            args = argparse.Namespace(
                reference=ref,
                candidate=candidate,
                max_probability_drift=0.002,
                max_score_drift=0.002,
                max_flips=0,
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(bench.compare(args), 0)
                records[0]["answers"]["route"]["probabilities"] = {
                    "billing": 0.5,
                    "technical": 0.5,
                }
                (candidate / "responses.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in records)
                )
                self.assertEqual(bench.compare(args), 1)
            for path in (ref, candidate):
                (path / "responses.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in records[:-1])
                )
            with self.assertRaises(ValueError):
                bench.compare(args)
