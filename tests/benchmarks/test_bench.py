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
            answers[qid] = {"type": kind, "noul": float(target)}
        else:
            labels = (
                list(question["criteria"])
                if kind == "choice"
                else [str(i) for i in range(len(question["criteria"]))]
            )
            answers[qid] = {
                "type": kind,
                kind: target,
                "probabilities": {k: float(k == str(target)) for k in labels},
            }
    return {"answers": answers}


def saved_inputs(path, cases=CASES):
    path.mkdir()
    (path / "requests.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    bench.write_json(
        path / "config.json",
        {
            "manifest_sha256": bench.digest(path / "requests.jsonl"),
            "warmup_manifest_sha256": None,
            "latency_basis": "post_to_body_json_and_schema_validation",
        },
    )


def run_args(root):
    metadata = {
        key: "mock-only"
        for key in (
            "gpu", "gpu_ids", "driver", "cuda", "precision", "model_revision",
            "runtime_revision", "cache_policy", "cuda_evidence", "reservation",
        )
    }
    bench.write_json(root / "metadata.json", metadata)
    return argparse.Namespace(
        metadata=root / "metadata.json", output=root / "run",
        manifest=Path(bench.__file__).with_name("smoke.jsonl"),
        endpoint="http://localhost/v1/systemone", model="english",
        phase="measured", concurrency=1, warmup=1, timeout=1,
        warmup_manifest=None,
    )


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_run_namespace_without_optional_warmup_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            args = run_args(Path(directory))
            del args.warmup_manifest

            def handler(request):
                body = json.loads(request.content)
                body.pop("model")
                case = next(c for c in CASES if c["request"] == body)
                return httpx.Response(200, json=response_for(case))

            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            with (
                patch.object(bench.httpx, "AsyncClient", return_value=client),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(await bench.run(args, CASES), 0)

    async def test_warmup_failures_preserve_all_measured_unattempted(self):
        for kind in ("missing-type", "wrong-type", "nonobject", "http", "timeout", "cancel"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                args = run_args(Path(directory))
                raw = response_for(CASES[0])
                if kind == "missing-type":
                    raw["answers"]["route"].pop("type")
                elif kind == "wrong-type":
                    raw["answers"]["route"]["type"] = "noul"
                elif kind == "nonobject":
                    raw["answers"]["route"] = []

                def handler(request):
                    if kind == "timeout":
                        raise httpx.ReadTimeout("controlled warmup timeout", request=request)
                    if kind == "cancel":
                        raise asyncio.CancelledError
                    return httpx.Response(503, text="busy") if kind == "http" else httpx.Response(200, json=raw)

                client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
                with patch.object(bench.httpx, "AsyncClient", return_value=client):
                    with self.assertRaisesRegex(ValueError, "readiness/warmup failed"):
                        await bench.run(args, CASES)
                self.assertFalse((args.output / "responses.jsonl").exists())
                saved = json.loads((args.output / "summary.json").read_text())
                self.assertEqual(saved, bench.summarize_saved(args.output))
                self.assertEqual(saved["planned_requests"], 4)
                self.assertEqual(saved["unattempted_requests"], 4)
                self.assertEqual(saved["planned_decisions"], 6)
                self.assertEqual(saved["unattempted_decisions"], 6)
                for field in ("attempted_requests", "incomplete_requests", "successful_requests",
                              "failed_requests", "attempted_decisions", "incomplete_decisions"):
                    self.assertEqual(saved[field], 0)
                for field in ("wall_seconds", "requests_per_second", "decisions_per_second"):
                    self.assertIsNone(saved[field])
                state = json.loads((args.output / "completion.json").read_text())
                self.assertIsNone(state["started_at"])
                self.assertIsNone(state["finished_at"])
                self.assertEqual(state["attempted_ids"], [])
                self.assertEqual(state["stop_reason"], "warmup_interrupted" if kind == "cancel" else "warmup_failed")
                self.assertFalse(saved["evidence_complete"])

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
                warmup_manifest=None,
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
            self.assertEqual(bench.summarize_saved(args.output), summary)
            self.assertEqual(
                len(json.loads((args.output / "warmup.json").read_text())), 2
            )
            self.assertEqual(
                (args.output / "requests.jsonl").read_bytes(),
                args.manifest.read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                await bench.run(args, CASES)

    async def test_independent_warmup_does_not_enter_measurement(self):
        warmup_case = json.loads(json.dumps(CASES[0]))
        warmup_case["id"] = "warmup-only"
        warmup_case["request"]["state"] = "independent warmup"
        calls = []

        def handler(request):
            body = json.loads(request.content)
            body.pop("model")
            calls.append(body)
            case = (
                warmup_case if body == warmup_case["request"] else next(
                    c for c in CASES if c["request"] == body
                )
            )
            return httpx.Response(200, json=response_for(case))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                key: "mock-only"
                for key in (
                    "gpu", "gpu_ids", "driver", "cuda", "precision",
                    "model_revision", "runtime_revision", "cache_policy",
                    "cuda_evidence", "reservation",
                )
            }
            bench.write_json(root / "metadata.json", metadata)
            warmup_manifest = root / "warmup.jsonl"
            warmup_manifest.write_text(json.dumps(warmup_case) + "\n")
            args = argparse.Namespace(
                metadata=root / "metadata.json",
                output=root / "run",
                manifest=Path(bench.__file__).with_name("smoke.jsonl"),
                endpoint="http://localhost/v1/systemone",
                model="english",
                phase="measured",
                concurrency=1,
                warmup=2,
                timeout=1,
                warmup_manifest=warmup_manifest,
            )
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            with (
                patch.object(bench.httpx, "AsyncClient", return_value=client),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(await bench.run(args, CASES), 0)
            self.assertEqual(calls[:2], [warmup_case["request"]] * 2)
            records = [
                json.loads(line)
                for line in (args.output / "responses.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["id"] for r in records], [c["id"] for c in CASES])
            self.assertEqual(
                (args.output / "warmup-requests.jsonl").read_bytes(),
                warmup_manifest.read_bytes(),
            )
            self.assertEqual(bench.summarize_saved(args.output)["planned_requests"], 4)
            (args.output / "warmup-requests.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "warmup manifest checksum"):
                bench.summarize_saved(args.output)

    async def test_body_and_validation_are_inside_latency_clock(self):
        clock = [10.0]
        body = json.dumps(response_for(CASES[0])).encode()

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                clock[0] += 1
                yield body

        def validate(case, payload):
            clock[0] += 2
            return original(case, payload)

        original = bench.read_answers
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=Body())
            )
        ) as client:
            with (
                patch.object(bench.time, "perf_counter", side_effect=lambda: clock[0]),
                patch.object(bench, "read_answers", side_effect=validate),
            ):
                record = await bench.request_one(
                    client, "http://localhost/v1/systemone", CASES[0], "english", 1
                )
        self.assertEqual(record["latency_ms"], 3000)
        self.assertIsNone(record["error"])

    async def test_cancel_after_body_retains_raw_without_completed_latency(self):
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=response_for(CASES[0]))
            )
        ) as client:
            with patch.object(
                bench, "read_answers", side_effect=asyncio.CancelledError
            ):
                record = await bench.request_one(
                    client, "http://localhost/v1/systemone", CASES[0], "english", 1
                )
        self.assertEqual(record["state"], "incomplete")
        self.assertEqual(json.loads(record["response"]), response_for(CASES[0]))
        self.assertIsNone(record["latency_ms"])
        self.assertEqual(record["answers"], {})

    async def test_total_deadline(self):
        async def handler(request):
            await asyncio.sleep(1)
            return httpx.Response(200)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            record = await bench.request_one(
                client, "http://localhost/v1/systemone", CASES[0], "english", 0.01
            )
        self.assertEqual(record["error_kind"], "timeout")

    async def test_transport_failure_with_empty_error_is_excluded_from_quality(self):
        def handler(request):
            raise httpx.ConnectError("", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            record = await bench.request_one(
                client, "http://localhost/v1/systemone", CASES[0], "english", 1
            )
        self.assertEqual(record["error_kind"], "transport")
        summary = bench.summarize(CASES[:1], [record], 1)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(
            summary["quality_on_successful_requests"]["choice_accuracy"]["count"], 0
        )


class PersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_bom_and_invalid_utf8_online_offline_text_decoding_agree(self):
        valid = json.dumps(response_for(CASES[0])).encode("utf-8")
        for label, body, error_kind in (
            ("bom", b"\xef\xbb\xbf" + valid, "invalid_response"),
            ("invalid-utf8", valid[:-1] + b', "unused": "\xff"}', None),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "run"
                saved_inputs(path, CASES[:1])
                expected_text = httpx.Response(200, content=body).text
                async with httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, content=body)
                )) as client:
                    records, elapsed = await bench.replay(
                        client, "http://localhost/v1/systemone", CASES[:1], "english", 1, 1, path
                    )
                self.assertEqual(records[0]["response"], expected_text)
                self.assertEqual(records[0]["error_kind"], error_kind)
                self.assertTrue(expected_text.startswith("\ufeff") if label == "bom" else "\ufffd" in expected_text)
                expected = bench.summarize(CASES[:1], records, elapsed)
                expected["stop_reason"] = "completed"
                self.assertEqual(bench.summarize_saved(path), expected)

    async def collect(self, path):
        saved_inputs(path)

        def handler(request):
            body = json.loads(request.content)
            body.pop("model")
            case = next(c for c in CASES if c["request"] == body)
            return httpx.Response(200, json=response_for(case))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await bench.replay(
                client, "http://localhost/v1/systemone", CASES, "english", 2, 1, path
            )

    def replace_records(self, path, records):
        (path / "responses.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records)
        )
        state = json.loads((path / "completion.json").read_text())
        state["responses_sha256"] = bench.digest(path / "responses.jsonl")
        bench.write_json(path / "completion.json", state)

    async def test_saved_http_timeout_and_typed_validation_failures(self):
        calls = 0
        missing_type = response_for(CASES[1])
        missing_type["answers"]["refund"].pop("type")

        def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(503, text="busy")
            if calls == 2:
                return httpx.Response(200, json=missing_type)
            if calls == 3:
                raise httpx.ReadTimeout("controlled timeout", request=request)
            return httpx.Response(200, json=response_for(CASES[3]))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            saved_inputs(path)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                records, elapsed = await bench.replay(
                    client, "http://localhost/v1/systemone", CASES, "english", 1, 1, path
                )
            summary = bench.summarize_saved(path)
            self.assertEqual(
                summary["errors"],
                {"http_503": 1, "invalid_response": 1, "timeout": 1},
            )
            self.assertEqual(summary["completed_requests"], 4)
            self.assertEqual(summary["failed_requests"], 3)
            self.assertEqual(summary["failed_decisions"], 3)
            self.assertIsNotNone(summary["failed_latency_p50_ms"])
            self.assertEqual(json.loads(records[1]["response"]), missing_type)
            self.assertEqual(records[1]["answers"], {})
            expected = bench.summarize(CASES, records, elapsed)
            expected["stop_reason"] = "completed"
            self.assertEqual(summary, expected)

    async def test_out_of_order_records_recompute_by_frozen_id(self):
        release = asyncio.Event()
        later_completed = asyncio.Event()

        async def handler(request):
            body = json.loads(request.content)
            body.pop("model")
            case = next(c for c in CASES if c["request"] == body)
            if case["id"] == CASES[0]["id"]:
                await release.wait()
            else:
                later_completed.set()
            return httpx.Response(200, json=response_for(case))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            saved_inputs(path)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                task = asyncio.create_task(bench.replay(
                    client, "http://localhost/v1/systemone", CASES, "english", 2, 1, path
                ))
                await later_completed.wait()
                partial = [json.loads(line) for line in (path / "responses.jsonl").read_text().splitlines()]
                self.assertTrue(partial)
                self.assertNotIn(CASES[0]["id"], [r["id"] for r in partial])
                partial_summary = bench.summarize_saved(path)
                self.assertEqual(partial_summary["successful_requests"], len(partial))
                self.assertEqual(partial_summary["incomplete_requests"], 1)
                self.assertIsNone(partial_summary["wall_seconds"])
                release.set()
                records, elapsed = await task
            saved = [
                json.loads(line)
                for line in (path / "responses.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["id"] for r in saved], [c["id"] for c in CASES])
            self.assertEqual([r["id"] for r in records], [c["id"] for c in CASES])
            expected = bench.summarize(CASES, records, elapsed)
            expected["stop_reason"] = "completed"
            self.assertEqual(bench.summarize_saved(path), expected)
            args = argparse.Namespace(
                reference=path,
                candidate=path,
                max_probability_drift=0,
                max_score_drift=0,
                max_flips=0,
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(bench.compare(args), 0)

    async def test_cancel_saves_completed_incomplete_and_unattempted(self):
        slow_started = asyncio.Event()
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(200, json=response_for(CASES[0]))
            slow_started.set()
            await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            saved_inputs(path)
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                task = asyncio.create_task(bench.replay(
                    client, "http://localhost/v1/systemone", CASES, "english", 1, 60, path
                ))
                await slow_started.wait()
                before = [
                    json.loads(line)
                    for line in (path / "responses.jsonl").read_text().splitlines()
                ]
                self.assertEqual([r["id"] for r in before], [CASES[0]["id"]])
                state = json.loads((path / "completion.json").read_text())
                self.assertEqual(state["attempted_ids"], [c["id"] for c in CASES[:2]])
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            summary = bench.summarize_saved(path)
            for name, count in (
                ("planned_requests", 4), ("attempted_requests", 2),
                ("successful_requests", 1), ("failed_requests", 0),
                ("incomplete_requests", 1), ("unattempted_requests", 2),
            ):
                self.assertEqual(summary[name], count)
            self.assertEqual(summary["planned_decisions"], 6)
            self.assertEqual(summary["attempted_decisions"], 2)
            self.assertEqual(summary["unattempted_decisions"], 4)
            self.assertEqual(summary["incomplete_decisions"], 1)
            self.assertEqual(summary["stop_reason"], "interrupted")
            self.assertFalse(summary["evidence_complete"])
            self.assertIsNotNone(summary["wall_seconds"])
            records = [
                json.loads(line)
                for line in (path / "responses.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[1]["error_kind"], "interrupted")
            self.assertIsNone(records[1]["latency_ms"])

    async def test_unfinished_measurement_never_invents_wall_or_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            await self.collect(path)
            records = [
                json.loads(line)
                for line in (path / "responses.jsonl").read_text().splitlines()
            ]
            self.replace_records(path, records[:-1])
            state = json.loads((path / "completion.json").read_text())
            state.update(
                finished_at=None, wall_seconds=None, stop_reason=None,
                completed_ids=[r["id"] for r in records[:-1]],
                active_ids=[records[-1]["id"]],
            )
            bench.write_json(path / "completion.json", state)
            summary = bench.summarize_saved(path)
            self.assertEqual(summary["incomplete_requests"], 1)
            self.assertIsNone(summary["wall_seconds"])
            self.assertIsNone(summary["requests_per_second"])
            self.assertIsNone(summary["decisions_per_second"])
            self.assertFalse(summary["evidence_complete"])

    async def test_id_coverage_duplicate_and_unknown_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            await self.collect(path)
            original = [
                json.loads(line)
                for line in (path / "responses.jsonl").read_text().splitlines()
            ]
            for records in (
                original[:-1], original + [original[0]],
                [{**original[0], "id": "unknown"}] + original[1:],
            ):
                self.replace_records(path, records)
                with self.assertRaises(ValueError):
                    bench.summarize_saved(path)

    async def test_checksums_and_raw_normalized_consistency_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            await self.collect(path)
            files = {
                name: (path / name).read_bytes()
                for name in (
                    "requests.jsonl", "config.json", "responses.jsonl", "completion.json"
                )
            }
            for name in ("requests.jsonl", "config.json", "responses.jsonl"):
                (path / name).write_bytes(files[name] + b"\n")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    bench.summarize_saved(path)
                (path / name).write_bytes(files[name])
            for change in ("raw", "normalized", "validation"):
                records = [
                    json.loads(line)
                    for line in files["responses.jsonl"].decode().splitlines()
                ]
                if change == "raw":
                    records[0]["response"] += " "
                elif change == "normalized":
                    qid = next(iter(records[0]["answers"]))
                    records[0]["answers"][qid]["value"] = "changed"
                else:
                    records[0]["error_kind"] = "invalid_response"
                    records[0]["error"] = "changed"
                self.replace_records(path, records)
                with self.assertRaises(ValueError):
                    bench.summarize_saved(path)
                (path / "completion.json").write_bytes(files["completion.json"])


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
                "type": "choice",
                "choice": max(probabilities, key=probabilities.get),
                "probabilities": probabilities,
            }
            bench.read_answers(case, {"answers": {"q": answer}})
        off = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.09}
        with self.assertRaises(ValueError):
            bench.read_answers(
                case,
                {"answers": {"q": {
                    "type": "choice", "choice": "A", "probabilities": off
                }}},
            )

    def test_missing_wrong_and_nonobject_answer_types(self):
        for case in CASES:
            for qid in case["request"]["questions"]:
                for mutation in (None, "wrong", []):
                    payload = response_for(case)
                    if mutation is None:
                        payload["answers"][qid].pop("type")
                    elif isinstance(mutation, str):
                        payload["answers"][qid]["type"] = mutation
                    else:
                        payload["answers"][qid] = mutation
                    with self.assertRaises(ValueError):
                        bench.read_answers(case, payload)

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
