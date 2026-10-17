"""Score wire contracts: model precision and Decider's isolated zero-fit output."""

import argparse
import asyncio
import copy
import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx

from benchmarks import bench


def case_for(count):
    return {
        "request": {
            "questions": {
                "q": {"type": "score", "criteria": [str(i) for i in range(count)]}
            }
        }
    }


def payload(score, probabilities):
    return {"answers": {"q": {"score": score, "probabilities": probabilities}}}


class ScoreWireTests(unittest.TestCase):
    def test_two_decimal_contract_is_explicit(self):
        response = payload(0.88, {"0": 0.1234, "1": 0.8766})
        with self.assertRaisesRegex(ValueError, "expected level"):
            bench.read_answers(case_for(2), response)
        self.assertEqual(
            bench.read_answers(case_for(2), response, score_decimals=2)["q"]["value"],
            0.88,
        )
        with self.assertRaisesRegex(ValueError, "expected level"):
            bench.read_answers(
                case_for(2), payload(0.9, {"0": 0.1234, "1": 0.8766}), score_decimals=2
            )

    def test_observed_laya_independent_rounding(self):
        probabilities = {"0": 0.0337, "1": 0.0974, "2": 0.701, "3": 0.1678}
        result = bench.read_answers(case_for(4), payload(2.003, probabilities))
        self.assertEqual(result["q"]["probabilities"], probabilities)
        with self.assertRaisesRegex(ValueError, "expected level"):
            bench.read_answers(case_for(4), payload(2.007, probabilities))

    def test_independently_rounded_distributions(self):
        rng = random.Random(93)
        for count in range(2, 11):
            for decimals in (2, 3, 4):
                for _ in range(32):
                    weights = [rng.random() for _ in range(count)]
                    probabilities = [w / sum(weights) for w in weights]
                    score = round(
                        sum(i * p for i, p in enumerate(probabilities)), decimals
                    )
                    wire = {
                        str(i): round(probabilities[i], 4)
                        for i in reversed(range(count))
                    }
                    self.assertEqual(
                        bench.read_answers(
                            case_for(count),
                            payload(score, wire),
                            score_decimals=decimals,
                        )["q"]["probabilities"],
                        wire,
                    )

    def test_exact_score_range_and_invalid_distribution(self):
        for score in (-0.00001, 1.00001, float("nan"), True):
            with self.subTest(score=score), self.assertRaises(ValueError):
                bench.read_answers(
                    case_for(2), payload(score, {"0": 0, "1": 1}), score_decimals=2
                )
        for probabilities in (
            {"0": 0, "1": 0},
            {"0": 0.2, "1": 0.2},
            {"0": True, "1": 0},
        ):
            with self.assertRaises(ValueError):
                bench.read_answers(case_for(2), payload(0, probabilities))

    def test_only_complete_zero_fit_metadata_is_accepted(self):
        response = payload(0, {"0": 0, "1": 0})
        answer = response["answers"]["q"]
        answer.update(
            type="score",
            level_fit={"0": 0, "1": 0},
            confidence=0,
            x_p_max=0,
            fit_mass=0,
            certainty=1,
        )
        bench.read_answers(case_for(2), response)
        for key in (
            "type",
            "level_fit",
            "confidence",
            "x_p_max",
            "fit_mass",
            "certainty",
        ):
            missing = copy.deepcopy(response)
            del missing["answers"]["q"][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.read_answers(case_for(2), missing)
        for key in ("score", "confidence", "x_p_max", "fit_mass", "certainty"):
            bad = copy.deepcopy(response)
            bad["answers"]["q"][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.read_answers(case_for(2), bad)
        bad = copy.deepcopy(response)
        bad["answers"]["q"]["level_fit"]["0"] = 10**1000
        with self.assertRaises(ValueError):
            bench.read_answers(case_for(2), bad)

    def test_transport_threads_model_precision(self):
        async def check():
            response = payload(0.88, {"0": 0.1234, "1": 0.8766})
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(200, json=response)
                )
            ) as client:
                records, _ = await bench.replay(
                    client,
                    "http://localhost/v1/systemone",
                    [case_for(2) | {"id": "score"}],
                    "decider",
                    1,
                    1,
                    score_decimals=2,
                )
                self.assertIsNone(records[0]["error"])

            bad = payload(0, {"0": 0, "1": 0})
            bad["answers"]["q"].update(
                type="score",
                level_fit={"0": 10**1000, "1": 0},
                confidence=0,
                x_p_max=0,
                fit_mass=0,
                certainty=1,
            )
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: httpx.Response(200, json=bad))
            ) as client:
                record = await bench.request_one(
                    client,
                    "http://localhost/v1/systemone",
                    case_for(2) | {"id": "score"},
                    "decider",
                    1,
                    score_decimals=2,
                )
                self.assertEqual(record["error_kind"], "invalid_response")

        asyncio.run(check())

    def test_run_saves_precision_and_uses_it_for_warmup(self):
        async def check():
            case = case_for(2) | {"id": "score", "expected": {"q": 0.88}}
            response = payload(0.88, {"0": 0.1234, "1": 0.8766})
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                metadata = {
                    key: "CPU fixture"
                    for key in (
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
                bench.write_json(root / "manifest.jsonl", case)
                args = argparse.Namespace(
                    metadata=root / "metadata.json",
                    output=root / "run",
                    manifest=root / "manifest.jsonl",
                    endpoint="http://localhost/v1/systemone",
                    model="decider",
                    phase="feasibility",
                    concurrency=1,
                    warmup=1,
                    timeout=1,
                    score_decimals=2,
                )
                client = httpx.AsyncClient(
                    transport=httpx.MockTransport(
                        lambda _: httpx.Response(200, json=response)
                    )
                )
                with (
                    patch.object(bench.httpx, "AsyncClient", return_value=client),
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(await bench.run(args, [case]), 0)
                import json

                config = json.loads((args.output / "config.json").read_text())
                self.assertEqual(config["score_decimals"], 2)
                self.assertEqual(
                    config["validation_protocol"], "probabilities-4dp-model-score-v1"
                )

        asyncio.run(check())
