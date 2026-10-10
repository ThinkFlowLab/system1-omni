#!/usr/bin/env python3
"""Tests for the scoring reference and for the kernel's algorithm.

Two things are checked here. First, that the reference behaves the way a decision
readout has to: a distribution, 1/K when the candidates are indistinguishable, no
NaN for a degenerate input, no overflow for a large one. Second, that the
algorithm the kernel implements agrees with that reference, which is what fixes
the tolerance the real kernel has to meet.

What is **not** checked: whether candidate_scoring.cu compiles or runs. There is
no CUDA toolkit here; see README.md.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kernel_simulation  # noqa: E402
import reference  # noqa: E402


def case(name):
    for item in reference.vectors():
        if item["name"] == name:
            return item
    raise KeyError(name)


class ReferenceTest(unittest.TestCase):
    """The oracle has to be right before anything is compared to it."""

    def test_probabilities_are_a_distribution(self):
        for item in reference.vectors():
            probs = reference.probabilities(item["query"], item["candidates"], item["spec"])
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=10, msg=item["name"])
            self.assertTrue(np.all(probs >= 0.0), item["name"])

    def test_a_single_candidate_gets_probability_one(self):
        item = case("single-candidate")
        probs = reference.probabilities(item["query"], item["candidates"], item["spec"])
        self.assertAlmostEqual(float(probs[0]), 1.0, places=12)

    def test_identical_candidates_split_evenly(self):
        item = case("identical-candidates")
        probs = reference.probabilities(item["query"], item["candidates"], item["spec"])
        for value in probs:
            self.assertAlmostEqual(float(value), 1.0 / 3.0, places=12)

    def test_output_is_relative_to_the_candidate_set(self):
        # A question's distribution says nothing about another question's, which
        # is what #9 and #27 both promise.
        rng = np.random.default_rng(11)
        query = rng.standard_normal(64)
        spec = reference.ScoreSpec()
        few = reference.probabilities(query, rng.standard_normal((2, 64)), spec)
        many = reference.probabilities(query, rng.standard_normal((9, 64)), spec)
        self.assertEqual(len(few), 2)
        self.assertEqual(len(many), 9)
        self.assertNotAlmostEqual(float(few.max()), float(many.max()), places=6)

    def test_softmax_survives_logits_that_overflow_without_the_max(self):
        # The overflow-prone case is only meaningful if it really would overflow.
        item = case("overflow-prone")
        raw = reference.logits(item["query"], item["candidates"], item["spec"])
        with np.errstate(over="ignore", invalid="ignore"):
            naive = np.exp(raw)
            self.assertFalse(np.all(np.isfinite(naive / naive.sum())),
                             "the case must break a naive softmax to be worth having")
        probs = reference.probabilities(item["query"], item["candidates"], item["spec"])
        self.assertTrue(np.all(np.isfinite(probs)))

    def test_a_zero_candidate_does_not_produce_nan(self):
        item = case("zero-candidate")
        norms = np.sqrt((item["candidates"] ** 2).sum(axis=1))
        self.assertEqual(float(norms.min()), 0.0, "the case needs a zero row")
        probs = reference.probabilities(item["query"], item["candidates"], item["spec"])
        self.assertTrue(np.all(np.isfinite(probs)))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=10)

    def test_the_largest_allowed_candidate_set(self):
        item = case("max-k")
        self.assertEqual(item["candidates"].shape[0], reference.MAX_K
                         if hasattr(reference, "MAX_K") else 255)
        probs = reference.probabilities(item["query"], item["candidates"], item["spec"])
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=10)

    def test_scale_and_temperature_are_separable(self):
        # logit = scale * sim / temperature, so doubling the temperature must
        # equal halving the scale.
        rng = np.random.default_rng(12)
        query, candidates = rng.standard_normal(64), rng.standard_normal((3, 64))
        a = reference.logits(query, candidates, reference.ScoreSpec(scale=1.0, temperature=2.0))
        b = reference.logits(query, candidates, reference.ScoreSpec(scale=0.5, temperature=1.0))
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_invalid_specs_are_rejected(self):
        for kwargs in ({"scale": 0.0}, {"scale": -1.0}, {"temperature": 0.0}):
            with self.assertRaises(ValueError):
                reference.ScoreSpec(**kwargs)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            reference.probabilities(np.zeros(8), np.zeros((3, 4)))
        with self.assertRaises(ValueError):
            reference.probabilities(np.zeros(4), np.zeros((0, 4)))


class SimulationTest(unittest.TestCase):
    """The algorithm the kernel implements must agree with the reference."""

    TOLERANCE = 1e-4

    def test_every_fixed_case_agrees(self):
        results = kernel_simulation.parity(reference.vectors(), tolerance=self.TOLERANCE)
        failures = [r for r in results if not r["within_tolerance"]]
        self.assertEqual(failures, [], "cases over tolerance: %r" % failures)

    def test_the_worst_case_is_comfortably_inside_the_declared_tolerance(self):
        # A tolerance that the reference algorithm only just meets would be a
        # tolerance the real kernel fails on hardware.
        results = kernel_simulation.parity(reference.vectors())
        worst = max(result["max_abs"] for result in results)
        self.assertLess(worst, self.TOLERANCE / 100.0,
                        "worst absolute difference %.3e leaves too little headroom" % worst)

    def test_the_simulation_also_produces_distributions(self):
        for item in reference.vectors():
            probs, _logits = kernel_simulation.score(
                item["query"], item["candidates"], item["spec"].scale,
                item["spec"].temperature, item["spec"].normalize)
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=5, msg=item["name"])

    def test_identical_candidates_stay_exactly_even(self):
        # Same input through the same path must give bit-identical probabilities,
        # not merely close ones.
        item = case("identical-candidates")
        probs, _ = kernel_simulation.score(item["query"], item["candidates"],
                                           item["spec"].scale, item["spec"].temperature,
                                           item["spec"].normalize)
        self.assertEqual(float(probs[0]), float(probs[1]))
        self.assertEqual(float(probs[1]), float(probs[2]))

    def test_the_reduction_is_a_tree_not_a_sequential_sum(self):
        # The kernel reduces with warp shuffles, so the simulation has to as
        # well, or a comparison against the reference measures summation order
        # rather than the kernel. The two orders are checked to be genuinely
        # different on this input, not merely assumed to be.
        values = [-97851907.805664, -80883723.94256, 106089862.338608, -80753467.53319,
                  -3252170.494552, 88438986.738317, -58360043.27433, -11170194.958416,
                  11046414.324948, 6378177.425506, -122505582.641769, 7614023.037701,
                  135882342.174154, -154714467.812848, 85938268.80216, 11935402.569658,
                  -64147039.410722, 200041654.634242, 76225971.208471, -119928890.210522,
                  7451622.877146, 57668958.367019, -18878212.535075, 68291026.719521,
                  -6651732.014942, 66724756.083433, 143852259.165615, -67566225.100565,
                  20313861.038961, -46330757.653842, 12726841.122583, -118719452.785014]
        sequential = np.float32(0.0)
        for value in values:
            sequential = np.float32(sequential + np.float32(value))
        tree = kernel_simulation._reduce(values)
        self.assertNotEqual(float(tree), float(sequential),
                            "this input must distinguish the two orders")

    def test_the_reduction_matches_an_explicit_tree(self):
        # The structural property itself: halve the stride and add, which is what
        # __shfl_down_sync(WARP/2), (WARP/4), ... does.
        values = [1.0 + index * 0.5 for index in range(kernel_simulation.WARP - 6)]
        lanes = [np.float32(v) for v in values] + [np.float32(0.0)] * 6
        offset = kernel_simulation.WARP // 2
        while offset > 0:
            for index in range(offset):
                lanes[index] = np.float32(lanes[index] + lanes[index + offset])
            offset //= 2
        self.assertEqual(float(kernel_simulation._reduce(values)), float(lanes[0]))

    def test_an_out_of_range_candidate_count_is_rejected(self):
        rng = np.random.default_rng(13)
        with self.assertRaises(ValueError):
            kernel_simulation.score(rng.standard_normal(8), rng.standard_normal((0, 8)))
        with self.assertRaises(ValueError):
            kernel_simulation.score(rng.standard_normal(8),
                                    rng.standard_normal((kernel_simulation.MAX_K + 1, 8)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
