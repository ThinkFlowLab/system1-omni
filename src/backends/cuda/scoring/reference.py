#!/usr/bin/env python3
"""Reference implementation of fused candidate scoring.

A System 1 decision is a distribution over a question's candidates. Both models
that need this compute the same shape of thing:

- Kev (#27) scores ``scale * <k_proj(c_i), q_proj(s)>`` over that question's
  candidates, then softmaxes.
- CLM (#9) scores ``exp(logit_scale) * cos(state_head(s), action_head(c))`` and
  softmaxes.

They differ in the similarity (dot vs cosine) and in whether a projection happens
first. What they share is the fused primitive: turn a query vector and a matrix
of candidate vectors into probabilities, without materialising the intermediate
similarities.

This module is the oracle. It is deliberately the slow, obvious, float64 version
so that a kernel can be checked against it rather than against itself. ``vectors``
emits fixed inputs so a GPU test does not have to reproduce the RNG.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

# The reference accumulates in float64 and rounds once at the end, so a float32
# kernel's error is measured against the true value rather than against another
# float32 path.
ACCUMULATOR = np.float64


class ScoreSpec:
    """How one model's decision is computed from a query and its candidates.

    ``normalize`` selects cosine similarity (CLM) or a plain dot product (Kev
    before its scale). ``scale`` is applied to the similarity and ``temperature``
    divides the scaled logit, so the two knobs stay separable and each can be
    checked on its own.
    """

    def __init__(self, scale=1.0, temperature=1.0, normalize=False):
        if scale <= 0:
            raise ValueError("scale must be positive, got %r" % (scale,))
        if temperature <= 0:
            raise ValueError("temperature must be positive, got %r" % (temperature,))
        self.scale = float(scale)
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

    def as_dict(self):
        return {"scale": self.scale, "temperature": self.temperature,
                "normalize": self.normalize}


def l2_normalize(matrix, axis=-1, eps=1e-12):
    """Row-wise L2 normalisation, with a floor so a zero row stays finite.

    A zero candidate vector is degenerate rather than impossible (padding, a
    dropped option), and a kernel that divides by its norm without a floor
    produces NaN that then poisons the whole softmax. The reference defines the
    floor so the kernel has something to match.
    """
    values = np.asarray(matrix, dtype=ACCUMULATOR)
    norms = np.sqrt(np.sum(values * values, axis=axis, keepdims=True))
    return values / np.maximum(norms, eps)


def similarities(query, candidates, spec=None):
    """Scaled similarities, one per candidate. Shape: ``[K]``."""
    spec = spec or ScoreSpec()
    q = np.asarray(query, dtype=ACCUMULATOR)
    c = np.asarray(candidates, dtype=ACCUMULATOR)
    if c.ndim != 2:
        raise ValueError("candidates must be [K, D], got shape %r" % (c.shape,))
    if q.shape != (c.shape[1],):
        raise ValueError("query must have D=%d elements, got %r" % (c.shape[1], q.shape))
    if c.shape[0] == 0:
        raise ValueError("candidates must not be empty")
    if spec.normalize:
        q = l2_normalize(q)
        c = l2_normalize(c, axis=1)
    return (c @ q) * spec.scale


def logits(query, candidates, spec=None):
    """Similarities divided by temperature. Shape: ``[K]``."""
    spec = spec or ScoreSpec()
    return similarities(query, candidates, spec) / spec.temperature


def softmax(values):
    """Numerically stable softmax.

    The max subtraction is not cosmetic: a kernel that exponentiates raw logits
    overflows to inf for inputs the reference handles, so the reference fixes the
    convention the kernel has to follow.
    """
    values = np.asarray(values, dtype=ACCUMULATOR)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def probabilities(query, candidates, spec=None):
    """The decision: a distribution over the candidates. Shape: ``[K]``.

    The distribution is relative to the candidate set in the request, which is
    what both #9 and #27 promise, so probabilities for different questions are
    not comparable with each other.
    """
    return softmax(logits(query, candidates, spec))


# ---- fixed inputs, so a GPU test does not have to reproduce an RNG ----

def _rng(seed):
    return np.random.default_rng(seed)


def vectors(cases=8, seed=20260928):
    """Deterministic (query, candidates, spec) cases covering the shape range.

    Includes the cases that break naive kernels: a single candidate, an empty
    difference (identical candidates), a zero candidate, a large-K question, and
    magnitudes that would overflow a softmax without max subtraction.
    """
    rng = _rng(seed)
    out = []

    for index in range(cases):
        # One long-vector case covers the D=2560 path. It keeps K small on
        # purpose: D x K is the size of a committed vector, so covering an axis
        # costs least when the other axis is narrow.
        k, d = int(rng.integers(1, 33)), int(rng.choice([64, 256]))
        if index == 0:
            k, d = 2, 2560
        query = rng.standard_normal(d)
        candidates = rng.standard_normal((k, d))
        spec = ScoreSpec(scale=float(rng.choice([1.0, 0.0625, 2.0])),
                         temperature=float(rng.choice([1.0, 2.406050072164233])),
                         normalize=bool(rng.integers(0, 2)))
        out.append({"name": "random-%d" % index, "query": query, "candidates": candidates,
                    "spec": spec})

    d = 256
    q = _rng(1).standard_normal(d)
    out.append({"name": "single-candidate", "query": q,
                "candidates": np.array([_rng(2).standard_normal(d)]),
                "spec": ScoreSpec(scale=0.0625, temperature=1.0)})

    base = _rng(3).standard_normal(d)
    out.append({"name": "identical-candidates", "query": _rng(4).standard_normal(d),
                "candidates": np.vstack([base, base, base]),
                "spec": ScoreSpec(scale=0.0625, temperature=1.0)})

    out.append({"name": "zero-candidate", "query": _rng(5).standard_normal(d),
                "candidates": np.vstack([np.zeros(d), _rng(6).standard_normal(d)]),
                "spec": ScoreSpec(scale=1.0, temperature=1.0, normalize=True)})

    # K is the axis this case covers (the in-block softmax over 255 values), so
    # D stays small: 255 x 256 would be 1.3 MB of committed vectors for a
    # dimensionality already covered by the long-vector case.
    out.append({"name": "max-k", "query": _rng(7).standard_normal(32),
                "candidates": _rng(8).standard_normal((255, 32)),
                "spec": ScoreSpec(scale=0.0625, temperature=2.406050072164233)})

    # A large scale on correlated vectors makes raw logits large enough that
    # exponentiating them without subtracting the max overflows.
    big = _rng(9).standard_normal(d) * 40.0
    out.append({"name": "overflow-prone", "query": big * 40.0,
                "candidates": np.vstack([big, big * 0.9, big * 0.5]),
                "spec": ScoreSpec(scale=1.0, temperature=1.0)})

    return out


def report(cases):
    """Run the cases and return a JSON-serialisable report."""
    results = []
    for case in cases:
        probs = probabilities(case["query"], case["candidates"], case["spec"])
        results.append({
            "name": case["name"],
            "k": int(case["candidates"].shape[0]),
            "d": int(case["candidates"].shape[1]),
            "spec": case["spec"].as_dict(),
            "logits": [float(value) for value in logits(case["query"], case["candidates"],
                                                        case["spec"])],
            "probabilities": [float(value) for value in probs],
            "probability_sum": float(np.sum(probs)),
        })
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the fixed vectors and expected outputs here")
    parser.add_argument("--cases", type=int, default=8)
    args = parser.parse_args(argv)

    cases = vectors(cases=args.cases)
    results = report(cases)

    # Inputs are serialised as float32, because float32 is what the kernel
    # receives: storing float64 would test a rounding the hardware never sees,
    # and roughly doubles the file.
    payload = {
        "generator": "src/backends/cuda/scoring/reference.py",
        "accumulator": "float64",
        "input_precision": "float32",
        "cases": [dict(result,
                       query=[float(value) for value in
                              np.asarray(case["query"], dtype=np.float32)],
                       candidates=[[float(value) for value in row] for row in
                                   np.asarray(case["candidates"], dtype=np.float32)])
                  for result, case in zip(results, cases)],
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
        print("wrote %s (%d cases)" % (args.json, len(results)))
    else:
        for result in results:
            print("%-20s K=%-4d D=%-5d sum=%.10f max=%.6f"
                  % (result["name"], result["k"], result["d"],
                     result["probability_sum"], max(result["probabilities"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
