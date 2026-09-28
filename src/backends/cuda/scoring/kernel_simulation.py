#!/usr/bin/env python3
"""The kernel's algorithm in numpy, so its logic can be checked without a GPU.

This is not a simulation of CUDA. It reproduces what ``candidate_scoring.cu``
computes — float32 accumulation, the max-subtracted softmax, the norm floored as
``max(sqrt(sum), eps)`` and not as ``sqrt(max(sum, eps))`` — so that a parity
failure on real hardware is a bug in the kernel rather than in the arithmetic it
was designed to perform.

It exists because the authoring machine has no CUDA toolkit. It cannot tell you
the kernel compiles, runs, or is fast. It can tell you the algorithm agrees with
the reference, and it produces the tolerance the real kernel has to meet.

One deliberate difference from the kernel: the kernel uses ``fmaf``, which rounds
once, while numpy has no fused multiply-add here and rounds twice. This
simulation's error is therefore an upper bound on the kernel's, which is the
useful direction for a tolerance.
"""

from __future__ import annotations

import numpy as np

WARP = 32
MAX_K = 255
NORM_EPS = np.float32(1e-12)

# float32, because the kernel accumulates in float32.
F = np.float32


def _reduce(values):
    """Sum float32 values the way the kernel's warp shuffle does.

    ``__shfl_down_sync`` sums in a tree: step WARP/2, then WARP/4, ... Each step
    adds a partial to a partial, so the order differs from a sequential sum and
    the rounding differs with it. Reproducing the tree is the point — a kernel
    checked against a sequential sum would show error that is really just a
    different summation order.
    """
    lanes = [F(v) for v in values] + [F(0.0)] * (WARP - len(values))
    offset = WARP // 2
    while offset > 0:
        for i in range(offset):
            lanes[i] = F(lanes[i] + lanes[i + offset])
        offset //= 2
    return lanes[0]


def _row_moments(row, query, normalize):
    """The dot and the squared norm, accumulated per lane then reduced.

    The kernel walks ``d = lane; d < D; d += WARP`` and keeps two accumulators
    fed from the same load, which is the fusion: one read of the row yields both.
    """
    dot_partials = []
    norm_partials = []
    for lane in range(WARP):
        dot = F(0.0)
        norm2 = F(0.0)
        for d in range(lane, len(row), WARP):
            c = F(row[d])
            dot = F(dot + F(c * F(query[d])))
            if normalize:
                norm2 = F(norm2 + F(c * c))
        dot_partials.append(dot)
        norm_partials.append(norm2)
    return _reduce(dot_partials), _reduce(norm_partials)


def _query_norm(query, normalize):
    if not normalize:
        return F(1.0)
    partials = []
    for lane in range(WARP):
        partial = F(0.0)
        for d in range(lane, len(query), WARP):
            q = F(query[d])
            partial = F(partial + F(q * q))
        partials.append(partial)
    return F(max(np.sqrt(_reduce(partials)), NORM_EPS))


def score(query, candidates, scale=1.0, temperature=1.0, normalize=False):
    """Probabilities, computed the way the kernel computes them.

    Returns ``(probabilities, logits)``, both float32.
    """
    query = np.asarray(query, dtype=F)
    candidates = np.asarray(candidates, dtype=F)
    K, D = candidates.shape
    if not 1 <= K <= MAX_K:
        raise ValueError("K must be in [1, %d], got %d" % (MAX_K, K))
    scale = F(scale)
    temperature = F(temperature)

    query_norm = _query_norm(query, normalize)

    similarity = np.zeros(K, dtype=F)
    for k in range(K):
        dot, norm2 = _row_moments(candidates[k], query, normalize)
        value = dot
        if normalize:
            value = F(value / F(query_norm * F(max(np.sqrt(norm2), NORM_EPS))))
        similarity[k] = F(F(value * scale) / temperature)

    maximum = similarity.max()
    exponentials = np.array([F(np.exp(F(s - maximum))) for s in similarity], dtype=F)
    total = F(0.0)
    for value in exponentials:
        total = F(total + value)
    inverse = F(F(1.0) / total) if total > 0 else F(0.0)
    probabilities = np.array([F(value * inverse) for value in exponentials], dtype=F)
    return probabilities, similarity


def parity(cases, tolerance=1e-4):
    """Compare the kernel algorithm against the reference over the fixed cases.

    Returns a list of dicts: the worst absolute and relative probability
    difference per case, and the logit spread that drives it. A wide spread makes
    the softmax more sensitive, so both are reported rather than a single number.
    """
    import reference as reference_module

    results = []
    for case in cases:
        spec = case["spec"]
        expected = reference_module.probabilities(case["query"], case["candidates"], spec)
        actual, _logits = score(case["query"], case["candidates"], spec.scale,
                                spec.temperature, spec.normalize)
        difference = np.abs(actual.astype(np.float64) - expected)
        denominator = np.maximum(np.abs(expected), 1e-12)
        results.append({
            "name": case["name"],
            "k": int(case["candidates"].shape[0]),
            "d": int(case["candidates"].shape[1]),
            "normalize": spec.normalize,
            "max_abs": float(difference.max()),
            "max_rel": float((difference / denominator).max()),
            "probability_sum": float(actual.astype(np.float64).sum()),
            "within_tolerance": bool(difference.max() <= tolerance),
        })
    return results


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--json")
    args = parser.parse_args(argv)

    import reference as reference_module

    results = parity(reference_module.vectors(), tolerance=args.tolerance)
    worst_abs = max(result["max_abs"] for result in results)
    worst_rel = max(result["max_rel"] for result in results)

    for result in results:
        flag = "ok " if result["within_tolerance"] else "OVER"
        print("%s %-20s K=%-4d D=%-5d max_abs=%.3e max_rel=%.3e sum=%.7f"
              % (flag, result["name"], result["k"], result["d"], result["max_abs"],
                 result["max_rel"], result["probability_sum"]))
    print()
    print("worst absolute difference: %.3e" % worst_abs)
    print("worst relative difference: %.3e" % worst_rel)
    print("all cases within %g: %s" % (args.tolerance,
                                       all(r["within_tolerance"] for r in results)))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"tolerance": args.tolerance, "worst_abs": worst_abs,
                       "worst_rel": worst_rel, "cases": results}, handle, indent=1)
        print("wrote %s" % args.json)
    return 0 if all(result["within_tolerance"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
