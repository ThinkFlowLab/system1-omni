#!/usr/bin/env python3
"""Check the compiled kernel against the reference, on a GPU.

Skips rather than fails when there is no library or no CUDA device, because the
authoring environment has neither and a test that cannot run must not look like
one that passed.

    python3 gpu_parity.py [--library PATH] [--tolerance 1e-4]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import reference  # noqa: E402

DEFAULT_LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libscoring.so")


def unavailable_reason(library):
    """Why this cannot run, or None when it can."""
    if not shutil.which("nvidia-smi"):
        return "no nvidia-smi: this machine has no NVIDIA driver"
    probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        return "nvidia-smi reports no devices"
    if not os.path.isfile(library):
        return "no %s: build it with build.sh first" % os.path.basename(library)
    return None


def gpu_name():
    """The first device's name, or None when there is none."""
    probe = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True)
    if probe.returncode != 0 or not probe.stdout.strip():
        return None
    return probe.stdout.strip().splitlines()[0]


def provenance():
    """What the numbers are attached to: device, driver, CUDA toolkit, nvcc."""
    out = {"gpu": gpu_name()}
    probe = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True)
    if probe.returncode == 0 and probe.stdout.strip():
        out["driver"] = probe.stdout.strip().splitlines()[0]
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if os.path.exists(nvcc):
        version = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
        for line in version.stdout.splitlines():
            if "release" in line:
                out["nvcc"] = line.strip()
    return out


def load_runtime():
    """The CUDA runtime, for device memory.

    The kernel takes device pointers. A numpy array's buffer is host memory, so
    handing its address straight to the kernel makes it dereference host memory
    as device memory -- which is an illegal access, not a wrong answer. The
    library under test exports only the scoring entry points, so the harness
    allocates and copies through cudart itself.
    """
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            runtime = ctypes.CDLL(name)
        except OSError:
            continue
        runtime.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        runtime.cudaMalloc.restype = ctypes.c_int
        runtime.cudaFree.argtypes = [ctypes.c_void_p]
        runtime.cudaFree.restype = ctypes.c_int
        runtime.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_size_t, ctypes.c_int]
        runtime.cudaMemcpy.restype = ctypes.c_int
        runtime.cudaDeviceSynchronize.restype = ctypes.c_int
        runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        runtime.cudaGetErrorString.restype = ctypes.c_char_p
        return runtime
    raise RuntimeError("libcudart not found; is the CUDA runtime installed?")


def check(runtime, status, what):
    if status != 0:
        raise RuntimeError("%s failed: %s (%d)"
                           % (what, runtime.cudaGetErrorString(status).decode(), status))


def load(library):
    lib = ctypes.CDLL(library)
    lib.cs_score_abi_version.restype = ctypes.c_uint32
    lib.cs_score_candidates.restype = ctypes.c_int
    lib.cs_score_candidates_batch.restype = ctypes.c_int
    lib.cs_score_candidates_batch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    # Device pointers arrive as c_void_p from cudaMalloc, not as float arrays:
    # the kernel reads device memory, and typing them as POINTER(c_float) makes
    # ctypes reject exactly that.
    lib.cs_score_candidates.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int,
        ctypes.c_void_p]
    return lib


def score_with_kernel(lib, runtime, query, candidates, spec):
    """One question through the C ABI, with the copies a GPU call needs."""
    q = np.ascontiguousarray(query, dtype=np.float32)
    c = np.ascontiguousarray(candidates, dtype=np.float32)
    k, d = c.shape
    out = np.zeros(k, dtype=np.float32)

    device = {}
    try:
        for name, array in (("q", q), ("c", c), ("out", out)):
            pointer = ctypes.c_void_p()
            check(runtime, runtime.cudaMalloc(ctypes.byref(pointer), array.nbytes),
                  "cudaMalloc(%s)" % name)
            device[name] = pointer
        check(runtime, runtime.cudaMemcpy(device["q"], q.ctypes.data, q.nbytes, 1),
              "cudaMemcpy(query to device)")
        check(runtime, runtime.cudaMemcpy(device["c"], c.ctypes.data, c.nbytes, 1),
              "cudaMemcpy(candidates to device)")

        status = lib.cs_score_candidates(device["q"], device["c"], device["out"], None,
                                         int(k), int(d), ctypes.c_float(spec.scale),
                                         ctypes.c_float(spec.temperature),
                                         1 if spec.normalize else 0, None)
        check(runtime, status, "cs_score_candidates")
        # Synchronise before reading: a launch error is asynchronous, and without
        # this the copy below reports it as its own failure.
        check(runtime, runtime.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        check(runtime, runtime.cudaMemcpy(out.ctypes.data, device["out"], out.nbytes, 2),
              "cudaMemcpy(result to host)")
        return out
    finally:
        for pointer in device.values():
            runtime.cudaFree(pointer)


def run_batch(lib, runtime, questions, k, d, spec, seed=99):
    """Score several questions in one grid, and check each against the reference."""
    rng = np.random.default_rng(seed)
    q = np.ascontiguousarray(rng.standard_normal((questions, d)), dtype=np.float32)
    c = np.ascontiguousarray(rng.standard_normal((questions, k, d)), dtype=np.float32)
    out = np.zeros((questions, k), dtype=np.float32)

    device = {}
    try:
        for name, array in (("q", q), ("c", c), ("out", out)):
            pointer = ctypes.c_void_p()
            check(runtime, runtime.cudaMalloc(ctypes.byref(pointer), array.nbytes),
                  "cudaMalloc(%s)" % name)
            device[name] = pointer
        check(runtime, runtime.cudaMemcpy(device["q"], q.ctypes.data, q.nbytes, 1), "copy q")
        check(runtime, runtime.cudaMemcpy(device["c"], c.ctypes.data, c.nbytes, 1), "copy c")
        status = lib.cs_score_candidates_batch(
            device["q"], device["c"], device["out"], None, int(questions), int(k), int(d),
            ctypes.c_float(spec.scale), ctypes.c_float(spec.temperature),
            1 if spec.normalize else 0, None)
        check(runtime, status, "cs_score_candidates_batch")
        check(runtime, runtime.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
        check(runtime, runtime.cudaMemcpy(out.ctypes.data, device["out"], out.nbytes, 2),
              "copy result out")
    finally:
        for pointer in device.values():
            runtime.cudaFree(pointer)

    worst = 0.0
    for index in range(questions):
        expected = reference.probabilities(q[index], c[index], spec)
        worst = max(worst, float(np.abs(out[index].astype(np.float64) - expected).max()))
    return worst, out


def rejection_checks(lib, runtime):
    """Every rejected argument must return an error, not a scored answer.

    Scoring the first 255 of a larger candidate set would return a confident
    wrong answer, which is worse than an error, so this is checked rather than
    assumed.
    """
    pointer = ctypes.c_void_p()
    check(runtime, runtime.cudaMalloc(ctypes.byref(pointer), 4096), "cudaMalloc")
    results = []
    try:
        cases = [
            ("K above 255", dict(K=256, D=8, scale=1.0, temperature=1.0)),
            ("K of zero", dict(K=0, D=8, scale=1.0, temperature=1.0)),
            ("D of zero", dict(K=2, D=0, scale=1.0, temperature=1.0)),
            ("scale of zero", dict(K=2, D=8, scale=0.0, temperature=1.0)),
            ("negative temperature", dict(K=2, D=8, scale=1.0, temperature=-1.0)),
            ("null query", dict(K=2, D=8, scale=1.0, temperature=1.0, query=None)),
        ]
        for name, spec in cases:
            query = spec.pop("query", pointer)
            status = lib.cs_score_candidates(query, pointer, pointer, None, spec["K"],
                                             spec["D"], ctypes.c_float(spec["scale"]),
                                             ctypes.c_float(spec["temperature"]), 0, None)
            results.append((name, status))
    finally:
        runtime.cudaFree(pointer)
    return results


def graph_checks(lib, runtime, questions=4, k=17, d=64, seed=7):
    """Capture one launch into a CUDA Graph, replay it, and compare with a direct call.

    The ABI is written for capture: it queues on the caller's stream and does not
    synchronize. So a capture must see exactly one node, and a replay must produce
    the same probabilities as the same call made directly -- not merely within
    tolerance, but identical, because the kernel has no atomics and its reduction
    order is fixed.
    """
    for name, argtypes in (
        ("cudaStreamCreate", [ctypes.POINTER(ctypes.c_void_p)]),
        ("cudaStreamBeginCapture", [ctypes.c_void_p, ctypes.c_int]),
        ("cudaStreamEndCapture", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]),
        ("cudaGraphGetNodes", [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]),
        ("cudaGraphInstantiate", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                  ctypes.c_ulonglong]),
        ("cudaGraphLaunch", [ctypes.c_void_p, ctypes.c_void_p]),
        ("cudaGraphExecDestroy", [ctypes.c_void_p]),
        ("cudaGraphDestroy", [ctypes.c_void_p]),
        ("cudaStreamDestroy", [ctypes.c_void_p]),
    ):
        getattr(runtime, name).argtypes = argtypes
        getattr(runtime, name).restype = ctypes.c_int

    rng = np.random.default_rng(seed)
    q = np.ascontiguousarray(rng.standard_normal((questions, d)), dtype=np.float32)
    c = np.ascontiguousarray(rng.standard_normal((questions, k, d)), dtype=np.float32)
    direct = np.zeros((questions, k), dtype=np.float32)
    replayed = np.zeros((questions, k), dtype=np.float32)
    spec = reference.ScoreSpec(scale=0.0625, temperature=1.5)

    device = {}
    stream, graph, graph_exec = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    try:
        for name, array in (("q", q), ("c", c), ("direct", direct), ("replayed", replayed)):
            pointer = ctypes.c_void_p()
            check(runtime, runtime.cudaMalloc(ctypes.byref(pointer), array.nbytes),
                  "cudaMalloc(%s)" % name)
            device[name] = pointer
        check(runtime, runtime.cudaMemcpy(device["q"], q.ctypes.data, q.nbytes, 1), "copy q")
        check(runtime, runtime.cudaMemcpy(device["c"], c.ctypes.data, c.nbytes, 1), "copy c")
        check(runtime, runtime.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")

        def score(out, on_stream):
            return lib.cs_score_candidates_batch(
                device["q"], device["c"], device[out], None, int(questions), int(k), int(d),
                ctypes.c_float(spec.scale), ctypes.c_float(spec.temperature), 1, on_stream)

        check(runtime, score("direct", None), "direct cs_score_candidates_batch")
        check(runtime, runtime.cudaDeviceSynchronize(), "sync after the direct call")

        # cudaStreamCaptureModeThreadLocal = 1: only this thread's work is captured.
        check(runtime, runtime.cudaStreamBeginCapture(stream, 1), "cudaStreamBeginCapture")
        check(runtime, score("replayed", stream), "cs_score_candidates_batch under capture")
        check(runtime, runtime.cudaStreamEndCapture(stream, ctypes.byref(graph)),
              "cudaStreamEndCapture")

        nodes = ctypes.c_size_t()
        check(runtime, runtime.cudaGraphGetNodes(graph, None, ctypes.byref(nodes)),
              "cudaGraphGetNodes")
        check(runtime, runtime.cudaGraphInstantiate(ctypes.byref(graph_exec), graph,
                                                    ctypes.c_ulonglong(0)), "cudaGraphInstantiate")
        check(runtime, runtime.cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch")
        check(runtime, runtime.cudaDeviceSynchronize(), "sync after the replay")

        check(runtime, runtime.cudaMemcpy(direct.ctypes.data, device["direct"], direct.nbytes, 2),
              "copy the direct result out")
        check(runtime, runtime.cudaMemcpy(replayed.ctypes.data, device["replayed"],
                                          replayed.nbytes, 2), "copy the replayed result out")
    finally:
        if graph_exec.value:
            runtime.cudaGraphExecDestroy(graph_exec)
        if graph.value:
            runtime.cudaGraphDestroy(graph)
        if stream.value:
            runtime.cudaStreamDestroy(stream)
        for pointer in device.values():
            runtime.cudaFree(pointer)

    return int(nodes.value), float(np.abs(direct.astype(np.float64)
                                          - replayed.astype(np.float64)).max())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", default=DEFAULT_LIBRARY)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv)

    reason = unavailable_reason(args.library)
    if reason:
        print("SKIP: %s" % reason)
        return 0

    lib = load(args.library)
    runtime = load_runtime()
    print("kernel ABI version: %d" % lib.cs_score_abi_version())

    worst = 0.0
    failures = []
    for case in reference.vectors():
        expected = reference.probabilities(case["query"], case["candidates"], case["spec"])
        actual = score_with_kernel(lib, runtime, case["query"], case["candidates"],
                                   case["spec"])
        difference = float(np.abs(actual.astype(np.float64) - expected).max())
        worst = max(worst, difference)
        ok = difference <= args.tolerance
        if not ok:
            failures.append(case["name"])
        print("%s %-20s K=%-4d D=%-5d max_abs=%.3e"
              % ("ok " if ok else "FAIL", case["name"], case["candidates"].shape[0],
                 case["candidates"].shape[1], difference))

    print()
    print("--- batch path (grid over questions) ---")
    for questions, k, d in ((1, 4, 256), (8, 4, 256), (4, 17, 64), (3, 255, 32)):
        batch_worst, out = run_batch(lib, runtime, questions, k, d,
                                     reference.ScoreSpec(scale=0.0625, temperature=1.5))
        ok = batch_worst <= args.tolerance
        if not ok:
            failures.append("batch q=%d k=%d" % (questions, k))
        worst = max(worst, batch_worst)
        rows_ok = all(abs(float(row.sum()) - 1.0) < 1e-4 for row in out)
        print("%s questions=%-3d K=%-4d D=%-5d max_abs=%.3e  rows sum to 1: %s"
              % ("ok " if ok and rows_ok else "FAIL", questions, k, d, batch_worst, rows_ok))

    print()
    print("--- rejected arguments (must be errors, not answers) ---")
    for name, status in rejection_checks(lib, runtime):
        ok = status != 0
        if not ok:
            failures.append("not rejected: %s" % name)
        print("%s %-20s returned %d" % ("ok " if ok else "FAIL", name, status))

    print()
    print("--- CUDA Graph capture and replay ---")
    nodes, graph_difference = graph_checks(lib, runtime)
    graphs_ok = nodes == 1 and graph_difference == 0.0
    if not graphs_ok:
        failures.append("graph: %d node(s), max_abs=%.3e" % (nodes, graph_difference))
    print("%s captured %d node, replay matches the direct call exactly (max_abs=%.3e)"
          % ("ok " if graphs_ok else "FAIL", nodes, graph_difference))

    print("\nworst absolute difference: %.3e (tolerance %g)" % (worst, args.tolerance))
    if failures:
        print("cases over tolerance: %s" % ", ".join(failures))
        return 1
    print("all cases within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
