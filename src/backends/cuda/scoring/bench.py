#!/usr/bin/env python3
"""Measure what fusing the candidate scoring is worth, on a GPU.

Compares the shipped kernel with the two-pass baseline in `bench.cu`. Both are
given the same inputs, both are checked against `reference.py` first, and only
then are they timed -- a kernel that computes the wrong thing can be arbitrarily
fast, so a correctness check that fails aborts the run rather than producing a
number.

    python3 bench.py [--library PATH] [--iterations N] [--out results.json]

Writes a JSON report; `docs/benchmarks/scoring-fusion/README.md` holds the
protocol and the results of the recorded run.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gpu_parity  # noqa: E402
import reference  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIBRARY = os.path.join(HERE, "libscoring.so")
BASELINE_SOURCE = os.path.join(HERE, "bench.cu")
BASELINE_LIBRARY = os.path.join(HERE, "libscoring_bench.so")

# The shapes the two models actually ask for, each with the similarity it uses.
# CLM is a cosine over 512-wide projected vectors with up to 255 candidates; Kev
# is a plain dot over 2560-wide hidden states, one question per prefill row.
#
# Kev is the control. It has no norm to compute, so there is nothing for this
# fusion to remove from the traffic, and the measurement should say so.
SHAPES = [
    ("clm-cosine-5x512", 1, 5, 512, True),
    ("clm-cosine-64x512", 1, 64, 512, True),
    ("clm-cosine-255x512", 1, 255, 512, True),
    ("clm-cosine-batch8-255x512", 8, 255, 512, True),
    ("kev-dot-255x2560", 1, 255, 2560, False),
]

SCALE, TEMPERATURE = 0.0625, 1.5


def build_baseline(nvcc, arch):
    """Compile bench.cu, with the same flags build.sh uses for the kernel."""
    targets = [arch] + os.environ.get("CUDA_ARCH_LIST", "").split()
    gencode = []
    for target in targets:
        if target:
            gencode += ["-gencode", "arch=compute_%s,code=sm_%s" % (target, target)]
    command = ([nvcc, "-O3", "-std=c++17"] + gencode
               + ["-shared", "-Xcompiler", "-fPIC", "-Xcompiler", "-Wall,-Wextra",
                  BASELINE_SOURCE, "-lcudart", "-o", BASELINE_LIBRARY])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("bench.cu did not compile:\n%s" % result.stderr[-2000:])
    return " ".join(command)


def load_bench_library(path):
    lib = ctypes.CDLL(path)
    lib.bench_unfused.restype = ctypes.c_int
    lib.bench_unfused.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
    return lib


def timed(runtime, call, iterations, stream=None):
    """Median milliseconds per call, over `iterations` calls after a warmup.

    Both the direct launch and the graph replay are timed on the same stream, so
    the only difference between them is the path taken to the kernel.
    """
    for _ in range(max(3, iterations // 10)):
        call()
    runtime.cudaDeviceSynchronize()

    start, stop = ctypes.c_void_p(), ctypes.c_void_p()
    gpu_parity.check(runtime, runtime.cudaEventCreate(ctypes.byref(start)), "cudaEventCreate")
    gpu_parity.check(runtime, runtime.cudaEventCreate(ctypes.byref(stop)), "cudaEventCreate")
    samples = []
    try:
        for _ in range(iterations):
            gpu_parity.check(runtime, runtime.cudaEventRecord(start, stream), "record start")
            call()
            gpu_parity.check(runtime, runtime.cudaEventRecord(stop, stream), "record stop")
            gpu_parity.check(runtime, runtime.cudaEventSynchronize(stop), "sync stop")
            elapsed = ctypes.c_float()
            gpu_parity.check(runtime, runtime.cudaEventElapsedTime(ctypes.byref(elapsed), start, stop),
                             "cudaEventElapsedTime")
            samples.append(elapsed.value)
    finally:
        runtime.cudaEventDestroy(start)
        runtime.cudaEventDestroy(stop)
    return float(np.median(samples)), samples


def capture_fused(runtime, lib, device, stream, questions, k, d, spec):
    """Capture one fused launch into a graph, and instantiate it.

    The ABI is written for this: it queues on the caller's stream and never
    synchronizes, so the capture sees exactly one node.
    """
    graph, graph_exec = ctypes.c_void_p(), ctypes.c_void_p()
    gpu_parity.check(runtime, runtime.cudaStreamBeginCapture(stream, 1), "cudaStreamBeginCapture")
    gpu_parity.check(runtime, lib.cs_score_candidates_batch(
        device["q"], device["c"], device["fused"], None, questions, k, d,
        ctypes.c_float(spec.scale), ctypes.c_float(spec.temperature),
        1 if spec.normalize else 0, stream), "launch under capture")
    gpu_parity.check(runtime, runtime.cudaStreamEndCapture(stream, ctypes.byref(graph)),
                     "cudaStreamEndCapture")
    gpu_parity.check(runtime, runtime.cudaGraphInstantiate(ctypes.byref(graph_exec), graph,
                                                           ctypes.c_ulonglong(0)),
                     "cudaGraphInstantiate")
    return graph, graph_exec


def traffic_bytes(questions, k, d, passes):
    """Mandatory DRAM traffic: the query, `passes` reads of the candidates, and the output."""
    return questions * d * 4 + passes * questions * k * d * 4 + questions * k * 4


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", default=DEFAULT_LIBRARY)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--arch", default=os.environ.get("CUDA_COMPUTE_CAP", "89"))
    parser.add_argument("--nvcc", default=shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    reason = gpu_parity.unavailable_reason(args.library)
    if reason:
        print("SKIP: %s" % reason)
        return 0

    build = build_baseline(args.nvcc, args.arch)
    fused = gpu_parity.load(args.library)
    unfused = load_bench_library(BASELINE_LIBRARY)
    runtime = gpu_parity.load_runtime()
    for name in ("cudaEventCreate", "cudaEventRecord", "cudaEventSynchronize",
                 "cudaEventElapsedTime", "cudaEventDestroy"):
        getattr(runtime, name).restype = ctypes.c_int
    runtime.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float),
                                             ctypes.c_void_p, ctypes.c_void_p]

    provenance = dict(gpu_parity.provenance())
    provenance.update({
        "library": args.library,
        "baseline_build": build,
        "iterations": args.iterations,
        "arch": args.arch,
    })

    rows = []
    for name, questions, k, d, cosine in SHAPES:
        spec = reference.ScoreSpec(scale=SCALE, temperature=TEMPERATURE, normalize=cosine)
        rng = np.random.default_rng(1234)
        q = np.ascontiguousarray(rng.standard_normal((questions, d)), dtype=np.float32)
        c = np.ascontiguousarray(rng.standard_normal((questions, k, d)), dtype=np.float32)
        device, outputs = {}, {}
        try:
            for key, array in (("q", q), ("c", c)):
                pointer = ctypes.c_void_p()
                gpu_parity.check(runtime,
                                 runtime.cudaMalloc(ctypes.byref(pointer), array.nbytes),
                                 "cudaMalloc(%s)" % key)
                device[key] = pointer
                gpu_parity.check(runtime,
                                 runtime.cudaMemcpy(pointer, array.ctypes.data, array.nbytes, 1),
                                 "copy %s" % key)
            for key, count in (("fused", questions * k), ("unfused", questions * k),
                               ("norms", questions * k)):
                pointer = ctypes.c_void_p()
                gpu_parity.check(runtime,
                                 runtime.cudaMalloc(ctypes.byref(pointer), count * 4),
                                 "cudaMalloc(%s)" % key)
                device[key] = pointer
            for key in ("fused", "unfused"):
                outputs[key] = np.zeros((questions, k), dtype=np.float32)

            # One stream for every timing below, so the direct launch and the
            # graph replay differ only in the path to the kernel. Measuring on
            # the legacy default stream instead would add its synchronisation
            # semantics to both and make the columns incomparable.
            stream = ctypes.c_void_p()
            gpu_parity.check(runtime, runtime.cudaStreamCreate(ctypes.byref(stream)),
                             "cudaStreamCreate")

            def call_fused():
                return fused.cs_score_candidates_batch(
                    device["q"], device["c"], device["fused"], None, questions, k, d,
                    ctypes.c_float(spec.scale), ctypes.c_float(spec.temperature),
                    1 if spec.normalize else 0, stream)

            def call_unfused():
                return unfused.bench_unfused(
                    device["q"], device["c"], device["norms"], device["unfused"], questions, k, d,
                    ctypes.c_float(spec.scale), ctypes.c_float(spec.temperature),
                    1 if spec.normalize else 0, stream)

            # Correctness first. A wrong kernel can be arbitrarily fast.
            for label, call, key in (("fused", call_fused, "fused"),
                                     ("unfused", call_unfused, "unfused")):
                gpu_parity.check(runtime, call(), "%s launch" % label)
                gpu_parity.check(runtime, runtime.cudaDeviceSynchronize(), "sync")
                gpu_parity.check(runtime,
                                 runtime.cudaMemcpy(outputs[key].ctypes.data, device[key],
                                                    outputs[key].nbytes, 2),
                                 "copy %s out" % label)
                worst = 0.0
                for index in range(questions):
                    expected = reference.probabilities(q[index], c[index], spec)
                    worst = max(worst, float(np.abs(
                        outputs[key][index].astype(np.float64) - expected).max()))
                if worst > 1e-4:
                    raise SystemExit("%s %s is wrong (max_abs=%.3e); not timing it"
                                     % (name, label, worst))
                if label == "fused":
                    fused_worst = worst
                else:
                    unfused_worst = worst

            fused_ms, _ = timed(runtime, call_fused, args.iterations, stream)
            unfused_ms, _ = timed(runtime, call_unfused, args.iterations, stream)

            # The same launch, replayed from a captured graph. The small shapes
            # are launch-bound, so this is where the difference should show.
            graph, graph_exec = capture_fused(runtime, fused, device, stream,
                                              questions, k, d, spec)
            try:
                def call_replay():
                    return runtime.cudaGraphLaunch(graph_exec, stream)

                direct_ms, _ = timed(runtime, call_fused, args.iterations, stream)
                graph_ms, _ = timed(runtime, call_replay, args.iterations, stream)
            finally:
                runtime.cudaGraphExecDestroy(graph_exec)
                runtime.cudaGraphDestroy(graph)
                runtime.cudaStreamDestroy(stream)
        finally:
            for pointer in device.values():
                runtime.cudaFree(pointer)

        # The baseline only reads the candidates twice when it has a norm to
        # compute; a plain dot reads once, like the fused kernel.
        fused_bytes = traffic_bytes(questions, k, d, 1)
        unfused_bytes = traffic_bytes(questions, k, d, 2 if spec.normalize else 1)
        rows.append({
            "shape": name,
            "questions": questions, "K": k, "D": d, "cosine": bool(spec.normalize),
            "fused_ms": fused_ms,
            "unfused_ms": unfused_ms,
            "speedup": unfused_ms / fused_ms,
            "fused_gbs": fused_bytes / fused_ms / 1e6,
            "unfused_gbs": unfused_bytes / unfused_ms / 1e6,
            "fused_bytes": fused_bytes,
            "unfused_bytes": unfused_bytes,
            "fused_max_abs": fused_worst,
            "unfused_max_abs": unfused_worst,
            "direct_ms": direct_ms,
            "graph_replay_ms": graph_ms,
            "graph_gain": direct_ms / graph_ms,
        })
        print("%-26s %-4s K=%-4d D=%-5d q=%d  fused %7.1f us (%5.0f GB/s)  "
              "unfused %7.1f us (%5.0f GB/s)  %.2fx"
              % (name, "cos" if spec.normalize else "dot", k, d, questions,
                 fused_ms * 1e3, rows[-1]["fused_gbs"],
                 unfused_ms * 1e3, rows[-1]["unfused_gbs"], rows[-1]["speedup"]))
        print("%-26s      the same launch: direct %7.1f us, graph replay %7.1f us  %.2fx"
              % ("", direct_ms * 1e3, graph_ms * 1e3, direct_ms / graph_ms))

    report = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "provenance": provenance, "results": rows}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
