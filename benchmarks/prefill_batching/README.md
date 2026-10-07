# Open-Jev selective prefill packing on H200

Packing input and gate/up projections across candidates reduces mean warm HTTP
latency by **18.04% for four-question requests**, **20.11% for eight-question
requests**, and **10.67% for mixed Choice/Noul/Score examples**. Compared outputs
match the native reference exactly. Single-question mean/p95 increases
**0.21%/1.09%**, within the declared 2% regression limits.

These are concurrency-1, within-request observations on one H200 with BF16,
from 2026-10-06. Repeated-question workloads are synthetic transformations of
real JevBench cases. Two measured passes show observed variability, not
confidence intervals or production-traffic performance.

## Implementation

The adapter preserves candidate order in groups of at most 16 sequences and
4096 tokens. Longer prompts execute alone at their original length. Input and
gate/up GEMMs share packed rows; output/down projections keep per-prompt shapes
to preserve cuBLASLt reduction order. Attention, positions, convolution, GDN
state and final-token readout remain independent. Scores are regrouped by
question before calibration. Runtime admission still covers one whole request.

CUDA Graph keys contain ordered sequence lengths. Cache misses retain the eager
result; cache hits read freshly embedded inputs. The upstream CUDA ABI5 and
native vision support are preserved. Graph performance is unmeasured here.

## Matched warm HTTP results

Each latency cell gives measured pass 1 / 2 means in milliseconds. Reduction
uses the arithmetic mean of those pass means.

| Workload | Requests/pass | Current main | Packed | Mean reduction |
| --- | ---: | ---: | ---: | ---: |
| Real single-question JevBench | 74 | 48.048 / 48.143 | 48.307 / 48.083 | -0.21% |
| Four repeated questions | 60 | 98.703 / 98.618 | 80.800 / 80.915 | 18.04% |
| Eight repeated questions | 60 | 197.470 / 197.502 | 157.558 / 157.978 | 20.11% |
| Mixed Choice/Noul/Score | 12 | 258.378 / 257.593 | 230.449 / 230.490 | 10.67% |

The single-question slice contains 74 Noul cases spanning 80–3399 tokens from
[JevBench at f8ce713](https://github.com/fstandhartinger/jevbench/tree/f8ce71361165846101d02ebc83ad44e47ae44fc3).
The four/eight-question slices duplicate the 60 cases of at most 400 tokens
under distinct question IDs. The mixed slice uses the repository's
[three-question fixture](../../tests/open_jev/data/contract.json), appending
0–33 copies of a fixed billing-context sentence across 12 cases.

The measured passes contain **824 successful requests / 3320 decisions**.
Answers, token usage, model identity and metadata except inference time match
the baseline feasibility reference exactly: maximum probability/Score drift
**0.0**, zero decision flips and zero failures. All declared gates pass:
at least 15% reduction for both repeated-question slices, at most 2% single
mean/p95 regression, drift at most 0.001, zero flips and zero failures.

## Controls and evidence

[summary.json](artifacts/20261006/summary.json) records per-pass metrics, gates,
frozen source/checkpoint revisions, source hashes and execution controls.
[timings.csv](artifacts/20261006/timings.csv) retains all 824 measured request
latencies in seconds: one row per case, with baseline/candidate pass 1/2 columns
and the question-answer count for each request.
P95 uses nearest rank within each pass; reported reductions average pass metrics.

Baseline: `47eff9cdeda01e4847a4fb9634a43f2cab6a233f`.
Measured candidate runtime: `82b5e7e60363405614bccd79ba83ab774e7513f5`;
its source hashes match the final runtime files. Both arms use the same merged
export, tokenizer, trained head, temperature `2.5343690298472983`, max length
16384, CUDA ABI5 library, frontend and 32 MiB GEMM workspace. GPU 2 UUID
`GPU-cbf66259-f4ab-0ede-1811-82037dde5924`, SM90, NUMA 0 / CPUs 0–15 are fixed.
The driver/scheduler labels this H200 device L20X.

Each arm reuses one worker/frontend pair. Real readiness, validated first
inference, preparation and one feasibility pass per slice are excluded, followed
by exactly two measured passes. Graphs and profiling are disabled during HTTP
measurements; no shared caches are reset or clocks changed. Timing includes
localhost forwarding, tokenization, inference and UTF8 response receipt;
request construction and response JSON parsing are excluded.

The complete [raw evidence snapshot at 7d03326](https://github.com/ThinkFlowLab/system1-omni/tree/7d03326d51debde070f2a0d3a925405032275270/benchmarks/prefill_batching/artifacts/20261006)
preserves requests, all measured/feasibility responses, collector, frozen plan,
validation logs and JevBench's MIT notice. These files remain in Git history
and a verified local archive; duplicate inputs, process logs and rejected-attempt
records are excluded from the current diff.

For reproduction, build baseline/candidate worktrees at the pinned runtime SHAs
with matched release options and prepare the pinned export using the
[native recipe](../../recipe/open_jev/native.md). Copy the archived plan and
harness into a fresh run directory and create its `analysis/` directory. Update
host paths, hashes and collector GPU/affinity assertions together, freeze those
controls, then run the collector through the verified scheduler with fixed exact
GPU IDs, NUMA and CPU affinity. Reuse the declared feasibility/two-pass budget.

## Validation and limits

Formatting, strict Clippy, locked workspace tests (**90 passed, 13 ignored**),
release build and strict docs pass. Six reserved CUDA reference tests and the
checkpoint packing test pass, covering unequal lengths, reordered shapes,
17-candidate splitting, changed-input cache hits and later singleton execution.
Nsight Systems records **four actual graph launches** in the correctness test;
this trace is outside measured HTTP runs. Task-owned processes exited and the
GPU returned available with 0 MB used.

Other architectures, higher concurrency, production distributions, peak memory,
cold capture and combined graph performance remain unmeasured. Output fidelity
is against native main; this is not a new full-precision accuracy evaluation.
