# Public231 request latency leaderboard

This board ranks deployed model/service configurations on the same 231 public
text tasks. It measures complete HTTP requests at concurrency 1 on one H800;
it does not isolate kernel speed, model size or same-model acceleration.

## H800 group: Slurm 415076, 2026-10-07 UTC

Rank uses pooled p50, ascending. Equal full-precision values share a rank. p95 has
its own ordering: a low median does not imply low tail latency.

| p50 rank | Deployment | p50 ms | p95 ms | p95 rank | Successful / planned |
|---:|---|---:|---:|---:|---:|
| 1 | Open-Jev-9B, Omni native | 42.688 | 327.057 | 4 | 693 / 693 |
| 2 | Open-Jev-27B-v1.1, Omni native | 68.877 | 1068.817 | 5 | 693 / 693 |
| 3 | Intern-Decision-2B, official HF | 79.054 | 175.276 | 2 | 693 / 693 |
| 4 | Intern-Decision-0.8B, official HF | 80.493 | 175.237 | 1 | 693 / 693 |
| 5 | Intern-Decision-4B, official HF | 104.552 | 235.291 | 3 | 693 / 693 |

There were no measured failures, invalid responses or missing requests. Each
deployment had five client warmups once, followed by three 231-request traversals.
Warmups and process startup are excluded. Repetition gives 693 timing samples
per deployment, with 231 unique tasks.

## Clock and controls

The historical client starts immediately before POST and stops after HTTPX reads
the complete body and assigns `response.text`. JSON decoding and schema validation
occur after the timer stops. Quantiles use sorted samples, k=(n−1)q and linear
interpolation; pooled p95 is recomputed from all samples, not averaged from round
p95 values. The [immutable report](https://github.com/linear3735/system1-agents/blob/40bf742e2b84ad6157bf2990dd5e4bab73571702/benchmarks/results/public231-h800.md)
includes per-round measurements and exact deployment revisions.

Inputs and scorer are fixed to [JevBench 7ce310c7](https://github.com/fstandhartinger/jevbench/tree/7ce310c7262ed49cc85853339a8a42459298e3f3),
canonical dataset SHA-256 (`dataset_hash` of the pinned tasks, recorded as
`canonical_dataset_sha256` in the numeric attachment's source manifest):
`dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`.
All arms used BF16, concurrency 1, thinking off, no retries and the same request
order. Intern used upstream HF at T=1 with Torch fallback for unavailable optional
fast paths. Open-Jev used native Rust/CUDA with checkpoint temperatures, merged
LoRA/head and graph/prefix cache off; 9B was sequential and 27B packed candidates.
Different architectures, templates and context policies limit what ratios mean.

The published report is available now. A numeric audit attachment is prepared but
not yet published; it contains all 3465 measured timings and 25 warmups, probability
vectors and source hashes without question/option text. Raw responses remain
outside the repository. Accuracy is maintained separately in
[System1-Agents](https://github.com/ThinkFlowLab/system1-agents/pull/69).

## New runner group: Slurm 416135

The [serving-evidence PR](https://github.com/ThinkFlowLab/system1-omni/pull/131)
reports a separate two-model run: 9B pooled p50/p95 43.082/326.968 ms and
27B 69.459/1066.037 ms. That client includes body decoding, JSON and schema
validation and uses nearest-rank p95. It performs five warmups per traversal.
Those values do not enter the historical five-model ranking above. No speedup
ratio is derived between these clocks or campaigns.

## Add a result

Use the existing [runner and offline summary](README.md#run-against-a-ready-gpu-server),
with a pinned manifest, actual deployment metadata and fresh external output
directories. Predeclare hardware, precision, concurrency, request order, cache
state, timeouts, warmups, repeats, clock boundaries and quantile method.

Publish per-round and pooled p50/p95 with full planned/attempted/successful/failed/
incomplete/unattempted counts. Keep failure latencies separate. In this board, a
run with failures or incomplete coverage remains visible but unranked; a fast
success subset cannot establish a winner. Startup and throughput use their own
boundaries, and ticket episode duration is not request latency.

Append a dated campaign and auditable evidence without replacing prior results.
Compare new deployments with a reference measured in the same campaign. Different
hardware, workloads, precision, concurrency, clocks or quantile methods need
separate groups. Image replacement, repeated-image/cache workloads and higher
concurrency have not been measured here; add their own frozen inputs before ranking.
