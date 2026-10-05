# Bounded native admission validation

Actual runs on 2026-10-05 compared merged main `9adb0701132d259a9fcc8a559472f1c3dc5590b0`
with runtime candidate `b5c443c9310f2118289500b6088822e7bba1fb72`. All 1,536 measured
requests succeeded with exact response/key-order parity, excluding Open-Jev
`metadata.inference_seconds`. All six predeclared performance gates passed.
The limit-one functional burst accepted one request and rejected seven with
HTTP 503 for each model; a subsequent inference returned 200 with exact parity.

The [protocol](protocol.json) records source/build/binary hashes, pinned models,
controls, input hashes, budget and thresholds. The [compressed evidence](raw-results.json.gz)
contains every measured/feasibility response, warmup, first-inference, near-limit,
expected-error and overload/recovery record, plus fixed input manifests,
per-run summaries, startup and raw device-memory samples.

## Change and controls

The independent variable is bounded pending execution and its overload mapping.
Execution remains serial. Normal-load comparisons set `OMNI_NATIVE_MAX_PENDING=64`
(the baseline ignores this new setting); candidate-only functional bursts use 1.
The bound counts Cua-S1 question forwards and Open-Jev complete requests. It does
not bound ingress requests or prepared-token memory; GPU batching remains planned.

Controls: one reserved SM90 NVIDIA L20X, exact GPU5, driver 570.133.20, CUDA driver
API 13000, NUMA memory node 1/CPUs 56–71, Rust 1.98.1,
Python 3.12, httpx 0.28.1, tokenizers 0.22.2; BF16 backbone, existing FP32 CPU heads/f64
normalization, unchanged CUDA ABI4 library, eager CUDA, 16 Tokio workers and disabled
tokenizer parallelism. Checkpoint revisions and input hashes are in the protocol.
No caches were dropped, models downloaded or exports repeated. Baseline binaries
reuse an isolated build whose entire tree equals the merged baseline; all comparison
measurements are fresh. Candidate uses its own Cargo target and matching package flags.

The PR later integrated upstream documentation/recipes and added this report.
All Rust sources, Cargo manifests/lockfile and CUDA build inputs still match the
measured candidate; `native_source_hashes` lets reviewers verify that equivalence.

## Warmed native HTTP comparison

Per model/revision: one real-warmed server, first inference validation, one excluded
64-request feasibility pass, then two measured 64-request passes at each concurrency
1/8/16, with three excluded warmups per pass. There are 1,536 measured requests and
256 excluded feasibility requests. All warmup and first-inference responses also
passed parity. Each configuration then validated a 16,368-token input and four
expected errors outside performance timing. No extra runs or discarded failures.

Gates: mean of per-run mean HTTP latency ≤105% baseline, mean requests/s ≥95%, and
mean per-run p95 ≤110%, for each model/concurrency. Timing is direct native HTTP;
frontend overhead and individual kernel timings are not measured.

| Model | Concurrency | Baseline mean ms | Candidate mean ms | Latency change | Throughput change | p95 change | Observed gate |
|---|---:|---:|---:|---:|---:|---:|---|
| cua | 1 | 23.47 | 23.53 | +0.27% | -0.27% | +0.39% | pass |
| cua | 8 | 167.52 | 167.81 | +0.17% | -0.19% | +0.09% | pass |
| cua | 16 | 320.86 | 321.56 | +0.22% | -0.25% | +0.43% | pass |
| jev | 1 | 108.68 | 108.69 | +0.00% | -0.00% | -0.77% | pass |
| jev | 8 | 812.37 | 812.04 | -0.04% | +0.04% | +3.42% | pass |
| jev | 16 | 1515.66 | 1516.07 | +0.03% | -0.02% | -0.08% | pass |


Relative ranges across the two repetitions, `(max - min) / mean`:

| Model | Concurrency | Baseline mean run range | Candidate mean run range | Baseline p95 run range | Candidate p95 run range |
|---|---:|---:|---:|---:|---:|
| cua | 1 | 0.04% | 0.02% | 0.42% | 0.30% |
| cua | 8 | 0.04% | 0.36% | 0.32% | 0.18% |
| cua | 16 | 0.28% | 0.06% | 0.29% | 1.29% |
| jev | 1 | 0.04% | 0.18% | 0.21% | 1.25% |
| jev | 8 | 0.03% | 0.19% | 0.22% | 0.12% |
| jev | 16 | 0.00% | 0.06% | 0.04% | 0.08% |

## Functional overload and recovery

After normal-load comparisons, one candidate server per model used limit 1, with
real warmup and a validated first inference. One burst of eight concurrent
16,368-token requests yielded one 200 and seven 503 responses per model. The
successful response matched the ordinary reference; each 503 had the native worker
error shape and `native execution capacity exhausted`. The next ordinary request
returned 200 with exact parity; `/health` remained ready. These 16 burst requests
and recovery/startup validation are outside measured performance denominators.
Queued/dispatched cancellation and error/panic release are covered by CPU tests.

## Startup and memory

CPU builds/copies and existing checkpoint exports are preparation, excluded from
process-to-readiness and replay. Startup and first inference are separate:

| Configuration | Process to readiness (s) | First inference (ms) | Sampled GPU peak incl. near-limit (MiB) |
|---|---:|---:|---:|
| cua-baseline | 3.38 | 143.32 | 12417 |
| cua-candidate | 3.67 | 50.16 | 12417 |
| jev-baseline | 15.71 | 189.11 | 55465 |
| jev-candidate | 15.31 | 199.80 | 55465 |

Memory is device footprint sampled every 200 ms, including near-limit validation;
it is not an allocator peak or a measured-only peak. All 40 task-owned processes
exited and the scheduler reservation was released. Full cleanup/probes remain in
the private original archive; machine/user locations are omitted from public records.

## Reproduce and inspect

Build each frozen source in a separate checkout/target using:

```sh
cargo build --release --locked -j 16 -p omni-cua-s1-native -p omni-open-jev-native
```

Prepare the pinned checkpoints with the
[Cua-S1 recipe](../../../recipe/cua_s1/native.md) and
[Open-Jev recipe](../../../recipe/open_jev/native.md). Reuse one unchanged CUDA
library for both revisions and honor the execution host's GPU reservation rules.
The archive can be inspected and its fixed input files extracted without GPU use:

```python
import gzip, json
from pathlib import Path
root = Path("docs/benchmarks/native-admission")
evidence = json.loads(gzip.decompress((root / "raw-results.json.gz").read_bytes()))
for name, text in evidence["inputs"].items():
    Path(name).write_text(text)
print(evidence["summary"]["comparisons"])
```

For each reserved model/revision server, use the
[repository runner](../../../benchmarks/bench.py), metadata requirements and
[comparison protocol](../../../benchmarks/README.md). Run a feasibility pass
and then two passes at each concurrency, reusing that server:

```sh
python benchmarks/bench.py run cua-requests.jsonl --endpoint http://127.0.0.1:8000/v1/systemone --model cua-s1-4b-0.2 --metadata metadata.json --output run-c1-r1 --phase measured --concurrency 1 --warmup 3 --timeout 180
```

Use `jev-requests.jsonl` with model `open-jev` for Open-Jev. Create metadata from
actual reserved hardware and the protocol's pinned revisions/precision; readiness
requires the worker's successful real warmup. Preserve every repeat and failure.
The private launcher used `gpu run --gpu-ids 5 --wait 10m --timeout 30m` and
`numactl --membind=1 --physcpubind=56-71`. For the functional check, restart each
candidate with limit 1, validate its first inference, send the archive's near-limit
request eight times concurrently, and validate the following ordinary inference.

## Limits and disclosure

Two repetitions provide descriptive ranges, not a confidence interval or
statistically proven equivalence. This synthetic workload tests regression/parity,
not model accuracy or general serving capacity; no speed winner is claimed.
Sequential configuration blocks leave time/order, CPU and thermal effects possible.
Metal, CUDA Graphs, full frontend performance, aggregate token budgets and GPU
batching are unverified here. Labelled accuracy metrics and demo video are N/A for
this admission change. Inputs are task-generated synthetic data; the public archive
retains original response bodies and timings while removing machine-local identity,
paths and process/reservation details. No third-party images or private/customer
inputs are included.
