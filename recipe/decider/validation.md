# Decider-2B v11 validation on RTX 4090

Validation date: 2026-10-07. This records one BF16 CUDA device and a frozen synthetic
contract corpus. It does not establish task quality, every supported GPU, performance
improvement or universal bit-identical agreement with Transformers.

## Controls and fidelity gates

Checkpoint/tokenizer/calibration: Mapika/decider-2b
`533964dae8be954c5b5e19fa4948e48408094c1e`. Checkpoint SHA-256:
`acaef2228b134dcdc20cad4ee79219482c927ec819aa3687b9b8a575c338817f`.
Native checkpoint/config/tokenizer checks are in `src/models/decider/native/src/checkpoint.rs`.
Reference runtime: Mapika/decider `50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f`,
decider-ai 1.8.1. The comparison harness verifies fixed hashes of all five imported
reference files before running either implementation and saves them in its protocol.

RTX 4090, sm_89, 24 GiB; driver 595.71.05, CUDA toolkit 13.0.88, Rust 1.98.1,
Python 3.12, Torch 2.14.0+cu130 and Transformers 5.17.0. The original eager baseline used one complete unpadded row at a time and ABI5.
The recorded dependent prefix campaign rebuilt the backend and consumers for ABI7; followup
controls below cover unchanged eager/Graph behavior and fixed/shared execution. The reference uses `DecisionModel.slot_logits`
with unpadded independent rows and disabled cache. Its missing causal-conv1d and
flash-linear-attention packages use Transformers' PyTorch reference fallbacks.
No optimized-reference speed comparison is claimed.

Before execution, the harness fixes these gates:

- Maximum absolute answer-probability and per-level fit drift: **0.02**.
- Maximum Score expectation drift: **0.1**.
- Identical Choice decisions when reference top-two probability margin is at least
  **0.05**; report all lower-margin disagreements.
- Exact prepared tokens, candidate IDs/order, readout positions, usage and answer
  identities. Complete responses retain all model-specific fields.

The current helpers record validation protocol **version 2** before execution.
The three fidelity gates above are unchanged. Both native and reference responses
must have the complete typed schema, finite non-boolean numeric values, valid
ranges and the pinned decimal rounding. Confidence, certainty and `x_p_max` are
checked against their formulas using the intervals implied by four-decimal
probability rounding (half a unit is 0.00005); Score confidence accounts for modal
ties hidden by that rounding. Score expectation also receives its two-decimal
formula check. These checks do not impose a common empirical tolerance on all
auxiliary fields.

For `n` Score levels, `fit_mass` must agree with the sum of reported level fits
within `(n + 1) * 0.00005`, accounting for rounding each fit and the sum. Its
reference drift bound is `n * 0.02 + 2 * (n + 1) * 0.00005`, derived from the
unchanged per-level gate and both response rounding allowances. The protocol saves
these methods explicitly. Historical inference records retain their original run
protocol; applying the revised validator to archived records is a CPU recheck,
not a new execution or evidence for the consolidated current head.

## Recorded eager baseline results

The 18-request corpus has **41 independent inference rows**. It covers Choice with
2/3/10/11/26/255 options, Score with 2/3/10 isolated levels, true/false and
criteria-only Noul, structured Unicode and long arrays, mixed questions, reversed
question order, repeated input, empty questions and long context. The longest
reference-compared row is 4,140 tokens.

Both the initial run and the source-hash-verified rerun pass the same gates:
maximum probability/per-level-fit drift **0.0119**, maximum Score drift **0.01**,
and **zero Choice disagreements**, including lower-margin cases. Raw logit drift
is reported separately (maximum 0.125); logits are not bit-identical. These are
observed results for this corpus, not a general numerical-error bound.

CPU opt-in tests check 12 pinned token/row/usage fixtures, every label through JT,
12 response assemblies including zero-fit Score, and state-prefix truncation.
Normal CPU tests cover selected-row ordering/padding, invalid shapes/IDs/nonfinite
weights, file content/size hashes, alternate index rejection and complete-row
validation before dispatch. A real GPU microtest checks the BF16 projection tie
`1 + 0.00390625 -> 1`, before FP32 calibration, and excludes the padded label.

Real model regression checks verify mixed answer order, exact repeated output,
validation-error recovery and empty requests without inference. The separate native GPU boundary test executes a **36,864-token complete row**
twice, rejects the next token before execution, and then reproduces an earlier short
request exactly after scratch growth. It does not expand the reference-parity claim
beyond 4,140 tokens.

HTTP testing passes **27 checks**: 18 direct/native diagnostic/frontend comparisons,
six invalid-input cases, the worker body limit, concurrent requests and recovery
health. The concurrent check makes 12 requests across direct and frontend paths
with four client threads; all returned bytes match the serial baseline. The first
HTTP harness run stopped on a Python variable-shadowing error after its earlier
checks; the corrected full run passes. The worker binaries were unchanged.

The test-only CUDA wrapper lets the genuine warmup projection complete, then fails
only the second `1 x 256 x 2048` head GEMM. The worker returns 503, health becomes
unavailable, two later requests are refused, and exactly two projections are seen.
Sampled per-process GPU residency drops from **4,192 MiB to 394 MiB**, demonstrating
retirement of the model allocation; these are two lifecycle samples, not peak-memory
or performance measurements. The wrapper delegates all other operations to the
real library and is never used by normal serving.

Cancellation semantics inherit the tested shared SerialScheduler. Decider-specific
caller cancellation and injected Rust panic are not separately hardware-tested.
CPU/Metal inference, other GPUs/precisions, dynamic batching, prefix reuse, Decider
Graph replay, model-quality datasets, throughput and cold/warm latency comparisons
are outside the original eager campaign; the followup checks below cover batching,
Graph and request-local prefix reuse.

## Reproduce

Build and prepare the checkpoint using [the native recipe](README.md). For the
reference environment, use Python 3.12 and the recorded Torch/Transformers versions;
install Transformers 5.17.0, tokenizers 0.22.2 and Torch 2.14.0 built for CUDA 13.0.
The native worker's preparation and serving require none of these Python packages.

Obtain the pinned reference in a separate directory:

```sh
git clone https://github.com/Mapika/decider.git /path/to/decider-reference
git -C /path/to/decider-reference checkout 50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f
```

From the System1-Omni repository root, with a reserved/available NVIDIA GPU:

```sh
cargo build --release --locked -p omni-decider-native --example decider-run
DECIDER_GRAPH=0 python tests/decider/verify_reference.py \
  --model /path/to/models/decider-2b-v11 \
  --reference /path/to/decider-reference \
  --binary "$PWD/target/release/examples/decider-run" \
  --library "$PWD/target/release/libqwen3_5_cuda.so" \
  --output /path/to/evidence/parity

python tests/decider/verify_http.py \
  --worker "$PWD/target/release/omni-decider" \
  --frontend "$PWD/target/release/omni-jev" \
  --model /path/to/models/decider-2b-v11 \
  --library "$PWD/target/release/libqwen3_5_cuda.so" \
  --parity-output /path/to/evidence/parity \
  --output /path/to/evidence/http
```

The HTTP runner starts local sockets on 18110/18111, checks readiness and shuts its
processes down even on failure. It retains worker/frontend logs and complete results.
Do not run it over unrelated processes already using these ports.

For the isolated failure test, build a wrapper beside the real library so its
runtime search path resolves the actual backend:

```sh
gcc -shared -fPIC -Wall -Wextra -I src/backends/cuda/qwen3_5 \
  tests/decider/fail_head.c -L target/release -Wl,--no-as-needed \
  -lqwen3_5_cuda -Wl,-rpath,'$ORIGIN' -ldl \
  -o target/release/decider-fail-head.so
python tests/decider/verify_retirement.py \
  --worker "$PWD/target/release/omni-decider" \
  --model /path/to/models/decider-2b-v11 \
  --wrapper "$PWD/target/release/decider-fail-head.so" \
  --output /path/to/evidence/retirement
```

This runner uses port 18112 and `nvidia-smi` per-process memory reporting. The recipe
also lists the explicit CPU and real-GPU Cargo tests. Ordinary workspace tests
compile/discover their ignored cases but do not establish hardware validation.

## Evidence

The task evidence archive retains both parity runs, the original HTTP harness
failure and its successful rerun, native/reference rows/logits/responses, protocol,
source/binary/library hashes, environment inventory, GPU tests and retirement logs.
Generated run output and binaries are kept out of the source tree; only maintained
fixtures and reproduction helpers are committed. The implementation PR links the
immutable evidence archive and its SHA-256. All requests are synthetic and carry
no private/customer content. A video is inapplicable to this text API integration.

## Request-local batching comparison

The followup runner `tests/decider/verify_modes.py` accepts a frozen JSON plan with
`model`, `library`, `parity_output` (the pinned reference run above),
`repetitions: 2`, `timed_cases` and `modes`. Each mode supplies `name`, an absolute
`binary` path and an `env` mapping. It hashes binaries, library, reference records
and comparison scripts before execution; retains complete records; checks prepared
rows exactly and the same response gates. Time is the diagnostic JSON-lines
roundtrip at concurrency 1, including processing, synchronization, serialization
and IPC. Startup to the empty diagnostic response and one workload warmup are
reported/excluded separately. These measurements do not establish HTTP latency,
peak memory or production throughput. Device memory is a whole-device sample at
the end of each mode, not a peak measurement.

On the RTX 4090 followup campaign, row limits 2 and 4 passed all 18 reference
requests. Row limit 8 failed the unchanged probability gate on structured Unicode
input (0.0208 > 0.02), so supported packing is capped at **4 rows**. The failed
campaign and narrowed-protocol rerun are retained in the PR evidence archive.
Defaults remain one row; do not extrapolate these small workloads to arbitrary
requests or hardware. Batch fault injection also verifies a two-row head failure
retires the complete worker after real startup warmup.

## Graph checks

`DECIDER_GRAPH=1` enables explicit backbone capture/replay. The ignored root
`tests/qwen3_5/graph.rs` test uses the pinned Decider checkpoint and compares eager
and graph hidden states exactly for ordered length vectors, token changes under
the same shape, scratch growth, replay and FIFO eviction beyond 64 shapes. Run:

```sh
DECIDER_MODEL=/path/to/models/decider-2b-v11 \
DECIDER_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" CUA_S1_GRAPH=1 \
  cargo test --release --locked -p omni-qwen3-5-native --test graph -- --ignored --test-threads=1
```

Set `require_replays: true` on each Graph mode in the comparison plan to require
actual captures and warm replays with zero fallbacks. The diagnostic runner preserves
these counters separately from the decision response. `tests/decider/fail_graph.c`
is a test-only wrapper that rejects capture startup while real CUDA eager inference
continues; `verify_graph_fallback.py` checks all pinned response cases and proves
one fallback, zero captures/replays, and effective eager execution. Build the wrapper
like `fail_head.c` above, then pass `--binary`, `--model`, `--wrapper`,
`--parity-output` and `--output` to that runner.

## Shared-prefix controls

The dependent prefix branch consumes the canonical continuation, fixed-GEMM and
executor implementation from Qwen PRs #98/#99 while preserving main's packed
prefill and Decider Graph statistics. The duplicate private continuation path and
its additional ABI8 entry points have been removed. Both workers use public
`PrefixState` and the same `run_window` implementation. The consolidated head
requires ABI7; current-head GPU/reference/HTTP/performance validation is unverified.
Rebuild all current-branch consumers with ABI7. The unchanged default GEMM handle serves the
normal/Graph path; fixed/shared modes use a separate no-split-K handle selected
at reference M64. Model tests compare shared branches bit for bit with independent
`forward_fixed` rows, including prefixes around 64-token boundaries, growth,
repetition, reordering, invalid IDs and interleaved Graph calls. Kernel tests now
include Decider's 2048 hidden width, 16 GDN heads and 8/2 attention heads alongside
the existing 4B/9B/27B shapes. Run ignored tests serially with `CUA_S1_CUDA_LIB`,
`QWEN3_5_MODEL`, `DECIDER_MODEL` and `DECIDER_CUDA_LIB` set to the ABI7 library
and pinned Decider checkpoint.

For response controls, put fixed4 before shared4 in the frozen modes plan and set
`exact_control: "fixed4"` and `require_shared: true` on shared4. Both use
`DECIDER_BATCH_MAX_ROWS=4` and Graph off; fixed4 sets `DECIDER_FIXED=1`, shared4
sets `DECIDER_PREFIX=1`. This verifies exact raw logits and complete responses
between fixed/shared as well as the original Transformers fidelity gates, and
requires actual request-local token savings. Keep normal eager4 in the plan to
measure algorithm-selection cost separately from prefix reuse.

The cached stock Qwen3.5-4B artifact has FP32 A_log and is not a valid merged BF16
worker export; its attempted legacy-loader check fails before inference and is
retained as unavailable coverage. The same constructor/explicit-mode regression
passes on Decider2B with the inherited CUA switch. Full Cua-S1/Open-Jev decisions
are not revalidated by the Decider campaign; they require their proper merged
exports and corresponding hardware.

The maintained `tests/decider/data/prefix-workloads.json` adds two targeted long
shared-context requests (mixed types and ten Score levels). Generate their pinned
reference records with `verify_reference.py --cases tests/decider/data/prefix-workloads.json`
and the same model/reference/binary/library/output arguments above. Then point
`verify_modes.py` at that parity directory, time exactly those two names, and keep
two repetitions per mode. This corpus is independent of the original 18-case run;
it resolves long-prefix reuse performance and does not replace broader fidelity
checks. The fixed/shared exact-logit gate also applies to measured repetitions.

## Followup measurements (diagnostic roundtrip)

The final bounded-row campaign measured `mixed` at 25.69 ms for single rows and
15.75 ms for 4-row packing; `score_10` at 51.03→29.44 ms. Independent Graph4 measured
`mixed` at 15.80→15.12 ms relative to matched eager4, while long-context single-row
latency was 89.76→89.73 ms. Each value is the median of two warm samples on the
same RTX 4090. These are narrow synthetic diagnostic timings, not HTTP latency
or a general speedup. Startup to the empty diagnostic record was measured
separately (approximately 5.5–6.4 seconds), with actual captures/replays recorded.

The ABI 7 shared-prefix comparison retains the fixed-algorithm control separately:

| Workload | Normal eager4 (ms) | Independent fixed4 (ms) | Shared4 (ms) |
| --- | ---: | ---: | ---: |
| Original mixed, short state |15.73|31.27|31.51|
| Original structured Unicode |32.43|47.57|41.44|
| Targeted long mixed, 5 rows / 17683 processed tokens |414.47|484.32|144.18|
| Targeted long Score10, 10 rows / 24460 processed tokens |576.17|672.89|151.96|

Every cell uses two warmed repetitions with raw min/max retained in the archive.
The long shared runs are bit-identical in raw candidate logits and complete
responses to their independent fixed controls, and satisfy the original reference
fidelity gates. The two long cases show 2.87× and 3.79× diagnostic-roundtrip speedup
against matched normal eager4; they do not establish task quality or a workload-wide
improvement. Short states can regress due to fixed GEMM selection and branch
launch overhead, so all optimizations remain opt-in. Shared and Graph are tested
as separate modes, not combined. End-of-mode whole-device memory samples for the
long corpus were 4626 MiB eager4, 4658 MiB fixed4 and 4746 MiB shared4; peak memory
and sustained throughput were not measured.
