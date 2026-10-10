# Open-Jev-27B-v1.1 H200 validation

## Raw HF Transformers comparison, 2026-10-03

With Open-Jev-27B-v1.1, the native Rust/CUDA worker delivers a **7.47× speedup over raw HF Transformers**
by mean warm HTTP latency: **362.21→48.50 ms (86.61% lower)** on one H200.
This comparison covers 74 real JevBench `noul` requests, one candidate each,
80–3399 tokens, BF16, max length 16384 and concurrency 1. Each backend reuses
one server for an excluded feasibility pass and two measured passes:
148 measured requests per backend.

| Configuration | Overall mean (ms) | Mean, pass 1 / pass 2 (ms) | P50, pass 1 / pass 2 (ms) | P95, pass 1 / pass 2 (ms) | Correct / 74 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw HF Transformers (unmerged LoRA; PyTorch fallback) | 362.209 | 362.238 / 362.180 | 324.822 / 321.321 | 651.999 / 652.911 | 64 |
| Native Rust/CUDA (cached RMSNorm + packed SiLU) | 48.503 | 48.471 / 48.535 | 25.797 / 24.895 | 215.696 / 218.882 | 64 |
| Original OpenJev-Fast | 50.936 | 51.097 / 50.775 | 24.651 / 24.622 | 248.121 / 251.193 | 63 |

Native and HF agree on all 74 thresholded decisions and both score **64/74**;
their maximum probability difference is **0.020423**. Fast scores 63/74 with
one different decision (`hard-opus-a-temporal_numeric-09`); maximum native/Fast
probability difference is 0.034353. Each backend's probabilities are exactly
unchanged across its feasibility and measured passes. These counts do not
establish statistical accuracy superiority or full numerical parity.

**Raw HF baseline:** the original Open-Jev server and `DecisionModel`, with
160 unmerged PEFT LoRA modules, the trained scalar head and saved temperature.
Full attention uses stock SDPA; runtime assertions verify all 48 linear-attention
layers use `torch_chunk_gated_delta_rule`, stock convolution and
`Qwen3_5RMSNormGated`. Because the shared environment contains optional kernels,
the baseline makes Transformers' `is_flash_linear_attention_available` and
`is_causal_conv1d_available` checks return false before importing model classes.
It uses no custom Fast model, `torch.compile`, CUDA Graph replay or prefix cache.
Native uses merged LoRA and eager execution (`CUA_S1_GRAPH=0`); original Fast
retains its custom kernels and CUDA Graph stack, reporting 30 retained graphs.
This comparison changes the complete backend; it does not isolate one optimization.

Timing includes localhost HTTP through the same frozen Rust frontend,
tokenization, worker execution and UTF8 response decoding. Client body
serialization and response JSON parsing are excluded. Downloads, preparation,
process-to-readiness, warmup, first inference after readiness and the complete
feasibility pass are excluded. No Nsight launcher or trace collection was used.
P50 is the median; P95 uses JevBench's `sorted[int(0.95*N)-1]` rank.
The observed mean pass ranges are disjoint; two-pass ranges are not confidence
intervals. Native's mean is 4.78% below Fast in this run, while Fast has a lower
median. Multi-candidate prefix sharing and broader JevBench coverage remain
unmeasured; native CUDA Graph replay is reported separately below. The author's
17.3 ms B300 result uses different hardware and workload.

### Frozen controls and reproduction

- Exact H200 GPU 2, UUID `GPU-cbf66259-f4ab-0ede-1811-82037dde5924`, NUMA 0,
  CPUs 0–15. The archived driver name is `NVIDIA L20X`, with 143771 MiB and
  SM90 / 132 SMs. The same device, affinity, requests, model and prepared
  environment are used for all three backends; no shared caches are dropped.
- Frozen native worker/frontend: `202c0e163f868334a99d88407056ebe61dbb2dce`;
  accepted packed-SiLU CUDA library SHA256:
  `e033315d4c67127809e62f41d991a149ac8ee0ce81827997af063815fd00d435`.
  PR source at measurement: `ad1cb818d2ea001bbd8c84643e1a7241d0fc26c8`.
- [Original Open-Jev](https://github.com/Zefan-Cai/Open-Jev/tree/3308a15ccd7eea1df7a37d6ddc39b023b801ba16)
  at `3308a15ccd7eea1df7a37d6ddc39b023b801ba16`. Fast, JevBench, request JSON,
  base, adapter and temperature use the same pins recorded in the October 2
  controls below and the [native recipe](native.md).
- Actual runtime: Torch 2.13.0+cu130, CUDA 13.0, Transformers 5.10.2,
  PEFT 0.19.1. Reuse prepared weights and SM90 extensions offline.
- Use `gpu run --gpu-ids 2 --wait 10m --timeout 45m --note <label> --`, then
  `numactl --membind=0 --physcpubind=0-15`. Run raw HF, native and Fast,
  preserving the 74-case request order, with one excluded feasibility pass
  and exactly two measured passes per configuration.

Raw commands, request/token IDs, timing rows, source snapshots and 48 verified
input hashes are archived locally in the benchmark worktree's
`profile/jev-hf-transformers-comparison-20261003/`, outside this PR.
A collector cleanup assertion rejected native's intentional SIGTERM exit after
all HF/native measurements were saved. Only the remaining Fast configuration
continued on a second reservation of the same GPU and affinity; no measured
passes were repeated or added. All task-owned processes exited and GPU 2 returned
to 0 MB used. The October 2 timings below are a separate experiment.

## Native CUDA Graph replay, 2026-10-03

Warm graph replay reduces mean HTTP latency from **48.086 to 47.112 ms (2.03%)**
on the same 74-case workload. The shared backend now retains up to 64 exact-length
graphs, enough for these 57 distinct lengths. Its previous eight-entry cache
regressed the mixed workload because evicted lengths require another eager
forward and graph capture. Graph mode remains opt-in with `CUA_S1_GRAPH=1`.

| Workload | Configuration | Mean (ms) | Mean, pass 1 / pass 2 (ms) |
| --- | --- | ---: | ---: |
| Fixed 107-token request | Eager | 19.829 | 19.835 / 19.823 |
| Fixed 107-token request | Graph, eight entries | 18.882 | 18.902 / 18.863 |
| Fixed 107-token request | Graph, 64 entries | 19.081 | 18.968 / 19.194 |
| 74 mixed-length requests | Eager | 48.086 | 48.062 / 48.110 |
| 74 mixed-length requests | Graph, eight entries | 95.289 | 95.425 / 95.153 |
| 74 mixed-length requests | Graph, 64 entries | 47.112 | 47.050 / 47.174 |

All 74 probabilities and decisions remain exactly unchanged (maximum delta 0.0).
The 64-entry candidate also reduces the fixed-short mean by 3.77%. Both measured
passes improve over eager for both workloads, satisfying the prespecified 3%
short and 2% mixed mean gates. The mixed improvement only narrowly exceeds its
gate; two passes are observed variability, not confidence intervals or evidence
for a general workload winner. Observed device memory after mixed passes is
50,947 / 50,967 / 51,089 MB for eager / graph-eight / graph-64: 122 MB more for
64 entries than eight. These are scheduler samples, not peak-memory measurements.

This is a separate native A/B experiment, not a newly measured HF/Fast comparison.
It keeps H200 GPU 2, UUID, NUMA affinity, BF16, max length 16384, model export,
CUDA library, frontend and request order fixed. Both graph capacities are built
with identical rustc options and the frozen `202c0e1` dependency artifacts;
their source copies differ only in the cache limit. Eager uses the eight-entry
binary with graph mode disabled. Each configuration validates its first long
inference after readiness to allocate the workload's maximum scratch size, then
validates the short request. One server is reused for an excluded 32-request
short feasibility pass and two measured 32-request passes, followed by an excluded
74-case feasibility pass and two measured 74-case passes. HTTP timing has no
Nsight launcher and uses the same boundary as the raw HF comparison above.

New-length capture cost remains significant: excluded mixed feasibility means
are 50.130 / 97.694 / 110.255 ms for eager / graph-eight / graph-64. These are
single feasibility observations, not measured cold-latency comparisons. Scratch
growth invalidates graphs, and more than 64 distinct lengths can still evict them.
No model downloads, cache drops or clock changes occur. All owned processes exit
and GPU 2 returns to 0 MB used.

A preceding experiment proves warm replay in Nsight Systems: one graph launch,
zero recaptures and zero individual runtime kernel-launch calls per short request,
with the same 834 kernels. Node-level graph tracing reports larger gaps despite
lower unprofiled HTTP latency; use unprofiled measurements for the speedup.
[NVIDIA documents graph-node tracing overhead](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-graph-trace).
Hardware counters and per-SM utilization remain unmeasured.

Raw plans, source copies, build commands, requests/responses, traces, memory
samples and verified input hashes are archived outside this PR in
`profile/jev-cuda-graph-20261003-074613/` and
`profile/jev-cuda-graph-cache64-20261003-075238/`. Run the native worker with
`CUA_S1_GRAPH=0/1` under the same reservation and affinity documented above;
warm the maximum workload length and all tested lengths before measured passes.
For new comparisons, declare the budget first and retain capture costs whenever
they occur inside measured requests.

## Packed-SiLU A/B, 2026-10-02

A matched comparison of 74 real JevBench `noul` requests, each with one candidate,
measured packed MLP SiLU with cached residual RMSNorm fixed in both native variants.
All 74 native decisions and probabilities were exactly unchanged by packed SiLU;
the native worker scored 64/74 correct. OpenJev-Fast scored 63/74, with one different
decision (`hard-opus-a-temporal_numeric-09`) and maximum native/Fast probability
difference 0.03435. These counts do not establish statistical accuracy superiority.

### Warm HTTP latency

| Configuration | Mean, pass 1 / pass 2 (ms) | P50, pass 1 / pass 2 (ms) |
| --- | --- | --- |
| Native with cached residual RMSNorm; scalar SiLU | 49.563 /49.490 | 25.434 /25.539 |
| Native with cached residual RMSNorm and packed SiLU | 48.286 /48.307 | 24.925 /24.657 |
| Original OpenJev-Fast | 57.345 /50.908 | 25.026 /24.770 |

Packed SiLU reduces native mean 49.527→48.297 ms (2.483%). Its two-pass mean range
is disjoint from baseline's. These are observed two-pass ranges, not confidence
intervals. Fast's 6.436 ms mean spread prevents claiming a stable aggregate winner.
This subset does not measure multi-candidate prefix sharing. The author's 17.3 ms
B300 result is a different hardware/workload measurement.

HTTP includes the same Rust frontend, tokenization and worker execution; excludes
client body serialization and response JSON parsing, includes UTF8 decoding. One
server per configuration is reused. Real warmup, first request after readiness and
one complete 74-case feasibility pass are excluded; two subsequent passes are
measured. Nsight collection is inactive during HTTP passes, although CUPTI
instrumentation may remain loaded. No shared caches are dropped or clocks changed.

### Separate CUDA timelines

Each entry is the two-trace mean of 64 MLP SiLU launches per request, in milliseconds.

| Tokens | Scalar SiLU | Packed SiLU | Fast SiLU lookup |
| --- | --- | --- | --- |
| 107 | 0.680 | 0.305 | 0.359 |
| 936 | 5.643 | 1.746 | 1.513 |
| 3399 | 21.217 | 6.380 | 5.466 |

Long-request MLP SiLU decreases 69.93%, closing 94.19% of that measured kernel
family's gap to Fast. Other kernel families vary between traces; their separate
timeline totals are not HTTP latency. Fast uses padded/tree layouts and lookup
tables, so this does not imply identical rows, fusion boundaries or arithmetic.

The prespecified acceptance gates were five GPU tests, unchanged native decisions,
maximum probability delta ≤0.01, ≥25% long SiLU duration reduction and ≥2% warm HTTP
mean reduction with disjoint observed ranges. All passed; no extra measured runs
were added. Final integration also passed the six-test shared CUDA ABI 4 suite.

### Frozen controls and reproduction

- Device: exact scheduler GPU 2, UUID `GPU-cbf66259-f4ab-0ede-1811-82037dde5924`,
  NVIDIA H200, 143771 MiB (reported as `NVIDIA L20X` in the archived device metadata);
  CUDA driver and Nsight identify SM90 / 132 SMs. NUMA 0, CPUs 0–15.
- BF16, max length 16384, HTTP concurrency 1, native `CUA_S1_GRAPH=0`. Fast retains
  its original graph/kernel stack. Build with nvcc 13.0.88, SM90, `-O3 -std=c++17
  -lineinfo`; use the same cuBLASLt/runtime libraries for both native variants.
- Frozen native worker/frontend: `202c0e163f868334a99d88407056ebe61dbb2dce`.
  Both native libraries include the cached RMSNorm source; only `elementwise.cu`
  differs for packed SiLU. The shared CUDA ABI remains 4.
- [JevBench](https://github.com/fstandhartinger/jevbench)
  at `f8ce71361165846101d02ebc83ad44e47ae44fc3`; select its 74 `noul` cases, preserving
  request bodies and order. Frozen request JSON SHA256:
  `0c756a7b0b4c1f1352225f2e01770b5b3a0646fe86d5cc6930ded9e292ef37df`.
- [OpenJev-Fast](https://github.com/lyuyiqi/open-jev-fast)
  at `c52b8bb958c1f0d241d4eb7fce4ecd8d885bf1e4`; original server/model/kernels, with
  prepared SM90 extensions, PyTorch 2.13cu130/Triton 3.7.1, Transformers 5.10.2,
  PEFT 0.19.1 and FLA 0.5.2. These differ from the author's B300 environment.
- Model/export uses the pinned base, adapter and calibration in the
  [native recipe](native.md); temperature 2.5343690298472983. Reuse prepared weights
  and extensions. Keep copies, downloads, compilation, warmup and process-to-
  readiness separate from measured execution.

Reserve the exact device with `gpu run --gpu-ids 2 --timeout 45m --note <label> --`,
then bind the command with `numactl --membind=0 --physcpubind=0-15`. Run baseline,
packed native and Fast once each, reusing each server for one excluded feasibility
and two measured passes. Separately trace two requests at each 107/936/3399 tokens
per configuration with Nsight Systems CUDA/node tracing: 18 traces total.

Raw bodies, timing rows, SHA256 manifests, reproduction commands,
18 `.nsys-rep`/SQLite pairs and plots are archived locally in the benchmark
worktree's `profile/jev-single-candidate-silu-pack8-20261002/`, outside this PR.
Nsight Compute counters are denied by the host policy. Geometry,
register/shared-memory metadata and CUDA timelines are available; achieved
occupancy, per-SM tails, stalls, Tensor Core utilization and bandwidth/cache
efficiency are unmeasured. No hardware-cause claim follows from timeline data alone.
