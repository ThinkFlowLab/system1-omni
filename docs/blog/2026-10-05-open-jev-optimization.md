---
title: "Open-Jev: 7.47× Faster Inference on H200"
date: 2026-10-05
author: "Hongsheng Liu"
summary: "362.21 → 48.50 ms versus raw HF on H200, with separate RMSNorm, SiLU, CUDA Graph and GDN A/B results."
tags:
  - performance
  - cuda
  - open-jev
---

# Open-Jev: 7.47× faster inference on H200

System1-Omni's native Rust/CUDA backend runs Open-Jev-27B-v1.1 at
**48.50 ms mean warm HTTP latency**, versus **362.21 ms with raw HF Transformers**.
That is **7.47× faster**, saving **313.71 ms per request**.

One H200 · BF16 · concurrency 1 · 74 real JevBench `noul` requests/pass ·
one candidate/request · 80–3,399 tokens · two measured passes per arm.

![PR 55's total HF-to-native gain and separate RMSNorm, SiLU and CUDA Graph A/B results](../assets/blog/open-jev-20261005/pr55-attribution.svg)

*Complete backend change on the left; isolated tuning pairs on the right.
Each pair has its own baseline: these gains cannot be added. Dots are two pass means.*

## The same-H200 comparison — PR #55, October 3

| Backend | Mean HTTP (ms) | Pass 1 / pass 2 (ms) | Correct / 74 |
| --- | ---: | ---: | ---: |
| Raw HF Transformers | 362.209 | 362.238 / 362.180 | 64 |
| Native Rust/CUDA, eager | **48.503** | 48.471 / 48.535 | 64 |
| OpenJev-Fast | 50.936 | 51.097 / 50.775 | 63 |

Native's mean is **4.78% below Fast**; Fast has the lower median.
HF/native match all 74 decisions, with maximum probability difference **0.020423**.
Fast differs on one decision. These are subset results.
[Three-backend figure](../assets/blog/open-jev-20261005/backend-http.svg).

Raw HF uses unmerged PEFT LoRA, stock SDPA and PyTorch linear-attention/conv
fallbacks; optional FLA/causal-conv dispatch is disabled. Native uses merged
weights and custom CUDA kernels; Fast uses its own kernel/graph stack.
[Frozen setup](../../recipe/open_jev/validation.md#raw-hf-transformers-comparison-2026-10-03).

## What PR #55 changes

| Optimization | How it removes work |
| --- | --- |
| Native prefill | Shared CUDA GDN/attention kernels and grouped projection GEMMs |
| LoRA merging | Compute adapter updates once at export; remove runtime low-rank projections |
| Attention-gate fusion | Sigmoid/multiply in the attention epilogue; remove a launch and intermediate write/read |
| Cached RMSNorm | Keep residuals in registers across reduction; avoid rereading them |
| Packed SiLU | Load/store eight BF16 values at a time; preserve `expf` and rounding |
| CUDA Graph replay | Submit a captured forward with one graph launch |

Native-only, LoRA-only and gate-only timings remain **unmeasured**.
RMSNorm and SiLU each pass their **≥2% HTTP / ≥25% kernel-family** gates;
all 74 native probabilities/decisions stay unchanged.
[Kernel A/B figure](../assets/blog/open-jev-20261005/pr55-isolated-ab.svg).

Code: [PR #55](https://github.com/ThinkFlowLab/system1-omni/pull/55),
[RMSNorm/SiLU](https://github.com/ThinkFlowLab/system1-omni/commit/61b83b380baca912c0dbd989d175cb3da8a881ae),
[LoRA export](https://github.com/ThinkFlowLab/system1-omni/blob/61b83b380baca912c0dbd989d175cb3da8a881ae/recipe/open_jev/export_merged.py),
[attention gate](https://github.com/ThinkFlowLab/system1-omni/blob/61b83b380baca912c0dbd989d175cb3da8a881ae/src/backends/cuda/qwen3_5/attention.cu).
The executor and graph ancestry are [#19](https://github.com/ThinkFlowLab/system1-omni/pull/19)
and [#52](https://github.com/ThinkFlowLab/system1-omni/pull/52).

## CUDA Graphs: fit the cache to the workload

**57 token lengths → 64 cache entries.** Eight entries repeatedly evict and
recapture graphs. Warm replay reduces host submission; the short trace still
executes the same **834 GPU kernels**, through one graph launch.

![Graph-cache results for mixed lengths and a fixed 107-token request](../assets/blog/open-jev-20261005/graph-cache.svg)

*Graph-64 saves **2.03% mixed / 3.77% short** versus matched eager controls.
The eight-entry mixed regression stays visible. Dots: two pass means.*

Graph mode is opt-in (`CUA_S1_GRAPH=1`). New lengths pay capture cost;
tokenization, transfers and CPU scoring stay outside capture.
All 74 native outputs match. [Capacity change](https://github.com/ThinkFlowLab/system1-omni/commit/3685d2037c6910a10ffa033db2262f734ba3fc11)
and [protocol/traces](../../recipe/open_jev/validation.md#native-cuda-graph-replay-2026-10-03).

## Gated DeltaNet preparation — PR #68, October 3

Pack TF32 Q/K once and write U/W directly as BF16. Shared memory drops from
**92 → 72 KiB**, preserving four-term accumulation and rounding.

![GDN complete-call gains beside the separate HTTP A/B result](../assets/blog/open-jev-20261005/gdn-kernel-http.svg)

*About **13% faster** at 936/3,399 tokens, but only **0.42% HTTP improvement**.
The ≥10% longer-call gate passes; the ≥2% HTTP gate is missed. Graphs disabled.*

All 74 native outputs match. BF16/FP16 alternatives failed numerical checks;
eight-term TF32 changed intermediate bits. October 4 integration passed six
GPU tests without a new latency measurement.
[PR #68](https://github.com/ThinkFlowLab/system1-omni/pull/68) ·
[raw runs, rejected variants and integration](../../benchmarks/gdn/README.md).

## Processing and runtime — PRs #78 / #80, October 5

[#78](https://github.com/ThinkFlowLab/system1-omni/pull/78) separates prepare/execute/finish;
[#80](https://github.com/ThinkFlowLab/system1-omni/pull/80) adds FIFO admission before
blocking dispatch. Both pass regression gates; neither establishes a speedup.

![Independent Open-Jev regression comparisons for PRs 78 and 80 at concurrency 1, 8 and 16](../assets/blog/open-jev-20261005/pr-regression-ab.svg)

*Synthetic multi-question/candidate requests, direct-worker HTTP; separate from
the JevBench frontend timer. Two 64-request passes per arm/model/concurrency.*

Each campaign: **1,536 requests** across Open-Jev/Cua-S1 · **zero failures** ·
responses match except elapsed-time metadata · mean/p95/throughput gates pass.
GPU batching remains planned.

## Setup, evidence and next measurements

The JevBench timer includes localhost HTTP, tokenization, worker execution and
response-body decoding. Preparation, readiness, first inference and warmups are
excluded; graph runs warm maximum scratch before measurement.

Next: a matched **raw HF → merged-LoRA HF → native scalar/unfused → fused gate →
cached RMSNorm → packed SiLU → Graph-64** ladder. Intermediate stages and
**merged GDN + Graph** remain unmeasured; prefix sharing needs its own comparison.

[Evidence ledger](../assets/blog/open-jev-20261005/source-data.json): frozen
revisions, device UUIDs/affinity, controls, per-run values and gates.
[Timing export](../assets/blog/open-jev-20261005/ab-passes.jsonl):
**78 measured passes / 5,040 timings** with provenance/parity hashes.
Full response/protocol/trace archives remain local. These October 1–5 results
are historical measurements, inspected against `f594d7d`; current main is untimed.

Regenerate figures without GPU use:

```sh
python docs/assets/blog/open-jev-20261005/plot.py
mkdocs build --strict
```

[Inference setup](../../recipe/open_jev/native.md). Presentation references:
[OpenJev-Fast](https://yiqilyu.me/open-jev-fast/),
[Qwen3-Omni](https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization),
[Kimi K3](https://vllm.ai/blog/2026-09-13-kimi-k3-performance-optimization).
