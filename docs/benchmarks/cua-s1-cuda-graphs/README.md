# Cua-S1 exact-length CUDA Graph experiment

This compares the native worker from PR #19, baseline
`8367333a5d0e115c4c4ad366b67a91a0553449ce`, with the Graph patch in this PR.
The same modified executable and CUDA library serve both configurations:
`CUA_S1_GRAPH=0` (eager) versus `CUA_S1_GRAPH=1` (Graph replay). There is no
GEMM tuning or change to kernel arithmetic. Since these measurements, the patch
has added capture-failure error cleanup, regression tests, comments and
documentation. The latency benchmark was not rerun for that recovery fix.

## Protocol and controls

The hypothesis was that Graph replay reduces CPU launch overhead on repeated
prompt lengths. Success required exact response equality and repeatable latency
reduction beyond run variability. Stop conditions were any failed correctness
check, 120-second readiness timeout, or 20-minute reservation timeout. No runs
were added after inspecting results.

- One scheduler-reserved GPU 3: NVIDIA L20X, compute capability 8.9; CUDA 13.0;
  driver 570.133.20; Rust 1.98.1. CPU affinity was inherited (CPUs 0–223) in both configurations;
  NUMA placement was not explicitly bound. Other GPUs had unrelated workloads.
- Pinned Qwen3.5-4B base `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Cua-S1
  text adapter `16818868b0cc7813808aae4e87b417657046ab79`, merged BF16 weights.
  Both modes reused exactly the same export. Export dependencies were torch
  2.13.0+cu129, transformers 5.14.1, peft 0.20.0, tokenizers 0.22.2,
  safetensors 0.8.0, huggingface-hub 1.30.0: an existing environment, differing
  from the recipe pins. The native runtime does not use these Python packages.
- One real model-loaded/warmed server per mode, reused for correctness,
  feasibility and two measured runs. No cache clearing. HTTP directly to the
  worker, concurrency 1; no frontend or Python baseline comparison.
- Each case: three warmups, twenty measured requests per run. A separate
  feasibility run with one request per case was excluded from measured results.
- Seventeen correctness requests: fourteen fixtures followed by the positive
  fixture with `Submit` replaced by `Cancel` in its state, the one-option
  fixture, then the changed fixture again. This exercises changed token ids,
  scratch growth and later shorter prompts, eviction and recapture. Scratch
  capacity only grows; shorter prompts reuse that allocation. All seventeen response bodies
  matched exactly, including probabilities and choices.

## Reproduction

Use the build and merged-weight steps in
[the native recipe](../../../recipe/cua_s1/native.md), reserving one exact GPU
through the host's scheduler for all device work. Retrieve the author's frozen
benchmark and earlier fixtures:

```sh
curl -fL https://raw.githubusercontent.com/twu3202/system1-omni/f9ab3fc808e27d92d26c45c9179ee2a0faf57963/recipe/cua_s1/bench_text.py -o /tmp/bench_text.py
curl -fL https://raw.githubusercontent.com/twu3202/system1-omni/a086a316babe796d41ba96cc4b4be0cc5123f15d/tests/cua_s1/data/text_inputs.json -o /tmp/text_inputs.json
```

For each mode, start one worker with `CUA_S1_GRAPH=0` or `1`, wait for its real
`/health`, validate the first inference, and run:

```sh
python /tmp/bench_text.py --direct http://127.0.0.1:8000 --inputs /tmp/text_inputs.json --warmup 3 --repeat 1 --out feasibility.json
python /tmp/bench_text.py --direct http://127.0.0.1:8000 --inputs /tmp/text_inputs.json --warmup 3 --repeat 20 --out run1.json
python /tmp/bench_text.py --direct http://127.0.0.1:8000 --inputs /tmp/text_inputs.json --warmup 3 --repeat 20 --out run2.json
```

The runner uses only Python's standard library. Preserve both measured runs;
compare the same case across modes. [results.json](results.json) contains all
measured samples, warmups, correctness summary and startup metadata. Full
response bodies are omitted; the correctness summary records 17/17 identical
responses.

## Results

P50 HTTP milliseconds, showing both measured runs:

| Prompt tokens | Eager | Graph replay |
| ---: | ---: | ---: |
| 139 | 6.31 / 6.24 | 5.73 / 5.74 |
| 154 | 6.60 / 6.60 | 5.82 / 5.87 |
| 218 | 7.25 / 7.27 | 6.63 / 6.93 |
| 292 | 8.58 / 8.69 | 7.99 / 8.01 |
| 712 | 16.17 / 16.11 | 16.04 / 16.07 |
| 15,446 | 355.10 / 355.05 | 355.50 / 355.73 |

For these short cases (139–292 tokens), the reduction in the mean of run
medians is 6.6–11.4%. The long case has no meaningful gain. Graph capture adds
first-use work: the first post-readiness fixture took 25.23 ms eager and
33.66 ms with Graphs. Process-to-readiness was 5.58 / 4.93 seconds respectively;
these single observations are not controlled startup comparisons. Export time
is excluded and the files were already prepared for both modes.

This is one GPU and a fixed fixture workload, not a general speedup claim.
Mixed lengths can evict captures; first-use and capture costs are excluded from
warm request measurements. Full fp32-reference accuracy, concurrent serving,
other CUDA versions and other hardware were not evaluated for this change.
