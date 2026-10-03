# Gated DeltaNet preparation on H200

Lossless TF32 operand packing and direct U/W stores reduce complete native GDN
call latency by about 13% at 936 and 3,399 tokens. The same change gives only a
0.42% mean HTTP latency reduction on the 74-request Open-Jev workload, below the
predeclared 2% end-to-end target. All compared model probabilities and decisions
are unchanged. These are separate kernel and serving results.

## Implementation

The preparation kernel previously kept normalized Q/K in float32 shared memory
and converted each loaded WMMA fragment to TF32. It now converts each component
once and stores the 19 meaningful TF32 bits in 16-bit and 3-bit planes, using
three bytes per component. This preserves the exponent range and TF32 operands.
Decayed Q/K still come from the original float32 normalization values.

Explicit `mma.m16n8k4` keeps the four-term accumulation used by the original
WMMA path in the measured SM90 build. Using `m16n8k8` changed intermediate
rounding despite identical TF32 operands; that candidate was rejected. U/W
accumulators are packed directly to bfloat16 instead of passing through an 8 KiB
float32 shared staging area. Gate loads are prefetched in groups of eight while
retaining the sequential float32 sum. Dynamic shared memory falls from 92 KiB
to 72 KiB. The inverse stays float32, and the workspace, C ABI, state and output
kernels keep their existing contracts.

## Fixed controls and run budget

Measured on 2026-10-03, on one H200, SM90, 132 SMs. The driver reports this
device as `NVIDIA L20X`; H200 is the corrected hardware label used here.
All runs used scheduler-reserved GPU 2, UUID
`GPU-cbf66259-f4ab-0ede-1811-82037dde5924`, NUMA node 0 and CPUs 0–15.
The driver was 570.133.20 and native libraries were built with CUDA 13.0 `nvcc`.
No shared caches were dropped or clocks changed.

The only native A/B variable was GDN preparation. The baseline source is main
at `58b8cbe9d4738f6c217f6fe584852dc9361b2c24`; its GDN source also matches PR #55.
The [archived plans and manifest](artifacts/20261003/manifest.json) freeze the
working patch, source/library hashes, commands, dependencies and original paths.
Builds and copies were completed before measurements.

Standalone inputs use batch 1, 16 Q/K heads, 48 value heads and head dimension
128. Q/K/V/beta are bfloat16 and log gates are float32. Seed 20261003 generates
correlated Q/K with weak decay, using PyTorch 2.13.0+cu129. The token counts are
representative workload shapes; the tensors are synthetic inputs rather than
captured model activations. Both libraries use the same workspace and stream.

Each shape/configuration has one excluded feasibility run and ten excluded
warmups, then exactly two measured passes of 100 calls. Pass 2 reverses arm
order. The complete call includes preparation, state and output kernels. Eager
CPU submission is included in wall time; CUDA event spans also include gaps
between submissions and must not be read as active kernel time.

## Complete GDN call latency

Wall time per call, in milliseconds:

| Tokens | Baseline pass 1 / 2 | Candidate pass 1 / 2 | Reduction pass 1 / 2 |
| --- | --- | --- | --- |
| 107 | 0.051291 / 0.050976 | 0.050914 / 0.050957 | 0.74% / 0.04% |
| 936 | 0.247565 / 0.247337 | 0.214176 / 0.214178 | 13.49% / 13.41% |
| 3,399 | 0.800775 / 0.798935 | 0.696768 / 0.697262 | 12.99% / 12.73% |

The predeclared kernel gate was at least 10% reduction at both longer shapes
in each pass, with no short-input regression. It passed. Short-input differences
are too small to support a meaningful speedup claim. Raw wall/event timings and
bitwise workspace comparisons are in [results.jsonl](artifacts/20261003/kernel/analysis/results.jsonl).

Feasibility checks compare every initialized U/W/QD/KD/PB/decay element by its
bit representation; all changed counts are zero. Final outputs also match.
The recurrent float64 reference retains its original tolerance and now covers
tile/chunk boundaries, grouped heads, 936/3,399 tokens, weak decay, zero Q/K,
and amplitudes 1e-9 and 1e-18. All three GPU tests, including the existing
attention and Graph tests, passed; see the [test log](artifacts/20261003/kernel/analysis/recurrent-reference.log).
These checks establish parity for the sampled inputs on this device, rather
than a proof of identical results for all inputs or architectures.

## Warm HTTP latency

The serving comparison reuses the prepared merged BF16 Open-Jev-27B-v1.1
checkpoint, temperature 2.5343690298472983, maximum length 16,384, one Noul
candidate and concurrency 1. The same 74 requests span 80–3,399 tokens. The
source JevBench revision is `f8ce71361165846101d02ebc83ad44e47ae44fc3`.

Both arms use the same frozen Rust worker/frontend from
[PR #55](https://github.com/ThinkFlowLab/system1-omni/pull/55), worker revision
`202c0e163f868334a99d88407056ebe61dbb2dce`. ABI4 library sources are frozen at
`7b935723c6fa17145e3d866675202fb6f7ea1f5d` except for the candidate GDN
translation unit. The branch on main uses an ABI3 library for its Rust GPU
tests; that library was not substituted into the ABI4 Open-Jev worker.

Model base revision is `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, and checkpoint
revision is `28cf73067d5b337860bbef3c85b8b82ba8730956`. Prepared weights and
tokenizer were reused. Graph capture and shared-prefix caching were disabled.
No Nsight/CUPTI instrumentation was active. The HTTP timer includes receiving
and decoding the response body, and excludes request JSON serialization and
response JSON parsing.

Each arm reuses one worker/frontend pair. Genuine readiness and the first
validated inference are followed by one excluded 74-request feasibility pass
and exactly two measured passes. Baseline precedes candidate. Worker readiness
took 11.63 s and 10.62 s respectively; these startup times are excluded.

| Configuration | Pass 1 mean | Pass 2 mean | Aggregate mean |
| --- | --- | --- | --- |
| Baseline | 48.375 ms | 48.441 ms | 48.408 ms |
| Candidate | 48.105 ms | 48.305 ms | 48.205 ms |

All 74 decisions, probabilities and token counts match the pinned native
reference in feasibility and both measured passes. The mean reduction is
0.42%; it misses the 2% gate despite both candidate passes being faster.
With two repetitions and this small difference, the result does not establish
a material end-to-end gain. The [summary](artifacts/20261003/e2e/summary.json)
and four adjacent JSONL files preserve every measured response and latency.
Task-owned processes exited and the GPU reservation was released.

This is an incremental comparison against native Rust/CUDA. Raw HF Transformers
and open-jev-fast were not remeasured in this campaign, so prior HF speedups must
not be presented as a new matched comparison for this kernel change.

## Ready kernels in vLLM

At vLLM revision `5f30fc7031cae49bf51073fc953d419b08f8887c`, the
[Qwen GDN dispatcher](https://github.com/vllm-project/vllm/blob/5f30fc7031cae49bf51073fc953d419b08f8887c/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)
selects FlashInfer on SM90 with `auto` or `flashinfer`. Its in-tree CuTe prefill
backend is opt-in and targets SM10x. The
[post-convolution preparation kernel](https://github.com/vllm-project/vllm/blob/5f30fc7031cae49bf51073fc953d419b08f8887c/vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py)
fuses layout preparation, Q/K normalization and gating.

FlashInfer provides [SM90 fused and context-parallel prefill](https://github.com/flashinfer-ai/flashinfer/blob/7eb86aa0fdc1248fab43c89801de4ed450e35e77/flashinfer/gdn_prefill.py).
The fused CuTe kernel uses TMA, WGMMA and warp specialization; the context-parallel
path splits work across sequence chunks. Python/CuTe entry points need an AOT
export and C ABI adapter for Rust; a native integration was not implemented.

The installed FlashInfer 0.6.16.post3 / CUTLASS DSL 4.6.2 comparison includes
required Q/K normalization, gate exponentiation and beta conversion. These
versions differ from the researched upstream revisions. Complete eager wall
time, in milliseconds:

| Tokens | Native pass 1 / 2 | FlashInfer fused pass 1 / 2 | FlashInfer CP pass 1 / 2 |
| --- | --- | --- | --- |
| 107 | 0.0521 / 0.0518 | 0.4190 / 0.4379 | 0.8345 / 0.8161 |
| 936 | 0.2482 / 0.2481 | 0.4392 / 0.4305 | 0.8468 / 0.8274 |
| 3,399 | 0.8019 / 0.8001 | 0.4259 / 0.5080 | 0.8382 / 0.8566 |

The fused path helps the longest shape, with substantial observed variation,
and is slower for the other two complete calls. CP is slower at these shapes.
This is a separate standalone experiment against the original native kernel;
it establishes neither full-model FlashInfer fidelity nor Rust integration
performance. [Raw results](artifacts/20261003/flashinfer/results.jsonl) include
numerical comparisons. No hardware-counter evidence attributes the differences
to a particular stall, occupancy or bandwidth cause.

## Reproduction and rejected variants

Build the current native library and register the GPU test binary without
initializing CUDA:

```sh
NVCC=/usr/local/cuda/bin/nvcc src/backends/cuda/qwen3_5/build.sh /tmp/gdn-current 90
cargo test --release --locked -p omni-cua-s1-native --test kernels --no-run
```

After checking the host scheduler, reserve an available exact device and run
the reference tests (GPU 2 is the recorded experiment device, not authorization
on another host):

```sh
gpu run --gpu-ids 2 --timeout 10m --note "GDN reference validation" -- \
  numactl --membind=0 --physcpubind=0-15 \
  env CUA_S1_CUDA_LIB=/tmp/gdn-current/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-cua-s1-native --test kernels -- \
    --ignored --nocapture --test-threads=1
```

The [frozen kernel runner](artifacts/20261003/kernel/harness/compare.py) records
the exact RNG, workspace comparisons and timing loop. For another run, copy it
into a fresh `<run>/harness/` directory, create `<run>/analysis/`, and write
`<run>/plan.json` with `libraries.baseline`, `libraries.candidate`,
`validation_library`, `kernel_test_binary` and a SHA256 `hashes` map for the
frozen files. Build a baseline library from the recorded main revision. The
runner's device/affinity checks match the recorded controls; adapt and record
them when changing hosts. Run `compare.py --probe` before device work, then use
the scheduler command preserved in [plan.json](artifacts/20261003/kernel/plan.json).
The serving plan preserves all worker/model paths and hashes; reproducing that
comparison also requires the frozen PR #55 worker and prepared 74-request
manifest. Archived result directories must not be reused for new measurements.

Earlier experiments remain in the local archives named in the manifest:

- TF32 tensor-core off-diagonal inverse: 2–4% core gain, below the kernel gate.
- BF16 pair products: unit reference passed, but a probability near 0.5 changed
  one model decision during feasibility; no measured candidate HTTP passes.
- RN-even FP16 pair products: probability drift exceeded 0.01 during feasibility.
- RNA-rounded FP16 pair products: one model decision changed during feasibility.
- Float32 staging with direct stores: exact comparisons passed, but 7–8% core
  gain missed the kernel gate; no end-to-end comparison was run for that variant.
- Packed TF32 with eight-term MMA: intermediate bitwise comparisons failed;
  timing stopped before measured passes. Restoring four-term MMA resolved the
  compared differences and produced the final results above.

No failed variant was promoted by relaxing the numerical gates or extending
its measured run budget. Hardware profiling counters were unavailable to this
account. SM80/SM89/SM90 compilation passed; current device execution and
performance evidence cover SM90. The four Rust CI commands also passed; their
exact commands and build outcomes are recorded in the manifest.
