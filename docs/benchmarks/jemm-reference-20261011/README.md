# JEMM 1–4-image numerical regression

The optional reference-numerics path passes the frozen 16-request / 25-question
corpus on one NVIDIA A800 80 GB. Maximum probability drift is
**0.0166172279642417**. The unchanged gates are probability/confidence/noul 0.02,
Score expected value 0.1, and winner agreement when the reference margin is at
least 0.05. Low-margin cases are retained. Two- and four-image results agree to
floating-point host-softmax noise; the maximum is the three-image Choice case.

This is a correctness campaign, not a throughput or latency benchmark. Each
candidate was declared before its one complete pass. Failed diagnostic
candidates remain in the evidence archive. No tolerance was widened and no
failed case was removed. Broader prompts, other accelerators/framework versions,
labelled task accuracy, speed and peak memory are unverified.

## Reproduce

The maintained [corpus](../../../tests/jemm/data/parity/corpus.jsonl) and
[official reference oracle](../../../tests/jemm/data/parity/reference.json) are
consumed by [the HTTP validator](../../../tests/jemm/parity.py). Images are small
synthetic deployment-console fixtures; request text contains no customer data.
The corpus SHA-256 is
`82c05de9cb21b7763864dde6e80cc6bed588219d4d53da2d8157c0ec27a3663c`.

Follow the [JEMM recipe](../../../recipe/jemm/README.md) to export the pinned BF16
base and separate FP32 rank-16 adapter, build with cuDNN 9, and start the worker.
The worker verifies all checkpoint hashes. Use an exclusively reserved GPU and
submit the corpus sequentially (concurrency 1):

```sh
python tests/jemm/parity.py --url http://127.0.0.1:8000 \
  --corpus tests/jemm/data/parity/corpus.jsonl \
  --reference tests/jemm/data/parity/reference.json \
  --report /tmp/jemm-parity.json
```

The reference oracle uses JEMM source
`6822fe0fd53c5e6670af6ba99fb2c857a661e532`, base
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, and adapter
`76e3c209e8441fa658221c7ba2725bad2f811176`. Its active GDN implementation is FLA,
not the FP32 fallback function whose name appears in the Transformers wrapper.
The measured environment used Torch 2.12.1+cu130, Transformers 5.17.0,
FLA 0.5.2, Triton 3.7.1, CUDA 13.0 and cuDNN 9.

## What was checked

| Layer | Result and scope |
| --- | --- |
| Full native forward | 16 requests / 25 questions pass, graphs disabled, no explicit warmup or retries. |
| Actual HTTP worker | The same corpus passes after the worker's two required startup forwards. Text-language and vision graphs enabled; multimodal language remains eager. Health READY before/after; no capture error. |
| Reference kernels | 11 fixed-seed checks: GDN and language attention at 63/278/475 tokens, RMSNorm at 1/278 rows, vision attention at 256/512 patches, and patch convolution. Every BF16 output equals the pinned framework; graph replay equals eager output. |
| Existing CUDA API | All 11 checkpoint-free kernel regressions pass, including prefix continuation and capture recovery. The checkpoint-dependent prefix test is not included. |
| Selected label head | Real CUDA constant-row projection test passes. |
| Compatibility build | CUDA library builds with and without cuDNN; reference numerics require the cuDNN-enabled variant. |

The [operator test](../../../tests/jemm/reference_kernels.py) requires the pinned
Torch/FLA environment. The existing CUDA tests remain under
[`tests/qwen3_5/kernels.rs`](../../../tests/qwen3_5/kernels.rs). Model-scale 4B
checkpoint execution was not repeated; legacy source entry points are preserved
and their kernel regressions were run.

## Evidence and review scope

[Raw evidence archive](https://github.com/Levius-Fubuki/system1-omni/releases/tag/jemm-reference-20261011)
contains native/HTTP responses, predeclared protocols, exact library/binary/source
archive hashes, operator results, build/test logs, and failed diagnostic records.
The measured source is parent `3a2ff031daf6e2a441902ff34cdf6efe72f3d24b` plus the
reference API integration recorded in `production-source-v1.tar.gz`; later
comment, documentation and test additions do not change that execution path.
The source archive, not a later untested commit label, identifies the run.

The fix couples export semantics, projection shapes and arithmetic: preserve
BF16 base/FP32 adapters, use original component GEMM dimensions, aggregate vision
projection rows with image-local attention, and select reference CUDA functions
only for JEMM. Review in that order, then inspect model resource ownership,
response reconstruction and tests. Existing Qwen API callers retain legacy
arithmetic. The reference path does not support prefix caching.
