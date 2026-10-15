# Cua-S1 text native/reference parity on A800

This investigation follows the five decision flips reported in
[PR #141](https://github.com/ThinkFlowLab/system1-omni/pull/141).
The original L20X run used a different GPU. On A800, the frozen 139-request corpus
produced two stable flips, not the original five. Two independent HTTP passes
per worker reproduced both flips. This report qualifies the A800 environment;
it does not establish that all five original L20X cases have been reproduced.

## PR dependencies and evidence provenance

The independently qualified source `27f77e49868d84b091aa804f02ccb4d3474a9a00`
was stacked on [#118](https://github.com/ThinkFlowLab/system1-omni/pull/118)
at `ed9f9d6eb566d226e2441c5e9d84187abe1e4d63` for the unmerged FP32 language
adapter loader and optional reference norm/GEMM/attention operations.
The 2026-10-15 parent synchronization used #118 at
`3be46fa270cb74c33d4db4ffd761405d0522018a`, which inherits
[#117](https://github.com/ThinkFlowLab/system1-omni/pull/117) and its merged
resident-feature increment #135. Rebuild the library and all consumers together
for ABI 8. The later ABI-8 source `7293c71c` and legacy-readout repair `f212c56f`
have separate source-pinned A800 campaigns described below. Decider modes and the separate #127 cache
increment are not included. The frozen corpus comes from #141; merging #141 is
not a runtime dependency.

The historical cumulative results in the Results section were measured on candidate
`76af07996f60905082b8d48199d4a7003f33f4f6`, whose parent is
`18e1a5a6f68b9d11d3e11c7ea7b939af87272f53`. That integrated build used ABI 8.
The separate independent qualification at `27f77e49` used ABI 6 with additive
optional FP32 GDN symbols. No cuDNN is needed for this text path. Its exact
source/library/binary hashes, repeated HTTP results and self-review are recorded
in [PR #143](https://github.com/ThinkFlowLab/system1-omni/pull/143).
Both campaigns remain historical, source-pinned evidence. ABI 8 in the cumulative
campaign does not establish qualification of the current ABI-8 branch; use each
run's source pin rather than substituting either result for a newer build.

## Frozen inputs and environment

- Baseline source: `3e9cc33a61ddf110d97512fb6f296cf540377c61`.
- Corpus: PR #141 `869260649f7dd5b56cac4d91eaae9087f2e985ee`,
  `benchmarks/e2e/choice.jsonl`; 139 requests/questions, 172–3988 prompt tokens.
- Historical GPU candidate parent: cumulative ABI-8/unmerged-LoRA shared executor,
  `18e1a5a6f68b9d11d3e11c7ea7b939af87272f53`.
- GPU: A800 PCIe 80 GB, sm_80; CUDA toolkit 13.0.88; NVIDIA driver 595.71.05.
- Python 3.12, Torch 2.14.0+cu130, Transformers 5.17.0, PEFT 0.21.0;
  TF32 disabled in the independent reference; no FLA or causal-conv1d installed.
- Base `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
  text adapter `cua-ai/cua-s1-4b-0.2@16818868b0cc7813808aae4e87b417657046ab79`.
  Pinned files were SHA-256 checked. The new export copies the original adapter
  unchanged, with hash `9b59c5aed96171a50b26526613766bbf44347a5c7af70f81efe6bcc6e9dbfb0e`.

The reference worker uses the repository's independent Transformers/PEFT text
implementation. No reference logits, labels or probabilities are modified.
Requests are sent through the Rust frontend to a real worker. The missing model
field in corpus rows is filled with the required `cua-s1-4b-0.2`, as the benchmark
wire adapter does. Raw request IDs, bodies, statuses, usage and model identities
are retained with source/build/launch manifests in the local investigation archive.

## Confirmed numerical causes

Token IDs matched on all 139 requests. The differences begin in execution:

1. The old export merges FP32 LoRA into BF16 base weights. PEFT instead computes
   the unmerged LoRA branch in FP32 and casts the combined projection to BF16.
2. The old GEMM reduction policy and normalization reductions differ from the
   pinned framework. With identical first-layer MLP down-projection input,
   244,845 BF16 base outputs differed; the existing reference GEMM handle matched.
   With identical normalization input, two BF16 values differed; reference norm
   matched after adding the 2560-hidden shape.
3. The old GDN uses TF32 pair products and BF16 chunk intermediates/state;
   the no-FLA Torch implementation uses FP32 chunked triangular solves and state.
   On identical first-layer input, old relative L2 error was 0.00217711 and
   326,687 of 954,368 BF16 outputs differed. Keeping recurrence FP32 alone did not
   match the reference's chunk arithmetic. Matching the chunked computation and
   Torch's cumulative-decay scan association produced identical BF16 outputs.
4. Native letter projection previously emitted FP32 logits. Reference `lm_head`
   emits BF16 before casting to FP32 softmax. Restoring that BF16 boundary matters
   for tied options and removes the remaining readout differences.

The fix selects the precision path using `cua-s1-text-unmerged/1`. Legacy merged
exports remain loadable with their original FP32 readout; BF16 letter-logit
rounding is restricted to the unmerged text profile. Other model adapters retain their existing numerical
mode. Prefix continuation is rejected in the new mode; eager is the default.
The probability gate of 0.02 was not relaxed.

## Legacy merged upgrade regression

The [2026-10-15 A800 ABI-8 evidence](https://github.com/Levius-Fubuki/system1-omni/releases/tag/cua-a800-abi8-20261015)
compares parent `5bdd625b4c27efb02f43a1f71150e1a7d7c0f071` with candidate
`7293c71cfb738504a30ff033ae3567f3368333c1`, using the same legacy merged artifact
and frozen 139-request corpus for two passes each. The candidate changed four
decisions, with maximum probability change 0.02441754937171936; correctness fell
from 97/139 to 96/139. Each changed decision collapsed the two leading options
into equal probabilities, then selected the earlier option in request order:

| Request | Legacy choice | Rounded merged choice | Frozen task label |
| --- | --- | --- | --- |
| `original-routing-06-1` | `tools` | `general` | `general` |
| `hard-opus-a-temporal_numeric-07` | `sep_28` | `sep_24` | `sep_26` |
| `hard-sol-b-long_policy-05` | `delete_profile_omar_april_and_table` | `delete_everything` | `delete_profile_omar_april_and_table` |
| `hard-sol-b-temporal_numeric-03` | `9_credits` | `12_credits` | `9_credits` |

The cause is the text executor's unconditional BF16 conversion of its FP32 dot
products. The model's merged loader still selects the old backbone arithmetic;
the unmerged reference-profile operations are opt-in. The compatibility fix
keeps the validated `TextExport` on each executor and applies BF16 readout only
to `Unmerged`. `Merged` retains the parent's FP64 accumulation followed by FP32
readout. Softmax, option order, tie handling and screenshot execution are unchanged.
Restoring parent behavior reverses the candidate's one task-label improvement
and two regressions; the fourth case is wrong under both choices. Compatibility
is measured against the parent, independently of task labels.

A CPU regression first failed with `[20.0, 20.0]` instead of
`[20.03125, 20.046875]`, then passed with the profile-specific readout. Tests also
cover both profiles' FP64 accumulation, nonfinite rejection, finite FP32 values
outside BF16 range, and the four requests' option order through real response
finishing with explicitly synthetic BF16 projection inputs. These synthetic
vectors are not captured checkpoint activations. The maintained four-request
device test uses the actual frozen inputs and parent responses; run it as in the
[native recipe](../../../recipe/cua_s1/native.md).

The [repair GPU campaign](https://github.com/Levius-Fubuki/system1-omni/releases/tag/cua143-merged-readout-f212c56f-20261015)
tested source `f212c56f23ac750ebfa38f4c72e9fcdc914bf1cd` on pinned A800/ABI 8.
First, two four-request passes on rejected source `7293c71c` reproduced all four
changes. The repaired four-request Engine regression then passed twice per
Graph setting, with zero probability error. Graph 1 logged 6 captures, 2 replays
and no fallback; the test verified the nine artifact files' SHA-256 before loading.

| Fresh full-corpus mode | Passes × requests | Comparator | Maximum probability difference | Decision changes | Correct |
| --- | --- | --- | --- | --- | --- |
| Legacy parent | 2 × 139 | Parent | 0 | 0 | 97/139 |
| Repaired merged eager | 2 × 139 | Parent | 0 | 0 | 97/139 |
| Repaired merged Graph | 2 × 139 | Parent | 0 | 0 | 97/139 |
| Repaired unmerged eager | 2 × 139 | Python reference | 5.960464477539063e-8 | 0 | 97/139 |
| Repaired unmerged Graph | 2 × 139 | Python reference | 5.960464477539063e-8 | 0 | 97/139 |
| Python reference | 2 × 139 | Reference | 0 | 0 | 97/139 |

Every repaired merged response body and content type matched the parent byte
for byte. All modes repeated identical response bytes; unmerged eager/Graph
bodies also matched each other. Each full-corpus Graph worker logged 207
captures and 73 replays, with no fallback. The original 0.02/zero-flip gates
were retained and legacy compatibility additionally required byte identity.
Counts are 1,668 full-corpus HTTP measurements, 8 rejected-source reproductions,
7 excluded feasibility requests and 16 separate direct Engine predictions.
Initial bootstrap rejected a packaging-version mismatch before any GPU request:
Torch metadata is `2.14.0` while runtime is `2.14.0+cu130`; both fields and CUDA
runtime `13.0` were then checked explicitly. The failure is preserved. No source
or tolerance changed during the campaign. Raw evidence was verified locally
before the server was shut down. Later documentation-only updates preserve all
non-Markdown source and fixture bytes from the tested repair commit.

Legacy compatibility remains distinct from reference parity: restoring the
original merged behavior does not remove its known approximation relative to
unmerged FP32 LoRA. These results do not establish full 27B, L20X/Hopper,
FLA-enabled, performance or general GUI task-quality qualification.

## Results

| Real HTTP mode | Passes × requests | Flips vs reference | Maximum probability difference | Correct |
| --- | --- | --- | --- | --- |
| Baseline native | 2 × 139 | 2 | 0.04872608184814453 | 97/139 |
| Fixed eager | 2 × 139 | 0 | 5.960464477539063e-8 | 97/139 |
| Fixed Graph | 2 × 139 | 0 | 5.960464477539063e-8 | 97/139 |

Every fixed request returned HTTP 200. Each mode's two passes produced identical
response bytes; model identity, usage and option/question order matched reference.
Graph trace independently records 207 captures and 72 replays, with no fallback.
The two baseline flips were `original-routing-06-1` (tools/general) and
`hard-sol-b-temporal_numeric-03` (9_credits/12_credits); both reference maxima
are ties. Accuracy staying at 97/139 is expected: this repairs implementation
parity and does not retrain or improve the model.

A controlled full-model probe additionally matched all 139 final BF16 hidden
vectors and selected BF16 logit rows exactly. The slight HTTP probability drift
comes after that BF16 boundary in FP32 softmax/response finishing.

Fourteen maintained GPU cases compare the new GDN with the pinned no-FLA Torch
implementation: 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 233, 513 and 2049 tokens
with 32 value/16 key heads, plus 129 tokens with 4 value/2 key heads. All BF16
outputs matched exactly. Repeated changed-input execution resets state, invalid
shapes/scales fail, and repeated context creation/destruction releases scratch.

## Reproduction and limits

Use the [native text recipe](../../../recipe/cua_s1/native.md) to build sm_80,
export original FP32 LoRA separately and start the worker. Use the pinned
[reference text recipe](../../../recipe/cua_s1/text.md) in an environment without
FLA/causal-conv1d, then replay the frozen corpus twice through each worker/frontend.
Run the maintained operator regression in that reference environment:

```sh
.venv/bin/python tests/qwen3_5/test_gdn_fp32.py \
  --library target/release/libqwen3_5_cuda.so
```

The historical integrated candidate passed Cargo fmt, full-workspace Clippy
with warnings denied, full-workspace tests and
release build with `--locked --features omni-laya/serve` where applicable.
Backend contract checker/tests, 21 pinned Python text contract/server tests, CUDA
operator regressions and strict MkDocs build also passed. The checkpoint-dependent
prefix test used canonical merged text weights with the original pinned multimodal
config for a zero-image contract fixture; it is not numerical-reference evidence.

Other GPUs, framework versions, FLA execution and screenshot-model numerical
parity require separate qualification. Independent-head evidence remains
source-pinned in #143 rather than inferred from these historical results. This corpus does not establish general
GUI accuracy or a performance improvement.
