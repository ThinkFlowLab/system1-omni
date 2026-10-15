# JEMM native worker

The Rust/CUDA worker serves ordered `choice`, `score` and `noul` questions over
text and up to four PNG/JPEG/WebP images. See the [recipe](../../../recipe/jemm/README.md)
for download, export, launch and validation. CPU inference and native Metal are
unsupported. Numerical validation is scoped to the frozen corpus below.

## Pinned artifacts

| Artifact | Revision |
| --- | --- |
| ypcypc/JEMM reference | `6822fe0fd53c5e6670af6ba99fb2c857a661e532` |
| MaestroYan/JEMM adapter | `76e3c209e8441fa658221c7ba2725bad2f811176` |
| Qwen/Qwen3.8-27B base/tokenizer/processor | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

The `jemm-native/2` exporter verifies pinned file SHA-256s and tensor inventory,
retains the 17 original BF16 language shards byte for byte, and preserves the raw
FP32 adapter separately (496 rank-16 pairs, scale 2). Its language-only index
retains original `model.language_model.*` keys. Vision tensors and 32 selected
rows from the untied LM head are extracted without merging adapter weights.
`jemm_export.json` records every exported file's SHA-256 and the adapter contract.
Startup verifies original shard and adapter pins before CUDA initialization;
legacy premerged `jemm-native/1` exports are explicitly rejected. Keep files immutable
while serving. This head requires the pinned temperatures 1.3480874159655591
(text), 1.3954832341582943 (images) and threshold 0.9872681877423998. Follow-up
#122 loads validated calibration from the checksum-covered decision config;
its calibration policy is a separate change. The threshold is provenance and
does not gate SystemOne responses.

## Request and response contract

`POST /v1/systemone` takes `state`, a nonempty ordered `questions` object and
optional base64 `images` (bare strings or data URLs). Missing state is empty;
structured state uses compact insertion-ordered JSON. Instructions/descriptions
use Python-style coercion and whitespace flattening. Candidate order selects
labels `A`–`Z`, then `0`–`5`; Choice and Score accept 2–32 candidates.

- Choice returns the first argmax label, probabilities and max-probability confidence.
- Noul orders yes/no and returns `noul = p(yes)`, probabilities and confidence.
- Score returns zero-based level probabilities, confidence and `expected_value`.

Every question gets one complete prompt; vision features are shared only within
that request. Usage sums complete prompt lengths, including image tokens, for
every question. `output_tokens` is zero; `latency_ms` is included in usage.
Text/image prompt limits are 8192/3072 tokens with no truncation. Each source
image is limited to 3,145,728 pixels. Native preparation additionally bounds
compiled prompt text to 16 MiB and question count to 64, before device work.

The **64-question limit is a native resource bound**, not an expansion of the
official HTTP bundle limit. At the pinned revision, `DecisionModel.respond`
calls `_decide` and `systemone.single_requests`, which iterate independent
questions without the eight-question cap. `MAX_BUNDLE_QUESTIONS = 8` applies
only to `contract.bundle_text`/`encode_bundle`, which this endpoint does not use.
The native bound deliberately rejects larger independent HTTP requests.

JSON duplicate keys, nonfinite values, integers outside i64/u64, mismatched
model names and excessive nesting are rejected. Integer `-0` normalizes to 0;
floating `-0.0` remains a float. Native Python repr has a known short-escape gap
for control characters inside container-shaped instruction text.

The HTTP body limit is 16,000,000 bytes (413 above it); invalid preparation
returns 422. JSON bodies are parsed regardless of Content-Type, like the pinned
reference. Errors have `error` (ValueError/TypeError/RuntimeError) and `detail`
bounded to 200 Unicode characters. Native retirement returns 503 rather than
the reference's 500, and some validation exception classifications/messages
differ. `/health` returns `{"status":"READY","model":"JEMM"}` only after
both a text and a synthetic-image warmup have completed. Startup errors prevent
listener binding; inference errors/panics retire the model and make health 503.

## Ownership and lifetimes

`contract.rs` renders prompts and reconstructs calibrated answers;
`processing.rs` validates the entire request, decodes/transforms images, tokenizes
unpadded rows and constructs three-axis positions. Prepared inputs own token
IDs, candidate counts, image positions/indices and shared image patches.
Response context owns question identities, input usage, calibration and timing.

`executor.rs` owns shared Qwen language/vision models, selected BF16 head weights
and CUDA state. Vision runs once per request, language once per question, and
head GEMM rounds to BF16 before FP32 host softmax. One `SerialScheduler` admission
covers the entire request. Both streams synchronize before resource release;
caught execution errors/panics clear atomic readiness and retire loaded state.
There is no cross-request batching or prefix cache.

## Validation

The unmerged FP32-adapter v2 path passes the frozen **16 requests / 25 questions**
covering text and 1–4 images on A800. Maximum probability drift is
**0.0166172279642417**, under the unchanged 0.02 gate. Score tolerance remains
0.1 and winner agreement is required at reference margins of at least 0.05;
low-margin cases remain in the corpus. See the
[requests, reference outputs and evidence](../../../docs/benchmarks/jemm-reference-20261011/README.md).
These results do not establish parity for every possible prompt, GPU or framework version.

JEMM opts into separate reference CUDA functions. They preserve the active
framework's BF16 boundaries, TF32 matrix-product order, normalization reductions,
FlashAttention split combination and cuDNN patch convolution. FP32 adapters stay
unmerged, component projections keep their original GEMM shapes, and multi-image
vision projections use aggregate rows while attention remains image-local.
Existing Cua/Open-Jev/JEV-VL CUDA entry points retain their previous behavior.

The historical premerged v1 corpus passed text/single-image checks but failed the
supplemental two-/four-image gate (maximum drift 0.042138323189940485).
That failed evidence remains in the
[original release](https://github.com/Levius-Fubuki/system1-omni/releases/tag/jemm-a800-20261008);
it is not evidence for the new v2 path.

Root `tests/jemm/` covers prompt order/coercion, candidate limits, the 64-question
resource boundary, response math, image decoding/positions, tokenizer and token
budgets, export integrity, BF16 head padding, error-envelope bounds, warmup image
decoding and synchronized retirement. The real CUDA head test is ignored in CPU
CI. CPU tests do not prove full-model parity or execute the startup image forward.
No fresh GPU campaign or performance claim accompanies the review fixes.
