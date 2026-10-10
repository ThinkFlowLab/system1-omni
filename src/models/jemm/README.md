# JEMM native worker

The Rust/CUDA worker serves ordered `choice`, `score` and `noul` questions over
text and up to four PNG/JPEG/WebP images. See the [recipe](../../../recipe/jemm/README.md)
for download, export, launch and validation. CPU inference and native Metal are
unsupported. Multi-image reference parity has a known limitation below.

## Pinned artifacts

| Artifact | Revision |
| --- | --- |
| ypcypc/JEMM reference | `6822fe0fd53c5e6670af6ba99fb2c857a661e532` |
| MaestroYan/JEMM adapter | `76e3c209e8441fa658221c7ba2725bad2f811176` |
| Qwen/Qwen3.8-27B base/tokenizer/processor | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

The exporter verifies pinned file SHA-256s and tensor inventory, merges language
LoRA in FP32 into cloned BF16 weights and extracts 32 rows from the untied LM
head. `jemm_export.json` records every exported file's SHA-256. Startup verifies
complete file/shard coverage before CUDA initialization; keep files immutable
while serving. Calibration loads from checksum-covered `decision_config.json`; temperatures
must be finite and positive, threshold finite and within [0,1], and all values
must agree with the export manifest. The pinned artifact values are
1.3480874159655591 (text), 1.3954832341582943 (images) and 0.9872681877423998
(threshold). The threshold is provenance and
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

## Optional CUDA Graph replay

`CUA_S1_GRAPH=1` enables language replay, with separate 64-entry FIFO caches
for text ordered lengths and multimodal `[tokens]`. `CUA_S1_VISION_GRAPH=1`
independently enables vision replay; its four-slot FIFO uses exact `[T,H,W]`
grids. Both switches default off. Current IDs/features/positions/pixels are
uploaded before replay. An eager miss completes before recording, preserving its
answer. Capture failure logs unconditionally, disables that graph path and keeps
the eager result. Eviction synchronizes before graph/buffer destruction.

Vision scratch persists for up to four grids even with graphs disabled, changing
resident activation memory from per-call allocation. This bounds grid slots,
not weights or GEMM plan memory. Startup's image warmup can now capture its own
shape when enabled; it does not prewarm every user grid. These mechanisms do not
establish lower latency on every workload or fix the multi-image numerical gap.

## Validation and known multi-image limitation

The historical A800 BF16 corpus (13 requests / 16 questions, text and single-image
inputs) passed exact tokens/positions/grids/pixels and predeclared probability
0.02 / Score 0.1 / winner-margin 0.05 gates. Max probability drift was 0.0015338802.
These results are scoped to the source revisions and corpus in
[the evidence release](https://github.com/Levius-Fubuki/system1-omni/releases/tag/jemm-a800-20261008).

**Two- and four-image reference parity is not established.** The supplemental
2/3/4-PNG-image feasibility corpus exceeded the unchanged 0.02 probability gate
for two and four images, with max drift **0.042138323189940485**; three images
passed. All nine question records had exact official tokens, positions, grids,
candidate counts and FP32 pixels. #118 and #122 produced identical answers and
usage for these requests. This narrows investigation to execution/merge numerics,
but does not identify the cause. Candidate repeated rounds stopped after the
failed feasibility gate; failures were retained. The worker accepts these input
classes, but applications requiring verified multi-image fidelity must validate
their own workloads or use the pinned reference. No tolerance was relaxed.

Root `tests/jemm/` covers prompt order/coercion, candidate limits, the 64-question
resource boundary, response math, image decoding/positions, tokenizer and token
budgets, export integrity, BF16 head padding, error-envelope bounds, warmup image
decoding and synchronized retirement. The real CUDA head test is ignored in CPU
CI. CPU tests do not prove full-model parity or execute the startup image forward.
No fresh GPU campaign or performance claim accompanies the review fixes.
