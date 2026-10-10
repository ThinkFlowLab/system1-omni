# JEV-27B-VL experimental native worker

This integration targets **autotrust/JEV-27B-VL**, with a native Rust/CUDA
language backbone and an offline Hugging Face vision encoder. The reviewed
integrated candidate passed the frozen-corpus H800 checks; it remains experimental
and has not completed broad quality or clean-install validation. See the [recipe](../../../recipe/jev_vl/README.md)
and [evidence and limitations](../../../recipe/jev_vl/validation.md).

## Pinned reference and ownership

The checkpoint, tokenizer, `adapter_vllm`, calibration and `serve_decide.py`
reference are from
[autotrust/JEV-27B-VL at f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc](https://huggingface.co/autotrust/JEV-27B-VL/tree/f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc).
The model card declares Apache-2.0. Prompt rendering and verbalizer selection
follow that release's `serve_decide.py`; source attribution, adapted portions and
the license are recorded in the [third-party notices](native/THIRD_PARTY_NOTICES.md).
The exporter records checkpoint-index, adapter, decision-head and calibration
hashes in `jev_vl_export.json`.

The language executor reuses the shared Qwen3.5/3.8 implementation established by
[Open-Jev PR #55](https://github.com/ThinkFlowLab/system1-omni/pull/55).
This checkpoint's verbalizer head and prompt differ from
ZefanCai/Open-Jev-27B-v1.1's independent-candidate scalar head. It is also distinct
from [openjev/openjev issue #95](https://github.com/ThinkFlowLab/system1-omni/issues/95).
[Cua-S1 PR #64](https://github.com/ThinkFlowLab/system1-omni/pull/64) supplies
upstream native vision infrastructure; this JEV worker currently consumes
preencoded embeddings and does not call that native vision tower.
[Shared-observation RFC #85](https://github.com/ThinkFlowLab/system1-omni/issues/85)
is related design context, not this model's acceptance specification.

## Request and response contract

`POST /v1/systemone` accepts the reference's **single-question** shape:

```json
{"kind":"choice","state":"The package arrived damaged.","question":"What should support do?","options":["Offer a replacement","Close the ticket"]}
```

This differs from the `questions`/`answers` envelope used by other workers. The
frontend transports this body unchanged; generic clients must use this worker's
schema. It exposes only the health and decision routes; unsupported generation
routes receive the frontend's own 404. An optional `model` must equal
`autotrust/JEV-27B-VL`.

| Kind | Candidate set and output |
| --- | --- |
| `choice` | 2–256 string options in request order; the first 16 use trained A–P slots, later labels use the exported single-token table with zero extra slot bias. |
| `noul` | Fixed options `"false"`, `"true"`. |
| `score` | Fixed options `"0"` through `"5"`; the selected score is the highest-probability category, not an expected value. |

Responses include `kind`, `effective_kind`, `options`, `probabilities`,
`choice_index`, `choice`, `adaptation`, `protocol`, `model`, `usage`,
`elapsed_seconds` and `num_model_requests`. Ties select the earliest option.
`usage` counts the full expanded prompt even on cache hits and reports one
completion token for the decision; no autoregressive decoding takes place.

The raw prompt is `[kind] …\n[state] …\n[question] …\n[options]\n…\n[decision]:`.
It uses no chat template. Structured state preserves Python-style JSON rendering;
list parts concatenate without an inserted separator. Each image part inserts
`<|vision_start|><|image_pad|><|vision_end|>` before image-token expansion.

The default exported limit is 16,384 expanded tokens. Export permits 1–32,768;
larger limits have not been benchmarked. HTTP bodies are limited to 4 MiB.
`thinking=auto/on`, tournament and permutation strategies, generation, audio,
video, dynamic batching and unprepared images are unsupported. `strategy=auto`
uses the single-pass path in this worker. The historical corpus covers only
2–16 choice options and one image, so the wider accepted range and multiple
images do not have full-checkpoint parity evidence.

## Execution, layouts and lifetimes

`processing.rs` validates and tokenizes one question and prepares either text
IDs, full multimodal inputs, or a cached-prefix continuation. Image assets hold
BF16 `[image_tokens, 5120]` adapted embeddings and their patch grid. The
processor constructs expanded token IDs and three position axes; image rows
replace placeholder-token embeddings. All vectors describe one unpadded prompt,
not a GPU batch.

`executor.rs` owns checkpoint loading, the model forward and selected output-head
rows. The final 5120-element hidden state is downloaded and projected against
FP32 merged head rows using FP64 accumulation. Per-kind probabilities are
`softmax((label_logit + slot_bias) / temperature)`; the common vocabulary
log-normalizer cancels. Response assembly stays in the processor. This is a
mathematical readout equivalence, not a claim of bitwise equivalence to vLLM.

The shared `SerialScheduler` admits one complete question per executor. A model
mutex protects mutable CUDA state; the scheduler retains admission until the
blocking forward and readout finish. Model-owned prefix snapshots contain
full-attention K/V, FP32 Gated DeltaNet state and convolution history. A
continuation owns references to its snapshot and image assets until execution
finishes. Restart the worker and use a fresh asset directory when changing
weights, tokenizer or preprocessing; caches are scoped to a loaded worker.

Three configurable caches retain processor prefix geometry (L1), parsed
preencoded image assets (L2), and language-prefix device state (L3). The current
L3 path targets a compatible single-image prefix aligned to a 64-token chunk;
other inputs use a full forward. L2 avoids file parsing and copying of prepared
assets; it does not run or cache a live vision encoder. Cache sizes and controls
are listed in the [recipe](../../../recipe/jev_vl/README.md#cache-controls).

Tests are registered under [`tests/jev_vl/`](../../../tests/jev_vl/) and
[`tests/qwen3_5/`](../../../tests/qwen3_5/). CPU tests establish contract and
bookkeeping behavior; ignored checkpoint/kernel tests and paired full-model
validation are separate gates.
