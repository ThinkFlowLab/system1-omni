# Valen-Preview-0923 reference contract

This directory contains the System1-Omni reference integration for
Valen-Preview-0923. It is a reference-serving boundary only: the Python worker,
processor, executor, and postprocessor serve the pinned text and single-image
`choice`, `noul`, and `score` questions. Native CUDA execution is not part of
it.

## Pinned artifacts

| Artifact | Revision |
| --- | --- |
| Valen source | [750bfcfbb48a5275534a9c912257ebe83ca57a97](https://github.com/Liuziyu77/Valen/commit/750bfcfbb48a5275534a9c912257ebe83ca57a97) |
| Valen checkpoint | [Valen-Team/Valen-Preview-0923](https://huggingface.co/Valen-Team/Valen-Preview-0923) |
| Preview checkpoint revision | `81b9c63` |
| Preview checkpoint SHA-256 | `836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012` |
| Qwen3.5-2B base revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |

The checkpoint is the upstream `checkpoint.pt` plus `config.json` format. It
does not contain the frozen Qwen base model. The published config uses
`stage=vision_top`, `dtype=bf16`, `projection_dim=256`, and `max_length=8192`.

## Current serving scope

- `state` is either one inline PNG/JPEG image (`{"image": data URL}`) or text
  (a non-empty, non-whitespace string; objects and arrays are serialized to
  JSON text, matching the shared text-state wire convention);
- image limits: single-frame PNG/JPEG, at most 4 MiB, 2048 pixels per side,
  1,048,576 pixels, a 200:1 aspect ratio; text state, instructions and
  criteria are limited to 16,384 characters;
- `instructions` must be non-empty after trimming; criterion keys and texts
  must be non-empty after trimming (a null criterion text defaults to its key);
- `choice`, `noul` and `score` questions, at most 8 per request, freely mixed
  against one shared state;
- `choice`: up to 255 named candidates per question;
- `noul`: fixed `true`/`false` candidates; `criteria` is optional and may
  replace either side's wording (`{"true": ..., "false": ...}`, null keeps
  the reference's fixed wording);
- `score`: an ordered list of 2 to 10 level descriptions in `criteria`;
- no video, multi-image, or remote URL inputs;
- no token generation: the decision head returns candidate logits directly.

The public request keeps the Cua-S1-compatible shape:

~~~json
{
  "model": "valen-preview-0923",
  "state": {"image": "data:image/png;base64,..."},
  "questions": {
    "move": {
      "type": "choice",
      "instructions": "Choose the next move.",
      "criteria": {"up": "Move up", "down": "Move down"}
    }
  }
}
~~~

A text-only request replaces the image object with plain text:

~~~json
{
  "model": "valen-preview-0923",
  "state": "The card was charged twice for one order.",
  "questions": {
    "refund": {
      "type": "choice",
      "instructions": "Decide the refund action.",
      "criteria": {"refund": "Refund the duplicate charge", "wait": "Wait for review"}
    }
  }
}
~~~

Preprocessing validates and decodes the data URL, writes it to a request-scoped
directory, hashes the bytes, and converts it to Valen's local-media
`messages` record. Text state skips materialization and compiles a text-only
user message with no assets. The worker owns that directory's lifetime.

The pinned compiler builds every question type from the same record family:

- `choice` and `noul` compile to one branch holding all candidates;
- `score` compiles one isolated branch per level — each branch sees only its
  own level description, never the level ID or the other levels — and the K
  per-level decision logits are softmaxed across branches;
- `noul` without custom wording is the reference's native noul record with
  the schema's fixed candidate wording; custom wording cannot flow through
  it, so it compiles as the equivalent `choice` record keyed `true`/`false`,
  with unspecified sides filled from the same fixed wording — `P(true)` still
  comes from the two-candidate softmax.

Postprocessing keeps Valen's original formulas:

~~~text
choice/noul confidence (n candidates):
  n = 1: 1
  n > 1: max(0, (max(p) - 1/n) / (1 - 1/n))    # noul is the n = 2 case
score confidence (K ordered levels, mode = argmax p):
  distance    = sum_i p_i * |i - mode|
  uniform_mad = (sum_i |i - (K-1)/2|) / K
  confidence  = max(0, 1 - distance / uniform_mad)
~~~

This confidence choice is intentionally temporary. The shared `/v1/systemone`
contract, including confidence semantics, error details, busy behavior, model
identity, and usage fields, remains coordinated with
[system1-omni#61](https://github.com/ThinkFlowLab/system1-omni/issues/61).

Each `choice` answer is `{"type": "choice", "choice": ...,
"probabilities": ..., "confidence": ...}`: `choice` is the candidate with the
highest probability, ties go to the earliest candidate in request order, and
`probabilities` maps every candidate key to its probability in request order.

Each `noul` answer is `{"type": "noul", "noul": ...,
"probabilities": ..., "confidence": ...}`: `noul` is `P(true)`, the reference
answer module's own field; the pair and confidence are this worker's
extension (the reference answer carries only `P(true)`).

Each `score` answer is `{"type": "score", "probabilities": ...,
"score": ..., "legend": ..., "confidence": ...}`: `probabilities` maps the
level keys `"0"`… to their cross-branch softmax probability, `score` is the
expected level `sum(i * p_i)`, and `legend` maps each level key back to its
request-order description.

The response `model` is `valen-preview-0923`, the worker's stable serving
identity. `usage.input_tokens` counts the compiled prompt's logical tokens:
the shared state tokens plus each question's task suffix, counted once.
`usage.output_tokens` is 0; the decision head returns logits without
generation. `internal_usage.compute_tokens` counts the full per-question
branch lengths the executor runs, with the shared state repeated per branch —
a K-level `score` question therefore counts K branches.

## Errors

An error rejects the whole request. Its body is `{"detail": "<message>"}`, and
the message names the problem.

The status is `400` when the body is not a usable JSON object (invalid JSON or
UTF-8, non-finite numbers, a repeated key) or the declared `Content-Length`
does not match the received bytes.

The status is `411` without exactly one `Content-Length`; chunked requests are
unsupported. `408` means the body read timed out, `413` that the body or image
exceeded its limit, `415` that the content type is not `application/json`,
and `503` that another inference is running.

The status is `422` when a well-formed request cannot be evaluated:

- a `model` other than `valen-preview-0923`;
- a question whose `type` is not `choice`, `noul` or `score`, or unsupported
  request or question fields;
- text over its limit, or containing a tokenizer control token — the media
  tokens such as `<|image_pad|>` at minimum, plus the loaded tokenizer's full
  special-token set (for example `<|im_end|>`) when the worker is serving;
  this guard covers text state, instructions, criterion texts, level
  descriptions and option keys (candidate names enter the prompt);
- empty or whitespace-only `instructions`, option keys, criterion texts,
  noul descriptions, score level descriptions, or text state;
- `noul` criteria that name anything other than `true`/`false`, or `score`
  criteria that are not an ordered list of 2 to 10 level descriptions;
- an empty text state, or a state that is neither an image data URL nor text;
- an image that is not a single-frame PNG/JPEG matching its MIME type and
  limits;
- anything else the pinned compiler rejects, for example a question whose
  compiled prompt exceeds the checkpoint's 8192-token `max_length`; the
  compiler's message is passed through with a `request rejected by the pinned
  compiler` prefix.

Any other failure is `500` with a generic detail and no request content in
the log (the exception type only).

## Checkpoint export

The pinned checkpoint and base also export to a native-loadable
safetensors artifact set — the language LoRA served separately as unmerged
FP32 adapters, vision-top and merger replaced, FP32 decision head — with
every transformation recorded in a JSON
manifest: see [recipe/valen/export.md](../../../recipe/valen/export.md).
Consuming that export does not change this worker.

## Ownership boundary

- `protocol.py` owns wire validation and safe inline-image limits.
- `preprocess.py` owns media materialization and the Valen compiler record.
- `engine.py` owns the pinned model, forward pass, learned head, device state,
  and warmup.
- `postprocess.py` owns probability normalization, confidence, and response
  reconstruction.
- `src/frontend/valen.py` owns HTTP lifecycle and worker orchestration.

The current tests are CPU-only and do not require model weights or a GPU.
The real worker path additionally requires the pinned source, checkpoint, base,
and a CUDA-capable PyTorch environment.
