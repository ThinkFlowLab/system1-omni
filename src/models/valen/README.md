# Valen-Preview-0923 reference contract

This directory contains the first System1-Omni integration slice for
Valen-Preview-0923. It is a reference-serving boundary only: the worker and
executor will be added after the protocol tests are stable. Native CUDA
execution is not part of this slice.

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

- one inline PNG/JPEG image in `state.image`;
- `choice` questions only;
- up to 255 named candidates;
- no `noul`, `score`, video, multi-image, or remote URL inputs;
- no token generation: the decision head returns candidate logits directly.

The public request keeps the Cua-S1-compatible shape:

```json
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
```

Preprocessing validates and decodes the data URL, writes it to a request-scoped
directory, hashes the bytes, and converts it to Valen's local-media
`messages` record. The worker owns that directory's lifetime. Postprocessing
uses Valen's original choice confidence formula for this reference slice:

```text
K = 1: confidence = 1
K > 1: confidence = max(0, (max(p) - 1/K) / (1 - 1/K))
```

This confidence choice is intentionally temporary. The shared `/v1/systemone`
contract, including confidence semantics, error details, busy behavior, model
identity, and usage fields, remains coordinated with
[system1-omni#61](https://github.com/ThinkFlowLab/system1-omni/issues/61).

## Ownership boundary

- `protocol.py` owns wire validation and safe inline-image limits.
- `preprocess.py` owns media materialization and the Valen compiler record.
- `postprocess.py` owns probability normalization, confidence, and response
  reconstruction.
- The future executor will own the pinned base, Valen checkpoint, forward pass,
  decision head, device state, and warmup.

The current tests are CPU-only and do not require model weights or a GPU.
