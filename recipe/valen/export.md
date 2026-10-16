# Valen-Preview-0923 checkpoint export

This recipe turns the pinned Preview checkpoint plus its pinned Qwen3.5-2B
base into a safetensors artifact set that any consumer — native or
otherwise — can load without the training-framework pickle or the reference
Python code. One deterministic command performs every transformation and
records each one in a JSON manifest, so the artifact set can be audited on
its own. Building, validating and parity-checking the artifact set is this
recipe's scope; serving from it is the consumer's side.

The export complements the
[reference worker recipe](reference.md): the worker serves through the
pinned Python reference stack, while the export is the path for everything
that consumes weights directly.

## Usage

Requirements: the pinned artifacts prepared exactly as in the reference
recipe's [install section](reference.md#install-and-prepare-the-artifacts),
and the repository venv. The export runs CPU-only and never needs a GPU.

~~~sh
PYTHONPATH=src .venv/bin/python recipe/valen/export_checkpoint.py \
  --checkpoint "$VALEN_CKPT" \
  --base "$VALEN_BASE" \
  --out weights/valen-export
~~~

`--out` names the output directory; it must not exist (or be empty), and
every path is overridable this way. The export streams tensor-by-tensor
(mmap reads, an incremental safetensors writer), so host memory stays well
below the ~4.5 GiB artifact size. Expected console output:

~~~text
exported weights/valen-export
  backbone: 578 tensors
  decision-head: 2 tensors
  lora: 372 tensors
  merger: 6 tensors
  vision-top: 48 tensors
  downcasts: 36 fp32 -> bf16
  lora: 372 unmerged fp32 keys over 186 targets
  manifest: weights/valen-export/valen_export.json
~~~

Before writing anything, the export enforces the artifact pins — the
checkpoint SHA-256 against `training_metadata.json`, the base weights
digest against both `valen_manifest.json` and the checkpoint's own
`base_manifest`, and the revision agreement between them — and rejects a
checkpoint whose `stage` is not `vision_top`. A missing pin or a mismatch
fails loudly instead of exporting the wrong weights.

## The artifact set

Five files, nothing else:

| File | Contents |
| --- | --- |
| `model.safetensors` | The backbone in the base's own key layout (`model.language_model.*` / `model.visual.*`), BF16 only: the base's 632 tensors with the trained vision-top blocks 20–23 and the merger substituted as plain replacements (48 + 6), the remaining 578 passing through verbatim (36 of them downcast from FP32). The 186 language LoRA targets are base tensors here, untouched. |
| `lora.safetensors` | The language LoRA, served unmerged: 372 FP32 tensors under their PEFT key names, verbatim from the checkpoint. |
| `head.safetensors` | The decision head: two FP32 matrices `head.decision.weight` and `head.candidate.weight`, no bias. |
| `config.json` | The base config copied verbatim, so the reference builder loads the export unchanged. |
| `valen_export.json` | The manifest (below). |

Inventory, totaling 1006 manifest tensor records: 578 `backbone`, 372
`lora`, 48 `vision-top`, 6 `merger`, 2 `decision-head`. The vision-top
blocks and merger land in the base's own `model.visual.*` layout — same
key tails and shapes as the base snapshot — so they are drop-in
replacements.

## The manifest

`valen_export.json` is format `valen-checkpoint-export/2` and is the
contract for auditing the export without the reference code. Its
top-level structure:

~~~text
format            "valen-checkpoint-export/2"
sources           pinned checkpoint and base revisions + SHA-256s, tool versions
lora              the adapter file, dtype, rank/alpha/scaling, target and key
                  counts, and the declared application semantics
downcasts         the names of every FP32 -> BF16 downcast tensor
head              head file, dtype, projection_dim, hidden_size, formulas
tensors           per-tensor name/shape/dtype/component/source/file records
files             size + SHA-256 of each artifact file
consumption       how to load the artifact set (transcribed below)
parity            the parity-gate result; null until the gate fills it
parity_slot_shape the documented shape of the parity record
~~~

`export_checkpoint.validate_export` cross-checks a finished export
against its manifest — file digests, tensor inventory against the
safetensors headers, downcast bookkeeping, head config, and the adapter
inventory (every `lora.safetensors` key an FP32 tensor that maps onto a
listed backbone tensor, one A and one B per target, at the declared rank)
— reading headers and digests only, so it also runs cheaply against the
multi-GiB real artifact set.

## The LoRA adapters

The export serves the language LoRA separately and unmerged: the 372 FP32
adapter tensors ride in `lora.safetensors` under their PEFT key names
(`backbone.language_model.base_model.model.<module>.lora_A.default.weight`
and the `lora_B` twins), verbatim from the checkpoint, and
`model.safetensors` carries the base weights of the 186 targeted linears
untouched. The pinned config is rank 32, alpha 64, scaling α/r = 64/32 =
2.0.

The application semantics a consumer must implement, recorded verbatim in
the manifest's `lora` block: per targeted linear,
`y = bf16(fp32(bf16(x @ W.t())) + scaling * ((x.float() @ A.t()) @ B.t()))`
— the base GEMM runs in BF16, the LoRA delta is computed in FP32 on the
activations and added to the upcast base output, and the sum is rounded
to BF16 exactly once. These are the unmerged peft 0.18.1 `Linear.forward`
semantics with the adapters kept FP32 at runtime — the same boundaries
the pinned reference serves with, and what
`export_checkpoint.unmerged_lora_output` computes (tested bit-equal
against the installed peft). Merging the adapters into the base weights
instead — `bf16(fp32(W) + 2·(B@A))` — changes the computed outputs
relative to the trained artifact's unmerged semantics, which is why the
export does not do it.

## The FP32 downcasts

The base stores 36 tensors in FP32 — the delta-rule normalizer logs and
norms, `model.language_model.layers.<i>.linear_attn.A_log` and
`...linear_attn.norm.weight` — and the export downcasts each to BF16,
listing every one in the manifest's `downcasts` block. The downcast is
lossless against the reference's own BF16 runtime: transformers has no
`keep_in_fp32_modules` entry for this architecture and upcasts these
tensors with `.float()` at use, so the reference already computes on the
BF16-rounded values. Any FP32 base tensor outside the declared families
is an error, not a silent downcast.

## The decision head

`head.safetensors` holds the two FP32 projections —
`head.decision.weight` and `head.candidate.weight` — each
256 × 2048 (`projection_dim` × `hidden_size`), no bias, with the
dimensions and formulas recorded in the manifest's head block:
logits are `(W_cand @ h_cand) . (W_dec @ h_dec) / sqrt(projection_dim)`
in FP32. The head is exported in FP32, unchanged from the checkpoint,
which stores it in FP32.

## Consumption

The manifest's `consumption` block records how to load the export; in
full:

~~~text
backbone: Qwen3_5ForConditionalGeneration.from_pretrained(out, dtype=bf16,
          attn_implementation='eager', local_files_only=True).model
lora:     wrap backbone.language_model in peft get_peft_model(LoraConfig(r=32,
          lora_alpha=64, lora_dropout=0, target_modules=<every language
          Linear>, bias='none')), load lora.safetensors' tensors by key and
          keep the adapters fp32 at runtime (the application semantics
          recorded in the lora block)
head:     head.safetensors: DecisionHead(hidden_size=2048,
          projection_dim=256), fp32, no bias
~~~

The copied `config.json` makes the `from_pretrained` call work against
the export directory directly; the LoRA is re-applied exactly the way the
pinned reference builds it (the same peft wrap at the recorded rank and
alpha, adapters kept FP32), and the head is rebuilt from its two matrices
and recorded dimensions. Nothing else is needed to reconstruct the model.

## Pins and licenses

The export consumes the same pinned artifacts as the
[reference worker](reference.md#pinned-artifacts) — the Preview
checkpoint revision `81b9c639...` (SHA-256
`836622ef...f42f5a012`) and the Qwen3.5-2B base revision
`15852e8c...` — and records both, with per-file digests, in the
manifest's `sources` block on every run. The Valen source, the Preview
checkpoint, and the Qwen base report Apache-2.0 license metadata for
these pins (recorded with the serving slice in
[PR #115](https://github.com/ThinkFlowLab/system1-omni/pull/115));
re-check the metadata when changing a revision. The exported artifact
set inherits those terms.

## Parity evidence

Whether the exported stack reproduces the original checkpoint's answers
is the parity gate's question, not this document's claim: the verdict —
numbers, device, tolerances — lives in the manifest's
`parity` slot (left `null` by the export, shaped by
`parity_slot_shape`) and in the evidence blocks of
[reference.md](reference.md), where the parity run is recorded. Consult
those for the result; do not treat an export as parity-proven until its
manifest's parity slot is filled.

## Determinism and regression

Two runs on the same inputs produce byte-identical artifacts and
identical manifests: no timestamps, no absolute paths, tensors and JSON
written in sorted key order, single-threaded CPU math. The regression is
wired into the recipe-test pattern:

~~~sh
PYTHONPATH=src .venv/bin/python -m pytest tests/valen -q
python recipe/valen/test_export_reference.py
~~~

The first is the CPU-only suite (105 tests passed, 2 artifact-gated skips
at the time of writing). The second chains that suite's export battery plus
the env-gated real-artifact determinism gate — set `VALEN_CKPT` and
`VALEN_BASE` to run it; it exports twice and compares every byte — and,
with `VALEN_EXPORT` pointing at an exported directory (defaulting to
`weights/valen-export` when present), validates the finished artifact set
against its manifest. Without the artifacts, both gated phases print a
clear skip line and the script still passes.
