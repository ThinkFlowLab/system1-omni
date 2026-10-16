# Valen-Preview-0923 reference worker

This recipe serves Valen-Preview-0923 through the Python reference worker and
the existing Rust frontend. The model-side implementation is under
[`src/models/valen/`](../../src/models/valen/), and the worker is
[`src/frontend/valen.py`](../../src/frontend/valen.py).

Run the commands from the repository root on Linux with an NVIDIA GPU and
Python 3.12. The validation record below was collected on WSL2 x86_64 with an
RTX 5060 Laptop GPU. Other hardware is not covered by these measurements.

## Scope

The validated slice is:

- model id: `valen-preview-0923`;
- the Python reference worker behind the Rust frontend;
- text state (a non-empty string; objects and arrays are serialized to JSON
  text) or one inline PNG or JPEG image in `state.image`;
- textual instructions and criteria for `choice`, `noul`, and `score`
  questions, freely mixed in one request against one shared state;
- BF16 CUDA inference with the pinned Preview checkpoint. Image and text
  state are serving-validated on the GPU host, with recorded reference parity
  for all three question types.

The current worker does not implement video, multiple images, remote image
URLs, or a native Rust/CUDA Valen executor.

The current protocol limits requests to an 8 MiB JSON body, one PNG/JPEG image
of at most 4 MiB, 2048 pixels per side, 1,048,576 pixels total, and a 200:1
maximum aspect ratio. Text state is limited to 16,384 characters, the same
limit as combined instruction and criterion text. A request may contain at
most 8 questions, 255 named candidates per `choice` question, and an ordered
list of 2 to 10 level descriptions per `score` question.

The first slice follows
[issue #84](https://github.com/ThinkFlowLab/system1-omni/issues/84): establish
a faithful reference service, document unsupported inputs, run a real
post-readiness inference, and verify the Rust frontend path.

## Architecture and ownership

~~~text
Rust frontend
  -> Python worker
     -> prepare: validate, materialize the image, compile Valen input
     -> execute: Qwen3.5, visual blocks, visual merger, LoRA, decision head
     -> finish: softmax, ordered choice, confidence, usage, response
~~~

The ownership boundary follows
[`docs/architecture.md`](../../docs/architecture.md):

- `protocol.py` owns wire validation and image limits.
- `preprocess.py` owns media materialization and the Valen compiler record.
- `engine.py` owns weights, forward, the learned head, device state, and warmup.
- `postprocess.py` owns probability normalization, choice, confidence, usage,
  and response reconstruction.
- `frontend.valen` owns HTTP lifecycle and orchestration.
- The Rust frontend owns transport and forwarding, not model state.

The processor/executor seam is preserved for a future native path. The Python
worker remains the reference serving path while shared processing and batching
layers are still being developed.

## Pinned artifacts

| Artifact | Revision or checksum |
| --- | --- |
| Valen source | `750bfcfbb48a5275534a9c912257ebe83ca57a97` |
| Preview checkpoint revision | `81b9c6396ded3a45a92ae8079af85f988aa40501` |
| `checkpoint.pt` SHA-256 | `836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012` |
| Qwen3.5-2B base revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Valen source | [Liuziyu77/Valen](https://github.com/Liuziyu77/Valen) |
| Preview checkpoint | [Valen-Team/Valen-Preview-0923](https://huggingface.co/Valen-Team/Valen-Preview-0923) |

The source, Preview checkpoint, and Qwen base report Apache-2.0 license
metadata for these pins. Re-check the metadata when changing a revision.

The base `valen_manifest.json` records artifact-preparation provenance,
separately from the serving environment:

~~~text
torch 2.6.0 · torchvision 0.21.0 · transformers 5.19.0
peft 0.18.1 · huggingface-hub 1.33.0 · av 16.1.0
~~~

## Validated environment

~~~text
OS: WSL2 x86_64
Python: 3.12.3
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
VRAM: 8151 MiB
PyTorch: 2.11.0+cu128
torchvision: 0.26.0+cu128
Transformers: 5.19.0
PEFT: 0.18.1
huggingface-hub: 1.33.0
Pillow: 12.3.0
av: 16.1.0
GPU capability: sm_120
dtype: bfloat16
system1-omni: 2fd8ce9 (GPU serving and smoke records); parity blocks re-verified verbatim at 507c448
~~~

FP32 and BF16 small-operator checks passed. Startup memory was approximately
`6593 MiB / 8151 MiB` on this host; this is not a general memory guarantee.
Without optional `causal_conv1d`, Valen falls back to its PyTorch reference
implementation. This affects performance, not the current correctness result.

## Install and prepare the artifacts

PyTorch wheels must match the host CUDA runtime. The validated environment used
the CUDA 12.8 build shown above.

~~~sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  torch==2.11.0 torchvision==0.26.0 \
  transformers==5.19.0 peft==0.18.1 \
  huggingface-hub==1.33.0 av==16.1.0 \
  Pillow==12.3.0 pytest==8.4.2
~~~

Set the artifact paths. Override these variables to use an external model
directory:

~~~sh
export VALEN_SRC="$PWD/weights/Valen"
export VALEN_CKPT="$PWD/weights/Valen-Preview-0923"
export VALEN_BASE="$PWD/weights/Qwen3.5-2B"
export VALEN_SOURCE_REV=750bfcfbb48a5275534a9c912257ebe83ca57a97
export VALEN_CHECKPOINT_REV=81b9c6396ded3a45a92ae8079af85f988aa40501
export VALEN_CHECKPOINT_SHA=836622efe78fe757e2627aa6050c223d461c424ea430a1c110c15e6f42f5a012
export VALEN_BASE_REV=15852e8c16360a2fea060d615a32b45270f8a8fc
mkdir -p weights
~~~

Fetch and verify the pinned source and artifacts:

~~~sh
git clone https://github.com/Liuziyu77/Valen.git "$VALEN_SRC"
git -C "$VALEN_SRC" checkout --detach "$VALEN_SOURCE_REV"

.venv/bin/hf download Valen-Team/Valen-Preview-0923 \
  --revision "$VALEN_CHECKPOINT_REV" --local-dir "$VALEN_CKPT"
.venv/bin/hf download Qwen/Qwen3.5-2B \
  --revision "$VALEN_BASE_REV" --local-dir "$VALEN_BASE"

.venv/bin/python "$VALEN_SRC/scripts/setup/prepare_model.py" \
  --output "$VALEN_BASE" --revision "$VALEN_BASE_REV" --verify-local
~~~

The preparation script verifies Qwen safetensors and the local processor, then
writes `valen_manifest.json`. Verify the final identity before serving:

~~~sh
test "$(git -C "$VALEN_SRC" rev-parse HEAD)" = "$VALEN_SOURCE_REV"
test -z "$(git -C "$VALEN_SRC" status --porcelain --untracked-files=no)"
test -z "$(git -C "$VALEN_SRC" ls-files --others --exclude-standard | grep '\.py$')"
test "$(sha256sum "$VALEN_CKPT/checkpoint.pt" | cut -d' ' -f1)" = "$VALEN_CHECKPOINT_SHA"
.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
checkpoint = Path(os.environ["VALEN_CKPT"])
base = Path(os.environ["VALEN_BASE"])
config = json.loads((checkpoint / "config.json").read_text())
manifest = json.loads((base / "valen_manifest.json").read_text())
assert config["stage"] == "vision_top"
assert config["projection_dim"] == 256
assert config["max_length"] == 8192
assert manifest["repo"] == "Qwen/Qwen3.5-2B"
assert manifest["revision"] == os.environ["VALEN_BASE_REV"]
print("artifact verification: PASS")
print(manifest["versions"])
PY
~~~

The pinned checkpoint and base can also be exported to a manifest-described
safetensors artifact set for non-Python consumers: [export.md](export.md).

## Start the worker

The worker loads all model-owned state and performs a real representative
warmup before binding port 8000:

~~~sh
PYTHONPATH=src .venv/bin/python -m frontend.valen \
  --valen-source "$VALEN_SRC" --checkpoint "$VALEN_CKPT" \
  --base "$VALEN_BASE" --device cuda --dtype bf16 --port 8000
~~~

~~~sh
curl -sS -i http://127.0.0.1:8000/health
~~~

Validated response:

~~~json
{"status":"ready","model":"valen-preview-0923","modality":"multimodal"}
~~~

## Example request

The complete runnable request is in
[`example-request.json`](example-request.json). It contains a tiny inline PNG.

~~~sh
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/valen/example-request.json
~~~

Observed response:

~~~json
{
  "model": "valen-preview-0923",
  "answers": {"color": {
    "type": "choice",
    "choice": "black",
    "probabilities": {"white": 0.4459854086288687, "black": 0.5540145913711313},
    "confidence": 0.10802918274226259
  }},
  "usage": {"input_tokens": 108, "output_tokens": 0},
  "internal_usage": {"compute_tokens": 108}
}
~~~

This is a serving smoke test, not an accuracy result. The numbers follow the
[worker contract](../../src/models/valen/README.md). Confidence semantics remain coordinating
with [system1-omni#61](https://github.com/ThinkFlowLab/system1-omni/issues/61).

A text-only request uses a plain string state:

~~~sh
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"valen-preview-0923","state":"The card was charged twice for one order.","questions":{"refund":{"type":"choice","instructions":"Decide the refund action.","criteria":{"refund":"Refund the duplicate charge","wait":"Wait for review"}}}}'
~~~

Observed response, same host as the image fixture:

~~~json
{
  "model": "valen-preview-0923",
  "answers": {"refund": {
    "type": "choice",
    "choice": "refund",
    "probabilities": {"refund": 0.5159631044115068, "wait": 0.4840368955884932},
    "confidence": 0.03192620882301367
  }},
  "usage": {"input_tokens": 48, "output_tokens": 0},
  "internal_usage": {"compute_tokens": 48}
}
~~~

A yes/no judgment uses `noul`, optionally replacing the reference's fixed
true/false wording through `criteria`. The complete runnable request is in
[`example-noul-request.json`](example-noul-request.json); it asks one
fixed-wording and one custom-wording question against one state:

~~~sh
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/valen/example-noul-request.json
~~~

Expected response shape (the observed response for this fixture is recorded
in the integration pull request):

~~~json
{
  "model": "valen-preview-0923",
  "answers": {
    "duplicate": {"type": "noul", "noul": "<p(true)>", "probabilities": {"true": "<p(true)>", "false": "<p(false)>"}, "confidence": "<confidence>"},
    "authorized": {"type": "noul", "noul": "<p(true)>", "probabilities": {"true": "<p(true)>", "false": "<p(false)>"}, "confidence": "<confidence>"}
  },
  "usage": {"input_tokens": "<count>", "output_tokens": 0},
  "internal_usage": {"compute_tokens": "<count>"}
}
~~~

A graded assessment uses `score` with an ordered list of 2 to 10 level
descriptions. The complete runnable request is in
[`example-score-request.json`](example-score-request.json):

~~~sh
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/valen/example-score-request.json
~~~

Expected response shape (the observed response for this fixture is recorded
in the integration pull request):

~~~json
{
  "model": "valen-preview-0923",
  "answers": {"severity": {
    "type": "score",
    "probabilities": {"0": "<p(0)>", "1": "<p(1)>", "2": "<p(2)>"},
    "score": "<expected level>",
    "legend": {"0": "<level 0 text>", "1": "<level 1 text>", "2": "<level 2 text>"},
    "confidence": "<confidence>"
  }},
  "usage": {"input_tokens": "<count>", "output_tokens": 0},
  "internal_usage": {"compute_tokens": "<count>"}
}
~~~

## Start and verify the Rust frontend

~~~sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  target/release/omni-jev
~~~

Send the same request through port 8080:

~~~sh
curl -sS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/valen/example-request.json
~~~

The generic `recipe/compare_with_backend.py` also covers `score`/`noul`
cases; the byte-identical comparison below uses the Valen fixture directly:

~~~sh
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/valen/example-request.json > /tmp/valen-worker.json
curl -fsS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/valen/example-request.json > /tmp/valen-frontend.json
cmp /tmp/valen-worker.json /tmp/valen-frontend.json
~~~

The validated runs returned 200 from both endpoints with matching JSON content
types and byte-identical response bodies, for the image fixture above and for
the text-only example. The noul and score fixtures were re-checked the same
way on the same host: byte-identical through the frontend for all three
fixtures.

## Error behavior

The request, response and error contract — status codes, `detail` bodies,
confidence, and usage counting — is the
[worker contract](../../src/models/valen/README.md).

Worker responses are forwarded unchanged. Frontend-generated 502/504 responses
are frontend errors, not model responses.

## Validation

~~~sh
PYTHONPATH=src .venv/bin/python -m pytest tests/valen -q
PYTHONPATH=src .venv/bin/python -m compileall -q \
  src/models/valen src/frontend/valen.py tests/valen
cargo fmt --all --check
cargo test -p omni-jev --test frontend --locked
cargo build --release --locked
python recipe/test_compare_with_backend.py
python recipe/valen/test_compare_reference.py
~~~

Validated results:

~~~text
pytest tests/valen: 47 passed in 1.21s (at 2192c27, before the export battery)
compileall: passed
cargo fmt --all --check: passed
frontend integration: 12 passed; 0 failed
cargo build --release --locked: passed
compare_with_backend: ok
compare_reference tests: ok
~~~

## Reference parity

A real image fixture was compared with the pinned Valen reference using the
same checkpoint, base revision, image, question, candidate order, and
temperature. The block below is the verbatim `compare_reference.py` output:

~~~text
case: image
reference model: Valen
worker model: valen-preview-0923
input_tokens: 108 == 108
output_tokens: 0 == 0
compute_tokens: 108 == 108
max probability delta: 2.7810645453e-08
confidence delta: 5.5621290906e-08
decision agreement: PASS
probability tolerance <= 1e-06: PASS
confidence tolerance <= 1e-06: PASS
token accounting: PASS
~~~

The text-state case, same host, methodology and tolerances:

~~~text
case: text
reference model: Valen
worker model: valen-preview-0923
input_tokens: 48 == 48
output_tokens: 0 == 0
compute_tokens: 48 == 48
max probability delta: 2.6866340819e-08
confidence delta: 5.3732681637e-08
decision agreement: PASS
probability tolerance <= 1e-06: PASS
confidence tolerance <= 1e-06: PASS
token accounting: PASS
~~~

The reference model field is `Valen`; the system1-omni worker deliberately
uses `valen-preview-0923` as its stable serving identity.

Reproduce with [`compare_reference.py`](compare_reference.py). Both paths
share one pinned model instance: a Valen-native record compiled by the pinned
`Compiler` from the worker's own record rules, against the production
`frontend.valen.decide` pipeline; `noul` and `score` answers are decoded by
the pinned reference answer module (`valen.evaluation.inference.answer`). The
criteria are exact discrete agreement (candidate order, the requested answer
type, chosen key, score legend), equal token accounting, and the declared
tolerance (default `1e-6`, `--tolerance` to override) on probabilities,
confidences and the scalar answer fields (`noul` P(true), `score` expected
level). An answer whose public `type` drifts from the requested question
type fails the run instead of skipping the noul/score checks:

~~~sh
PYTHONPATH=src .venv/bin/python recipe/valen/compare_reference.py \
  --valen-source "$VALEN_SRC" --checkpoint "$VALEN_CKPT" \
  --base "$VALEN_BASE" --device cuda --dtype bf16 --case image

PYTHONPATH=src .venv/bin/python recipe/valen/compare_reference.py \
  --valen-source "$VALEN_SRC" --checkpoint "$VALEN_CKPT" \
  --base "$VALEN_BASE" --device cuda --dtype bf16 --case text

PYTHONPATH=src .venv/bin/python recipe/valen/compare_reference.py \
  --valen-source "$VALEN_SRC" --checkpoint "$VALEN_CKPT" \
  --base "$VALEN_BASE" --device cuda --dtype bf16 --case noul

PYTHONPATH=src .venv/bin/python recipe/valen/compare_reference.py \
  --valen-source "$VALEN_SRC" --checkpoint "$VALEN_CKPT" \
  --base "$VALEN_BASE" --device cuda --dtype bf16 --case score
~~~

The image case uses [`example-request.json`](example-request.json); the text
case uses the recipe's text example unless `--text-body` points elsewhere.
The `noul` and `score` cases use [`example-noul-request.json`](example-noul-request.json)
and [`example-score-request.json`](example-score-request.json) unless
`--noul-body`/`--score-body` point elsewhere; `--case all` runs all four in
one model load. Both blocks above are verbatim output of the committed tool;
the `noul` and `score` blocks are recorded in the integration pull request.

## Export parity

The exported artifact set ([`export.md`](export.md)) was compared against the
original pinned checkpoint stack with
[`verify_export_parity.py`](verify_export_parity.py) on the same host as the
records above. The two stacks are loaded serially, each in its own subprocess
(the ~6.6 GiB BF16 stack does not fit twice in 8 GiB VRAM):

- original: the worker's own `ValenExecutor` build — pinned base,
  PEFT-wrapped language model with the FP32 LoRA applied unmerged, trained
  vision-top blocks, merger and head from `checkpoint.pt`;
- exported: the consumption path the export manifest documents —
  `Qwen3_5ForConditionalGeneration.from_pretrained(weights/valen-export,
  dtype=bf16, eager).model`, the language LoRA re-applied through the same
  peft build the pinned reference uses (every language Linear wrapped at the
  manifest-declared rank 32 / alpha 64, the 372 FP32 tensors of
  `lora.safetensors` loaded by key, adapters kept FP32), plus the decision
  head rebuilt from `head.safetensors` (DecisionHead 2048→256, FP32, no
  bias), run through the pinned `Valen` class.

Both stacks answer the text example and the image fixture compiled by the
pinned `Compiler`; the declared gate is exact answer agreement with
probability deltas ≤ 1e-6. The block below is the verbatim tool output
(device `cuda:0`, NVIDIA GeForce RTX 5060 Laptop GPU, sm_120, 7.96 GiB,
torch 2.11.0+cu128; recorded 2026-10-15):

~~~text
case: text
logical tokens: 48
compute tokens: 48
max probability delta: 0.0000000000e+00
answer agreement: PASS
probability tolerance <= 1e-06: PASS
token accounting: PASS

case: image
logical tokens: 108
compute tokens: 108
max probability delta: 0.0000000000e+00
answer agreement: PASS
probability tolerance <= 1e-06: PASS
token accounting: PASS

device: cuda:0 (NVIDIA GeForce RTX 5060 Laptop GPU (sm_120, 7.96 GiB))
parity slot filled: weights/valen-export/valen_export.json
verify_export_parity: ok
~~~

Per-candidate probabilities (original stack → exported stack, printed by the
tool's own stage runner):

~~~text
text/refund: 0.5159630775 0.4840368927 -> 0.5159630775 0.4840368927  (both choose refund)
image/color: 0.4459854066 0.5540145636 -> 0.4459854066 0.5540145636  (both choose black)
~~~

Verdict: **the parity gate passes at bit level**. The unmerged export
answers both fixtures with probabilities identical to the original stack's
(max delta exactly 0.0 on every candidate), the discrete answers and token
accounting agree exactly, and the original side matches the committed
serving records above (0.5159631044… / 0.5540145914…) within the ~3e-8
worker-vs-reference spread. Both stacks apply the LoRA through
the same peft 0.18.1 unmerged `Linear.forward` with FP32 adapters, and the
exported backbone differs from the reference's `from_pretrained` state only
in tensors that are bit-equal by construction (the trained vision-top and
merger replacements, and the 36 FP32→BF16 downcasts the reference's own
BF16 runtime performs at load), so the two forward graphs are numerically
identical. The verdict — per-case deltas, device, tolerance — is recorded
with `parity.status = "passed"` in
`weights/valen-export/valen_export.json`.

Reproduce (the export itself per [`export.md`](export.md) first):

~~~sh
PYTHONPATH=src .venv/bin/python recipe/valen/verify_export_parity.py \
  --valen-source "$VALEN_SRC" --checkpoint "$VALEN_CKPT" \
  --base "$VALEN_BASE" --export weights/valen-export --device cuda
~~~

The script validates the artifact set and its recorded source pins before
loading, refuses to overwrite an existing manifest parity record unless
`--force` is given, and exits 0 only when the gate passes.

## Troubleshooting and limits

- The source checkout must be exactly the pinned commit.
- The checkpoint must contain `config.json` and `checkpoint.pt` with the SHA-256
  listed above.
- The base must contain `valen_manifest.json` for the pinned Qwen revision.
- One PNG or JPEG data URL is accepted. The image is materialized into a
  request-scoped temporary directory because the pinned compiler requires a
  local media path.
- `causal_conv1d` is optional and only changes performance.
- An RTX 5060 requires a PyTorch/CUDA build supporting `sm_120`.
- `/health` is exposed only after model loading and real warmup complete.
- The memory and smoke-response observations are host-specific; they are not
  general accuracy or performance claims.
- Video and native execution remain outside this recipe.
