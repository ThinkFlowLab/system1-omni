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
- textual instructions and criteria for `choice` questions;
- BF16 CUDA inference with the pinned Preview checkpoint. Image and text
  state are serving-validated on the GPU host, with recorded reference parity
  for both.

The current worker does not implement `noul`, `score`, video, multiple images,
remote image URLs, or a native Rust/CUDA Valen executor.

The current protocol limits requests to an 8 MiB JSON body, one PNG/JPEG image
of at most 4 MiB, 2048 pixels per side, 1,048,576 pixels total, and a 200:1
maximum aspect ratio. Text state is limited to 16,384 characters, the same
limit as combined instruction and criterion text. A request may contain at
most 8 questions and 255 candidates per question.

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

The generic `recipe/compare_with_backend.py` also sends `score`/`noul` cases,
which this worker rejects, so compare the Valen fixture directly:

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
the text-only example.

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
pytest tests/valen: 16 passed in 0.14s
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
temperature:

~~~text
reference model: Valen
worker model: valen-preview-0923
choice agreement: exact
input_tokens: 108 == 108
output_tokens: 0 == 0
compute_tokens: 108 == 108
max probability delta: 1.1299724378e-08
confidence delta: 2.2599448757e-08
decision agreement: PASS
probability tolerance <= 1e-6: PASS
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
token accounting: PASS
~~~

The reference model field is `Valen`; the system1-omni worker deliberately
uses `valen-preview-0923` as its stable serving identity.

Reproduce with [`compare_reference.py`](compare_reference.py). The script loads
one pinned model instance and drives each request through two paths: a
Valen-native record compiled by the pinned `Compiler` and decoded with a plain
softmax, and the production `frontend.valen.decide` pipeline. The decision
criteria are exact choice agreement, equal token accounting and the declared
probability tolerance (default `1e-6`, `--tolerance` to override):

~~~sh
PYTHONPATH=src .venv/bin/python recipe/valen/compare_reference.py \
  --valen-source "$VALEN_SRC" --checkpoint "$VALEN_CKPT" \
  --base "$VALEN_BASE" --device cuda --dtype bf16 --case image

PYTHONPATH=src .venv/bin/python recipe/valen/compare_reference.py \
  --valen-source "$VALEN_SRC" --checkpoint "$VALEN_CKPT" \
  --base "$VALEN_BASE" --device cuda --dtype bf16 --case text
~~~

The image case uses [`example-request.json`](example-request.json); the text
case uses the recipe's text example unless `--text-body` points elsewhere.
The block above is the recorded image-case output.

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
- Video, native execution, `noul`, and `score` remain outside this recipe.
