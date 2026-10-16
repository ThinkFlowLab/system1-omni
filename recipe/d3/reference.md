# D3 reference worker

This recipe serves D3 (Decision 3.0) through the Python reference worker and
the existing Rust frontend. The model-side implementation is under
[`src/models/d3/`](../../src/models/d3/), and the worker is
[`src/frontend/d3.py`](../../src/frontend/d3.py).

Run the commands from the repository root on Linux with an NVIDIA GPU and
Python 3.12. The recipe records the procedure; measured results and verbatim
run output live in the pull request description, not here.

## Scope

The slice is:

- package: [`vllm-sr/d3`](https://huggingface.co/vllm-sr/d3) at the pinned
  revision below;
- the Python reference worker behind the Rust frontend, executing the
  checkpoint's own pinned runtime (`d3_runtime.py` and siblings) after a
  full-hash manifest gate;
- text/JSON state plus inline base64 media — `images` and `videos` lists on
  the five-field request envelope — and the three typed kinds — `choice`,
  `noul`, `score` — in one request;
- BF16 backbone with FP32 readout, noncausal full attention, last-position
  pooling, one forward pass per question.

Out of scope for this slice: upstream's FastAPI server features (API key,
request ids, server timing, `/v1/models`), and any native Rust/CUDA
executor.

## Media budgets and the transport cap

Per-item validation is the pinned runtime's strict loading
(`load_images`/`load_videos` with `strict=True`); a media item it cannot
decode rejects the whole request with HTTP 422 naming the item
(`images[i]: ...` / `videos[i]: ...`). The budgets, transcribed from the
pinned `d3_runtime.py` constants of this revision:

| Media | Budget |
| --- | --- |
| one image | PNG, JPEG or WebP data URL; at most 8,000,000 decoded bytes (base64 length ≤ 4·⌈8,000,000/3⌉ ≈ 10.7 MB, decoded length re-checked) and 16,000,000 source pixels |
| one video | MP4, WebM, QuickTime or Matroska data URL; at most 32,000,000 bytes (≈ 42.7 MB of base64), 300 seconds, 8,294,400 pixels per frame |
| video sampling | 2 frames per second, 4–32 frames spread over the video, each sampled frame at most 200,704 pixels |
| all videos of one request | at most 16,384 video input tokens |

Media capability follows the loaded checkpoint: when the vision tower, the
`qwen3_5` backbone or the video processor is missing, non-empty `images`/
`videos` are refused with the upstream server's messages (`This model reads
text only; images are not supported.` / `This model does not read videos.`).
Empty or absent lists never trigger the capability check.

On top of the per-item budgets this worker enforces a **64 MiB total request
body cap** (HTTP 413). Upstream sets no total-body limit — its budgets are
per item — so this is a deliberate difference, sized for one full-budget
video (~42.7 MB of base64) plus envelope headroom; a second full-budget
video in one request is refused with 413, not decoded. The cap also bounds
the memory the stdlib worker holds while reading a body.

## Timeout ceilings

- The Rust frontend applies a **60 s total timeout** per forwarded request
  (default, no environment override; only `OMNI_JEV_BIND`,
  `OMNI_JEV_BACKEND_URL`, `OMNI_JEV_HEALTH_TIMEOUT_MS` and
  `OMNI_JEV_MAX_RESPONSE_BYTES` are configurable). Uploading a near-budget
  video over a slow link plus video inference can exceed it and surface as
  a frontend 504.
- The worker's 15 s connection timeout is per `recv` on the request stream:
  a slow but continuously flowing upload does not trip it; only a stall
  longer than 15 s produces the 408 body timeout.

## Architecture and ownership

~~~text
Rust frontend
  -> Python worker
     -> admit: envelope (protocol), pinned-runtime question errors (preprocess)
     -> decode: media decode stage — empty short-circuit, capability gates,
        strict loading via the executor (frontend.d3 mirroring d3_server.py)
     -> verify: manifest identity pin (repo id + runtime digests), full
        SHA-256 of the manifest runtime table, then import it (engine)
     -> execute: the checkpoint's own d3_runtime prepare/run/respond
     -> finish: response validation + embedded-failure warning (postprocess)
~~~

The ownership boundary follows
[`docs/architecture.md`](../../docs/architecture.md) and the
[model contract](../../src/models/d3/README.md):

- `protocol.py` owns the five-field envelope and the media list-shape check;
- `preprocess.py` owns the pre-execution 422 mapping
  (`invalid_questions` map, plain `max_length_exceeded` message);
- `engine.py` owns the pinned runtime, the manifest gate stronger than the
  upstream default, the capability flags and the strict-loading wrappers;
- `postprocess.py` owns the pure answer formulas and the warning on
  embedded `invalid_model_output` answers (partial success stays HTTP 200
  but never passes silently);
- `frontend.d3` owns HTTP lifecycle and orchestration, including the media
  decode stage;
- The Rust frontend forwards `/v1/systemone` and `/health` unchanged.

## Pinned artifacts

| Artifact | Value |
| --- | --- |
| Package | [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) |
| Pinned revision | `cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75` |
| Manifest | `d3-package-manifest/1`, built 2026-10-11T03:30:34Z |
| Runtime gate | `repo_id` + `runtime.files_sha256` pinned per revision in `engine.py`, 9 files, full SHA-256 before import, no unlisted importables in the package root |
| License | Apache-2.0 (package metadata) |

The pinned revision predates later upstream commits (the repository's
`lastModified` is later than the manifest build time), which is exactly why
the revision pin plus the two manifest hash tables exist. Re-check before
changing the pin:

~~~sh
curl -sS "https://huggingface.co/api/models/vllm-sr/d3/revision/cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75" | jq -r '.sha'
.venv/bin/python - <<'PY'
import json, hashlib, pathlib
root = pathlib.Path("weights/d3")
manifest = json.loads((root / "MODEL_MANIFEST.json").read_text())
for name, digest in manifest["runtime"]["files_sha256"].items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    assert actual == digest, f"{name}: {actual} != {digest}"
print("runtime table: 9 files verified")
PY
~~~

## Install and prepare the artifacts

The environment installs from [`recipe/d3/requirements.txt`](requirements.txt),
a verbatim copy of the pinned package's own file (SHA-256 `57a19384…be8a32`,
matching the manifest's `runtime.files_sha256` entry; the worker's startup
gate re-verifies the package copy against the same digest). PyTorch wheels
must match the host CUDA runtime; the file's comment records that upstream
validated with transformers 5.17.0 on ROCm (torch 2.13).

~~~sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r recipe/d3/requirements.txt pytest
~~~

~~~sh
export D3_PKG="$PWD/weights/d3"
export D3_REV=cbff3a3cb0b074b6585fe9f747b9f9f5557a0e75
mkdir -p weights
.venv/bin/hf download vllm-sr/d3 --revision "$D3_REV" --local-dir "$D3_PKG"
~~~

The package is about 52 GB (26.09B BF16 parameters plus the FP32
(255, 5120) readout). The worker refuses to start when
`MODEL_MANIFEST.json` is missing, when the manifest is not the pinned
revision's own (a self-consistent manifest from a later snapshot carries a
different `repo_id` or runtime table), when any of the 9 runtime files
fails its full-hash check, or when the package root holds an importable
entry the runtime table does not name (an extra `*.py` file or
`__init__.py` package, which the `sys.path` setup would let shadow a
dependency) — that gate runs before any package code is imported.

## Start the worker

~~~sh
PYTHONPATH=src .venv/bin/python -m frontend.d3 \
  --model "$D3_PKG" --device cuda:0 --port 8000
~~~

The worker verifies the runtime table, loads the package through the
pinned runtime's own `verify="fast"` path (pass `--verify full` to hash
every file, a minutes-level operation over 52 GB), compiles the kernels via
warmup, and only then binds the port:

~~~sh
curl -sS -i http://127.0.0.1:8000/health
~~~

Expected shape (the pinned package reports no input limit):

~~~json
{"status":"ready","model":"d3","max_input_tokens":null,"modalities":["text","image","video"]}
~~~

## Example request

The complete runnable request is in
[`example-request.json`](example-request.json): one text state with one
`choice`, one `noul` and one `score` question.

~~~sh
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/d3/example-request.json
~~~

Expected response shape:

~~~json
{
  "model": "d3",
  "answers": {
    "refund": {"type": "choice", "choice": "...", "probabilities": {"...": 0.0}, "confidence": 0.0},
    "urgent": {"type": "noul", "noul": 0.0},
    "severity": {"type": "score", "score": 0.0, "probabilities": {"0": 0.0}, "confidence": 0.0,
                  "legend": {"0": "minor", "1": "moderate", "2": "major", "3": "critical"}}
  },
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
~~~

### Media request

[`example-request-media.json`](example-request-media.json) is the same shape
with one inline PNG data URL in `images` — a screenshot-style input every
question of the request sees:

~~~sh
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/d3/example-request-media.json
~~~

The response shape is unchanged; `usage.input_tokens` accounts for the image
tokens. For a video request, generate a small in-budget clip and build the
data URL yourself (the base64 is too large to commit as an example):

~~~sh
ffmpeg -f lavfi -i testsrc=duration=4:size=320x240:rate=8 \
  -pix_fmt yuv420p -c:v libx264 -preset veryfast -crf 30 /tmp/d3-clip.mp4
base64 -w0 /tmp/d3-clip.mp4 > /tmp/d3-clip.b64
python3 - <<'PY'
import json, pathlib
encoded = pathlib.Path("/tmp/d3-clip.b64").read_text().strip()
request = json.loads(pathlib.Path("recipe/d3/example-request.json").read_text())
request["videos"] = [f"data:video/mp4;base64,{encoded}"]
pathlib.Path("/tmp/d3-video-request.json").write_text(json.dumps(request))
PY
curl -sS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/d3-video-request.json
~~~

Both are serving smoke tests, not accuracy results; the exact numbers come
from the checkpoint and are recorded by the parity tool below.

## Start and verify the Rust frontend

~~~sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  target/release/omni-jev
~~~

Send the same request through port 8080 and compare byte-for-byte:

~~~sh
curl -fsS http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/d3/example-request.json > /tmp/d3-worker.json
curl -fsS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/d3/example-request.json > /tmp/d3-frontend.json
cmp /tmp/d3-worker.json /tmp/d3-frontend.json
~~~

Worker responses are forwarded unchanged. Frontend-generated 502/504
responses are frontend errors, not model responses.

## Error behavior

The request, response and error contract — the three error layers, the
partial-success warning, answer shapes and usage counting — is the
[model contract](../../src/models/d3/README.md). In short: transport errors
follow the shared worker discipline; invalid or over-limit questions reject
the whole request with 422 (`{"invalid_questions": {...}}` or a plain
`detail` naming the token count); a question with undecodable model output
keeps its 200 with the failure embedded and a worker warning logged.

## Repository-side validation

~~~sh
PYTHONPATH=src .venv/bin/python -m pytest tests/d3 -q
PYTHONPATH=src .venv/bin/python -m compileall -q src/models/d3 src/frontend/d3.py tests/d3
.venv/bin/python recipe/d3/test_compare_reference.py
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
~~~

Run these from the repository root. The recorded outputs of a run belong in
the pull request description, not in this recipe.

## Reference parity (author-side, opt-in)

[`compare_reference.py`](compare_reference.py) drives one fixture through
two paths that share the loaded pinned model: the upstream
`d3_server.py` `system_one` body (media decode, prepare, 422 admission, run,
respond) and the production `frontend.d3.decide` pipeline. The criteria are
exact decision agreement including embedded `invalid_model_output` shapes
and agreed 422 rejections (media decode errors among them), equal token
accounting, and a numeric tolerance of 1e-6. Point `--fixture` at
`example-request-media.json` (or any media request) to compare the media
paths. `--export` freezes the oracle fixture (token ids, probabilities,
answers) for the future native worker, and writes it only after a passing,
non-rejected comparison — a failed run or an agreed 422 never touches the
existing fixture.

~~~sh
PYTHONPATH=src .venv/bin/python recipe/d3/compare_reference.py \
  --model "$D3_PKG" --device cuda:0 \
  --export tests/d3/fixtures/oracle.json
~~~

Run it by hand on a GPU host and record the verbatim output, host and
package revision in the pull request description. The flash-linear-attention
kernels and the PyTorch reference GDN implementation are not assumed to
agree numerically; measure the delta on this fixture before freezing any
tolerance for a native worker. The exported oracle fixture
(`tests/d3/fixtures/oracle.json`) is the frozen comparison baseline for the
stage-3 native worker and is committed with the integration.

## Troubleshooting and limits

- The package directory must be the pinned revision's download; the worker
  refuses missing manifests, foreign manifests (wrong `repo_id` or runtime
  table) and any runtime-file hash mismatch at startup.
- Upstream moved after the manifest was built — do not track `main` of the
  repository; re-pin deliberately and re-record the revision and the pinned
  runtime table (`PINNED_RUNTIME_SHA256`) in `src/models/d3/engine.py`.
- Media 422s name the item (`images[0]: ...`, `videos[1]: ...`) with the
  pinned runtime's message; non-empty media against a checkpoint without
  the capability is refused whole with the upstream server's message
  (`This model reads text only; images are not supported.` /
  `This model does not read videos.`). `/health`'s `modalities` shows what
  the loaded checkpoint reads.
- Over-budget totals are 413 (`request exceeds the 64 MiB body limit`):
  the cap is this worker's deliberate difference from upstream, sized for
  one full-budget video plus envelope headroom.
- An inline base64 video is written to a named temporary file by the pinned
  runtime before OpenCV decodes it (`d3-video-*` in the temp directory);
  that is upstream behavior, not a worker-side staging step.
- Decoding runs inside the worker's inference lock: one request at a time
  covers decode, prepare, run and respond (upstream decodes outside its
  lock in a thread pool — a deliberate throughput difference of this
  worker, recorded in the integration PR).
- The Rust frontend's 60 s total timeout has no override; a near-budget
  video upload plus inference can surface as a frontend 504 (see
  [Timeout ceilings](#timeout-ceilings)).
- `/health` has no `authentication` field: this worker has no API key
  (a deliberate difference from upstream, which reads `DECISION_API_KEY`).
- Without optional `causal_conv1d`, the pinned runtime falls back to its
  PyTorch reference implementation of that op (the runtime prints a notice);
  correctness is unaffected, only speed.
- The noncausal full-attention hook inside the pinned runtime is adapted
  from perplexity-ai/pplx-decider-v1.1-27b (Apache-2.0).
- GPU evidence for the supported-models table is recorded in the
  integration pull request, not in this recipe; media serving evidence is
  author-side opt-in and likewise lives there until measured.
