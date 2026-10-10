# Native Decider-2B v11 text decisions

Run these commands from the repository root on Linux with Rust, an NVIDIA GPU,
CUDA toolkit/nvcc and cuBLASLt. Rebuild the library and every native Qwen consumer together for ABI **7**.
The existing Qwen backend targets compute capability
8.0 or newer; hardware validation is described in [validation.md](validation.md).
The worker itself uses no Python/PyTorch serving process. Python 3.10+ is used only
for downloading and optional reference validation. Keep at least 8 GB disk free for
weights and Rust build products; reference Python packages need additional space.
Device memory includes about 3.76 GB of BF16 checkpoint data plus model/head buffers
and length-dependent activation/workspace storage.

## Download and build

```sh
python3 recipe/decider/download_weights.py /path/to/models/decider-2b-v11
cargo build --release --locked -p omni-decider-native -p omni-jev
NVCC=/usr/local/cuda/bin/nvcc \
  bash src/backends/cuda/qwen3_5/build.sh target/release 89
```

Replace `89` with the target GPU's compute capability. The download helper pins
`Mapika/decider-2b` revision `533964dae8be954c5b5e19fa4948e48408094c1e` and checks
all five downloaded files; it never downloads unpinned remote Python code. Its
`--endpoint` option can select an accessible Hugging Face mirror; fixed hashes
remain required. Interrupted downloads leave no usable unverified final file.

Startup independently hashes the required checkpoint/config/calibration/tokenizer
and rejects changed artifacts before loading CUDA. Use the original single
`model.safetensors`, with no `model.safetensors.index.json`; no export or LoRA merge
is needed. Keep files immutable while loading and serving.

## Start the worker and frontend

Worker terminal:

```sh
DECIDER_GRAPH=0 \
DECIDER_MODEL=/path/to/models/decider-2b-v11 \
DECIDER_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
DECIDER_HOST=127.0.0.1 DECIDER_PORT=8000 \
  ./target/release/omni-decider
```

Decider defaults to eager. `DECIDER_GRAPH=1` opts the backbone into CUDA Graph
capture/replay; only 0/1 are accepted and an unset switch is off. `CUA_S1_GRAPH`
has no effect on Decider; other workers retain their existing switch.
`DECIDER_CUDA_LIB` defaults to the library beside the binary. The worker uses CUDA
device 0; use `CUDA_VISIBLE_DEVICES` to select a physical device.

The socket binds only after artifact validation, CUDA loading and a real warmup
request. Startup failure does not advertise readiness. In another terminal:

```sh
curl --fail http://127.0.0.1:8000/health
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

Health returns the model identity, checkpoint/reference revisions, BF16/effective execution mode
and effective per-type temperatures. A retired executor returns HTTP 503 and
`status: unavailable`. The frontend exposes its existing health behavior; see the
[frontend configuration](../../src/frontend/README.md) for its own limits/timeouts.

## First decision

```sh
curl --fail http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @recipe/decider/example-request.json
```

The example asks a Choice, Noul and three-level Score question about one text
state. The response has model `decider-2b-v11`, ordered typed answers and usage.
Question identities are omitted from model text. The exact output depends on the
checkpoint computation; this example is not a task-quality guarantee.

Errors use JSON `detail`: HTTP 415 for unsupported Content-Type, 413 for a body
above 8 MiB, 422 for invalid/unsupported requests, and 503 when the model becomes
unavailable after an execution failure. Request-preparation task failures return
500. Invalid inputs do not retire the executor. Empty questions return HTTP 200
with empty answers and zero input/output usage. No partial answers are returned.

Supported inputs are plain text/JSON state, independent Choice (2–255 options),
Noul and isolated Score (2–10 levels). State-prefix truncation is 32,768 tokens;
complete rows can include question suffixes up to 36,864 tokens. At most 1,024
expanded rows and 1,048,576 processed tokens are accepted per request. These bounds
do not bound all waiting requests together; runtime pending-queue limits and
cross-request batching remain separate work.

The pinned worker requires released calibration. Images, video, chat/schema-first,
packed questions, cross-request prefix caching and quantized/CPU/Metal execution
are unsupported. Request-local prefix reuse is opt-in through `DECIDER_PREFIX=1`
or the conservative `DECIDER_PREFIX=auto` policy; both require Graph off.
CUDA Graph replay is opt-in through `DECIDER_GRAPH=1`, as described below. See the [model contract](../../src/models/decider/README.md)
for native JSON restrictions and the processing/execution boundary.

## Validation and diagnostics

```sh
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
DECIDER_MODEL=/path/to/models/decider-2b-v11 \
  cargo test --release --locked -p omni-decider-native --test contract -- --ignored
DECIDER_MODEL=/path/to/models/decider-2b-v11 \
DECIDER_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" DECIDER_GRAPH=0 \
  cargo test --release --locked -p omni-decider-native --test gpu -- --ignored
DECIDER_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
  cargo test --release --locked -p omni-decider-native --lib \
  bf16_head_rounds_before_fp32_calibration -- --ignored
```

The CPU contract target requires only tokenizer/config; the GPU target requires the
complete pinned checkpoint. The projection test needs a built backend and a GPU.

For exact prepared rows and raw BF16-rounded logits, the diagnostic binary reads
one JSON request per line and emits rows, logits and complete response:

```sh
cargo build --release --locked -p omni-decider-native --example decider-run
python3 -c 'import json; print(json.dumps(json.load(open("recipe/decider/example-request.json"))))' > /tmp/decider-request.jsonl
DECIDER_GRAPH=0 ./target/release/examples/decider-run \
  /path/to/models/decider-2b-v11 "$PWD/target/release/libqwen3_5_cuda.so" \
  < /tmp/decider-request.jsonl
```

See [validation.md](validation.md) for full-checkpoint reference comparison,
HTTP/frontend checks, prerequisites and the recorded scope.

## Optional request-local packing

Set `DECIDER_BATCH_MAX_ROWS=4 DECIDER_BATCH_MAX_TOKENS=4096` on `omni-decider`
to pack independent rows within one admitted request. Rows default to 1; accepted
limits are 1–4 rows and 1–4096 packed tokens. Longer complete rows execute alone.
Malformed or out-of-range settings fail startup. These limits do not change HTTP
admission, token usage, or the maximum complete-row length.

The selected-label head uses persistent buffers sized to the row limit and projects
each batch in one BF16 GEMM. No cross-request batching is introduced. Keep the
single-row default until numerical and performance results fit your workload.

## Optional CUDA Graph replay

`DECIDER_GRAPH=1` works with the request-local batch limits above. The backbone
caches at most 64 captures by the **ordered sequence-length vector**, uploads
current token IDs before replay, and clears captures after synchronizing before
scratch growth. A shape miss runs eagerly once and records the warmed layer loop;
it does not launch that capture on the already-computed residual. Label projection
and response assembly remain outside the graph.

Health's `graph` object and the diagnostic runner's outer `graph` record report
requested/enabled mode, captures, replays, fallbacks, invalidations and cached shapes.
Decision-response fields remain unchanged. Capture failure disables Graph for the
worker lifetime and keeps the completed eager result; health then reports `eager`.
Launch/inference failures still retire the worker. Capture and startup costs must
be measured separately from warm replay.

## Optional request-local shared prefixes (pending integration)

This integration depends on the pending Qwen continuation/fixed-GEMM/executor
stack (#98/#99). Build this complete branch with ABI7; ABI5 and ABI6 libraries are
rejected. `DECIDER_PREFIX=1 DECIDER_GRAPH=0` plans exact shared token prefixes
within the admitted request and restores attention KV, GDN recurrent state and
three convolution-history rows for every branch. Contiguous rows for one question
can also share a longer question prefix. The executor ends reusable spans on
64-token boundaries and repeats their remainder in each suffix. Every branch
retains at least one token and original row order. Buffers persist but their
contents are valid only within one call; there is no cross-request cache.

Prefix execution uses the dependency's fixed GEMM algorithms. Set `DECIDER_FIXED=1`
with prefix/Graph off for the independent full-row control. With no usable prefix
(less than one reusable 64-token chunk), prefix mode executes independent fixed
rows. Both modes preserve the batch adapter's head grouping; backbone branches
run eagerly. Combining prefix/fixed with Graph or enabling prefix and fixed
together rejects startup. All new switches default off. Health reports effective
`shared`/`fixed` mode and cumulative `prefix.shared_requests`/`saved_tokens`.
The diagnostic outer record includes the same counters; decision output and usage
do not count execution savings differently.

## Conservative automatic prefix selection

`DECIDER_PREFIX=auto` is a default-off alternative to forced `DECIDER_PREFIX=1`.
The worker uses shared fixed-GEMM execution only when the existing aligned prefix
plan saves at least 4096 tokens and at least one third of the original row-token
work. Other requests use the normal independent/packed eager path, including
single-row and short/no-sharing requests. This avoids forcing short work through
the slower fixed-GEMM path. These are conservative host-policy thresholds, not a
guarantee of faster execution on every GPU or workload.

`DECIDER_PREFIX=1` and `DECIDER_FIXED=1` retain their existing forced semantics.
Auto, forced prefix and fixed modes reject `DECIDER_GRAPH=1`; all switches default
off. Auto retains request-local state only. Input/output contracts and token usage
are unchanged, but normal and fixed GEMM algorithms can differ numerically: check
the original reference gates for both selected paths. Health reports `prefix_mode:
auto`, total shared requests/saved tokens and `auto_independent_requests`; counts
include the readiness warmup.

```sh
DECIDER_PREFIX=auto DECIDER_GRAPH=0 \
DECIDER_BATCH_MAX_ROWS=4 DECIDER_BATCH_MAX_TOKENS=4096 \
DECIDER_MODEL=/path/to/models/decider-2b-v11 \
DECIDER_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
  target/release/omni-decider
```
