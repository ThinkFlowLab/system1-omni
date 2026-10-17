# Native Cua-S1 screenshot serving

The `omni-cua-s1-vision` worker serves the Cua-S1 4B 0.2 `multimodal`
adapter with native Rust/CUDA vision and language execution. It accepts one
inline PNG/JPEG screenshot and 1–8 independent `choice` questions. The vision
encoder runs once per request; each question keeps its own language prompt.
See the [model contract](../../src/models/cua_s1/README.md) for pinned revisions
and the [vision Graph recipe](native_vision_graph.md) for device tests and controls.

## Prepare and build

Prepare the pinned base and adapter using the [reference setup](text.md), keeping
the upstream `weights.lock.json` alongside the downloaded artifacts. Use the
reference Python environment only for this one-time language export. The vision
worker uses the original BF16 vision tensors and unmerged FP32 vision LoRA:

```sh
PYTHONPATH=src .venv/bin/python recipe/cua_s1/export_multimodal_language.py \
  --base weights/Qwen3.5-4B \
  --adapter weights/cua-s1-4b-0.2/multimodal \
  --out weights/cua-s1-multimodal-language

src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-cua-s1-native -p omni-jev
```

The export refuses an existing output directory. Keep the export and original
weights immutable while serving. `89` targets the validated RTX 4090; choose your
device's supported compute capability when building. Rebuild the CUDA library
and every native consumer together when the ABI changes. This branch requires
ABI 8, including the default host-feature path.

## Start the worker and frontend

```sh
CUA_S1_BASE=weights/Qwen3.5-4B \
CUA_S1_VISION_ADAPTER=weights/cua-s1-4b-0.2/multimodal \
CUA_S1_MODEL=weights/cua-s1-multimodal-language \
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
CUA_S1_HOST=127.0.0.1 CUA_S1_PORT=8000 \
CUA_S1_GRAPH=0 CUA_S1_VISION_GRAPH=0 CUA_S1_DEVICE_FEATURES=0 \
target/release/omni-cua-s1-vision
```

In another terminal, start the existing Rust frontend:

```sh
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 target/release/omni-jev
curl --fail http://127.0.0.1:8080/health
```

Create a request from a local screenshot, then send it through the frontend:

```sh
python3 - screenshot.png > /tmp/cua-screenshot-request.json <<'PY'
import base64, json, pathlib, sys
image = base64.b64encode(pathlib.Path(sys.argv[1]).read_bytes()).decode()
json.dump({
    "model": "cua-s1-4b-0.2",
    "state": {"image": "data:image/png;base64," + image},
    "questions": {"pick": {
        "type": "choice", "instructions": "Keep the files.",
        "criteria": {"delete": "Click Delete", "cancel": "Click Cancel"}
    }}
}, sys.stdout)
PY
curl --fail-with-body http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/cua-screenshot-request.json
```

Use matching `data:image/jpeg;base64,` for a JPEG. Returned option probabilities
and the selected key depend on the screenshot. The response identifies the
multimodal adapter and reports prompt-token usage with zero generated tokens.

## Execution controls and limits

`CUA_S1_GRAPH=1` enables host-feature language Graph replay;
`CUA_S1_VISION_GRAPH=1` independently enables vision Graph replay.
`CUA_S1_DEVICE_FEATURES=1` keeps vision features on the GPU and copies image rows
directly into language scratch. Its language path remains eager; it can be paired
with either vision Graph setting. All three controls default off. See the
[resident-feature contract](native_vision_graph.md#gpu-resident-vision-features)
for strict control parsing, synchronization and reproduction commands.

Requests are serialized on the loaded GPU executor. HTTP concurrency includes
preparation and queueing; there is no cross-request GPU batch. Input is limited
to an 8 MiB body, 4 MiB image file, dimensions up to 2048 on each axis
and 1,048,576 pixels, one image, and 1–26 options per question. Animated images,
remote image URLs, multi-image state, `score`, `noul` and native Metal are unsupported.

Recorded RTX 4090 evidence covers native screenshot parity and Graph regressions;
the resident-feature comparison covers host/device output equality on its fixed
real-screenshot corpus. These scopes do not establish general GUI task accuracy
or native equivalence to every reference precision and environment.
