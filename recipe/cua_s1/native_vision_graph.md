# Native Cua-S1 vision CUDA Graph

The `omni-cua-s1-vision` screenshot worker can capture the complete vision
encoder, including all 24 blocks, unmerged FP32 LoRA branches and the merger.
Set `CUA_S1_VISION_GRAPH=1` before loading the worker to enable it. Leave it
unset or set it to `0` for eager execution. This switch is independent of
`CUA_S1_GRAPH`, which controls language execution. Enabling vision Graph alone
does not enable language Graph replay.

Both modes use the shared Qwen vision executor and retain at most four exact
[T,H,W] scratch/Graph pairs in FIFO order. Equal patch counts with different
heights or widths remain distinct. Before eviction the worker synchronizes,
destroys the old executable, and releases its buffers. Pixel values are uploaded
on every request, including replay; geometry tables belong to their exact grid.
Each retained grid has at most 4608 patches. This bounds the number of retained
activations, not their total bytes, loaded weights or the GEMM plan cache.

The first ordinary forward for a grid finishes eagerly, then captures operators
for subsequent requests and returns the completed eager result. Capture time is
part of that first request. Uploads, downloads and CPU processing stay outside
the Graph. Capture failure always logs to stderr and disables vision Graph for
that loaded model; the completed eager result is preserved. Replay failures
propagate as inference errors. `forward_with_trace` runs eagerly with all stage
downloads and leaves a matching Graph available for later ordinary forwards.

For optional capture/replay messages, set `CUA_S1_GRAPH_TRACE=1`. Failure messages
are unconditional. Requests execute serially through the existing worker runtime;
this feature does not add batching or concurrent forwards.

## Build and launch

Use the pinned base and multimodal adapter revisions listed in the
[model contract](../../src/models/cua_s1/README.md#pinned-revisions), and the
verified language export produced by `export_multimodal_language.py`. Keep all
checkpoint files immutable during execution. Build the worker and the current
CUDA library together; this change adds no CUDA ABI symbols.

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-cua-s1-native

CUA_S1_BASE=weights/Qwen3.5-4B \
CUA_S1_VISION_ADAPTER=weights/cua-s1-4b-0.2/multimodal \
CUA_S1_MODEL=weights/cua-s1-multimodal-language \
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
CUA_S1_VISION_GRAPH=1 target/release/omni-cua-s1-vision
```

Set `CUA_S1_PORT` to choose the listening port (default 8000). Only the existing
single-image native CUDA path is covered. Python workers and Metal are unchanged.

## Regressions and native measurements

Export the same `CUA_S1_BASE`, `CUA_S1_VISION_ADAPTER`, `CUA_S1_MODEL`, and
`CUA_S1_CUDA_LIB` variables for these commands:

```sh
cargo test --release --locked -p omni-qwen3-5-native --lib vision_graph_ \
  -- --ignored --test-threads=1 --nocapture
CUA_S1_VISION_GRAPH=0 cargo run --release --locked -p omni-cua-s1-native \
  --example vision_graph_bench -- /tmp/vision-eager.json
CUA_S1_VISION_GRAPH=1 cargo run --release --locked -p omni-cua-s1-native \
  --example vision_graph_bench -- /tmp/vision-graph.json
```

The capture-failure test uses a separate process and a test-only wrapper library.
It delegates all math to the real CUDA library, ends the real capture, destroys
the executable, then reports an instantiation failure. No production backend is
modified. Set an absolute path for the real library:

```sh
export CUA_S1_REAL_CUDA_LIB="$CUA_S1_CUDA_LIB"
g++ -std=c++17 -shared -fPIC -Wall -Wextra tests/cua_s1/fail_graph_capture.cpp \
  -Wl,--no-as-needed "$CUA_S1_REAL_CUDA_LIB" -ldl \
  -o /tmp/cua-fail-capture.so
CUA_S1_CUDA_LIB=/tmp/cua-fail-capture.so CUA_S1_TEST_CAPTURE_FAILURE=1 \
CUA_S1_GRAPH_TRACE=0 cargo test --release --locked -p omni-qwen3-5-native \
  --lib vision_capture_failure_ -- --ignored --test-threads=1 --nocapture
```

The root GPU tests cover changed pixels, exact-grid residency and eviction, retirement,
trace bypass, invalid input and capture-failure fallback. Ordinary CI compiles
these tests but skips device execution. Run them explicitly on a reserved GPU.

The benchmark prepares ten deterministic RGB screenshots for each of three
sizes (256x256, 512x256, 512x512), with two choice questions each. It reports model
loading and first vision/capture time separately, excludes three warmup decisions
per shape, and retains every timed vision call and complete native response.
`vision_ms` includes validation, BF16 conversion, upload, forward, synchronized
download and output conversion. `native_decision_ms` includes vision, both eager
language forwards, letter projections and response construction; it excludes CPU
preparation, HTTP and model loading. Feature SHA256 and complete responses must
match between modes. These serial synthetic measurements do not establish
labelled task quality, production throughput, HTTP latency or a universal speedup.
Preserve two repetitions with reversed mode order, raw outputs, binary/source
hashes and failures in external evidence rather than committing generated runs.
