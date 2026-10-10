# Native Cua-S1 vision CUDA Graph

The `omni-cua-s1-vision` screenshot worker can capture the complete vision
encoder, including all 24 blocks, unmerged FP32 LoRA branches and the merger.
Set `CUA_S1_VISION_GRAPH=1` before loading the worker to enable it. Leave it
unset or set it to `0` for eager execution. This switch is independent of
`CUA_S1_GRAPH`, which controls language execution; this change does not enable
multimodal language Graph replay.

Both modes retain scratch by the exact `[T,H,W]` patch grid. The default entry
limit is 1; set `CUA_S1_VISION_CACHE_ENTRIES=4` to retain four geometries across
size changes. The entry setting is capped at 16; zero disables retention. Equal
patch counts with different heights or widths are distinct keys. The active
scratch is most recently used; a hit restores its existing buffers and Graph,
while a miss synchronizes and evicts least recently used entries before allocating
the new shape. Each executable is destroyed before its referenced buffers.
Pixel values are uploaded on every request, including replay. Geometry tables
remain valid only for that exact grid.

`CUA_S1_VISION_CACHE_BYTES` bounds total requested bytes in retained scratch,
defaulting to 268435456 (256 MiB). The fixed 4B layout requests 64096 bytes per
patch, including pixels, position/rotary tables, LoRA work, activations and output.
The four tested grids contain 2304 patches in total and retain 147677184 bytes
(140.84 MiB). This budget excludes weights, GEMM workspace/plans, Graph metadata
and allocator/runtime overhead. Trace messages report retained entries/bytes;
measure actual device memory separately.

Both limits are read once when the model loads. Unset values use their defaults;
an explicitly invalid unsigned integer (including trailing whitespace or a
negative value) logs a warning to stderr and uses that limit's default. The
warning names the setting, supplied value and effective fallback, independently
of `CUA_S1_GRAPH_TRACE`. Zero for either limit disables retention.

A valid geometry larger than the byte budget, or any geometry with retention
disabled, executes eagerly using transient scratch. It preserves existing
cache entries, synchronizes on success or error and releases the transient
buffers after the result is read. The default byte budget therefore prevents
retention even for a single oversized grid (the processor permits up to 4608
patches); increase it deliberately if that workload needs retention. Temporary
scratch required to answer such a request is additional peak memory, so this is
a retained-activation budget rather than a whole-device memory limit.

The first ordinary forward for a retained grid finishes eagerly, then captures operators
for subsequent requests and returns the completed eager result. Capture time is
part of that first request. Uploads, downloads and CPU processing stay outside
the Graph. Capture failure always logs to stderr and disables vision Graph for
that loaded model; the completed eager result is preserved. Replay failures
propagate as inference errors. `forward_with_trace` runs eagerly with all stage
downloads and preserves an already matching Graph for later ordinary forwards;
it does not capture a new Graph itself.

For optional cache and capture/replay messages, set `CUA_S1_GRAPH_TRACE=1`. Failure messages
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
CUA_S1_VISION_CACHE_ENTRIES=4 \
CUA_S1_VISION_CACHE_BYTES=268435456 \
CUA_S1_VISION_GRAPH=1 target/release/omni-cua-s1-vision
```

Set `CUA_S1_PORT` to choose the listening port (default 8000). Only the existing
single-image native CUDA path is covered. Python workers and Metal are unchanged.

## Regressions and native measurements

Export the same `CUA_S1_BASE`, `CUA_S1_VISION_ADAPTER`, `CUA_S1_MODEL`, and
`CUA_S1_CUDA_LIB` variables for these commands:

```sh
cargo test --release --locked -p omni-cua-s1-native --lib vision_graph_ \
  -- --ignored --test-threads=1 --nocapture
CUA_S1_VISION_CACHE_ENTRIES=4 cargo test --release --locked \
  -p omni-cua-s1-native --lib vision_cache_ \
  -- --ignored --test-threads=1 --nocapture
CUA_S1_VISION_CACHE_ENTRIES=4 CUA_S1_GRAPH_TRACE=1 \
  cargo test --release --locked -p omni-cua-s1-native --lib vision_cache_aba_probe \
  -- --ignored --test-threads=1 --nocapture > /tmp/vision-cache-aba.log 2>&1
python3 tests/cua_s1/vision_cache_trace_probe.py /tmp/vision-cache-aba.log
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
CUA_S1_VISION_CACHE_ENTRIES=4 CUA_S1_GRAPH_TRACE=0 \
  cargo test --release --locked -p omni-cua-s1-native \
  --lib vision_capture_failure_ -- --ignored --test-threads=1 --nocapture
```

The root GPU tests cover changed pixels, exact-grid identity, A/B/A reuse,
count/byte LRU eviction, transient oversize and zero retention, retirement,
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
