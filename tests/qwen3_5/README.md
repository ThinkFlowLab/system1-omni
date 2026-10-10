# Native multimodal language Graph regressions

These private-item tests live under root `tests/` and are wired into the
`omni-qwen3-5-native` library with `#[cfg(test)]`. Cargo compiles and discovers
all three cases without a generated source copy, a preparation script or manifest
changes. The wiring is excluded from release builds.

Ordinary `cargo test --workspace --locked` leaves the three cases ignored:
standard CI has no CUDA device or model weights. It still compiles their bodies.
Actual execution requires a real NVIDIA CUDA device, a supported BF16 language
checkpoint with `image_token_id` and at least 2048 positions, and the Qwen native
CUDA library. The base checkpoint's F32 tensors need the existing native export;
see [the multimodal language exporter](../../recipe/cua_s1/export_multimodal_language.py)
and [native build instructions](../../recipe/cua_s1/native.md).

```sh
export CUA_S1_TEST_MODEL=/absolute/path/to/merged-bf16-language
export CUA_S1_CUDA_LIB=/absolute/path/to/libqwen3_5_cuda.so
python3 tests/qwen3_5/run_graph_regressions.py \
  --log /tmp/native-mm-graph-gpu.log --result /tmp/native-mm-graph-result.json
```

The runner uses locked, offline, release Cargo execution. Fetch dependencies
beforehand (`cargo fetch --locked`) when a populated Cargo cache/vendor directory
is unavailable. Tests run serially in one process, with `CUA_S1_GRAPH_TRACE`
unset, to avoid shared CUDA device/API state interference. It requires exactly
three passing cases and one unconditional capture-failure diagnostic per mode.
It returns nonzero on missing hardware/weights, Cargo failure or a failed gate;
none of those are treated as skips.

Coverage:

- Capture misses and replay agree exactly with finite eager hidden states;
  changed tokens, BF16 image rows and distinct T/H/W positions are uploaded;
  text/multimodal interleave and scratch growth permit fresh captures. Text
  singleton `[4]` and packed `[1, 3]` / `[3, 1]` keep distinct ordered cache
  keys despite equal total tokens, and replay with changed token IDs.
- 65 lengths exercise the separate 64-entry multimodal FIFO, unchanged scratch
  addresses, hits without insertion-order refresh, eviction/recapture and a
  surviving text replay.
- A real CUDA recording closure returns an injected error. The production
  capture-result handler preserves the completed eager result, clears both
  caches, disables Graph, emits diagnostics and allows repeated public eager
  forwards in both modes. This does not simulate every CUDA failure class.

These are executor regressions, not a screenshot-encoder or agent-accuracy
benchmark. No latency, capture-cost or memory claim is made by these tests.
