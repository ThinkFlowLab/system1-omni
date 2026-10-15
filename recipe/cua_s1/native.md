# Cua-S1 4B 0.2 native text worker

The native worker ([`src/models/cua_s1/native/`](../../src/models/cua_s1/native/)) serves the `text` adapter like the reference worker in [`text.md`](text.md), with the Qwen3.5-4B forward pass on the CUDA kernels in [`src/backends/cuda/qwen3_5/`](../../src/backends/cuda/qwen3_5/) and no Python or PyTorch. It needs an NVIDIA GPU with compute capability 8.0 or newer.
The unmerged precision profile has source-pinned A800 80 GB (sm_80), CUDA 13.0
evidence at ABI-8 source `7293c71cfb738504a30ff033ae3567f3368333c1`, plus earlier
cumulative ABI-8 and independent ABI-6 campaigns linked in
[PR #143](https://github.com/ThinkFlowLab/system1-omni/pull/143).
The current branch uses ABI 8. The legacy-readout compatibility fix was separately
GPU-tested at `f212c56f23ac750ebfa38f4c72e9fcdc914bf1cd` on the same pinned A800.
See the [source-pinned repair evidence](https://github.com/Levius-Fubuki/system1-omni/releases/tag/cua143-merged-readout-f212c56f-20261015).
The earlier merged path was checked on an RTX 6000 Ada (sm_89) with CUDA 13.2.

Run the commands from the repository root. Pass your GPU's compute capability to `build.sh` (89 for Ada, 80 for A100, 90 for H100); the worker finds `libqwen3_5_cuda.so` next to its executable, or at `CUA_S1_CUDA_LIB`:

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89   # needs nvcc, cuBLAS and cuBLASLt
cargo build --release --locked -p omni-cua-s1-native
```

With the pinned reference environment and weights from `text.md`, export the BF16 language weights and retain the original FP32 text adapter separately (about 8.5 GB):

```sh
PYTHONPATH=src .venv/bin/python recipe/cua_s1/export_text_unmerged.py \
  --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text \
  --out weights/cua-s1-4b-0.2-text-unmerged
```

Start the worker (`CUA_S1_HOST` and `CUA_S1_PORT` default to `127.0.0.1` and `8000`), then the frontend and requests as in `text.md`:

```sh
CUA_S1_MODEL=weights/cua-s1-4b-0.2-text-unmerged target/release/omni-cua-s1-native
```

For the local CUDA Graph experiment, also set `CUA_S1_GRAPH=1`. The first use of
each exact ordered prompt-length vector warms the GEMM plans and captures the forward pass;
later requests replay it with freshly uploaded token ids. Each language mode keeps a separate 64-entry FIFO cache. Growing the scratch allocation clears the captures before freeing
their buffers. Capture adds first-use latency; leave the variable unset to use
the eager control. Rebuild both the worker and CUDA library together (ABI 8).
If capture fails, the worker returns the completed eager result and disables
Graph capture/replay for its remaining lifetime, logging the failure to stderr.

Each question runs one forward pass over its prompt. The `cua-s1-text-unmerged/1`
marker selects framework-compatible BF16 normalization and cuBLASLt GEMM,
FP32 LoRA branches, and FP32 chunked Gated DeltaNet. The GDN cumulative decay
uses the pinned Torch scan order; state is reset for every prompt. The letter
projection rounds its result to the BF16 `lm_head` output dtype before the existing
per-question softmax. Native finishing normalizes in FP64 and returns FP32
probabilities.
The marker records scale 2 and the SHA-256 of the unchanged adapter; incompatible
markers or altered adapter bytes fail before CUDA loading. Prefix continuation
is unsupported in this precision mode.

Existing `cua-s1-text-merged/1` exports remain loadable with the earlier merged
backbone execution path and retain FP32 letter logits after FP64 accumulation.
The BF16 readout boundary is restricted to the validated unmerged export profile,
not a request flag or process-global setting. An earlier #143 build applied it to
merged exports too, creating four decision changes and reducing frozen-corpus
correctness from 97/139 to 96/139. At repaired source `f212c56f`, merged eager and
Graph each matched every parent response byte in two 139-request passes, with
zero upgrade flips and 97/139 correct. Unmerged eager/Graph retained zero flips
against the pinned reference. Results remain scoped to that source and artifact.
Merging the adapter, TF32/BF16 GDN intermediates, and different
GEMM reduction policies can change decisions on close or tied options. Re-export
with the command above to use the precision profile. Rebuild worker and library
together: this branch now uses ABI 8, and the precision profile also requires the new optional
FP32 GDN symbols. No cuDNN or Python is needed at serving time.

The validation scope and tolerances are in
[`src/models/cua_s1/README.md`](../../src/models/cua_s1/README.md#validation).
Error messages are worded differently, and bodies nested more than 127 levels
deep are refused.

The request tests need no GPU; the kernel tests compare attention and the chunked Gated DeltaNet prefill with float64 references:

```sh
cargo test -p omni-cua-s1-native
# In the pinned Python reference environment, on the same GPU:
.venv/bin/python tests/qwen3_5/test_gdn_fp32.py \
  --library target/release/libqwen3_5_cuda.so
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release -p omni-qwen3-5-native --test kernels -- --ignored
```

The maintained legacy-upgrade regression uses the four frozen requests and
parent-baseline responses in `tests/cua_s1/fixtures/legacy_merged_readout.json`.
Those responses are compatibility oracles, not task ground truth. With the same
pinned A800 merged artifact, run each Graph setting in a separate process:

```sh
for graph in 0 1; do
  CUA_S1_GRAPH=$graph NVIDIA_TF32_OVERRIDE=0 \
    CUA_S1_MERGED_TEST_MODEL=$PWD/weights/cua-s1-4b-0.2-text-merged \
    CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
    cargo test --release --locked -p omni-cua-s1-native --lib \
      engine::legacy_merged_tests::legacy_merged_upgrade_preserves_four_decisions \
      -- --ignored --exact --test-threads=1 --nocapture
done
```

This selects one device test per process, running all four requests twice through
prepare/execute/finish. It requires unchanged baseline decisions, probability
error at most 0.02, identity/usage/option order and exact repeat bytes. It does
not substitute for the full 139-request legacy upgrade and unmerged-reference
comparisons. See the [four-case diagnosis and retest scope](../../docs/benchmarks/cua-s1-text-parity-20261013/README.md#legacy-merged-upgrade-regression).

For the screenshot worker, [the vision Graph recipe](native_vision_graph.md)
documents `CUA_S1_VISION_GRAPH`, retained four-grid FIFO scratch, memory/lifetime
limits and the real CUDA capture-failure wrapper build and test commands.

The ABI-8 parent synchronization preserves the unmerged precision profile.
The source-pinned GPU campaigns are linked in
[PR #143](https://github.com/ThinkFlowLab/system1-omni/pull/143); the
[numerical investigation](../../docs/benchmarks/cua-s1-text-parity-20261013/README.md)
records their scopes and the legacy-readout follow-up. Rebuild consumers together and rerun
the pinned device checks before extending qualification to this build or
another device.
