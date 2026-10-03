# Cua-S1 4B 0.2 native text worker

The native worker ([`src/models/cua_s1/native/`](../../src/models/cua_s1/native/)) serves the `text` adapter like the reference worker in [`text.md`](text.md), with the Qwen3.5-4B forward pass on the CUDA kernels in [`src/backends/cuda/qwen3_5/`](../../src/backends/cuda/qwen3_5/) and no Python or PyTorch. It needs an NVIDIA GPU with compute capability 8.0 or newer; only an RTX 6000 Ada (sm_89) with CUDA 13.2 has been run.

Run the commands from the repository root. Pass your GPU's compute capability to `build.sh` (89 for Ada, 80 for A100, 90 for H100); the worker finds `libqwen3_5_cuda.so` next to its executable, or at `CUA_S1_CUDA_LIB`:

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89   # needs nvcc and cuBLASLt
cargo build --release --locked -p omni-cua-s1-native
```

The worker loads the weights with the `text` adapter merged in. With the reference worker's environment and weights from `text.md`, export them once (about 8.5 GB):

```sh
PYTHONPATH=src .venv/bin/python recipe/cua_s1/export_text_merged.py \
  --base weights/Qwen3.5-4B --adapter weights/cua-s1-4b-0.2/text \
  --out weights/cua-s1-4b-0.2-text-merged
```

Start the worker (`CUA_S1_HOST` and `CUA_S1_PORT` default to `127.0.0.1` and `8000`), then the frontend and requests as in `text.md`:

```sh
CUA_S1_MODEL=weights/cua-s1-4b-0.2-text-merged target/release/omni-cua-s1-native
```

For the local CUDA Graph experiment, also set `CUA_S1_GRAPH=1`. The first use of
each exact prompt length warms the GEMM plans and captures the forward pass;
later requests replay it with freshly uploaded token ids. At most eight lengths
are cached. Growing the scratch allocation clears the captures before freeing
their buffers. Capture adds first-use latency; leave the variable unset to use
the eager control. Rebuild both the worker and CUDA library together (ABI 3).
If capture fails, the worker returns the completed eager result and disables
Graph capture/replay for its remaining lifetime, logging the failure to stderr.

Each question runs one forward pass over its prompt, eagerly by default or through exact-length CUDA Graph replay when enabled; the final hidden state at the last position times the 26 letter rows of the output projection gives the option probabilities. The probabilities are not bitwise identical to the reference worker's, since the adapter is merged and the kernels differ; they are held to the tolerance in [`src/models/cua_s1/README.md`](../../src/models/cua_s1/README.md#validation). Error messages are worded differently, and bodies nested more than 127 levels deep are refused.

The request tests need no GPU; the kernel tests compare attention and the chunked Gated DeltaNet prefill with float64 references:

```sh
cargo test -p omni-cua-s1-native
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release -p omni-cua-s1-native --test kernels -- --ignored
```
