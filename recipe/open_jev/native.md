# Open-Jev-27B-v1.1 native text worker

The worker owns request compilation, tokenization, candidate scoring and typed
responses in Rust. It uses the native CUDA prefill implementation introduced in
[PR #19](https://github.com/ThinkFlowLab/system1-omni/pull/19), shared with Cua-S1
under [`src/models/qwen3_5/native/`](../../src/models/qwen3_5/native/).
Python is required only to prepare the merged checkpoint.

It supports `choice` (1–255 candidates), `score` (2–10 levels), and `noul`
(yes/no). Each candidate has an independent prompt; the last hidden state goes
through Open-Jev's trained FP32 scalar head. Noul uses logits `[0, score]`.
The saved calibration temperature is applied before normalizing each complete
question. There is no autoregressive generation. Structured state and descriptions
use Open-Jev's sorted JSON rendering; question and candidate order is preserved.

## Prepare the checkpoint

Use the reference dependencies from
[Open-Jev @ 3308a15](https://github.com/Zefan-Cai/Open-Jev/tree/3308a15ccd7eea1df7a37d6ddc39b023b801ba16):
PyTorch 2.8 or newer, Transformers 5.10.2, PEFT 0.19.1, Accelerate 1.13.0,
and safetensors. An optional `kernels` installation must be compatible with that
Transformers release. Run these commands from the repository root:

```sh
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir weights/Qwen3.8-27B
hf download ZefanCai/Open-Jev-27B-v1.1 \
  --revision 28cf73067d5b337860bbef3c85b8b82ba8730956 \
  --include 'package/checkpoint/*' --local-dir weights/Open-Jev-27B-v1.1
CUDA_VISIBLE_DEVICES='' python recipe/open_jev/export_merged.py \
  --base weights/Qwen3.8-27B \
  --checkpoint weights/Open-Jev-27B-v1.1/package/checkpoint \
  --out weights/open-jev-27b-merged
```

CPU export needs roughly 110 GB of RAM and 52 GB of output storage. It merges
LoRA in BF16 and saves the trained head, temperature, and single-user chat
template in `open_jev_export.json`. The worker refuses a plain base checkpoint
or an incomplete export. The saved limit defaults to 4096 tokens per candidate;
`--max-length` may raise it to 16384. Oversize prompts fail before inference.

## Build and serve

The CUDA kernels require compute capability 8.0 or newer. The current build
target below is Ada (`89`); pass your GPU's compute capability explicitly.
The CUDA shared library and both Rust workers must be rebuilt together because
the gated-attention entry point updates the library ABI to version 4 alongside
the shared CUDA Graph entry points.

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-open-jev-native -p omni-jev
OPEN_JEV_MODEL=weights/open-jev-27b-merged \
  target/release/omni-open-jev-native
```

`OPEN_JEV_HOST` and `OPEN_JEV_PORT` default to `127.0.0.1` and `8000`.
`OPEN_JEV_CUDA_LIB` overrides the default library next to the executable.
The worker loads all text weights onto visible CUDA device 0, performs a real
warmup inference, then exposes `/health` and `/v1/systemone`.
Use a reservation before any GPU command on hosts with a GPU scheduler.

In another terminal, start the existing Rust frontend:

```sh
OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  target/release/omni-jev
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @recipe/open_jev/example-request.json
```

The worker accepts the model's base name `Qwen/Qwen3.8-27B`, `open-jev`,
`jev-latest`, and `open-jev-27b-v1.1`; the response model is the base name,
matching Open-Jev. Error wording and metadata differ from the reference service.
Requests are bounded to 4 MiB, 4096 questions, and 65536 candidate sequences.

## Validation and optimization scope

Tests and fixtures live in the repository-level `tests/` tree: Open-Jev's typed
contract and tokenizer cases are in
[`tests/open_jev/`](../../tests/open_jev/), and shared Qwen JSON, configuration
and CUDA reference tests are in [`tests/qwen3_5/`](../../tests/qwen3_5/).
The default suites below run on CPU without downloading model weights:

```sh
cargo test --locked -p omni-open-jev-native -p omni-qwen3-5-native
cargo test --locked -p omni-jev --test frontend
```

The frontend mock-worker API coverage is tracked in
[issue #46](https://github.com/ThinkFlowLab/system1-omni/issues/46) and
[PR #58](https://github.com/ThinkFlowLab/system1-omni/pull/58). Checkpoint tokenizer
and CUDA kernel tests are opt-in; the latter require a GPU reservation:

```sh
# Inside a GPU reservation, after building the library:
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-qwen3-5-native --test kernels -- --ignored
```

CPU golden fixtures come from Open-Jev's request compiler and response formatter
at the revision above. Kernel tests compare attention and Gated DeltaNet with
float64 references and require exact BF16 equality between fused attention gating
and a separate gate pass. Residual RMSNorm and packed SiLU are checked against
rounded references, including odd widths and unaligned pointers. The shared
kernel tests retain PR #19's
`CUA_S1_CUDA_LIB` environment variable.

The worker reuses PR #19's fused norm, activation, QK/RoPE and chunked Gated
DeltaNet operations. Attention's sigmoid gate is fused into its output epilogue,
preserving both BF16 rounding points and removing one launch and one output
read/write pass per full-attention layer (16 layers for this model). Residual
RMSNorm keeps thread values in registers at widths 2560/5120. MLP SiLU uses
16-byte BF16 loads/stores when width, stride and pointers permit it, retaining
both BF16 rounding points; other layouts use the scalar path.

This recipe leaves `CUA_S1_GRAPH` unset and runs one eager forward pass per
candidate. Set `CUA_S1_GRAPH=1` on the worker to enable CUDA Graph replay. The
shared backend retains at most 64 graphs, keyed by exact candidate token length;
growing the scratch buffer clears them. Capturing a new length first runs an
eager forward to initialize its plans, then captures and replays the forward.
This adds cost for new lengths, so graph mode remains opt-in. Warm replay is
validated on the 74-case H200 workload: its mean HTTP latency is 2.03% below
eager execution after all workload lengths are warmed. Tokenization, transfers
and the CPU scalar head remain outside the graph. Prefix sharing, GEMM autotuning,
quantization and multimodal inference are not implemented. The
[H200 validation](validation.md) reports full-checkpoint results for 74
single-candidate requests, including probability differences and timing
variability. It does not establish general accuracy parity or a speedup over
OpenJev-Fast; the author's B300 results use different hardware and workloads.
