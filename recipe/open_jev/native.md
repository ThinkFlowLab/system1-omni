# Open-Jev native text worker

The worker owns request compilation, tokenization, candidate scoring and typed
responses in Rust. It uses the native CUDA prefill implementation introduced in
[PR #19](https://github.com/ThinkFlowLab/system1-omni/pull/19), shared with Cua-S1
under [`src/models/qwen3_5/native/`](../../src/models/qwen3_5/native/).
Python is required only to prepare the merged checkpoint.

The worker serves Open-Jev-27B-v1.1 or Open-Jev-9B. Both checkpoints use the
same method, prompt format and head design, each with its own trained head; the
export's `model_id` selects the pinned revisions, the accepted request model
names and the expected backbone dimensions.

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

For Open-Jev-9B, download its pinned base and checkpoint and export them the
same way:

```sh
hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir weights/Qwen3.5-9B
hf download ZefanCai/Open-Jev-9B \
  --revision 47e966881e489511c0c7f5633a9e1960a676a551 \
  --include 'package/checkpoint/*' --local-dir weights/Open-Jev-9B
CUDA_VISIBLE_DEVICES='' python recipe/open_jev/export_merged.py \
  --base weights/Qwen3.5-9B \
  --checkpoint weights/Open-Jev-9B/package/checkpoint \
  --out weights/open-jev-9b-merged
```

CPU export needs roughly 110 GB of RAM and 52 GB of output storage for 27B,
and about 20 GB of RAM and 16 GB of storage for 9B. It merges LoRA in BF16 and
saves the trained head, temperature, and single-user chat template in
`open_jev_export.json`. The worker refuses a plain base checkpoint
or an incomplete export. The saved limit defaults to 4096 tokens per candidate;
`--max-length` may raise it to 16384. Oversize prompts fail before inference.

## Build and serve

The CUDA kernels require compute capability 8.0 or newer. The current build
target below is Ada (`89`); pass your GPU's compute capability explicitly.
The CUDA shared library and Rust workers must be rebuilt together for ABI
version 7, which includes shared vision, CUDA Graph, canonical prefix continuation and fixed-GEMM selection.

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89
cargo build --release --locked -p omni-open-jev-native -p omni-jev
OPEN_JEV_MODEL=weights/open-jev-27b-merged \
  target/release/omni-open-jev-native
```

For 9B, set `OPEN_JEV_MODEL=weights/open-jev-9b-merged`.

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

The worker accepts the loaded checkpoint's base name (`Qwen/Qwen3.8-27B` or
`Qwen/Qwen3.5-9B`), `open-jev`, `jev-latest`, and its own alias
(`open-jev-27b-v1.1` or `open-jev-9b`); other names get HTTP 422. The response
and `/health` model is the base name, matching Open-Jev. The checkpoint aliases,
error wording and metadata differ from the reference service.
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
# Tokenizer cases against either merged export, on CPU:
OPEN_JEV_MODEL=weights/open-jev-9b-merged \
  cargo test --locked -p omni-open-jev-native --lib -- --ignored
# Inside a GPU reservation, after building the library:
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  cargo test --release --locked -p omni-qwen3-5-native --test kernels -- --ignored

# Full-checkpoint packing, ordering and graph-shape checks, in the reservation:
OPEN_JEV_MODEL=$PWD/weights/open-jev-27b-merged \
OPEN_JEV_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so CUA_S1_GRAPH=1 \
  cargo test --release --locked -p omni-open-jev-native --test prefill_batch \
  -- --ignored --test-threads=1
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
read/write pass per full-attention layer (16 layers on 27B, 8 on 9B). Residual
RMSNorm keeps thread values in registers at widths 2560/5120; 9B's width 4096
uses the general kernel. MLP SiLU uses
16-byte BF16 loads/stores when width, stride and pointers permit it, retaining
both BF16 rounding points; other layouts use the scalar path.

This recipe leaves `CUA_S1_GRAPH` unset and uses eager prefill. For
Open-Jev-27B-v1.1, candidates within a request are packed in prepared order, up
to 16 sequences and 4096 total tokens per group; longer prompts execute alone
without truncation. Input and gate/up
projections share GEMMs. Output/down projections preserve their per-prompt shapes
and reduction order, and each sequence retains independent attention, positions,
convolution and GDN state. Calibration still uses every candidate in its question.
The [H200 packing comparison](../../benchmarks/prefill_batching/README.md)
records latency, exact output checks and frozen controls.
Packing validation covers H200 (sm_90); other CUDA architectures remain unverified.
Open-Jev-9B runs one forward pass per candidate: packing is not validated for it.

Set `CUA_S1_GRAPH=1` on the worker to enable CUDA Graph replay. The
shared backend retains at most 64 graphs, keyed by ordered sequence token lengths;
growing the scratch buffer clears them. A new shape first runs an eager forward
to initialize its plans and captures the layer loop for later replay.
This adds cost for new lengths, so graph mode remains opt-in. The earlier
single-prompt graph comparison with Open-Jev-27B-v1.1 on the 74-case H200 workload
measured mean HTTP latency 2.03% below eager execution after all lengths were
warmed. Combined
packing and graph performance remains unmeasured. Tokenization, transfers
and the CPU scalar head remain outside the graph. Prefix sharing, GEMM autotuning,
quantization and multimodal inference are not implemented. The
[H200 validation](validation.md) reports Open-Jev-27B-v1.1 results for 74
single-candidate requests, including probability differences and timing
variability. It does not establish general accuracy parity or a speedup over
OpenJev-Fast; the author's B300 results use different hardware and workloads.
The [Open-Jev-9B validation](validation-9b.md) compares the 9B worker with the
reference on one RTX 6000 Ada; it makes no claim about 27B.
