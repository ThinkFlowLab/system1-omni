# Native Laya

Run the English Laya model with Rust and CUDA. The serving process does not load Python, PyTorch, TileLang or a Python worker. Python is used only to build kernels/tables and generate reference tests.

The first implementation targets Laya 0.3.20, checkpoint `convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, and Hopper `sm_90a`. It preserves the official BF16 fast encoder, decision head and scoring behavior. It includes the selected RoPE geometry and an Attention shared-memory wait fix for empty key ranges. See [source attribution](../../../src/backends/cuda/THIRD_PARTY.md).

## Build

Use an allocated Hopper GPU and an existing environment with Laya 0.3.20, PyTorch, TileLang, CUDA Toolkit and a Rust toolchain. The validated versions and hashes are recorded in the generated manifests and acceptance report. Set `CUDA_VISIBLE_DEVICES` explicitly. Keep checkpoint files read-only.

```sh
export CHECKPOINT=/path/to/laya/snapshot
export BUNDLE=/path/to/personal/laya-cuda
python src/backends/cuda/tools/export.py "$BUNDLE"
python src/backends/cuda/tools/build.py "$BUNDLE"
python src/backends/cuda/tools/export_tables.py "$CHECKPOINT" "$BUNDLE"
cargo build --release --locked -p omni-laya --features serve
```

Deployment needs `target/release/omni-laya`, the checkpoint (config, tokenizer and safetensors), the bundle, and compatible CUDA/cuBLAS libraries. No Python environment is needed to start the server. Bundle/table hashes and checkpoint configuration hashes are checked at startup.

```sh
target/release/omni-laya "$CHECKPOINT" "$BUNDLE" 127.0.0.1:8080
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer request a refund?"}}}'
```

`laya-run CHECKPOINT BUNDLE` accepts JSON lines on stdin. `--eager` disables Graph; `--original-rope` selects the original geometry. Kernel failures do not trigger Python or CPU fallback. The existing `omni-jev` proxy remains available as a separate binary.

## Supported inputs and ownership

English `choice`, `score` and `noul`; at most 16 questions, 2048 total options, 512 tokens per row and a 1 MiB HTTP body. Unsupported model/language/configuration fails explicitly. This does not implement image/audio/video inference or language routing.

A single GPU worker owns the model, stream and buffers. The queue holds at most 32 requests. Requests time out after 30 seconds, including upload and queueing. Cancelled or expired queued requests are skipped; submitted CUDA work finishes before buffers can be reused. SIGTERM stops new admissions and drains accepted work. `/health` succeeds only after model loading and prewarm. Graphs cover the encoder and decision transformer; the scorer remains outside Graph. The cache is limited to four shapes and 512 MiB of workspaces; a new shape pays allocation, warmup and capture costs.

## Validate

```sh
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets --all-features -- -D warnings
cargo test --workspace --locked --all-features
LAYA_CHECKPOINT="$CHECKPOINT" cargo test -p omni-laya --test packing -- --ignored
python recipe/laya/native/http_acceptance.py "$CHECKPOINT" "$BUNDLE" http-results.json
python recipe/laya/native/benchmark.py native "$CHECKPOINT" "$BUNDLE" native-results.json
```

The CPU oracle checks token IDs, option markers, lengths, type IDs, padding and usage against the official tokenizer. GPU acceptance must separately compare eager/Graph outputs, intermediate tensors and warmed paired performance. Repeated requests do not add independent model-quality samples. Sub-millisecond latency and business quality are not implied by native execution.
