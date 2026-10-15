# JEMM setup and validation

Run from the repository root on Linux with a CUDA GPU (compute capability 8.0+),
nvcc/cuBLASLt, cuDNN 9 headers/library and Rust. Historical validation used one A800 80 GB. The raw base
and adapter need about 55.6 GB disk. V2 exports hardlink unchanged files on the
same filesystem; cross-filesystem exports copy the 17 language shards and raw
FP32 adapter, requiring additional disk for those files. Export is an
offline CPU operation using torch, safetensors and Transformers; the running
Rust worker needs no Python serving process. Keep inputs/exports immutable.

## Download and export

In a Python environment with `huggingface-hub`, torch, safetensors and Transformers:

```sh
export JEMM_DATA=/data/jemm
hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --include '*.safetensors' '*.json' '*.jinja' '*.txt' 'README.md' 'LICENSE' \
  --local-dir "$JEMM_DATA/base"
hf download MaestroYan/JEMM \
  --revision 76e3c209e8441fa658221c7ba2725bad2f811176 \
  --include 'adapter_config.json' 'adapter_model.safetensors' 'decision_config.json' 'README.md' 'LICENSE' \
  --local-dir "$JEMM_DATA/adapter"
cat > "$JEMM_DATA/provenance.json" <<'JSON'
{
  "base_model_id": "Qwen/Qwen3.8-27B",
  "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  "checkpoint_revision": "76e3c209e8441fa658221c7ba2725bad2f811176",
  "source_revision": "6822fe0fd53c5e6670af6ba99fb2c857a661e532"
}
JSON
python recipe/jemm/export.py --base "$JEMM_DATA/base" \
  --adapter "$JEMM_DATA/adapter" --out "$JEMM_DATA/native" \
  --provenance "$JEMM_DATA/provenance.json" --threads 4
```

The checked-in pinned inventory verifies every required input file and tensor.
Interrupted exports may resume only with matching inputs, producer and recorded
runtime. Do not substitute similarly named checkpoints or newer revisions.
The `jemm-native/2` format keeps the original BF16 language weights and FP32
adapter separate, with rank 16, 496 pairs and scale 2. Use a new directory for v2;
legacy premerged v1 directories are rejected rather than reused. Hardlinked
inputs and outputs share file contents and must remain immutable.

## Build, launch and request

```sh
CUDNN_INCLUDE_DIR=/path/to/cudnn/include \
CUDNN_LIB_DIR=/path/to/cudnn/lib \
  src/backends/cuda/qwen3_5/build.sh target/release 80
cargo build --release --locked -p omni-jemm-native -p omni-jev
JEMM_MODEL="$JEMM_DATA/native" \
JEMM_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
  target/release/omni-jemm-native
```

Replace `80` with the GPU's compute capability. Rebuild the CUDA library and
consumers together; JEMM requires the additive reference-numerics symbols and a
cuDNN-enabled build. The other Qwen workers keep their existing CUDA entry points.
The default listener is `127.0.0.1:8000`, set by `JEMM_HOST`/`JEMM_PORT`.
It binds after text and image warmup complete. In another terminal:

```sh
curl --fail http://127.0.0.1:8000/health
OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  target/release/omni-jev
curl --fail http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"JEMM","state":"Update installed.","questions":{"action":{"instructions":"Choose the next action.","criteria":{"close":"Close dialog","wait":"Wait"}}}}'
```

Optional `images` contains up to four base64 PNG/JPEG/WebP strings; images precede
text and are shared across questions. The frozen 16-request / 25-question corpus
passes the original numerical gates, including 2/3/4-image cases. See the
[validation scope and evidence](../../docs/benchmarks/jemm-reference-20261011/README.md).
Native CPU/Metal execution is unsupported.

## Checks and reference

```sh
# CPU exporter tests, in the environment containing torch and safetensors:
python -m unittest discover -s tests/jemm -p 'test_export.py' -v
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
# Requires an available CUDA device and real library:
JEMM_CUDA_LIB="$PWD/target/release/libqwen3_5_cuda.so" \
  cargo test --release --locked -p omni-jemm-native bf16_label_head -- \
  --ignored --test-threads=1
```

CPU checks compile the worker and exercise host/export contracts. Device checks
require a separately reserved GPU; normal CI compiles and ignores CUDA tests.
The [frozen numerical campaign](../../docs/benchmarks/jemm-reference-20261011/README.md)
records the unmerged adapter path and its validation scope.

```sh
# Pinned Torch/FLA environment; tests graph replay as well as eager values:
python tests/jemm/reference_kernels.py target/release/libqwen3_5_cuda.so \
  --report /tmp/jemm-operators.json
# Against the running native worker, on the reserved device:
python tests/jemm/parity.py --url http://127.0.0.1:8000 \
  --corpus tests/jemm/data/parity/corpus.jsonl \
  --reference tests/jemm/data/parity/reference.json \
  --report /tmp/jemm-parity.json
```

For the official reference, clone ypcypc/JEMM and detach at
`6822fe0fd53c5e6670af6ba99fb2c857a661e532`, then use `python -m jemm.serve
--adapter "$JEMM_DATA/adapter" --base "$JEMM_DATA/base" --device cuda:0`
with the source on PYTHONPATH and its dependencies installed. Its default port
is 8790. Freeze requests before comparing native/reference outputs and retain
tokens, grids, pixels, raw responses and source/library/model hashes. Predeclare
probability 0.02, Score 0.1 and winner margin 0.05 gates; stop at a failed gate.
The [historical evidence release](https://github.com/Levius-Fubuki/system1-omni/releases/tag/jemm-a800-20261008)
retains runners and raw results at their original source revisions. It does not
validate new code or general model accuracy.
