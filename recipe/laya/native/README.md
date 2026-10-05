# Laya native CUDA worker

Run commands from the repository root on Linux with an allocated Hopper GPU.
This implements English Laya 0.3.20 text decisions. CUDA targets `sm_90a` only;
CPU/Metal inference, automatic language routing and multimodal inputs are not supported.

## Prepare the checkpoint and bundle

The recorded build uses Rust 1.98.1, CUDA 13.0, PyTorch 2.11.0 (`cu128`),
Laya 0.3.20 and TileLang 0.1.14. TileLang's host-stub ABI is checked
by the exporter: incompatible generated layouts fail instead of guessing.
Python is used for build/table export and measurement, not by native inference.

```sh
hf download convaiinnovations/laya \
  --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851 --local-dir weights/laya
export LAYA_CHECKPOINT="$PWD/weights/laya"
export LAYA_CUDA_BUNDLE="$PWD/weights/laya-cuda"
PYTHON=python3 src/backends/cuda/build.sh "$LAYA_CUDA_BUNDLE" 90
python3 src/backends/cuda/tools/export_tables.py "$LAYA_CHECKPOINT" "$LAYA_CUDA_BUNDLE"
cargo build --release --locked -p omni-laya --features serve
cargo build --release --locked -p omni-jev
```

Table export uses an allocated GPU to reproduce official BF16 rotary-table
rounding. Keep the five checkpoint artifacts immutable after export. Startup
checks their hashes, four tables and library before loading CUDA. The executor
owns resident weights and bounded shape workspaces; allow several GiB of device
memory for the checkpoint, workspace and CUDA/cuBLAS context.

## Start and check

```sh
LAYA_HOST=127.0.0.1 LAYA_PORT=8000 target/release/omni-laya
```

The listener opens after a real warmup. In another terminal:

```sh
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  OMNI_JEV_BIND=127.0.0.1:8080 target/release/omni-jev
```

Check the frontend from a third terminal:

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"model":"english","state":"Please refund the duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}'
python3 recipe/compare_with_backend.py --model english \
  --backend http://127.0.0.1:8000 --frontend http://127.0.0.1:8080
```

The worker admits one complete request with `SerialScheduler`. Questions within
that request retain their order; this is not cross-request or continuous batching.
Inputs are limited to 16 questions, 512 tokens per sequence and 2048 markers.
Invalid inputs return 422; unsupported content types return 415. Native failure
returns 500 and stops this executor, making health return 503. The unchanged
frontend retains its documented timeout/forwarding limits.

For direct CLI execution, send one JSON request per line:

```sh
printf '%s\n' '{"model":"english","state":"Refund it.","questions":{"q":{"type":"noul","instructions":"Refund?"}}}' \
  | target/release/laya-run "$LAYA_CHECKPOINT" "$LAYA_CUDA_BUNDLE"
```

`--eager` disables Graph. `--original-rope` selects the original RoPE for a
controlled comparison; neither changes checkpoint precision. Graph caches only
Encoder/Decision; scorer/action head and readback remain outside capture.

## Verify and measure

```sh
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets --features omni-laya/serve -- -D warnings
cargo test --workspace --locked --features omni-laya/serve
python3 -m unittest discover -s tests/cuda -p 'test_*.py'
python3 recipe/laya/native/benchmark.py --checkpoint "$LAYA_CHECKPOINT" \
  --bundle "$LAYA_CUDA_BUNDLE" --output /tmp/laya-optimized-1
```

The benchmark uses five B=1 inputs, 20 warmups and 100 samples per input.
It stores every measured engine time, startup/request logs, binary/library/input
hashes and actual loaded mappings. Run without Nsight; use profiler captures
only to explain bottlenecks. Startup, first Graph construction and HTTP are
outside its timing boundary. Response stability is not reference-model parity.

To reconstruct the original GEMM configuration, copy the backend to an isolated
directory and reverse the supplied exporter patch; keep the current checkout intact:

```sh
baseline_source=$(mktemp -d)
mkdir -p "$baseline_source/src/backends"
cp -R src/backends/cuda "$baseline_source/src/backends/cuda"
git -C "$baseline_source" apply -R "$PWD/recipe/laya/native/optimized-export.patch"
PYTHON=python3 "$baseline_source/src/backends/cuda/build.sh" "$PWD/weights/laya-cuda-original" 90
cp "$LAYA_CUDA_BUNDLE"/rope_*.f32 "$LAYA_CUDA_BUNDLE/tables.json" weights/laya-cuda-original/
python3 recipe/laya/native/benchmark.py --checkpoint "$LAYA_CHECKPOINT" \
  --bundle weights/laya-cuda-original --original-rope --output /tmp/laya-original-1
```

Measure original→optimized→optimized→original, with unique output directories,
the same CLI and inputs. Repeat in eager mode separately. Do not sum historical
kernel gains or mix HTTP and CLI measurements. See [validation](VALIDATION.md).
