# JEV-27B-VL experimental native recipe

This recipe serves `autotrust/JEV-27B-VL` System-1 decisions with Rust/CUDA.
Text runs natively; images must first be encoded offline by Transformers.
The [model contract](../../src/models/jev_vl/README.md) explains the single-question
API, verbalizer head and cache ownership. The reviewed candidate passed a
[bounded H800 validation](validation.md#historical-h800-validation);
this remains an experimental integration with the documented coverage limits.

## Prepare a pinned checkpoint

Run all commands from the repository root. Preparation used Linux, Python 3.10,
PyTorch 2.13.0+cu130, Transformers 5.17.0 and safetensors 0.8.0. The
[requirements file](requirements.txt) pins observed direct dependencies; a clean
installation of those pins has not been revalidated. Use an isolated environment:

```sh
python3.10 -m venv .venv-jev-vl
.venv-jev-vl/bin/python -m pip install -r recipe/jev_vl/requirements.txt
.venv-jev-vl/bin/hf download autotrust/JEV-27B-VL \
  --revision f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc \
  --local-dir weights/JEV-27B-VL
CUDA_VISIBLE_DEVICES='' .venv-jev-vl/bin/python recipe/jev_vl/export_merged.py \
  --model weights/JEV-27B-VL --out weights/jev-vl-merged --max-length 16384
```

The CPU exporter streams shards, merges the backbone LoRA in FP32 before BF16
rounding, and separately exports selected merged LM-head rows as FP32. Keep the
original source checkpoint for image preprocessing. The historical language
export occupied about 48 GiB in addition to the source checkpoint. Peak host
RAM was not measured; the export job requested 64 GiB. Do not assume the whole
pipeline fits on a low-memory workstation from the streaming implementation
alone. The output directory must not exist before export.

## Build and launch

The author measured the worker on one H800 80 GB (`sm_90`). A maintainer also
[reported a bounded L20X replay](https://github.com/ThinkFlowLab/system1-omni/pull/96#issuecomment-6018324526)
on an earlier revision. These runs do not validate later code changes or maximum
context lengths. Use an allocated GPU on
scheduled hosts. Build the CUDA library and Rust workers from the same revision;
the integrated backend uses **ABI 6** and older libraries must be rebuilt.

```sh
src/backends/cuda/qwen3_5/build.sh target/release 90
cargo build --release --locked -p omni-jev-vl-native -p omni-jev
JEV_VL_MODEL=weights/jev-vl-merged \
  JEV_VL_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  JEV_VL_CACHE=0 target/release/omni-jev-vl-native
```

`JEV_VL_HOST` defaults to `127.0.0.1`, `JEV_VL_PORT` to `8001`, and the CUDA
library defaults to the file beside the executable. The worker uses visible
CUDA device 0. `/health` becomes available after model loading and a successful
text warmup; that health check does not validate image assets.

In separate terminals, start the frontend and send the same request directly and
through it:

```sh
OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8001 \
  target/release/omni-jev
```

```sh
curl --fail-with-body http://127.0.0.1:8001/health
curl --fail-with-body http://127.0.0.1:8080/health
curl --fail-with-body http://127.0.0.1:8001/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @recipe/jev_vl/example-request.json
curl --fail-with-body http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @recipe/jev_vl/example-request.json
```

Expect a 200 response with two finite `probabilities`, their sum approximately
one, an in-range `choice_index`, and the corresponding string `choice`. Compare
decision fields and usage between both responses; `elapsed_seconds` varies.
This worker takes `kind`, `state`, `question` and `options`, rather than the
`questions`/`answers` envelope in the generic comparison recipe. A fixed expected
choice is not asserted here; use the frozen comparison corpus for numerical
checks. The frontend routes `/health` and `/v1/systemone`; it does not proxy
generation routes such as `/v1/chat/completions`. An unknown route returns the
frontend's own 404 rather than this worker's error envelope.

## Prepare images before serving

The worker does not decode images, fetch URLs or invoke a vision tower on cache
miss. First prepare a JSONL manifest with one object per line containing a
`request` whose `state` includes `{"image":"data:image/png;base64,..."}`. The
preencoder accepts base64 data URIs; a fresh URL or changed image needs a new
asset. It runs on a GPU and should finish before starting the language worker
on the same device:

```sh
.venv-jev-vl/bin/python recipe/jev_vl/preencode.py \
  --model weights/JEV-27B-VL --manifest /path/to/requests.jsonl \
  --out weights/jev-vl-image-assets
JEV_VL_MODEL=weights/jev-vl-merged JEV_VL_IMGCACHE=weights/jev-vl-image-assets \
  JEV_VL_CACHE=1 target/release/omni-jev-vl-native
```

Replace the manifest path with your own prepared workload. The preencoder writes
`sha256(exact_data_uri)/emb.safetensors` and `grid.json`; the worker requires the
exact same URI string. Use a new output directory for a changed checkpoint or
processor. Complete assets are skipped; interrupted writes are regenerated on
the next run. Source identity is not a cryptographic runtime compatibility check. Preencoding,
including image decoding and vision execution, is excluded from the reported
worker timing. This is useful for repeated decisions on a prepared image; it is
not an end-to-end live screenshot service.

## Cache controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `JEV_VL_CACHE` | `1` | Master enable; `0` uses a full language forward and rereads prepared image assets. |
| `JEV_VL_L1`, `JEV_VL_L2`, `JEV_VL_L3` | `1` | Processor records, parsed image assets, and language-prefix state. |
| `JEV_VL_L1_MAX` | `256` | Maximum resident processor records. |
| `JEV_VL_L2_BYTES` | `1073741824` | Parsed image-asset cache budget. |
| `JEV_VL_L3_BYTES` | `2147483648` | Cached language-prefix device-state budget. |

Cache budgets do not include weights, model scratch, in-flight request inputs or
total process memory. `GET /v1/cache/stats` exposes counters and resident cache
accounting. Use those counters to assess L2 hits; `x-jev-cache` does not report
an L2 hit field. `POST /v1/cache/reset` resets counters only; restart the worker for
a cold cache. Request preparation may overlap, while GPU execution stays serial.
Use cache modes as experimental controls within the documented validation scope.

## Local checks

CPU checks do not need weights or CUDA:

```sh
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
python3 -m unittest discover -s tests/benchmarks -p 'test_jev_vl_*.py' -v
python3 recipe/jev_vl/preencode.py --help
```

The image-prefix tokenizer check requires the exported checkpoint. First
[restore the frozen corpus](validation.md#restore-the-frozen-corpus).
It uses the corpus's fixed `[1, 60, 60]` grid and synthetic embedding
rows, so it checks token/position splitting rather than the vision encoder.
CUDA kernel tests require an allocated GPU and rebuilt ABI 6 library:

```sh
JEV_VL_EXPORT=$PWD/weights/jev-vl-merged \
  JEV_VL_MANIFEST="$jev_vl_evidence/manifest.jsonl" \
  cargo test --locked -p omni-jev-vl-native --test jev_vl_prefix -- --ignored
CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
  QWEN3_5_CHECKPOINT=$PWD/weights/jev-vl-merged \
  cargo test --release --locked -p omni-qwen3-5-native --test kernels -- \
  --ignored --test-threads=1
```

These checks do not replace full-checkpoint parity or direct/frontend HTTP
validation. The [validation page](validation.md) lists the remaining gates and
separates historical measurements from the current revision.
