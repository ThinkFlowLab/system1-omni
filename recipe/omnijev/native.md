# OmniJev-4B v1.1 native worker

The native worker ([`src/models/omnijev/native/`](../../src/models/omnijev/native/))
serves OmniJev-4B v1.1 ([#114](https://github.com/ThinkFlowLab/system1-omni/issues/114))
on `/v1/systemone`: one PNG or JPEG image and any mix of Choice, Noul and Score
questions, with the vision tower and the Qwen3.5-4B language model on the CUDA kernels
in [`src/backends/cuda/qwen3_5/`](../../src/backends/cuda/qwen3_5/) and no Python or
PyTorch. It needs an NVIDIA GPU with compute capability 8.0 or newer and the export from
[the preparation recipe](README.md). [The model README](../../src/models/omnijev/README.md)
describes the contract and what the worker computes.

Run the commands from the repository root. Pass your GPU's compute capability to
`build.sh`; the worker finds `libqwen3_5_cuda.so` next to its executable, or at
`OMNIJEV_CUDA_LIB`:

```sh
src/backends/cuda/qwen3_5/build.sh target/release 89   # needs nvcc and cuBLASLt
cargo build --release --locked -p omni-omnijev-native
OMNIJEV_MODEL=$PWD/omnijev-export target/release/omni-omnijev-native
```

`OMNIJEV_HOST` and `OMNIJEV_PORT` default to `127.0.0.1` and `8000`. At start the
worker checks every file of the export against its manifest, loads the weights and
answers one request with all three question types before it listens, so `/health`
answers only once a request has run. Put the frontend in front of it with
`OMNI_JEV_BACKEND_URL` (see [the frontend README](../../src/frontend/README.md)), or
send requests to the worker directly:

```sh
IMAGE=$(base64 < screenshot.png | tr -d '\n')
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d @- <<EOF
{"model": "tinnel123/OmniJev",
 "state": {"images": ["data:image/png;base64,$IMAGE"]},
 "questions": {
   "error": {"type": "noul", "instructions": "This screen shows an error dialog."},
   "next": {"type": "choice", "instructions": "Which operation comes next?",
            "criteria": {"click": null, "type text": null, "scroll": null}},
   "risk": {"type": "score", "instructions": "How risky is acting on this screen?",
            "levels": ["safe", "check first", "dangerous"]}}}
EOF
```

The response is `{"model": "tinnel123/OmniJev", "answers": {id: answer},
"usage": {"input_tokens": n, "output_tokens": 0}}`: each answer has the reference's
fields, listed in [the model README](../../src/models/omnijev/README.md#request-and-response),
and `input_tokens` is counted as the reference counts it.

Malformed JSON gets 400, a body over 12 MiB 413, and a request the contract refuses
422, each with a `detail` message. The worker prepares two requests at a time and holds
at most 16, preparing, waiting or running; beyond that it answers 503 with
`Retry-After: 1`.

## Tests

The CPU tests need only the committed fixtures; the GPU tests need the export and the
CUDA library, as absolute paths:

```sh
cargo test --locked -p omni-omnijev-native
OMNIJEV_EXPORT=$PWD/omnijev-export OMNIJEV_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
    cargo test --release --locked -p omni-omnijev-native --test worker -- --ignored --test-threads=1
QWEN3_5_CHECKPOINT=$PWD/omnijev-export CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so \
    cargo test --release --locked -p omni-qwen3-5-native --test readout -- --ignored --test-threads=1
```

## Comparison with the reference

[`validate.py`](validate.py) runs the reference and the worker on the same images and
questions (`bench/speed_bench.py`'s twelve by default) and compares every answer. The
reference step needs the reference environment from [the preparation
recipe](README.md#reference-environment) and a GPU; the other two need only Python:

```sh
.venv-omnijev/bin/python recipe/omnijev/validate.py reference --reference OmniJev \
    --checkpoint omnijev-v1.1 --base qwen3.5-4b --images images --out reference.json
python3 recipe/omnijev/validate.py native --url http://127.0.0.1:8000 --images images \
    --against reference.json --out native.json
python3 recipe/omnijev/validate.py compare reference.json native.json
```

[`gate_questions.json`](gate_questions.json) adds eight harder questions for `--questions`:
options that differ in one token or share a long start, a Choice where no option fits, a
32-option Choice, a 10-level Score, and plain and region Noul.

On one RTX 6000 Ada (sm_89, CUDA 13.2), with the reference's seven v1.1 demo stills and
both question sets (140 answers), the input-token counts equal the reference's. The
largest probability difference per answer, against the reference in float32 (`--fp32`)
as the common baseline:

| Run | Median | p90 | Max | Decisions unlike the float32 reference |
| --- | ---: | ---: | ---: | ---: |
| Reference, BF16 | 0.019 | 0.074 | 0.188 | 2 |
| Worker, the stills as PNG | 0.014 | 0.073 | 0.209 | 2 |
| Worker, the stills as JPEG | 0.015 | 0.071 | 0.208 | 3 |

The PNG files carry the RGB values Pillow decodes from the JPEG files, so the reference
sees the same pixels in both runs. Merging the LoRA into BF16 (`--merge`) moves the
reference by at most 0.029, and running its rows one at a time by at most 0.019, on the
first set. Most of the rest comes from the vision tower: in BF16, the worker's and the
reference's image features both differ from float32 by about 14% (relative L2), and from
each other by 6%; given the reference's image features, the worker's question and option
states are within 1.1% of the reference's. Of the worker's decisions that differ, one (`web`,
`one_token`, float32 margin 0.19) differs for the BF16 reference too; the others have a
float32 margin under 0.03, except `video`'s speed_bench question 9 in the PNG run (0.18),
which also changes between the worker's PNG and JPEG runs.

Latency on the same GPU, with a 1600×900 still: the worker with one client, and the
reference with one stream (`bench/speed_bench.py`'s first 1, 3, 6 and 12 questions, and
Choices of 32 and 255 options). Both run the prefix once; the reference also batches a
question's options, which the worker runs one after another:

| Workload | Worker, median | Worker, requests per second | Reference, median |
| --- | ---: | ---: | ---: |
| 1 question | 0.16 s | 6.4 | 0.17 s |
| 3 questions | 0.21 s | 4.9 | 0.18 s |
| 6 questions | 0.35 s | 2.9 | 0.22 s |
| 12 questions | 0.66 s | 1.5 | 0.36 s |
| 32 options | 0.70 s | 1.4 | 0.25 s |
| 255 options | 5.1 s | 0.20 | 1.10 s |

With 8 or 16 clients the worker's throughput stays about the same and the median latency
grows with the queue: 5.0 s and 10.1 s for 12 questions, and 40.6 s for 255 options with
8 clients.
