# System1-Omni

Documentation: <https://thinkflowlab.github.io/system1-omni/>

A community-maintained inference engine for prefill-only System1-Omni models, designed around a Rust frontend, model-owned execution, and high-performance CUDA and Metal backends.

The Rust frontend forwards requests to a separately running model worker. The Cua-S1 4B 0.2 `text` adapter and Open-Jev-27B-v1.1 have native workers using shared CUDA kernels in this repository.

## News

- **2026-10-03:** Added [Open-Jev-27B-v1.1](recipe/open_jev/native.md)
  support through a native Rust/CUDA worker: **7.47× faster than raw HF Transformers**
  by mean warm HTTP latency, **362.21→48.50 ms** on one H200. Measured over
  74 single-candidate JevBench `noul` requests per pass, with two measured passes
  per backend (BF16, concurrency 1). See the
  [HF Transformers baseline, results and OpenJev-Fast comparison](recipe/open_jev/validation.md).

## Run the frontend

From the repository root, with stable Rust installed:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

Start the worker separately. See the [frontend documentation](src/frontend/README.md)
for the HTTP interface and configuration, or the [Laya recipe](recipe/laya/README.md)
for a CPU text worker and response checks.

## Architecture

Share serving infrastructure; let each model own its execution.

![System1-Omni architecture: Rust frontend, model-owned execution, and CUDA and Metal backends](docs/assets/architecture.svg)

| Layer | Responsibility |
| --- | --- |
| Rust frontend | API, request lifecycle, and response delivery through a small engine interface. |
| System1-Omni models | Model-specific preprocessing and postprocessing, batching, state, execution, and kernel selection. |
| CUDA backend | High-performance GPU operations for NVIDIA GPUs. |
| Metal backend | High-performance GPU operations for Apple GPUs. |

Each model owns its complete request-to-result path. Shared utilities stay minimal and are extracted when implementations need the same functionality. Backends can optimize for their hardware without requiring identical internal implementations.

## Repository layout

Implementation code lives under `src/`; recipes and documentation stay at the repository root.

| Directory | Responsibility |
| --- | --- |
| [`src/frontend/`](src/frontend/) | Rust serving code, Python worker adapters, and the small engine interface. |
| [`src/models/`](src/models/) | Model implementations, one directory per model: preprocessing, batching, state, execution, and output processing. |
| [`src/backends/cuda/`](src/backends/cuda/) | NVIDIA GPU operations and kernel integration. |
| [`src/backends/metal/`](src/backends/metal/) | Apple GPU operations and kernel integration. |
| [`recipe/`](recipe/) | Model setup instructions, launch commands, configuration examples, and example requests. |
| [`docs/`](docs/) | Project documentation and architecture assets. |

The frontend, both native workers, their shared Qwen3.5/3.8 prefill implementation and the Laya checkpoint reader are Cargo workspace members. The other model and backend directories currently document planned work; they do not prescribe process boundaries.

## Supported models

LAYA can run as an external Python worker for text requests; its in-repository model engine is still planned. The Cua-S1 4B 0.2 `text` adapter runs as a Python worker or as a native worker on CUDA:

| Model | Status |
| --- | --- |
| LAYA | [External worker](recipe/laya/README.md); [Python worker on Apple Silicon (MPS) and CPU](recipe/laya/apple-silicon.md); [CPU checkpoint reader](src/models/laya/README.md); model execution planned |
| Cua-S1 4B 0.2 (`text` adapter) | [Python worker](recipe/cua_s1/text.md); [native worker](recipe/cua_s1/native.md), CUDA, run on sm_89 |
| Open-Jev-27B-v1.1 | [Native Rust/CUDA worker](recipe/open_jev/native.md); eager independent text candidates; [H200 validation](recipe/open_jev/validation.md) |

[Supported models and hardware](docs/supported-models.md) lists the devices and where each worker has been run.

## Benchmarks

See the [GPU serving benchmark](benchmarks/README.md) for request replay,
output-fidelity checks, and the CUDA comparison protocol. The
[Open-Jev H200 results](recipe/open_jev/validation.md) cover 74 single-candidate
requests and a matched comparison with raw HF Transformers and OpenJev-Fast.

## Stay Tuned with Us

If you find system1-omni useful, [give us a star on GitHub](https://github.com/ThinkFlowLab/system1-omni)
to support the project and help others discover it!

[![GitHub repository screenshot demonstrating a click on Star, turning the star yellow and showing Starred](docs/assets/stay-tuned.gif)](https://github.com/ThinkFlowLab/system1-omni)
