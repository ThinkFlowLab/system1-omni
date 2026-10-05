<h1 align="center">System1-Omni</h1>

<p align="center">
  <a href="https://thinkflowlab.github.io/system1-omni/"><img src="https://img.shields.io/badge/Docs-site-2563eb?style=flat-square&amp;logo=github&amp;logoColor=white" alt="Documentation site"></a>
  <a href="https://github.com/ThinkFlowLab/system1-omni/blob/main/Cargo.toml"><img src="https://img.shields.io/badge/Rust-stable-ce6229?style=flat-square&amp;logo=rust&amp;logoColor=white" alt="Rust stable toolchain"></a>
  <a href="https://github.com/ThinkFlowLab/system1-omni/blob/main/recipe/cua_s1/native.md"><img src="https://img.shields.io/badge/CUDA-sm__80%2B-76b900?style=flat-square&amp;logo=nvidia&amp;logoColor=white" alt="CUDA compute capability 8.0 or newer"></a>
  <a href="#contributing"><img src="https://img.shields.io/badge/Contributions-welcome-0d9488?style=flat-square" alt="Contributions welcome"></a>
</p>

<p align="center">
  <strong>A community-maintained inference engine for prefill-only System1-Omni models.</strong>
</p>

<p align="center">
  <a href="#how-it-works">How It Works</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="https://thinkflowlab.github.io/system1-omni/">Documentation</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#roadmap">Roadmap</a> ·
  <a href="#contributing">Contributing</a>
</p>

## About

**System1-Omni** is a community-maintained inference engine for prefill-only
System1-Omni models, designed around a Rust frontend, model-owned execution,
and high-performance CUDA and Metal backends. The target architecture separates
processing and scheduling from model execution.

The Rust frontend forwards requests to a separately running model worker. The
Cua-S1 4B 0.2 `text` adapter and Open-Jev-27B-v1.1 have native workers using
shared CUDA kernels in this repository.

## News

- **2026-10-03:** Added [Open-Jev-27B-v1.1](recipe/open_jev/native.md)
  support through a native Rust/CUDA worker: **7.47× faster than raw HF Transformers**
  by mean warm HTTP latency, **362.21→48.50 ms** on one H200. Measured over
  74 single-candidate JevBench `noul` requests per pass, with two measured passes
  per backend (BF16, concurrency 1). See the
  [HF Transformers baseline, results and OpenJev-Fast comparison](recipe/open_jev/validation.md).

## Features

- **Rust serving frontend.** API, request lifecycle, and response delivery
  through a small engine interface, forwarding requests to separately running
  model workers.
- **Model-owned execution.** Model executors own weights, forward passes,
  learned heads, device state, and kernel selection. Native workers have separate
  processing and executor modules, with shared bounded FIFO admission and blocking
  dispatch in the [native runtime](src/runtime/README.md).
- **Native CUDA workers.** The Cua-S1 4B 0.2 `text` adapter and
  Open-Jev-27B-v1.1 run as native workers with shared CUDA kernels. Cua-S1
  also has a Python worker that serves as the correctness reference.
- **LAYA text serving.** LAYA runs as an external Python worker for text
  requests, with an in-repository CPU checkpoint reader.
- **CUDA and Metal backends.** High-performance GPU operations for NVIDIA
  GPUs, with an Apple-GPU backend planned alongside it.
- **Benchmark harness.** Request replay, output-fidelity checks, and a CUDA
  comparison protocol across serving backends.

## How It Works

Share processing and scheduling; let each model own its execution.

![System1-Omni target architecture: Rust frontend, independent processing and batching layers, model executors, and CUDA and Metal backends](docs/assets/architecture.svg)

The diagram shows the **target architecture**. Today the frontend forwards HTTP
requests to separately running workers, whose handlers coordinate independent
processors and executors. Both native workers use shared FIFO admission and
blocking dispatch, with configurable pending-unit limits per loaded executor.
Processing orchestration, token/batch budgets,
compatibility grouping and dynamic batching remain planned.

| Layer | Responsibility | Native target implementation |
| --- | --- | --- |
| Rust frontend | API transport, request forwarding, and response delivery. | Rust. |
| Processing layer | Independent pre/postprocessing modules with model-specific processors for input preparation and output interpretation. | Rust CPU processing; GPU transforms use backends. |
| Scheduler / batcher | Queue admission, batch budgets, compatibility grouping, batch assembly, request bookkeeping, and result routing. | Rust host policy; GPU packing uses backends. |
| Model executors | Weights, forward passes, learned heads, device state, and kernel selection. | Rust orchestration calling backend operations. |
| CUDA backend | High-performance GPU operations for NVIDIA GPUs. | Rust bindings/dispatch and CUDA C++ kernels. |
| Metal backend (planned) | High-performance GPU operations for Apple GPUs. | Rust bindings/dispatch and Metal shaders. |

In the target design, the shared worker runtime invokes processors, schedules
compatible work, calls the model executor, and routes each output back to its
request. Tokenization, modality transforms, and response interpretation remain
model-specific plugins, separate from the forward implementation. Models
declare batch constraints; batch adapters pack inputs and unpack outputs using
the executor's supported layout. A shared scheduler must not batch incompatible
models or inputs, and dynamic batching requires executor support for real batches.

These are logical layers: the runtime and executor can share a worker process.
Request bookkeeping belongs to the runtime; model device state belongs to the
executor. Backends can optimize for their hardware without requiring identical
internal implementations.

The [architecture and integration contracts](docs/architecture.md) define
processor/executor boundaries, compatibility grouping, state and buffer
lifetimes, and result reconstruction. They also distinguish the native target
from current single-prompt execution and CPU decision heads.

## Repository Layout

Implementation code lives under `src/`; recipes and documentation stay at the
repository root.

| Directory | Responsibility |
| --- | --- |
| [`src/frontend/`](src/frontend/) | Rust serving code, Python worker adapters, and the small engine interface. |
| [`src/runtime/`](src/runtime/README.md) | Shared native execution admission and blocking dispatch. |
| [`src/models/`](src/models/) | Model contracts, existing worker pipelines, and model executors, including the shared Qwen3.5/3.8 prefill implementation. |
| [`src/backends/cuda/`](src/backends/cuda/) | NVIDIA GPU operations and kernel integration. |
| [`src/backends/metal/`](src/backends/metal/) | Apple GPU operations and kernel integration. |
| [`recipe/`](recipe/) | Model setup instructions, launch commands, configuration examples, and example requests. |
| [`docs/`](docs/) | Project documentation and architecture assets. |

The frontend, native runtime, both native workers, their shared Qwen3.5/3.8
prefill implementation and the Laya checkpoint reader are Cargo workspace members.
The other model and backend directories currently document planned work;
they do not prescribe process boundaries.

## Getting Started

From the repository root, with stable Rust installed:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 \
OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
```

Start the worker separately. See the [frontend documentation](src/frontend/README.md)
for the HTTP interface and configuration, the [Laya recipe](recipe/laya/README.md)
for a CPU text worker and response checks, or the Cua-S1 recipes for the
[text worker](recipe/cua_s1/text.md) and [native worker](recipe/cua_s1/native.md).

## Supported Models

LAYA can run as an external Python worker for text requests; its
in-repository model engine is still planned. The Cua-S1 4B 0.2 `text` adapter
runs as a Python worker or as a native worker on CUDA. Open-Jev-27B-v1.1
runs as a native Rust/CUDA worker:

| Model | Status |
| --- | --- |
| LAYA | [External worker](recipe/laya/README.md); [Python worker on Apple Silicon (MPS) and CPU](recipe/laya/apple-silicon.md); [CPU checkpoint reader](src/models/laya/README.md); model execution planned |
| Cua-S1 4B 0.2 (`text` adapter) | [Python worker](recipe/cua_s1/text.md); [native worker](recipe/cua_s1/native.md), CUDA, run on sm_89 |
| Open-Jev-27B-v1.1 | [Native Rust/CUDA worker](recipe/open_jev/native.md); eager independent text candidates; [H200 validation](recipe/open_jev/validation.md) |

[Supported models and hardware](docs/supported-models.md) lists the devices
and where each worker has been run.

## Benchmarks

See the [GPU serving benchmark](benchmarks/README.md) for request replay,
output-fidelity checks, and the CUDA comparison protocol. The
[Open-Jev H200 results](recipe/open_jev/validation.md) cover 74 single-candidate
requests and a matched comparison with raw HF Transformers and OpenJev-Fast.

## Roadmap

The current focus is the native Cua-S1 and Open-Jev CUDA workers and the serving
benchmark harness. Planned work extends the shared runtime with processing
orchestration, token budgets, compatibility grouping and bounded dynamic
batching with batch-capable executors,
the in-repository LAYA model engine, additional model engines and GPU backends
including Metal, and per-model
performance measurements as implementations are added and validated.

<a id="contributing"></a>

## 🤝 Contributing

System1-Omni is open to contributions across serving, models, backends,
benchmarks, and documentation. A reproducible bug report, a carefully measured
benchmark, or a clearer recipe can be just as useful as a kernel optimization.

Review the [contributing guide](CONTRIBUTING.md) before opening a pull
request: self-review the full diff, keep serving, processing, scheduling, and
model execution separate according to the [architecture contracts](docs/architecture.md),
and run the checks appropriate to your changes.

**Have an idea or found a problem?** [Open an
issue](https://github.com/ThinkFlowLab/system1-omni/issues/new) with the
details, or [send a pull
request](https://github.com/ThinkFlowLab/system1-omni/compare). For larger
changes, start a discussion in an issue so we can work through the design
together.

## Stay Tuned with Us

If you find system1-omni useful, [give us a star on GitHub](https://github.com/ThinkFlowLab/system1-omni)
to support the project and help others discover it!

[![GitHub repository screenshot demonstrating a click on Star, turning the star yellow and showing Starred](docs/assets/stay-tuned.gif)](https://github.com/ThinkFlowLab/system1-omni)
