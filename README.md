# System1-Omni

A community-maintained inference engine for prefill-only System1-Omni models, designed around a Rust frontend, model-owned execution, and high-performance CUDA and Metal backends.

The Rust frontend can proxy requests to a separately running model worker. The native English Laya engine runs in Rust with a CUDA backend on Hopper sm_90a; see the [native recipe](recipe/laya/native/README.md).

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
| [`src/frontend/`](src/frontend/) | Rust serving code and the small engine interface. |
| [`src/models/laya/`](src/models/laya/) | LAYA preprocessing, batching, state, execution, and output processing. |
| [`src/backends/cuda/`](src/backends/cuda/) | NVIDIA GPU operations and kernel integration. |
| [`src/backends/metal/`](src/backends/metal/) | Apple GPU operations and kernel integration. |
| [`recipe/`](recipe/) | Model setup instructions, launch commands, configuration examples, and example requests. |
| [`docs/`](docs/) | Project documentation and architecture assets. |

The frontend, Laya model and CUDA backend are Cargo workspace members. The Metal backend remains planned.

## Supported models

LAYA supports text requests through a native Rust/CUDA engine or an external Python worker:

| Model | Status |
| --- | --- |
| LAYA | [Native English engine on Hopper sm_90a](recipe/laya/native/README.md); [external worker](recipe/laya/README.md) |

The native CUDA engine targets the frozen Laya 0.3.20 English checkpoint. Metal support is not implemented.

## Stay Tuned with Us

If you find system1-omni useful, [give us a star on GitHub](https://github.com/ThinkFlowLab/system1-omni)
to support the project and help others discover it!

[![GitHub repository screenshot demonstrating a click on Star, turning the star yellow and showing Starred](docs/assets/stay-tuned.gif)](https://github.com/ThinkFlowLab/system1-omni)

## Native Laya

The Rust/CUDA English engine, build steps and input limits are documented in [recipe/laya/native](recipe/laya/native/README.md).
