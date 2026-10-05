# CUDA backend

Home for high-performance NVIDIA GPU operations and kernel integration. Implement
operations needed by supported models, with hardware-specific optimizations
where needed.

In the [architecture contracts](../../../docs/architecture.md), the shared worker
runtime owns processing orchestration, batching policy, and request bookkeeping.
Model executors own forward passes, device state, and kernel selection. The
shared runtime layers are planned; current workers retain model-specific
pipelines. CUDA and Metal implementations do not need identical internal
structures or a universal tensor abstraction.

The native target combines Rust bindings/dispatch with CUDA C++ kernels and
native host glue. GPU transforms and tensor packing can use this backend while
retaining their processing/batching ownership.

Status: [`qwen3_5/`](qwen3_5/) provides the shared Qwen3.5/3.8 prefill operations
used by the Cua-S1 and Open-Jev native workers. See the
[supported-model coverage](../../../docs/supported-models.md) for validation scope.
