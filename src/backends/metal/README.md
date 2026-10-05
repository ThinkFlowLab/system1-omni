# Metal backend

Planned home for high-performance Apple GPU operations and Metal kernel integration. Implement the operations required by the first model, with hardware-specific optimizations where needed.

In the [architecture contracts](../../../docs/architecture.md), the shared worker
runtime owns processing orchestration, batching policy, and request bookkeeping.
Model executors own forward passes, device state, and kernel selection. The
shared runtime layers are planned; current workers retain model-specific
pipelines. CUDA and Metal implementations do not need identical internal
structures or a universal tensor abstraction.

The native target combines Rust bindings/dispatch with Metal shaders. GPU
transforms and tensor packing can use this backend while retaining their
processing/batching ownership.

Status: planned; no Metal implementation or validated hardware coverage yet. Reference results from other runtimes do not establish native Metal backend support.
