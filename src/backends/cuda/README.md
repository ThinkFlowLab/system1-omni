# CUDA backend

Native CUDA backend for the English Laya engine, with generated BF16 kernels, selected RoPE, runtime ownership and bounded CUDA Graph caching.

Model orchestration, batching policy, state management, and kernel selection remain with the model engine. CUDA and Metal implementations do not need identical internal structures or a universal tensor abstraction.

The initial target is Hopper sm_90a. See [build and validation instructions](../../../recipe/laya/native/README.md) and [source attribution](THIRD_PARTY.md). Other architectures are unsupported.
