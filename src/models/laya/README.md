# LAYA model engine

LAYA is the first native System1-Omni model. This directory owns its complete request-to-result path: preprocessing, postprocessing, batching policy, state, execution, and backend-specific kernel selection.

GPU operations and kernel implementations belong in [`backends/cuda/`](../../backends/cuda/) and [`backends/metal/`](../../backends/metal/). Setup and usage examples belong in the top-level [`recipe/`](../../../recipe/) directory.

The Rust/CUDA English engine supports the frozen Laya 0.3.20 checkpoint on Hopper sm_90a. See [native build, usage and validation](../../../recipe/laya/native/README.md). Numerical evidence is scoped to the tested fixtures; native review and parent acceptance remain required.
