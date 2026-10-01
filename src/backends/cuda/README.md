# CUDA backend

Planned home for high-performance NVIDIA GPU operations and kernel integration. Implement the operations required by the first model, with hardware-specific optimizations where needed.

Model orchestration, batching policy, state management, and kernel selection remain with the model engine. CUDA and Metal implementations do not need identical internal structures or a universal tensor abstraction.

Status: [`qwen3_5/`](qwen3_5/) has the operations of a prefill-only Qwen3.5 forward pass, used by the Cua-S1 native worker and measured on sm_89. Other models are planned.
