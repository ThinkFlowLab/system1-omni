//! A native `/v1/systemone` worker for Cua-S1 4B 0.2 (`text` adapter): request
//! handling, tokenization and scoring in Rust, the Qwen3.5 forward pass on the CUDA
//! kernels of `src/backends/cuda/qwen3_5`, loaded at run time.

pub mod contract;
pub mod cuda;
pub mod engine;
pub mod json;
pub mod model;
