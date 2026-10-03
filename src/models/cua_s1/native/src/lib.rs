//! A native `/v1/systemone` worker for Cua-S1 4B 0.2 (`text` adapter): request
//! handling, tokenization and scoring in Rust, the Qwen3.5 forward pass on the CUDA
//! kernels of `src/backends/cuda/qwen3_5`, loaded at run time.

pub mod contract;
pub use omni_qwen3_5_native::{cuda, json, model};
pub mod engine;
