//! A native `/v1/systemone` worker for Cua-S1 4B 0.2 (text and multimodal adapters): request
//! handling, tokenization and scoring in Rust, the Qwen3.5 forward pass on the CUDA
//! kernels of `src/backends/cuda/qwen3_5`, loaded at run time.

pub mod contract;
pub mod cuda;
pub mod engine;
pub mod image_preprocess;
pub mod image_request;
pub mod inputs;
pub mod json;
pub mod model;
pub mod multimodal;

pub mod vision;

pub mod vision_engine;

mod provenance;
