//! Qwen3.5/3.8 prefill, runtime-loaded CUDA operations, and request JSON helpers
//! shared by the Cua-S1 and Open-Jev native workers.
pub mod cuda;
pub mod json;
pub mod model;

pub mod inputs;

pub mod image_preprocess;
pub mod vision;

mod lora;
