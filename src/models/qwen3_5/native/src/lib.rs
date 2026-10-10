//! Qwen3.5/3.8 prefill, runtime-loaded CUDA operations, image decoding and
//! preprocessing, vision, and request JSON helpers shared by the native workers.
pub mod cuda;
pub mod json;
pub mod model;

pub mod inputs;

pub mod image_decode;
pub mod image_preprocess;
pub mod vision;
