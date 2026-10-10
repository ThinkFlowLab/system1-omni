//! Native Cua-S1 text and screenshot workers using shared Qwen execution.

pub mod contract;
pub use omni_qwen3_5_native::{cuda, inputs, json, model};
pub mod engine;
pub mod executor;
pub mod image_preprocess;
pub mod image_request;
pub mod multimodal;
pub mod processing;
mod provenance;
pub mod vision;
pub mod vision_engine;
pub mod vision_processing;
