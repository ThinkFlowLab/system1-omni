//! Native Rust/CUDA System-1 worker for autotrust/JEV-27B-VL: the merged text
//! backbone runs one prefill per request on the shared Qwen language kernels, and
//! the trained verbalizer readout (24-slot head + calibration) turns the last
//! hidden state into the official per-kind probability distribution.

pub mod caches;
pub mod contract;
pub mod engine;
pub mod executor;
pub mod images;
pub mod processing;
mod vision;
