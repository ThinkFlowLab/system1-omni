//! Decider-2B v11's CPU request compiler and response assembler, without model execution.
//!
//! Reference: Mapika/decider 50d0be0 (decider-ai 1.8.1), checkpoint 533964d.
//! Only plain state-first, independent questions and isolated Score levels are supported.
//! The executor must return one candidate-logit vector per complete unpadded row,
//! in prepared order, after the reference BF16 LM projection output rounding.
//! Calibration and whole-question normalization belong here; no GPU/HTTP worker is provided.
//!
//! Native input policy: duplicate keys, depth >=128, nonfinite numbers and integer
//! literals outside i64/u64 are rejected. Integer -0 renders as 0; float -0.0 is retained.
//! Choice list aliases accept strings; canonical maps support arbitrary JSON descriptions.
//! Score legend map keys accept Python whitespace, digit separators and Unicode decimal
//! digits, but nonfinite keys are rejected. Scalar numeric-string temperatures are accepted;
//! per-type values must be numbers. Extreme temperatures must remain positive/finite in FP32.
//! Rust shortest round-trip float rendering may differ on rare shortest-decimal ties;
//! CPU FP32 softmax exp/summation can differ from torch by a few ulps. Golden parity is
//! evidence for the recorded corpus, not universal bit-identical numerical output.
//! Config and tokenizer validation establishes CPU artifact compatibility, not weight identity.
mod config;
mod contract;
mod json;
mod math;
mod processing;
mod text;

pub use config::Config;
pub use contract::Kind;
pub use processing::{Label, Limits, PreparedRequest, Processor, ResponseContext, RowInput};
pub const MODEL_ID: &str = "decider-2b-v11";
pub const RUNTIME_REVISION: &str = "50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f";
pub const CHECKPOINT_REVISION: &str = "533964dae8be954c5b5e19fa4948e48408094c1e";
