//! CLM: the second System1-Omni model engine.
//!
//! A CLM decision is made over embeddings the engine does not compute. A frozen
//! Qwen3-8B encoder sits behind an OpenAI-compatible `/v1/embeddings` endpoint and the
//! engine owns everything after it: load the two projection heads, normalise their
//! output, score candidates by cosine similarity under the trained temperature, and
//! assemble `choice`, `noul` and `score` answers.
//!
//! This crate currently covers the checkpoint side and the decision maths. It does not
//! call an embeddings endpoint and does not serve HTTP; those belong with the runtime
//! that owns the request path. Nothing here needs a GPU.
pub mod config;
pub mod scoring;
pub mod weights;

pub use config::{Config, HeadConfig};
pub use scoring::{Answer, Kind, Question, answer, confidence, distribution};
pub use weights::{Head, Heads, Weights, head_tensors};
