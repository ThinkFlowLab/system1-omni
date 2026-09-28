//! CLM: the second System1-Omni model engine.
//!
//! A CLM decision is made over embeddings the engine does not compute. A frozen
//! Qwen3-8B encoder sits behind an OpenAI-compatible `/v1/embeddings` endpoint and the
//! engine owns everything after it: load the two projection heads, normalise their
//! output, score candidates by cosine similarity under the trained temperature, and
//! assemble `choice`, `noul` and `score` answers.
//!
//! That split is why this model is implemented second — LAYA's engine owns one forward
//! pass, while this one owns a client to someone else's server.
//!
//! Nothing here needs a GPU. The encoder can be [`embedding::HashingEncoder`], whose
//! vectors are the Python oracle's, so the decision path is checked on a CPU-only machine.
pub mod config;
pub mod embedding;
pub mod scoring;
pub mod serve;
pub mod weights;

pub use config::{Config, HeadConfig};
pub use embedding::{Encoder, HashingEncoder, HttpEncoder};
pub use scoring::{Answer, Kind, Question, answer, confidence, distribution};
pub use serve::{Decision, Engine, Request};
pub use weights::{Head, Heads, Weights, head_tensors};
