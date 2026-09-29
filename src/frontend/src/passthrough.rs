//! A stand-in engine that echoes its request. Not a model.
//!
//! This exists so the native path can be run and tested end to end without a GPU, a
//! checkpoint or CUDA: `OMNI_SYSTEMONE_ENGINE=passthrough` serves the same transport,
//! queue, readiness and shutdown behaviour a real engine gets, and answers every valid
//! request with its own body. That is how a deployment's wiring — bind address, readiness
//! probe, queue size, signal handling — can be checked before the model is available.

use std::{io, path::PathBuf};

/// The stand-in engine: it holds nothing and answers with the request it was given.
#[derive(Debug)]
pub struct Passthrough;

impl Passthrough {
    /// Builds the engine on the worker thread, as a real engine's loader would.
    pub fn load(_checkpoint: PathBuf) -> io::Result<Self> {
        Ok(Self)
    }
}
