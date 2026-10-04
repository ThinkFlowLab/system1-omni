//! Opt-in, bounded LRU storage for encoder graphs of supported shapes.
use crate::{
    encoder::{Encoder, validate_inputs},
    graph::ShapeEncoder,
    workspace::Workspace,
};
use anyhow::Result;
use std::collections::VecDeque;

/// Limits retained graph workspaces. Either zero limit disables graph caching.
#[derive(Clone, Copy, Debug)]
pub struct CacheConfig {
    pub max_shapes: usize,
    /// GPU workspace bytes only: excludes weights, graph/driver overhead, host
    /// staging, transient eager workspaces and buffers cloned by callers.
    pub max_bytes: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_shapes: 4,
            max_bytes: 512 * 1024 * 1024,
        }
    }
}

/// Each cache borrows one encoder; shapes cannot reuse another encoder's graph.
/// Returned workspaces remain valid until the next mutable access to the cache.
pub struct EncoderCache<'a> {
    encoder: &'a Encoder,
    config: CacheConfig,
    entries: VecDeque<ShapeEncoder<'a>>,
    cached_bytes: usize,
    eager: Option<Workspace>,
}

impl<'a> EncoderCache<'a> {
    pub fn new(encoder: &'a Encoder, config: CacheConfig) -> Self {
        Self {
            encoder,
            config,
            entries: VecDeque::new(),
            cached_bytes: 0,
            eager: None,
        }
    }

    /// Validates before changing cache state, then uploads the current inputs.
    /// Misses evict least recently used entries before allocation. Disabled or
    /// oversized shapes reuse one eager workspace while the shape stays the same.
    /// CUDA failures are always returned; failed requests do not promise outputs.
    pub fn run(
        &mut self,
        batch: usize,
        sequence: usize,
        ids: &[i64],
        lengths: &[i32],
        types: &[i64],
    ) -> Result<&Workspace> {
        let bytes = Workspace::required_bytes(batch, sequence)?;
        validate_inputs(ids, lengths, types, batch, sequence)?;
        if self.config.max_shapes == 0 || bytes > self.config.max_bytes {
            if !self.eager.as_ref().is_some_and(|workspace| {
                workspace.batch() == batch && workspace.sequence() == sequence
            }) {
                self.eager = None;
                self.eager = Some(Workspace::new(&self.encoder.cuda, batch, sequence)?);
            }
            let workspace = self.eager.as_ref().unwrap();
            self.encoder.run(ids, lengths, types, workspace)?;
            return Ok(workspace);
        }
        self.eager = None;
        if let Some(index) = self.entries.iter().position(|entry| {
            entry.workspace().batch() == batch && entry.workspace().sequence() == sequence
        }) {
            self.entries[index].run(ids, lengths, types)?;
            let entry = self.entries.remove(index).unwrap();
            self.entries.push_back(entry);
        } else {
            while self.entries.len() >= self.config.max_shapes
                || self.cached_bytes > self.config.max_bytes - bytes
            {
                let evicted = self.entries.pop_front().unwrap();
                self.cached_bytes -= evicted.workspace().bytes();
                drop(evicted);
            }
            let workspace = Workspace::new(&self.encoder.cuda, batch, sequence)?;
            let entry = ShapeEncoder::capture(self.encoder, workspace, ids, lengths, types)?;
            entry.run(ids, lengths, types)?;
            self.entries.push_back(entry);
            self.cached_bytes += bytes;
        }
        Ok(self.entries.back().unwrap().workspace())
    }

    /// Releases all cached graphs and the reusable eager workspace. Caller-held
    /// buffer clones remain alive; the encoder and cache limits are unchanged.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.cached_bytes = 0;
        self.eager = None;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total bytes owned by retained graph workspaces, subject to `max_bytes`.
    pub fn cached_bytes(&self) -> usize {
        self.cached_bytes
    }
}
