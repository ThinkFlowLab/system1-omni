use anyhow::{Result, ensure};
use omni_cuda::{Buffer, Cuda};

pub const MAX_MARKERS: usize = 2048;
const D: usize = 1024;

/// Fixed-shape scratch allocations. Contents are uninitialized until written.
pub struct Workspace {
    batch: usize,
    sequence: usize,
    bytes: usize,
    buffers: WorkspaceBuffers,
}

/// Buffer layouts consumed by Laya's encoder, decision head and scorer.
/// Access through `Workspace::buffers` keeps allocations fixed for its lifetime.
pub struct WorkspaceBuffers {
    pub ids: Buffer,
    pub lengths: Buffer,
    pub types: Buffer,
    pub residual: Buffer,
    pub hidden: Buffer,
    pub qkv: Buffer,
    pub attention: Buffer,
    pub gated: Buffer,
    pub feed_forward: Buffer,
    pub indices: Buffer,
    pub offsets: Buffer,
    pub markers: Buffer,
    pub scored: Buffer,
    pub logits: Buffer,
    pub features: Buffer,
    pub action_hidden: Buffer,
    pub actions: Buffer,
}

impl Workspace {
    pub fn new(cuda: &Cuda, batch: usize, sequence: usize) -> Result<Self> {
        ensure!(
            batch.is_power_of_two() && batch <= 16,
            "workspace batch must be 1, 2, 4, 8 or 16"
        );
        ensure!(
            (16..=512).contains(&sequence) && sequence.is_multiple_of(16),
            "workspace sequence must be a multiple of 16 in 16..=512"
        );
        let tokens = batch * sequence;
        let mut bytes = 0;
        let mut alloc = |size| {
            let buffer = cuda.alloc(size)?;
            bytes += size;
            Ok::<_, anyhow::Error>(buffer)
        };
        let buffers = WorkspaceBuffers {
            ids: alloc(tokens * 8)?,
            lengths: alloc(batch * 4)?,
            types: alloc(batch * 8)?,
            residual: alloc(tokens * D * 4)?,
            hidden: alloc(tokens * D * 2)?,
            qkv: alloc(tokens * D * 6)?,
            attention: alloc(tokens * D * 2)?,
            gated: alloc(tokens * 2624 * 2)?,
            feed_forward: alloc(tokens * 4096 * 2)?,
            indices: alloc(MAX_MARKERS * 4)?,
            offsets: alloc((batch + 1) * 4)?,
            markers: alloc(MAX_MARKERS * D * 2)?,
            scored: alloc(MAX_MARKERS * D * 2)?,
            logits: alloc(MAX_MARKERS * 2)?,
            features: alloc(batch * 1028 * 2)?,
            action_hidden: alloc(batch * 256 * 2)?,
            actions: alloc(batch * 2 * 2)?,
        };
        Ok(Self {
            batch,
            sequence,
            bytes,
            buffers,
        })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn sequence(&self) -> usize {
        self.sequence
    }

    /// Total scratch allocation bytes, excluding weights and CUDA overhead.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn buffers(&self) -> &WorkspaceBuffers {
        &self.buffers
    }
}

#[cfg(test)]
#[path = "../../../../tests/laya/unit/workspace.rs"]
mod tests;
