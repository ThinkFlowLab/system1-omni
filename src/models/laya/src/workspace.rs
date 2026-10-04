use anyhow::{Result, ensure};
use omni_cuda::{Buffer, Cuda};
use std::cell::RefCell;

pub const MAX_MARKERS: usize = 2048;
const D: usize = 1024;

/// Fixed-shape scratch allocations. Contents are uninitialized until written.
pub struct Workspace {
    batch: usize,
    sequence: usize,
    bytes: usize,
    buffers: WorkspaceBuffers,
    staging: RefCell<Vec<u8>>,
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
            staging: RefCell::new(vec![0; tokens * 8 + batch * 12]),
        })
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn sequence(&self) -> usize {
        self.sequence
    }

    /// GPU scratch allocation bytes, excluding weights, CUDA overhead and host staging.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn buffers(&self) -> &WorkspaceBuffers {
        &self.buffers
    }

    /// Validates and packs inputs into reusable host storage, then completes the
    /// grouped upload before releasing the staging borrow.
    pub(crate) fn upload(
        &self,
        cuda: &Cuda,
        ids: &[i64],
        lengths: &[i32],
        types: &[i64],
    ) -> Result<()> {
        crate::encoder::validate_inputs(ids, lengths, types, self.batch, self.sequence)?;
        let mut staging = self.staging.borrow_mut();
        let (id_bytes, rest) = staging.split_at_mut(ids.len() * 8);
        let (length_bytes, type_bytes) = rest.split_at_mut(lengths.len() * 4);
        for (out, value) in id_bytes.as_chunks_mut::<8>().0.iter_mut().zip(ids) {
            out.copy_from_slice(&value.to_le_bytes());
        }
        for (out, value) in length_bytes.as_chunks_mut::<4>().0.iter_mut().zip(lengths) {
            out.copy_from_slice(&value.to_le_bytes());
        }
        for (out, value) in type_bytes.as_chunks_mut::<8>().0.iter_mut().zip(types) {
            out.copy_from_slice(&value.to_le_bytes());
        }
        cuda.write_many(&[
            (&self.buffers.ids, id_bytes),
            (&self.buffers.lengths, length_bytes),
            (&self.buffers.types, type_bytes),
        ])
    }
}

#[cfg(test)]
#[path = "../../../../tests/laya/unit/workspace.rs"]
mod tests;
