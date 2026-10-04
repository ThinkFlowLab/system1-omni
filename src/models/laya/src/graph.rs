//! Opt-in graph execution for one fixed encoder shape.
use crate::{encoder::Encoder, workspace::Workspace};
use anyhow::Result;
use omni_cuda::graph::Graph;

/// An encoder graph owning its scratch buffers and borrowing its weights/code.
/// Each instance has one fixed batch and sequence length, chosen by its workspace.
pub struct ShapeEncoder<'a> {
    // Rust drops fields in declaration order: graph work finishes before scratch frees.
    graph: Graph,
    workspace: Workspace,
    encoder: &'a Encoder,
}

impl<'a> ShapeEncoder<'a> {
    /// Uploads representative inputs, warms up twice, then captures compute only.
    /// Input copies and synchronization are outside the captured graph.
    pub fn capture(
        encoder: &'a Encoder,
        workspace: Workspace,
        ids: &[i64],
        lengths: &[i32],
        types: &[i64],
    ) -> Result<Self> {
        workspace.upload(&encoder.cuda, ids, lengths, types)?;
        for _ in 0..2 {
            encoder.execute(&workspace, false)?;
            encoder.cuda.sync()?;
        }
        // The owned workspace and borrowed encoder keep all buffers and code alive.
        // execute disables test checkpoint readbacks and only launches stream work.
        let graph =
            unsafe { Graph::capture(&encoder.cuda, || encoder.execute(&workspace, false)) }?;
        Ok(Self {
            graph,
            workspace,
            encoder,
        })
    }

    /// Validates and uploads fresh inputs, then replays and waits for completion.
    /// Final FP32 hidden states are in `workspace().buffers().residual`.
    pub fn run(&self, ids: &[i64], lengths: &[i32], types: &[i64]) -> Result<()> {
        self.workspace
            .upload(&self.encoder.cuda, ids, lengths, types)?;
        self.graph.run()?;
        self.encoder.cuda.sync()
    }

    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }
}
