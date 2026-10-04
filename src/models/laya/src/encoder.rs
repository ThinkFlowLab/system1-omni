//! Eager execution of the fixed Laya encoder and decision transformer.
use crate::{
    artifacts, config::Config, resident::ResidentWeights, weights::Weights, workspace::Workspace,
};
use anyhow::{Result, ensure};
use omni_cuda::{
    Buffer, Cuda,
    kernels::{Kernel, Kernels},
};
use std::{fs, path::Path};

const NAMES: &[&str] = &[
    "embed",
    "qkv",
    "rope_original",
    "attn_full",
    "attn_local",
    "out",
    "addln",
    "geglu",
    "down",
    "type",
    "ln_bias",
    "head_in",
    "head_out",
    "addln_bias",
    "ffn1",
    "ffn2",
    "residual",
];

struct EncoderKernels {
    embed: Kernel,
    qkv: Kernel,
    rope_original: Kernel,
    attn_full: Kernel,
    attn_local: Kernel,
    out: Kernel,
    addln: Kernel,
    geglu: Kernel,
    down: Kernel,
    type_embedding: Kernel,
    ln_bias: Kernel,
    head_in: Kernel,
    head_out: Kernel,
    addln_bias: Kernel,
    ffn1: Kernel,
    ffn2: Kernel,
    residual: Kernel,
}

impl EncoderKernels {
    fn resolve(kernels: &Kernels) -> Result<Self> {
        Ok(Self {
            embed: kernels.resolve("embed")?,
            qkv: kernels.resolve("qkv")?,
            rope_original: kernels.resolve("rope_original")?,
            attn_full: kernels.resolve("attn_full")?,
            attn_local: kernels.resolve("attn_local")?,
            out: kernels.resolve("out")?,
            addln: kernels.resolve("addln")?,
            geglu: kernels.resolve("geglu")?,
            down: kernels.resolve("down")?,
            type_embedding: kernels.resolve("type")?,
            ln_bias: kernels.resolve("ln_bias")?,
            head_in: kernels.resolve("head_in")?,
            head_out: kernels.resolve("head_out")?,
            addln_bias: kernels.resolve("addln_bias")?,
            ffn1: kernels.resolve("ffn1")?,
            ffn2: kernels.resolve("ffn2")?,
            residual: kernels.resolve("residual")?,
        })
    }
}

struct EncoderLayer {
    qkv: Buffer,
    out: Buffer,
    mlp_norm: Buffer,
    up: Buffer,
    down: Buffer,
    next_norm: Buffer,
    rotary: [Buffer; 2],
    attention: Kernel,
}

struct HeadLayer {
    norm1: Buffer,
    norm1_bias: Buffer,
    qkv: Buffer,
    qkv_bias: Buffer,
    out: Buffer,
    out_bias: Buffer,
    norm2: Buffer,
    norm2_bias: Buffer,
    up: Buffer,
    up_bias: Buffer,
    down: Buffer,
    down_bias: Buffer,
}

pub struct Encoder {
    pub(crate) cuda: Cuda,
    kernels: EncoderKernels,
    _weights: ResidentWeights,
    embedding: Buffer,
    embedding_norm: Buffer,
    type_embedding: Buffer,
    layers: Vec<EncoderLayer>,
    head: Vec<HeadLayer>,
    zeros: Buffer,
    #[cfg(test)]
    checkpoints: tests::Checkpoints,
}

impl Encoder {
    /// # Safety
    /// `bundle` must contain trusted Laya native code built for this device.
    /// Hash validation binds files together; it does not establish code trust.
    pub unsafe fn load(cuda: &Cuda, checkpoint: &Path, bundle: &Path) -> Result<Self> {
        Config::load(checkpoint)?;
        artifacts::validate_bundle(checkpoint, bundle)?;
        let kernels = unsafe { Kernels::load(cuda, &bundle.join("liblaya_cuda.so"), NAMES) }?;
        let source = Weights::open(&checkpoint.join("model.safetensors"))?;
        let weights = ResidentWeights::upload(cuda, &source)?;
        let kernels = EncoderKernels::resolve(&kernels)?;
        let table = |kind: &str, part: &str| -> Result<Buffer> {
            let data = fs::read(bundle.join(format!("rope_{kind}_{part}.f32")))?;
            ensure!(data.len() == 512 * 32 * 4, "invalid rotary table size");
            cuda.upload(&data)
        };
        let full = [table("full", "cos")?, table("full", "sin")?];
        let local = [table("local", "cos")?, table("local", "sin")?];
        let w = |name: &str| weights.get(name).cloned();
        let mut layers = Vec::with_capacity(28);
        for i in 0..28 {
            let layer = |name: &str| w(&format!("encoder.layers.{i}.{name}"));
            let next_norm = if i < 27 {
                w(&format!("encoder.layers.{}.attn_norm.weight", i + 1))?
            } else {
                w("encoder.final_norm.weight")?
            };
            let (rotary, attention) = if i % 3 == 0 {
                (full.clone(), kernels.attn_full.clone())
            } else {
                (local.clone(), kernels.attn_local.clone())
            };
            layers.push(EncoderLayer {
                qkv: layer("attn.Wqkv.weight")?,
                out: layer("attn.Wo.weight")?,
                mlp_norm: layer("mlp_norm.weight")?,
                up: layer("mlp.Wi.weight")?,
                down: layer("mlp.Wo.weight")?,
                next_norm,
                rotary,
                attention,
            });
        }
        let mut head = Vec::with_capacity(2);
        for i in 0..2 {
            let layer = |name: &str| w(&format!("head.layers.{i}.{name}"));
            head.push(HeadLayer {
                norm1: layer("norm1.weight")?,
                norm1_bias: layer("norm1.bias")?,
                qkv: layer("self_attn.in_proj_weight")?,
                qkv_bias: layer("self_attn.in_proj_bias")?,
                out: layer("self_attn.out_proj.weight")?,
                out_bias: layer("self_attn.out_proj.bias")?,
                norm2: layer("norm2.weight")?,
                norm2_bias: layer("norm2.bias")?,
                up: layer("linear1.weight")?,
                up_bias: layer("linear1.bias")?,
                down: layer("linear2.weight")?,
                down_bias: layer("linear2.bias")?,
            });
        }
        Ok(Self {
            #[cfg(test)]
            checkpoints: Default::default(),
            cuda: cuda.clone(),
            kernels,
            embedding: w("encoder.embeddings.tok_embeddings.weight")?,
            embedding_norm: w("encoder.embeddings.norm.weight")?,
            type_embedding: w("type_emb.weight")?,
            _weights: weights,
            layers,
            head,
            zeros: cuda.upload(&vec![0; 3072 * 4])?,
        })
    }

    /// Writes FP32 final hidden states into `workspace.buffers().residual`.
    /// Inputs include padded rows; a zero length marks a dummy row.
    pub fn run(
        &self,
        ids: &[i64],
        lengths: &[i32],
        types: &[i64],
        workspace: &Workspace,
    ) -> Result<()> {
        validate_inputs(ids, lengths, types, workspace.batch(), workspace.sequence())?;
        let s = workspace.buffers();
        s.ids
            .write(&ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())?;
        s.lengths.write(
            &lengths
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        s.types.write(
            &types
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        self.execute(workspace)?;
        self.cuda.sync()
    }

    /// Enqueues compute after validated inputs have been uploaded; does not synchronize.
    pub(crate) fn execute(&self, workspace: &Workspace) -> Result<()> {
        let (b, l) = (workspace.batch(), workspace.sequence());
        let s = workspace.buffers();
        let k = &self.kernels;
        // Layouts are fixed by ResidentWeights and Workspace; indices were checked on CPU.
        macro_rules! call {
            ($kernel:expr, $args:expr $(,)?) => {
                unsafe { $kernel.launch($args, b, l) }
            };
        }
        let z = &self.zeros;
        call!(
            k.embed,
            [
                &s.ids,
                &self.embedding,
                &self.embedding_norm,
                &s.residual,
                &s.hidden,
            ],
        )?;
        #[cfg(test)]
        self.record("embedding", &s.residual)?;
        // Layer indices are consumed only by test checkpoint instrumentation.
        #[allow(clippy::unused_enumerate_index)]
        for (_i, layer) in self.layers.iter().enumerate() {
            call!(k.qkv, [&s.hidden, &layer.qkv, z, &s.qkv])?;
            call!(
                k.rope_original,
                [&s.qkv, &layer.rotary[0], &layer.rotary[1]],
            )?;
            call!(layer.attention, [&s.qkv, &s.lengths, &s.attention])?;
            call!(k.out, [&s.attention, &layer.out, z, &s.hidden])?;
            call!(
                k.addln,
                [&s.residual, &s.hidden, &layer.mlp_norm, z, &s.hidden],
            )?;
            call!(k.geglu, [&s.hidden, &layer.up, &s.gated])?;
            call!(k.down, [&s.gated, &layer.down, z, &s.hidden])?;
            call!(
                k.addln,
                [&s.residual, &s.hidden, &layer.next_norm, z, &s.hidden]
            )?;
            #[cfg(test)]
            if [0, 1, 2, 27].contains(&_i) {
                self.record(&format!("encoder{_i}"), &s.residual)?;
            }
        }
        call!(
            k.type_embedding,
            [&s.hidden, &self.type_embedding, &s.types, &s.residual],
        )?;
        #[allow(clippy::unused_enumerate_index)]
        for (_i, layer) in self.head.iter().enumerate() {
            call!(
                k.ln_bias,
                [
                    &s.residual,
                    &s.hidden,
                    &layer.norm1,
                    &layer.norm1_bias,
                    &s.hidden,
                ],
            )?;
            call!(k.head_in, [&s.hidden, &layer.qkv, &layer.qkv_bias, &s.qkv])?;
            call!(k.attn_full, [&s.qkv, &s.lengths, &s.attention])?;
            call!(
                k.head_out,
                [&s.attention, &layer.out, &layer.out_bias, &s.hidden],
            )?;
            call!(
                k.addln_bias,
                [
                    &s.residual,
                    &s.hidden,
                    &layer.norm2,
                    &layer.norm2_bias,
                    &s.hidden,
                ],
            )?;
            call!(
                k.ffn1,
                [&s.hidden, &layer.up, &layer.up_bias, &s.feed_forward],
            )?;
            call!(
                k.ffn2,
                [&s.feed_forward, &layer.down, &layer.down_bias, &s.hidden],
            )?;
            call!(k.residual, [&s.residual, &s.hidden])?;
            #[cfg(test)]
            self.record(&format!("head{_i}"), &s.residual)?;
        }
        Ok(())
    }
}

fn validate_inputs(ids: &[i64], lengths: &[i32], types: &[i64], b: usize, l: usize) -> Result<()> {
    ensure!(
        ids.len() == b * l && lengths.len() == b && types.len() == b,
        "encoder input shape mismatch"
    );
    ensure!(
        ids.iter().all(|id| (0..50368).contains(id)),
        "token ID outside vocabulary"
    );
    ensure!(
        lengths.iter().all(|n| *n >= 0 && *n as usize <= l),
        "invalid sequence length"
    );
    ensure!(
        types.iter().all(|t| (0..=2).contains(t)),
        "invalid question type"
    );
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/laya/unit/encoder.rs"]
mod tests;
