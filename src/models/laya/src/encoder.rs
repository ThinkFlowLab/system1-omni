//! Eager execution of the fixed Laya encoder and decision transformer.
use crate::{
    artifacts, config::Config, resident::ResidentWeights, weights::Weights, workspace::Workspace,
};
use anyhow::{Result, ensure};
use omni_cuda::{Buffer, Cuda, kernels::Kernels};
use std::{collections::HashMap, fs, path::Path};

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

pub struct Encoder {
    cuda: Cuda,
    kernels: Kernels,
    weights: ResidentWeights,
    tables: HashMap<String, Buffer>,
    zeros: Buffer,
    #[cfg(test)]
    checkpoints: std::cell::RefCell<Vec<(String, Vec<u8>)>>,
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
        let mut tables = HashMap::new();
        for kind in ["full", "local"] {
            for part in ["cos", "sin"] {
                let name = format!("rope_{kind}_{part}");
                let data = fs::read(bundle.join(format!("{name}.f32")))?;
                ensure!(data.len() == 512 * 32 * 4, "invalid rotary table size");
                tables.insert(name, cuda.upload(&data)?);
            }
        }
        Ok(Self {
            #[cfg(test)]
            checkpoints: Default::default(),
            cuda: cuda.clone(),
            kernels,
            weights,
            tables,
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

    fn execute(&self, workspace: &Workspace) -> Result<()> {
        let (b, l) = (workspace.batch(), workspace.sequence());
        let s = workspace.buffers();
        let w = |name: &str| self.weights.get(name);
        // Layouts are fixed by ResidentWeights and Workspace; indices were checked on CPU.
        let call = |name: &str, args: &[&Buffer]| unsafe { self.kernels.launch(name, args, b, l) };
        let z = &self.zeros;
        call(
            "embed",
            &[
                &s.ids,
                w("encoder.embeddings.tok_embeddings.weight")?,
                w("encoder.embeddings.norm.weight")?,
                &s.residual,
                &s.hidden,
            ],
        )?;
        #[cfg(test)]
        self.record("embedding", &s.residual)?;
        for i in 0..28 {
            let prefix = format!("encoder.layers.{i}");
            let layer = |name: &str| w(&format!("{prefix}.{name}"));
            call("qkv", &[&s.hidden, layer("attn.Wqkv.weight")?, z, &s.qkv])?;
            let kind = if i % 3 == 0 { "full" } else { "local" };
            call(
                "rope_original",
                &[
                    &s.qkv,
                    &self.tables[&format!("rope_{kind}_cos")],
                    &self.tables[&format!("rope_{kind}_sin")],
                ],
            )?;
            call(&format!("attn_{kind}"), &[&s.qkv, &s.lengths, &s.attention])?;
            call(
                "out",
                &[&s.attention, layer("attn.Wo.weight")?, z, &s.hidden],
            )?;
            call(
                "addln",
                &[
                    &s.residual,
                    &s.hidden,
                    layer("mlp_norm.weight")?,
                    z,
                    &s.hidden,
                ],
            )?;
            call("geglu", &[&s.hidden, layer("mlp.Wi.weight")?, &s.gated])?;
            call("down", &[&s.gated, layer("mlp.Wo.weight")?, z, &s.hidden])?;
            let norm = if i < 27 {
                format!("encoder.layers.{}.attn_norm.weight", i + 1)
            } else {
                "encoder.final_norm.weight".into()
            };
            call("addln", &[&s.residual, &s.hidden, w(&norm)?, z, &s.hidden])?;
            #[cfg(test)]
            if [0, 1, 2, 27].contains(&i) {
                self.record(&format!("encoder{i}"), &s.residual)?;
            }
        }
        call(
            "type",
            &[&s.hidden, w("type_emb.weight")?, &s.types, &s.residual],
        )?;
        for i in 0..2 {
            let prefix = format!("head.layers.{i}");
            let layer = |name: &str| w(&format!("{prefix}.{name}"));
            call(
                "ln_bias",
                &[
                    &s.residual,
                    &s.hidden,
                    layer("norm1.weight")?,
                    layer("norm1.bias")?,
                    &s.hidden,
                ],
            )?;
            call(
                "head_in",
                &[
                    &s.hidden,
                    layer("self_attn.in_proj_weight")?,
                    layer("self_attn.in_proj_bias")?,
                    &s.qkv,
                ],
            )?;
            call("attn_full", &[&s.qkv, &s.lengths, &s.attention])?;
            call(
                "head_out",
                &[
                    &s.attention,
                    layer("self_attn.out_proj.weight")?,
                    layer("self_attn.out_proj.bias")?,
                    &s.hidden,
                ],
            )?;
            call(
                "addln_bias",
                &[
                    &s.residual,
                    &s.hidden,
                    layer("norm2.weight")?,
                    layer("norm2.bias")?,
                    &s.hidden,
                ],
            )?;
            call(
                "ffn1",
                &[
                    &s.hidden,
                    layer("linear1.weight")?,
                    layer("linear1.bias")?,
                    &s.feed_forward,
                ],
            )?;
            call(
                "ffn2",
                &[
                    &s.feed_forward,
                    layer("linear2.weight")?,
                    layer("linear2.bias")?,
                    &s.hidden,
                ],
            )?;
            call("residual", &[&s.residual, &s.hidden])?;
            #[cfg(test)]
            self.record(&format!("head{i}"), &s.residual)?;
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
mod tests;
