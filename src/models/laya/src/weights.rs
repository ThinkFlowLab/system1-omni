use anyhow::{Result, ensure};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::{fs::File, path::Path};

pub struct Weights {
    data: Mmap,
}
impl Weights {
    /// Reject omitted, extra or duplicate names in the expected checkpoint inventory.
    pub fn validate_names<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Result<()> {
        let tensors = SafeTensors::deserialize(&self.data)?;
        let mut expected = std::collections::BTreeSet::new();
        for name in names {
            ensure!(expected.insert(name), "duplicate expected tensor: {name}");
        }
        let names = tensors.names();
        let actual: std::collections::BTreeSet<_> = names.into_iter().collect();
        if expected != actual {
            let joined = |set: std::collections::BTreeSet<&str>| {
                set.into_iter().collect::<Vec<_>>().join(", ")
            };
            let missing = joined(expected.difference(&actual).copied().collect());
            let extra = joined(actual.difference(&expected).copied().collect());
            anyhow::bail!(
                "tensor inventory does not match checkpoint: missing [{missing}], unexpected [{extra}]"
            );
        }
        Ok(())
    }
    /// The checkpoint must remain immutable while the mapping exists.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: model files are read-only inputs; no mutable mapping is created.
        let data = unsafe { Mmap::map(&file)? };
        SafeTensors::deserialize(&data)?;
        Ok(Self { data })
    }
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let tensors = SafeTensors::deserialize(&self.data)?;
        let t = tensors.tensor(name)?;
        ensure!(
            t.shape() == shape,
            "{name}: expected {shape:?}, got {:?}",
            t.shape()
        );
        let out = match t.dtype() {
            Dtype::F16 => t
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            Dtype::BF16 => t
                .data()
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            Dtype::F32 => t
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            dt => anyhow::bail!("{name}: unsupported dtype {dt:?}"),
        };
        Ok(out)
    }
    pub fn bf16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        Ok(self
            .f32(name, shape)?
            .into_iter()
            .map(|f| bf16::from_f32(f).to_bits())
            .collect())
    }
    pub fn f16(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        Ok(self
            .f32(name, shape)?
            .into_iter()
            .map(|f| f16::from_f32(f).to_bits())
            .collect())
    }
}

/// Names and shapes in the supported checkpoint; storage precision is chosen by the caller.
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
}
pub fn checkpoint_tensors() -> Vec<TensorSpec> {
    const D: usize = 1024;
    let mut tensors = Vec::new();
    let mut add = |name: &str, shape: &[usize]| {
        tensors.push(TensorSpec {
            name: name.into(),
            shape: shape.into(),
        });
    };
    add("encoder.embeddings.tok_embeddings.weight", &[50368, D]);
    add("encoder.embeddings.norm.weight", &[D]);
    add("encoder.final_norm.weight", &[D]);
    for i in 0..28 {
        let p = format!("encoder.layers.{i}");
        if i > 0 {
            add(&format!("{p}.attn_norm.weight"), &[D]);
        }
        add(&format!("{p}.mlp_norm.weight"), &[D]);
        for (name, shape) in [
            ("attn.Wqkv.weight", vec![3 * D, D]),
            ("attn.Wo.weight", vec![D, D]),
            ("mlp.Wi.weight", vec![5248, D]),
            ("mlp.Wo.weight", vec![D, 2624]),
        ] {
            add(&format!("{p}.{name}"), &shape);
        }
    }
    add("type_emb.weight", &[3, D]);
    for i in 0..2 {
        let p = format!("head.layers.{i}");
        for n in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"] {
            add(&format!("{p}.{n}"), &[D]);
        }
        for (n, rows, cols) in [
            ("self_attn.in_proj_weight", 3 * D, D),
            ("self_attn.out_proj.weight", D, D),
            ("linear1.weight", 4 * D, D),
            ("linear2.weight", D, 4 * D),
        ] {
            add(&format!("{p}.{n}"), &[rows, cols]);
        }
        for (n, len) in [
            ("self_attn.in_proj_bias", 3 * D),
            ("self_attn.out_proj.bias", D),
            ("linear1.bias", 4 * D),
            ("linear2.bias", D),
        ] {
            add(&format!("{p}.{n}"), &[len]);
        }
    }
    for n in ["scorer.0.weight", "scorer.0.bias"] {
        add(n, &[D]);
    }
    for (p, n, k) in [
        ("scorer.1", D, D),
        ("scorer.3", 1, D),
        ("act_head.0", 256, 1028),
        ("act_head.2", 2, 256),
    ] {
        add(&format!("{p}.weight"), &[n, k]);
        add(&format!("{p}.bias"), &[n]);
    }

    // Laya 0.3.20 common.py registers this legacy buffer but forward does not
    // consume it. Agent decoding uses fitted config temperatures instead.
    // Keep it in the inventory for complete checkpoint accounting.
    add("temperature", &[3]);
    tensors
}
