//! The CLM head tensors: inventory, loading and the expected shapes.
use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeSet, fs::File, path::Path};

use crate::config::{Config, HeadConfig};

/// The two projection heads.
pub const HEADS: [&str; 2] = ["state_head", "action_head"];

/// One head's parameters, all FP32 and row-major.
#[derive(Debug, Clone)]
pub struct Head {
    pub inp_weight: Vec<f32>,
    pub inp_bias: Vec<f32>,
    pub hidden_weight: Vec<f32>,
    pub hidden_bias: Vec<f32>,
    /// Absent when the configuration has no hidden blocks; LayerNorm weight when present.
    pub norm_weight: Option<Vec<f32>>,
    pub norm_bias: Option<Vec<f32>>,
    pub out_weight: Vec<f32>,
    pub out_bias: Vec<f32>,
}

pub struct Heads {
    pub state: Head,
    pub action: Head,
    pub config: Config,
}

pub struct Weights {
    data: Mmap,
}

impl Weights {
    /// The checkpoint must remain immutable while the mapping exists.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: model files are read-only inputs; no mutable mapping is created.
        let data = unsafe { Mmap::map(&file)? };
        SafeTensors::deserialize(&data)?;
        Ok(Self { data })
    }

    fn tensors(&self) -> Result<SafeTensors<'_>> {
        Ok(SafeTensors::deserialize(&self.data)?)
    }

    /// The safetensors metadata, which carries the head configuration.
    ///
    /// `SafeTensors` keeps its metadata private, so this reads the header directly from
    /// the mapping. No tensor data is touched.
    pub fn metadata(&self) -> Result<Option<std::collections::HashMap<String, String>>> {
        let (_, header) =
            SafeTensors::read_metadata(&self.data).context("read the safetensors header")?;
        Ok(header.metadata().clone())
    }

    /// Reject omitted, extra or duplicate names in the expected inventory.
    pub fn validate_names<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Result<()> {
        let tensors = self.tensors()?;
        let mut expected = BTreeSet::new();
        for name in names {
            ensure!(expected.insert(name), "duplicate expected tensor: {name}");
        }
        let actual: BTreeSet<&str> = tensors.names().into_iter().collect();
        if expected != actual {
            let joined = |set: BTreeSet<&str>| set.into_iter().collect::<Vec<_>>().join(", ");
            let missing = joined(expected.difference(&actual).copied().collect());
            let extra = joined(actual.difference(&expected).copied().collect());
            anyhow::bail!(
                "tensor inventory does not match checkpoint: missing [{missing}], unexpected [{extra}]"
            );
        }
        Ok(())
    }

    /// One tensor as FP32, checking the shape first so a mismatch is an error rather
    /// than a reinterpreted buffer.
    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let tensors = self.tensors()?;
        let t = tensors
            .tensor(name)
            .with_context(|| format!("read tensor {name}"))?;
        ensure!(
            t.shape() == shape,
            "{name}: expected {shape:?}, got {:?}",
            t.shape()
        );
        let out = match t.dtype() {
            Dtype::F32 => t
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            dt => anyhow::bail!("{name}: unsupported dtype {dt:?}, the export writes FP32"),
        };
        Ok(out)
    }
}

/// Names and shapes of every tensor the heads are built from.
pub fn head_tensors(cfg: &HeadConfig) -> Result<Vec<(String, Vec<usize>)>> {
    let blocks = cfg.hidden_blocks()?;
    let (h, w, p) = (cfg.hidden_size, cfg.width, cfg.projection_dim);
    let mut out = Vec::new();
    for head in HEADS {
        out.push((format!("{head}.inp.weight"), vec![w, h]));
        out.push((format!("{head}.inp.bias"), vec![w]));
        for i in 0..blocks {
            out.push((format!("{head}.hidden.{i}.weight"), vec![w, w]));
            out.push((format!("{head}.hidden.{i}.bias"), vec![w]));
            if cfg.layernorm {
                out.push((format!("{head}.norms.{i}.weight"), vec![w]));
                out.push((format!("{head}.norms.{i}.bias"), vec![w]));
            }
        }
        out.push((format!("{head}.out.weight"), vec![p, w]));
        out.push((format!("{head}.out.bias"), vec![p]));
    }
    Ok(out)
}

fn load_head(weights: &Weights, name: &str, cfg: &HeadConfig) -> Result<Head> {
    let (h, w, p) = (cfg.hidden_size, cfg.width, cfg.projection_dim);
    let blocks = cfg.hidden_blocks()?;
    ensure!(
        blocks <= 1,
        "the published checkpoint has {blocks} hidden blocks; the loader implements one"
    );
    let (hidden_weight, hidden_bias, norm_weight, norm_bias) = if blocks == 1 {
        (
            weights.f32(&format!("{name}.hidden.0.weight"), &[w, w])?,
            weights.f32(&format!("{name}.hidden.0.bias"), &[w])?,
            Some(weights.f32(&format!("{name}.norms.0.weight"), &[w])?),
            Some(weights.f32(&format!("{name}.norms.0.bias"), &[w])?),
        )
    } else {
        (Vec::new(), Vec::new(), None, None)
    };
    Ok(Head {
        inp_weight: weights.f32(&format!("{name}.inp.weight"), &[w, h])?,
        inp_bias: weights.f32(&format!("{name}.inp.bias"), &[w])?,
        hidden_weight,
        hidden_bias,
        norm_weight,
        norm_bias,
        out_weight: weights.f32(&format!("{name}.out.weight"), &[p, w])?,
        out_bias: weights.f32(&format!("{name}.out.bias"), &[p])?,
    })
}

impl Heads {
    /// Read the config from the metadata, check the inventory, and load every tensor.
    pub fn load(weights: &Weights) -> Result<Self> {
        let config = Config::from_metadata(weights.metadata()?.as_ref())?;
        let expected = head_tensors(&config.head)?;
        weights.validate_names(expected.iter().map(|(n, _)| n.as_str()))?;
        let mut loaded = Vec::new();
        for head in HEADS {
            loaded.push(load_head(weights, head, &config.head)?);
        }
        let action = loaded.pop().expect("two heads");
        let state = loaded.pop().expect("two heads");
        Ok(Self {
            state,
            action,
            config,
        })
    }
}
