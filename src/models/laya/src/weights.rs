use anyhow::{Result, ensure};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::{fs::File, path::Path};

pub struct Weights {
    data: Mmap,
}
impl Weights {
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
