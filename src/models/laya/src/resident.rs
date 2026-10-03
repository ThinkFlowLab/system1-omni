use crate::weights::{TensorSpec, Weights, checkpoint_tensors};
use anyhow::{Context, Result};
use omni_cuda::{Buffer, Cuda};
use std::collections::HashMap;

/// Checkpoint weights owned by one CUDA context; no inference workspace or tables.
pub struct ResidentWeights {
    buffers: HashMap<String, Buffer>,
    bytes: usize,
}

impl ResidentWeights {
    pub fn upload(cuda: &Cuda, source: &Weights) -> Result<Self> {
        let tensors = checkpoint_tensors();
        source.validate_names(tensors.iter().map(|t| t.name.as_str()))?;
        // Account for the legacy buffer, but do not keep unused calibration on GPU.
        source.f32("temperature", &[3])?;
        Self::upload_tensors(
            cuda,
            source,
            tensors.iter().filter(|t| t.name != "temperature"),
        )
    }

    fn upload_tensors<'a>(
        cuda: &Cuda,
        source: &Weights,
        tensors: impl IntoIterator<Item = &'a TensorSpec>,
    ) -> Result<Self> {
        let mut resident = Self {
            buffers: HashMap::new(),
            bytes: 0,
        };
        for spec in tensors {
            let data = packed(source, spec)?;
            let buffer = cuda
                .upload(&data)
                .with_context(|| format!("upload {}", spec.name))?;
            resident.bytes += buffer.bytes();
            resident.buffers.insert(spec.name.clone(), buffer);
        }
        Ok(resident)
    }

    pub fn get(&self, name: &str) -> Result<&Buffer> {
        self.buffers
            .get(name)
            .with_context(|| format!("no resident weight: {name}"))
    }

    /// Weight allocation bytes, excluding CUDA context and allocator overhead.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

fn packed(source: &Weights, spec: &TensorSpec) -> Result<Vec<u8>> {
    let name = spec.name.as_str();
    let shape = &spec.shape;
    if name == "encoder.embeddings.tok_embeddings.weight" {
        Ok(source
            .f16(name, shape)?
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect())
    } else if (shape.len() == 1 && (name.starts_with("encoder.") || name.starts_with("head.")))
        || name.starts_with("scorer.0.")
    {
        Ok(source
            .f32(name, shape)?
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect())
    } else {
        Ok(source
            .bf16(name, shape)?
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect())
    }
}

#[cfg(test)]
#[path = "../../../../tests/laya/unit/resident.rs"]
mod tests;
