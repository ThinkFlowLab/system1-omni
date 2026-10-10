//! Verify the released immutable artifacts before device initialization.
use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};

pub(crate) const HIDDEN: usize = 2048;
pub(crate) const VOCAB: usize = 248320;
pub(crate) const LABELS: usize = 255;
pub(crate) const PADDED_LABELS: usize = 256;
const FILES: &[(&str, u64, &str)] = &[
    (
        "config.json",
        1790,
        "6cb8daca9fb653c61485ff7452fc068bacd5c27cbee659ecd24b47186b0d1b52",
    ),
    (
        "decider_config.json",
        1240,
        "6e4891f2754a1c18a10f8dadb0c04e439e7f79fab0333d56641491bd4a05e722",
    ),
    (
        "tokenizer.json",
        19989325,
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
    (
        "model.safetensors",
        3763692048,
        "acaef2228b134dcdc20cad4ee79219482c927ec819aa3687b9b8a575c338817f",
    ),
];

pub(crate) struct Checkpoint {
    pub head: Vec<u8>,
}
impl Checkpoint {
    pub fn load(dir: &Path, labels: &[u32]) -> Result<Self> {
        ensure!(labels.len() == LABELS, "expected all 255 label IDs");
        ensure!(
            !dir.join("model.safetensors.index.json").exists(),
            "sharded checkpoint index would override the verified single-file release"
        );
        for &(name, size, digest) in FILES {
            verify_file(&dir.join(name), size, digest)?;
        }
        // In addition to provenance, check every dimension consumed by the shared kernels.
        let cfg = omni_qwen3_5_native::model::Config::load(dir)?;
        ensure!(
            (
                cfg.hidden,
                cfg.intermediate,
                cfg.heads,
                cfg.kv_heads,
                cfg.head_dim,
                cfg.lin_k_heads,
                cfg.lin_v_heads,
                cfg.lin_k_dim,
                cfg.lin_v_dim
            ) == (HIDDEN, 6144, 8, 2, 256, 16, 16, 128, 128)
                && cfg.full_attention.len() == 24
                && cfg
                    .full_attention
                    .iter()
                    .enumerate()
                    .all(|(i, &full)| full == (i % 4 == 3)),
            "expected released Decider-2B backbone layout"
        );
        let file = File::open(dir.join("model.safetensors"))?;
        // SAFETY: checkpoint artifacts must remain immutable while the worker runs.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;
        ensure!(
            tensors
                .names()
                .iter()
                .all(|name| tensors.tensor(name).is_ok_and(|v| v.dtype() == Dtype::BF16)),
            "expected BF16 tensors"
        );
        Ok(Self {
            head: selected_rows(&tensors, labels, HIDDEN, VOCAB)?,
        })
    }
}

fn verify_file(path: &Path, size: u64, expected: &str) -> Result<()> {
    let mut file =
        File::open(path).with_context(|| format!("missing pinned artifact {}", path.display()))?;
    ensure!(
        file.metadata()?.len() == size,
        "{} size mismatch",
        path.display()
    );
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    ensure!(
        format!("{:x}", hash.finalize()) == expected,
        "{} checksum mismatch",
        path.display()
    );
    Ok(())
}

fn selected_rows(
    tensors: &SafeTensors<'_>,
    ids: &[u32],
    hidden: usize,
    vocab: usize,
) -> Result<Vec<u8>> {
    ensure!(
        !ids.is_empty() && ids.len() <= LABELS && hidden > 0,
        "invalid head dimensions"
    );
    ensure!(
        ids.iter().all(|&id| (id as usize) < vocab)
            && ids.iter().collect::<std::collections::HashSet<_>>().len() == ids.len(),
        "invalid label IDs"
    );
    let view = tensors.tensor("model.language_model.embed_tokens.weight")?;
    ensure!(
        view.dtype() == Dtype::BF16 && view.shape() == [vocab, hidden],
        "tied embedding shape/dtype mismatch"
    );
    let mut rows = vec![0; ids.len().next_multiple_of(8) * hidden * 2];
    for (i, &id) in ids.iter().enumerate() {
        let start = id as usize * hidden * 2;
        let row = &view.data()[start..start + hidden * 2];
        ensure!(
            row.as_chunks::<2>()
                .0
                .iter()
                .all(|&bytes| half::bf16::from_le_bytes(bytes).is_finite()),
            "nonfinite selected weights"
        );
        rows[i * hidden * 2..(i + 1) * hidden * 2].copy_from_slice(row);
    }
    Ok(rows)
}

#[cfg(test)]
#[path = "../../../../../tests/decider/checkpoint.rs"]
mod tests;
