//! Bind a compiled bundle to its read-only checkpoint before CUDA startup.
use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path};

pub const CHECKPOINT_ARTIFACTS: [&str; 5] = [
    "rl_agent_config.json",
    "encoder/config.json",
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub fn validate_bundle(checkpoint: &Path, bundle: &Path) -> Result<()> {
    let tables: serde_json::Value = serde_json::from_slice(&fs::read(bundle.join("tables.json"))?)?;
    let build: serde_json::Value =
        serde_json::from_slice(&fs::read(bundle.join("build-manifest.json"))?)?;
    ensure!(
        tables["abi"] == 1
            && tables["laya"] == "0.3.20"
            && tables["hidden_size"] == 1024
            && tables["head_dim"] == 64
            && tables["max_len"] == 512
            && build["abi"] == 1
            && build["arch"] == "sm_90a",
        "unsupported CUDA bundle"
    );
    let check = |path: std::path::PathBuf, expected: Option<&str>| -> Result<()> {
        let hash = sha256_file(&path)?;
        ensure!(
            expected == Some(hash.as_str()),
            "bundle hash mismatch: {}",
            path.display()
        );
        Ok(())
    };
    for name in CHECKPOINT_ARTIFACTS {
        let expected = tables["checkpoint_sha256"][name]
            .as_str()
            .with_context(|| {
                format!("missing checkpoint hash for {name}; regenerate tables.json")
            })?;
        check(checkpoint.join(name), Some(expected))?;
    }
    for name in [
        "rope_full_cos.f32",
        "rope_full_sin.f32",
        "rope_local_cos.f32",
        "rope_local_sin.f32",
    ] {
        ensure!(
            fs::metadata(bundle.join(name))?.len() == 512 * 32 * 4,
            "invalid rotary table size: {name}"
        );
        check(bundle.join(name), tables["tables"][name].as_str())?;
    }
    check(
        bundle.join("liblaya_cuda.so"),
        build["library_sha256"].as_str(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
