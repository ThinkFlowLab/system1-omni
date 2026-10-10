//! Streaming integrity verification before native device initialization.
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Component, Path},
};
fn relative(name: &str) -> bool {
    !name.is_empty()
        && Path::new(name)
            .components()
            .all(|p| matches!(p, Component::Normal(_)))
}
pub fn verify_export(dir: &Path, manifest: &Value) -> Result<()> {
    let hashes = manifest["export_sha256"]
        .as_object()
        .context("export_sha256 must contain all native artifact hashes")?;
    ensure!(!hashes.is_empty(), "export_sha256 must not be empty");
    for (name, expected) in hashes {
        ensure!(
            relative(name),
            "export hash paths must be relative normal paths"
        );
        let hash = expected
            .as_str()
            .context("export SHA256 must be a string")?;
        ensure!(
            hash.len() == 64
                && hash
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "invalid export SHA256 for {name}"
        );
        ensure!(
            !["jemm_export.json", "export_progress.json"].contains(&name.as_str()),
            "export hash map must not include manifest/progress markers"
        );
    }
    for name in [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "decision_config.json",
        "model.safetensors.index.json",
        "vision.safetensors",
        "jemm_lm_head.safetensors",
    ] {
        ensure!(
            hashes.contains_key(name),
            "export_sha256 missing required artifact {name}"
        );
    }
    let index: Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
    let map = index["weight_map"]
        .as_object()
        .context("language index weight_map")?;
    ensure!(
        !map.is_empty(),
        "language index weight_map must not be empty"
    );
    for file in map.values() {
        let file = file.as_str().context("language shard path")?;
        ensure!(
            relative(file) && hashes.contains_key(file),
            "export_sha256 missing language shard {file}"
        );
    }
    let canonical_dir = dir.canonicalize()?;
    let mut buffer = vec![0u8; 1 << 20];
    for (name, expected) in hashes {
        let path = dir.join(name);
        let metadata =
            std::fs::symlink_metadata(&path).with_context(|| format!("artifact {name}"))?;
        ensure!(
            metadata.is_file()
                && !metadata.file_type().is_symlink()
                && path.canonicalize()?.starts_with(&canonical_dir),
            "artifact must be a regular file within export: {name}"
        );
        let mut file = File::open(&path)?;
        let mut digest = Sha256::new();
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        ensure!(
            format!("{:x}", digest.finalize()) == expected.as_str().expect("validated hash"),
            "export SHA256 mismatch: {name}"
        );
    }
    Ok(())
}
