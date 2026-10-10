//! Hash checks before assigning the pinned multimodal model identity.
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};
const LOCK_SHA: &str = "9820bd232c5762f114e19680c0f8203d7e1faaf8a60c196cfe01964d6d8a6c09";

fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn verify_files(
    root: &Path,
    files: &serde_json::Map<String, Value>,
    extra: Option<&str>,
) -> Result<()> {
    ensure!(!files.is_empty(), "empty artifact hash manifest");
    for (name, expected) in files {
        ensure!(
            Path::new(name).components().count() == 1
                && matches!(
                    Path::new(name).components().next(),
                    Some(std::path::Component::Normal(_))
                ),
            "invalid artifact name"
        );
        let path = root.join(name);
        ensure!(
            Some(fs::metadata(&path)?.len()) == expected["size"].as_u64(),
            "{} size mismatch",
            path.display()
        );
        ensure!(
            Some(hash_file(&path)?.as_str()) == expected["sha256"].as_str(),
            "{} checksum mismatch",
            path.display()
        );
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_str().context("non-UTF8 artifact")?;
        ensure!(
            name == ".cache" || Some(name) == extra || files.contains_key(name),
            "unlisted artifact: {name}"
        );
    }
    Ok(())
}

pub fn verify_sources(base: &Path, adapter: &Path) -> Result<()> {
    let lock = fs::read(
        base.parent()
            .context("base has no parent")?
            .join("weights.lock.json"),
    )?;
    ensure!(
        format!("{:x}", Sha256::digest(&lock)) == LOCK_SHA,
        "upstream weights manifest checksum mismatch"
    );
    let lock: Value = serde_json::from_slice(&lock)?;
    for artifact in lock["artifacts"].as_array().context("artifacts")? {
        let mut selected = serde_json::Map::new();
        let is_adapter = artifact["role"] == "adapter";
        for (name, value) in artifact["files"].as_object().context("files")? {
            if is_adapter {
                if let Some(name) = name.strip_prefix("multimodal/") {
                    selected.insert(name.into(), value.clone());
                }
            } else {
                selected.insert(name.clone(), value.clone());
            }
        }
        verify_files(if is_adapter { adapter } else { base }, &selected, None)?;
    }
    Ok(())
}

pub fn verify_export(dir: &Path, marker: &Value) -> Result<()> {
    let files = marker["files"]
        .as_object()
        .context("language export has no file hashes; rerun export_multimodal_language.py")?;
    for name in ["config.json", "tokenizer.json"] {
        ensure!(files.contains_key(name), "export must hash {name}");
    }
    ensure!(
        files.keys().any(|n| n.ends_with(".safetensors")),
        "export must hash model weights"
    );
    verify_files(dir, files, Some("cua_s1_language_export.json"))
}
