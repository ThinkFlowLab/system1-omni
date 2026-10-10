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
const ADAPTER_FILE: &str = "adapter_model.safetensors";
const PINNED_INVENTORY: &str = include_str!("../../../../../recipe/jemm/pinned_inventory.json");

/// Validate the unmerged adapter contract before opening a device or artifact file.
pub fn lora_contract(manifest: &Value) -> Result<(&'static str, f32)> {
    ensure!(
        manifest["format"] == "jemm-native/2",
        "expected jemm-native/2; legacy premerged BF16 exports are unsupported"
    );
    ensure!(
        manifest["lora_mode"] == "unmerged_fp32"
            && manifest["lora_file"] == ADAPTER_FILE
            && manifest["lora_rank"].as_u64() == Some(16)
            && manifest["lora_pairs"].as_u64() == Some(496)
            && manifest["lora_scale"].as_f64() == Some(2.)
            && manifest.get("lora_merge").is_none(),
        "expected unmerged FP32 JEMM LoRA: rank16, 496 pairs, scale2"
    );
    let pinned: Value = serde_json::from_str(PINNED_INVENTORY)?;
    let files = &pinned["adapter"]["files"];
    ensure!(
        manifest["lora_sha256"] == files[ADAPTER_FILE],
        "LoRA checksum differs from pinned JEMM adapter"
    );
    for name in [ADAPTER_FILE, "adapter_config.json"] {
        ensure!(
            manifest["source_sha256"]["adapter"][name] == files[name]
                && manifest["export_sha256"][name] == files[name],
            "LoRA source/export checksum differs from pinned artifact {name}"
        );
    }
    Ok((ADAPTER_FILE, 2.))
}

pub fn verify_export(dir: &Path, manifest: &Value) -> Result<()> {
    lora_contract(manifest)?;
    let pinned: Value = serde_json::from_str(PINNED_INVENTORY)?;
    let expected: serde_json::Map<String, Value> = pinned["base"]["tensors"]
        .as_object()
        .context("pinned tensor inventory")?
        .iter()
        .filter(|(key, _)| key.starts_with("model.language_model."))
        .map(|(key, spec)| (key.clone(), spec["file"].clone()))
        .collect();
    let index: Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
    ensure!(
        index["weight_map"].as_object() == Some(&expected),
        "expected language-only index with original pinned tensor keys/shards"
    );
    for filename in expected.values() {
        let filename = filename.as_str().context("pinned language shard")?;
        ensure!(
            manifest["export_sha256"][filename] == pinned["base"]["files"][filename]
                && manifest["source_sha256"]["base"][filename] == pinned["base"]["files"][filename],
            "expected original unmerged BF16 language shard {filename}"
        );
    }
    verify_export_files(dir, manifest)
}

/// Stream every checksum-covered file; model-specific pins are checked by verify_export.
pub fn verify_export_files(dir: &Path, manifest: &Value) -> Result<()> {
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
        "model.safetensors.index.json",
        "vision.safetensors",
        "jemm_lm_head.safetensors",
        ADAPTER_FILE,
        "adapter_config.json",
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
