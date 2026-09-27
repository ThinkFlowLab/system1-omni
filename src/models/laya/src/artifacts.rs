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
        check(bundle.join(name), tables["tables"][name].as_str())?;
    }
    check(
        bundle.join("liblaya_cuda.so"),
        build["library_sha256"].as_str(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture {
        root: PathBuf,
        checkpoint: PathBuf,
        bundle: PathBuf,
    }
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "laya-artifacts-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&root).unwrap();
            let checkpoint = root.join("checkpoint");
            let bundle = root.join("bundle");
            fs::create_dir_all(checkpoint.join("encoder")).unwrap();
            fs::create_dir_all(checkpoint.join("tokenizer")).unwrap();
            fs::create_dir(&bundle).unwrap();
            let mut hashes = serde_json::Map::new();
            for name in CHECKPOINT_ARTIFACTS {
                // The weight fixture crosses several bounded hash reads; no GPU or real model required.
                let data = if name == "model.safetensors" {
                    vec![42; 192 * 1024 + 1]
                } else {
                    name.as_bytes().to_vec()
                };
                fs::write(checkpoint.join(name), &data).unwrap();
                hashes.insert(name.into(), json!(format!("{:x}", Sha256::digest(&data))));
            }
            let mut tables = serde_json::Map::new();
            for name in [
                "rope_full_cos.f32",
                "rope_full_sin.f32",
                "rope_local_cos.f32",
                "rope_local_sin.f32",
            ] {
                fs::write(bundle.join(name), b"table").unwrap();
                tables.insert(
                    name.into(),
                    json!(format!("{:x}", Sha256::digest(b"table"))),
                );
            }
            let table_manifest = json!({
                "abi": 1,
                "laya": "0.3.20",
                "hidden_size": 1024,
                "head_dim": 64,
                "max_len": 512,
                "tables": tables,
                "checkpoint_sha256": hashes,
            });
            fs::write(
                bundle.join("tables.json"),
                serde_json::to_vec(&table_manifest).unwrap(),
            )
            .unwrap();
            fs::write(bundle.join("liblaya_cuda.so"), b"not loaded in CPU test").unwrap();
            let build_manifest = json!({
                "abi": 1,
                "arch": "sm_90a",
                "library_sha256": format!("{:x}", Sha256::digest(b"not loaded in CPU test")),
            });
            fs::write(
                bundle.join("build-manifest.json"),
                serde_json::to_vec(&build_manifest).unwrap(),
            )
            .unwrap();
            Self {
                root,
                checkpoint,
                bundle,
            }
        }
        fn validate(&self) -> Result<()> {
            validate_bundle(&self.checkpoint, &self.bundle)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn matching_artifacts_pass_without_loading_cuda() {
        Fixture::new().validate().unwrap();
    }

    #[test]
    fn each_checkpoint_artifact_is_bound_to_the_bundle() {
        for name in CHECKPOINT_ARTIFACTS {
            let f = Fixture::new();
            let path = f.checkpoint.join(name);
            let mut data = fs::read(&path).unwrap();
            data[0] ^= 1; // Same size: shape/file-size checks alone would not catch substitution.
            fs::write(path, data).unwrap();
            let error = f.validate().unwrap_err().to_string();
            assert!(
                error.contains("hash mismatch") && error.contains(name),
                "{error}"
            );
        }
    }

    #[test]
    fn missing_checkpoint_hashes_fail_closed() {
        for name in CHECKPOINT_ARTIFACTS {
            let f = Fixture::new();
            let path = f.bundle.join("tables.json");
            let mut manifest: serde_json::Value =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            manifest["checkpoint_sha256"]
                .as_object_mut()
                .unwrap()
                .remove(name);
            fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            let error = f.validate().unwrap_err().to_string();
            assert!(
                error.contains("missing checkpoint hash") && error.contains(name),
                "{error}"
            );
        }
        let f = Fixture::new();
        let path = f.bundle.join("tables.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        manifest
            .as_object_mut()
            .unwrap()
            .remove("checkpoint_sha256");
        fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(
            f.validate()
                .unwrap_err()
                .to_string()
                .contains("missing checkpoint hash")
        );
    }

    #[test]
    fn missing_checkpoint_file_fails_closed() {
        for name in CHECKPOINT_ARTIFACTS {
            let f = Fixture::new();
            fs::remove_file(f.checkpoint.join(name)).unwrap();
            let error = f.validate().unwrap_err().to_string();
            assert!(error.contains(name), "{error}");
        }
    }
}
