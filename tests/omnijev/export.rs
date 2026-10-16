//! The export manifest checks, on a stand-in export directory.
use std::fs;
use std::path::PathBuf;

use omni_omnijev_native::export;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// A directory with small stand-in files and a manifest that lists their hashes.
struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("omnijev-export-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let mut outputs = serde_json::Map::new();
        let shard = "model-00001-of-00001.safetensors";
        let files = [
            (
                "config.json",
                json!({"text_config": {"vocab_size": 248079}}).to_string(),
            ),
            (
                "model.safetensors.index.json",
                json!({"weight_map": {"embed_tokens.weight": shard}}).to_string(),
            ),
            (shard, "stand-in weights".to_string()),
            ("heads.safetensors", "stand-in heads".to_string()),
            ("tokenizer.json", "stand-in tokenizer".to_string()),
            ("vision.safetensors", "stand-in vision".to_string()),
        ];
        for (file, bytes) in files {
            fs::write(dir.join(file), &bytes).unwrap();
            outputs.insert(
                file.into(),
                json!(format!("{:x}", Sha256::digest(bytes.as_bytes()))),
            );
        }
        let manifest = json!({
            "format": "omnijev-merged/1",
            "model_id": "tinnel123/OmniJev",
            "checkpoint_revision": "ffe5f436eaf22e20e2f041f8e74e121fd057a6cb",
            "base_model_id": "Qwen/Qwen3.5-4B",
            "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "lora": {"rank": 32, "alpha": 64, "merged": "bfloat16"},
            "vocabulary": 248079,
            "option_tokens": {"<|opt|>": 248077, "<|/opt|>": 248078},
            "max_pixels": 602112,
            "heads": {"norm": "softmax", "lm_features": true, "ordinal": true},
            "calibration": {"temperatures": {"noul": 1.1991, "choice": 1.1199, "score": 1.1909},
                            "noul_bias": 0.05310856608843584},
            "inputs": {},
            "outputs": outputs,
        });
        let dir = Self(dir);
        dir.write(&manifest);
        dir
    }

    fn manifest(&self) -> Value {
        serde_json::from_slice(&fs::read(self.0.join("omnijev_export.json")).unwrap()).unwrap()
    }

    fn write(&self, manifest: &Value) {
        fs::write(
            self.0.join("omnijev_export.json"),
            serde_json::to_vec(manifest).unwrap(),
        )
        .unwrap();
    }

    fn error(&self) -> String {
        format!(
            "{:#}",
            export::load(&self.0)
                .err()
                .expect("the export must be refused")
        )
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn reads_a_matching_export_and_its_calibration() {
    let dir = Dir::new("ok");
    let export = export::load(&dir.0).unwrap();
    assert_eq!(export.calibration.temperatures, [1.1991, 1.1199, 1.1909]);
    assert_eq!(export.calibration.noul_bias, 0.05310856608843584);
}

#[test]
fn refuses_changed_files_and_other_releases() {
    let dir = Dir::new("changed");
    fs::write(dir.0.join("vision.safetensors"), "changed").unwrap();
    assert!(
        dir.error().contains("vision.safetensors does not match"),
        "{}",
        dir.error()
    );

    let dir = Dir::new("missing");
    fs::remove_file(dir.0.join("heads.safetensors")).unwrap();
    assert!(dir.error().contains("heads.safetensors"), "{}", dir.error());

    let edits: [(&str, Value, &str); 6] = [
        ("checkpoint_revision", json!("main"), "checkpoint_revision"),
        ("max_pixels", json!(401408), "max_pixels"),
        ("vocabulary", json!(248320), "vocabulary"),
        (
            "heads",
            json!({"norm": "sigmoid", "lm_features": true, "ordinal": true}),
            "heads",
        ),
        (
            "calibration",
            json!({"temperatures": {"noul": 0.0, "choice": 1.0, "score": 1.0}, "noul_bias": 0.0}),
            "temperatures",
        ),
        (
            "calibration",
            json!({"temperatures": {"noul": 1.0, "choice": 1.0, "score": 1.0}}),
            "noul_bias",
        ),
    ];
    for (key, value, reason) in edits {
        let dir = Dir::new(key);
        let mut manifest = dir.manifest();
        manifest[key] = value;
        dir.write(&manifest);
        assert!(dir.error().contains(reason), "{key}: {}", dir.error());
    }

    for (name, reason) in [
        ("tokenizer.json", "does not list tokenizer.json"),
        ("../config.json", "invalid file name"),
    ] {
        let dir = Dir::new("listing");
        let mut manifest = dir.manifest();
        let outputs = manifest["outputs"].as_object_mut().unwrap();
        let digest = outputs.remove("tokenizer.json").unwrap();
        if name != "tokenizer.json" {
            outputs.insert("tokenizer.json".into(), digest);
            outputs.insert(name.into(), json!("00"));
        }
        dir.write(&manifest);
        assert!(dir.error().contains(reason), "{name}: {}", dir.error());
    }
    // the shard the index names, and the vocabulary
    let dir = Dir::new("shard");
    let mut manifest = dir.manifest();
    manifest["outputs"]
        .as_object_mut()
        .unwrap()
        .remove("model-00001-of-00001.safetensors");
    dir.write(&manifest);
    assert!(
        dir.error()
            .contains("does not list model-00001-of-00001.safetensors"),
        "{}",
        dir.error()
    );
    let dir = Dir::new("vocab");
    let config = serde_json::json!({"text_config": {"vocab_size": 248320}}).to_string();
    fs::write(dir.0.join("config.json"), &config).unwrap();
    let mut manifest = dir.manifest();
    manifest["outputs"]["config.json"] = json!(format!("{:x}", Sha256::digest(config.as_bytes())));
    dir.write(&manifest);
    assert!(
        dir.error().contains("vocabulary is 248320"),
        "{}",
        dir.error()
    );
}
