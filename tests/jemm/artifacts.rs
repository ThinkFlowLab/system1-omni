use omni_jemm_native::artifacts::{lora_contract, verify_export, verify_export_files};
use serde_json::json;
#[test]
fn export_hash_map_is_mandatory_and_rejects_traversal() {
    let dir = std::env::temp_dir();
    assert!(
        verify_export_files(&dir, &json!({}))
            .unwrap_err()
            .to_string()
            .contains("export_sha256")
    );
    assert!(
        verify_export_files(&dir, &json!({"export_sha256":{"../escape":"0".repeat(64)}}))
            .unwrap_err()
            .to_string()
            .contains("relative")
    );
}
#[test]
fn hashes_every_exported_file_and_requires_all_language_shards() {
    use sha2::{Digest, Sha256};
    let dir = std::env::temp_dir().join(format!("jemm-artifacts-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let index = br#"{"weight_map":{"embed_tokens.weight":"model-1.safetensors"}}"#;
    let mut hashes = serde_json::Map::new();
    for name in [
        "config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "model.safetensors.index.json",
        "vision.safetensors",
        "jemm_lm_head.safetensors",
        "adapter_model.safetensors",
        "adapter_config.json",
        "model-1.safetensors",
        "decision_config.json",
    ] {
        let bytes = if name == "model.safetensors.index.json" {
            index.as_slice()
        } else {
            b"original"
        };
        std::fs::write(dir.join(name), bytes).unwrap();
        hashes.insert(name.into(), json!(format!("{:x}", Sha256::digest(bytes))));
    }
    let manifest = json!({"export_sha256":hashes});
    verify_export_files(&dir, &manifest).unwrap();
    std::fs::write(dir.join("decision_config.json"), b"tampered").unwrap();
    assert!(
        verify_export_files(&dir, &manifest)
            .unwrap_err()
            .to_string()
            .contains("mismatch: decision_config.json")
    );
    let mut incomplete = manifest;
    incomplete["export_sha256"]
        .as_object_mut()
        .unwrap()
        .remove("model-1.safetensors");
    assert!(
        verify_export_files(&dir, &incomplete)
            .unwrap_err()
            .to_string()
            .contains("missing language shard")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn malformed_unmerged_contracts_fail_before_file_or_device_initialization() {
    let file = "adapter_model.safetensors";
    let hash = "fac397c8af737bb8d259efbe16f771507f77276fb44ab74942cd2e1bc05c3fef";
    let config_hash = "7cfc1e46979f37a20012207b4ed72a720d24d74c9205952a5b62c1f81ee11427";
    let valid = json!({"format":"jemm-native/2","lora_mode":"unmerged_fp32","lora_file":file,
        "lora_rank":16,"lora_pairs":496,"lora_scale":2.0,"lora_sha256":hash,
        "source_sha256":{"adapter":{file:hash,"adapter_config.json":config_hash}},
        "export_sha256":{file:hash,"adapter_config.json":config_hash}});
    assert_eq!(lora_contract(&valid).unwrap(), (file, 2.));
    for (key, value) in [
        ("lora_merge", json!("FP32 delta into BF16 base")),
        ("format", json!("jemm-native/1")),
        ("lora_mode", json!("merged_bf16")),
        ("lora_file", json!("../adapter_model.safetensors")),
        ("lora_rank", json!(8)),
        ("lora_rank", json!("16")),
        ("lora_pairs", json!(495)),
        ("lora_scale", json!(1.0)),
        ("lora_sha256", json!("0".repeat(64))),
    ] {
        let mut manifest = valid.clone();
        manifest[key] = value;
        let error = verify_export(std::path::Path::new("/absent-jemm-export"), &manifest)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("jemm-native/2") || error.contains("LoRA"),
            "{key}: {error}"
        );
    }
}

#[tokio::test]
async fn executor_rejects_legacy_export_before_cuda_library_or_file_access() {
    let error = omni_jemm_native::executor::Executor::load(
        std::path::Path::new("/absent-jemm-export"),
        std::path::Path::new("/absent-cuda-library"),
        &json!({"format":"jemm-native/1"}),
    )
    .await
    .err()
    .unwrap();
    assert!(error.to_string().contains("jemm-native/2"), "{error}");
}

#[test]
fn original_language_index_and_adapter_provenance_are_mandatory() {
    let inventory: serde_json::Value =
        serde_json::from_str(include_str!("../../recipe/jemm/pinned_inventory.json")).unwrap();
    let adapter_hashes = inventory["adapter"]["files"].clone();
    let mut manifest = json!({"format":"jemm-native/2","lora_mode":"unmerged_fp32",
        "lora_file":"adapter_model.safetensors","lora_rank":16,"lora_pairs":496,"lora_scale":2.,
        "lora_sha256":adapter_hashes["adapter_model.safetensors"],
        "source_sha256":{"adapter":adapter_hashes},"export_sha256":inventory["adapter"]["files"]});
    for scope in ["source_sha256", "export_sha256"] {
        for name in ["adapter_model.safetensors", "adapter_config.json"] {
            let mut bad = manifest.clone();
            if scope == "source_sha256" {
                bad[scope]["adapter"][name] = json!("0".repeat(64));
            } else {
                bad[scope][name] = json!("0".repeat(64));
            }
            assert!(
                lora_contract(&bad)
                    .unwrap_err()
                    .to_string()
                    .contains("LoRA")
            );
        }
    }
    let dir = std::env::temp_dir().join(format!("jemm-original-index-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        br#"{"weight_map":{"embed_tokens.weight":"merged.safetensors"}}"#,
    )
    .unwrap();
    assert!(
        verify_export(&dir, &manifest)
            .unwrap_err()
            .to_string()
            .contains("original pinned tensor keys/shards")
    );
    let map: serde_json::Map<String, serde_json::Value> = inventory["base"]["tensors"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(key, _)| key.starts_with("model.language_model."))
        .map(|(key, spec)| (key.clone(), spec["file"].clone()))
        .collect();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&json!({"weight_map":map})).unwrap(),
    )
    .unwrap();
    assert!(
        verify_export(&dir, &manifest)
            .unwrap_err()
            .to_string()
            .contains("original unmerged BF16 language shard")
    );
    // A valid index alone cannot bless newly merged/re-serialized shards.
    manifest["source_sha256"]["base"] = inventory["base"]["files"].clone();
    for filename in map.values() {
        let name = filename.as_str().unwrap();
        manifest["export_sha256"][name] = inventory["base"]["files"][name].clone();
    }
    assert!(
        verify_export(&dir, &manifest)
            .unwrap_err()
            .to_string()
            .contains("required artifact config.json")
    );
    std::fs::remove_dir_all(dir).unwrap();
}
