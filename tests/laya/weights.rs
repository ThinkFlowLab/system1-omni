use omni_laya::weights::Weights;
use sha2::{Digest, Sha256};
#[test]
#[ignore = "requires LAYA_CHECKPOINT and LAYA_WEIGHT_ORACLE; CPU only"]
fn every_weight_conversion_matches_torch() {
    let checkpoint = std::path::PathBuf::from(std::env::var_os("LAYA_CHECKPOINT").unwrap());
    let oracle = std::fs::read(std::env::var_os("LAYA_WEIGHT_ORACLE").unwrap()).unwrap();
    let rows: Vec<serde_json::Value> = serde_json::from_slice(&oracle).unwrap();
    assert_eq!(rows.len(), 206, "oracle must cover the frozen checkpoint");
    let mut names = std::collections::HashSet::new();
    let weights = Weights::open(&checkpoint.join("model.safetensors")).unwrap();
    let inventory = omni_laya::weights::checkpoint_tensors();
    weights
        .validate_names(inventory.iter().map(|t| t.name.as_str()))
        .unwrap();
    let checkpoint_names: std::collections::HashSet<_> =
        inventory.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(checkpoint_names.len(), rows.len());
    for row in rows {
        let name = row["name"].as_str().unwrap();
        assert!(
            checkpoint_names.contains(name),
            "unaccounted checkpoint tensor: {name}"
        );
        assert!(
            names.insert(name.to_owned()),
            "duplicate oracle tensor: {name}"
        );
        let shape: Vec<usize> = serde_json::from_value(row["shape"].clone()).unwrap();
        let spec = inventory.iter().find(|spec| spec.name == name).unwrap();
        assert_eq!(spec.shape, shape, "{name}: checkpoint shape");
        for dtype in ["f32", "f16", "bf16"] {
            let bytes: Vec<u8> = match dtype {
                "f32" => weights
                    .f32(name, &shape)
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                "f16" => weights
                    .f16(name, &shape)
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
                _ => weights
                    .bf16(name, &shape)
                    .unwrap()
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            };
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                row[dtype].as_str().unwrap(),
                "{name} {dtype}"
            );
        }
    }
}

#[test]
#[ignore = "requires LAYA_CHECKPOINT; CPU only"]
fn checkpoint_inventory_covers_legacy_temperature() {
    let checkpoint = std::path::PathBuf::from(std::env::var_os("LAYA_CHECKPOINT").unwrap());
    let weights = Weights::open(&checkpoint.join("model.safetensors")).unwrap();
    let inventory = omni_laya::weights::checkpoint_tensors();
    weights
        .validate_names(inventory.iter().map(|t| t.name.as_str()))
        .unwrap();
    assert!(
        weights
            .validate_names(
                inventory
                    .iter()
                    .filter(|t| t.name != "temperature")
                    .map(|t| t.name.as_str())
            )
            .is_err()
    );
    assert!(
        weights
            .validate_names(inventory.iter().map(|t| t.name.as_str()).chain(["unknown"]))
            .is_err()
    );
    let legacy = weights.f32("temperature", &[3]).unwrap();
    assert_eq!(legacy, [1.0, 1.0, 1.0]);
    let config = omni_laya::config::Config::load(&checkpoint).unwrap();
    assert_ne!(
        legacy, config.agent.temperature,
        "legacy buffer is not the fitted calibration source"
    );
}
