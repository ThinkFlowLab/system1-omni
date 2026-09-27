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
    for row in rows {
        let name = row["name"].as_str().unwrap();
        assert!(
            names.insert(name.to_owned()),
            "duplicate oracle tensor: {name}"
        );
        let shape: Vec<usize> = serde_json::from_value(row["shape"].clone()).unwrap();
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
