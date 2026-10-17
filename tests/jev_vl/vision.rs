use super::*;

fn processor() -> Value {
    json!({
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "processor_class": "Qwen3VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast"
    })
}

#[test]
fn source_selection_checks_presence_and_never_falls_back_from_online() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing");
    assert!(ImageSource::from_paths(dir.path(), Some(missing.clone()), None).is_err());
    assert!(
        ImageSource::from_paths(dir.path(), Some(dir.path().into()), Some(missing.into())).is_err()
    );
    assert!(matches!(
        ImageSource::from_paths(dir.path(), Some(dir.path().into()), None).unwrap(),
        ImageSource::Online(_)
    ));
    assert!(matches!(
        ImageSource::from_paths(dir.path(), None, None).unwrap(),
        ImageSource::Prepared(None)
    ));
}

#[cfg(unix)]
#[test]
fn non_utf8_imgcache_preserves_default_selection_and_online_mode() {
    use std::os::unix::ffi::OsStringExt;

    let dir = tempfile::tempdir().unwrap();
    let default = dir.path().join("imgcache");
    std::fs::create_dir(&default).unwrap();
    let non_utf8 = std::ffi::OsString::from_vec(vec![b'i', 0xff]);
    let source = ImageSource::from_paths(dir.path(), None, Some(non_utf8.clone())).unwrap();
    let ImageSource::Prepared(Some(selected)) = source else {
        panic!("non-UTF8 imgcache must retain the default directory")
    };
    assert_eq!(selected, default);

    assert!(ImageSource::from_paths(dir.path(), Some(dir.path().join(&non_utf8)), None).is_err());
    let vision = dir.path().join("vision");
    std::fs::create_dir(&vision).unwrap();
    let source = ImageSource::from_paths(dir.path(), Some(vision.clone()), Some(non_utf8)).unwrap();
    let ImageSource::Online(selected) = source else {
        panic!("non-UTF8 imgcache must not conflict with online vision")
    };
    assert_eq!(selected, vision);
}

#[test]
fn processor_requires_the_frozen_resize_and_normalization_contract() {
    let valid = processor();
    validate_processor(&valid).unwrap();
    for (key, value) in [
        ("merge_size", json!(1)),
        ("temporal_patch_size", json!(1)),
        (
            "size",
            json!({"longest_edge": 1048576, "shortest_edge": 65536}),
        ),
        ("image_mean", json!([0.0, 0.0, 0.0])),
        ("do_resize", json!(false)),
        ("do_rescale", json!(false)),
        ("resample", json!(2)),
    ] {
        let mut changed = valid.clone();
        changed[key] = value;
        assert!(validate_processor(&changed).is_err(), "accepted {key}");
    }
}

#[test]
fn config_rejects_other_models_mismatches_and_unknown_vision_semantics() {
    let valid: Value = serde_json::from_str(include_str!("../qwen3_5/data/config.json")).unwrap();
    validate_config(&valid, &valid).unwrap();
    let mut wrong = valid.clone();
    wrong["model_type"] = json!("qwen3_5_moe");
    assert!(validate_config(&wrong, &wrong).is_err());
    let mut wrong = valid.clone();
    wrong["vision_config"]["out_hidden_size"] = json!(2560);
    assert!(validate_config(&wrong, &valid).is_err());
    let mut wrong = valid.clone();
    wrong["text_config"]["model_type"] = json!("another_text_model");
    assert!(validate_config(&wrong, &wrong).is_err());
    let mut wrong = valid.clone();
    wrong["vision_config"]["window_size"] = json!(112);
    assert!(validate_config(&wrong, &wrong).is_err());
}

fn export() -> (tempfile::TempDir, Value, Value) {
    let dir = tempfile::tempdir().unwrap();
    let pins = json!({
        "model_index_sha256": "a".repeat(64),
        "adapter_config_sha256": "b".repeat(64),
        "adapter_model_sha256": "c".repeat(64)
    });
    let language = json!({"pins": pins});
    let mut manifest = json!({
        "format": "jev-vl-vision/1", "model_id": crate::contract::MODEL_ID,
        "pins": pins, "files": {}
    });
    std::fs::write(
        dir.path().join("config.json"),
        include_str!("../qwen3_5/data/config.json"),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        serde_json::to_vec(&processor()).unwrap(),
    )
    .unwrap();
    // An empty tensor file proves that metadata reaches CPU inventory validation;
    // these tests never need a large checkpoint or CUDA.
    std::fs::write(
        dir.path().join("vision.safetensors"),
        [2u64.to_le_bytes().as_slice(), b"{}"].concat(),
    )
    .unwrap();
    for name in [
        "config.json",
        "preprocessor_config.json",
        "vision.safetensors",
    ] {
        manifest["files"][name] = json!(sha256(&dir.path().join(name)).unwrap());
    }
    (dir, language, manifest)
}

fn validate_fixture(dir: &Path, language: &Value, manifest: &Value) -> Result<()> {
    std::fs::write(
        dir.join("jev_vl_vision.json"),
        serde_json::to_vec(manifest).unwrap(),
    )
    .unwrap();
    validate_export(dir, dir, language)
}

#[test]
fn export_checks_all_source_pins_and_exact_files_before_weights() {
    let (dir, language, manifest) = export();
    validate_fixture(dir.path(), &language, &manifest).unwrap();
    let error = omni_qwen3_5_native::vision::VisionModel::load(
        dir.path(),
        &dir.path().join("missing-cuda-library"),
    )
    .err()
    .expect("empty inventory must fail before loading CUDA");
    assert!(format!("{error:#}").contains("missing visual tensor"));
    for pin in [
        "model_index_sha256",
        "adapter_config_sha256",
        "adapter_model_sha256",
    ] {
        let mut changed = manifest.clone();
        changed["pins"][pin] = json!("d".repeat(64));
        assert!(
            validate_fixture(dir.path(), &language, &changed)
                .unwrap_err()
                .to_string()
                .contains("source pin mismatch")
        );
    }
    let mut changed = manifest.clone();
    changed["files"]["other.json"] = json!("d".repeat(64));
    assert!(
        validate_fixture(dir.path(), &language, &changed)
            .unwrap_err()
            .to_string()
            .contains("pin exactly")
    );
    for name in [
        "config.json",
        "preprocessor_config.json",
        "vision.safetensors",
    ] {
        let mut changed = manifest.clone();
        changed["files"][name] = json!("d".repeat(64));
        assert!(
            validate_fixture(dir.path(), &language, &changed)
                .unwrap_err()
                .to_string()
                .contains(&format!("file hash mismatch: {name}"))
        );
    }
}

#[test]
fn streaming_hash_matches_known_sha256() {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), b"abc").unwrap();
    assert_eq!(
        sha256(file.path()).unwrap(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
