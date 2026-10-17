use omni_cua_s1_native::vision_engine::device_features_enabled;

#[test]
fn device_features_accepts_only_unset_zero_and_one() {
    assert!(!device_features_enabled(None).unwrap());
    assert!(!device_features_enabled(Some("0")).unwrap());
    assert!(device_features_enabled(Some("1")).unwrap());
    for value in ["", "true", " 1", "1 ", "2"] {
        assert!(device_features_enabled(Some(value)).is_err());
    }
}

fn rejects_before_loading(value: &std::ffi::OsStr) {
    let missing =
        std::env::temp_dir().join(format!("cua-device-control-missing-{}", std::process::id()));
    assert!(!missing.exists());
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_omni-cua-s1-vision"))
        .env("CUA_S1_DEVICE_FEATURES", value)
        .env("CUA_S1_BASE", &missing)
        .env("CUA_S1_VISION_ADAPTER", &missing)
        .env("CUA_S1_MODEL", &missing)
        .env("CUA_S1_CUDA_LIB", &missing)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("CUA_S1_DEVICE_FEATURES must be unset, 0, or 1"),
        "{error}"
    );
    assert!(!error.contains("export the multimodal language checkpoint"));
}

#[test]
fn invalid_feature_controls_fail_before_checkpoint_or_cuda_loading() {
    for value in ["", "true", "2"] {
        rejects_before_loading(value.as_ref());
    }
}

#[cfg(unix)]
#[test]
fn non_unicode_feature_control_fails_before_checkpoint_or_cuda_loading() {
    use std::os::unix::ffi::OsStrExt;
    rejects_before_loading(std::ffi::OsStr::from_bytes(&[0xff]));
}
