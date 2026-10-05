use std::process::Command;

fn worker() -> (&'static str, &'static str) {
    match (
        option_env!("CARGO_BIN_EXE_omni-cua-s1-native"),
        option_env!("CARGO_BIN_EXE_omni-open-jev-native"),
    ) {
        (Some(binary), None) => (binary, "CUA_S1"),
        (None, Some(binary)) => (binary, "OPEN_JEV"),
        other => panic!("expected one native worker binary, got {other:?}"),
    }
}

#[test]
fn invalid_capacity_fails_startup_before_model_or_cuda_loading() {
    let (binary, prefix) = worker();
    for limit in ["0", "-1", "invalid", "", "18446744073709551615"] {
        let output = Command::new(binary)
            .env("OMNI_NATIVE_MAX_PENDING", limit)
            .env(format!("{prefix}_MODEL"), "/nonexistent-model")
            .env(format!("{prefix}_CUDA_LIB"), "/nonexistent-cuda-library")
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("OMNI_NATIVE_MAX_PENDING"), "{stderr}");
        assert!(!stderr.contains("libcuda"), "{stderr}");
    }
}

#[test]
fn valid_capacity_and_default_proceed_to_checkpoint_validation() {
    let (binary, prefix) = worker();
    for limit in [None, Some("1"), Some("64")] {
        let mut command = Command::new(binary);
        command
            .env_remove("OMNI_NATIVE_MAX_PENDING")
            .env(format!("{prefix}_MODEL"), "/nonexistent-model")
            .env(format!("{prefix}_CUDA_LIB"), "/nonexistent-cuda-library");
        if let Some(limit) = limit {
            command.env("OMNI_NATIVE_MAX_PENDING", limit);
        }
        let output = command.output().unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("checkpoint"), "{stderr}");
        assert!(!stderr.contains("OMNI_NATIVE_MAX_PENDING"), "{stderr}");
    }
}
