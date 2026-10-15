use super::*;
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

struct Directory(PathBuf);
impl Directory {
    fn new(marker: serde_json::Value) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("cua-text-export-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("cua_s1_export.json"), marker.to_string()).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[tokio::test]
async fn unknown_export_is_rejected_before_tokenizer_or_cuda_loading() {
    let dir = Directory::new(json!({"format":"unknown"}));
    let error = Engine::load(&dir.0, Path::new("unused-cuda-library"))
        .await
        .err()
        .expect("must reject unknown export");
    assert!(
        error.to_string().contains("unsupported text export"),
        "{error}"
    );
}

#[tokio::test]
async fn reference_export_rejects_incorrect_adapter_scale_before_loading() {
    let dir = Directory::new(json!({"format":"cua-s1-text-unmerged/1", "lora_scale":2.1}));
    let error = Engine::load(&dir.0, Path::new("unused-cuda-library"))
        .await
        .err()
        .expect("must reject incorrect scale");
    assert!(error.to_string().contains("LoRA scale"), "{error}");
}

#[tokio::test]
async fn reference_export_rejects_changed_adapter_bytes_before_cuda_loading() {
    let dir = Directory::new(json!({"format":"cua-s1-text-unmerged/1", "lora_scale":2.0,
        "adapter_sha256":"0".repeat(64)}));
    fs::write(dir.0.join("adapter.safetensors"), b"changed checkpoint").unwrap();
    let error = Engine::load(&dir.0, Path::new("unused-cuda-library"))
        .await
        .err()
        .expect("must reject changed adapter");
    assert!(error.to_string().contains("adapter checksum"), "{error}");
}
