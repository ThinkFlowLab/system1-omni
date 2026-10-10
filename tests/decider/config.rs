use omni_decider_native::{Config, Kind};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

fn plain() -> Value {
    json!({"version":"2b-v11","layout":"plain","max_options":255,"max_state_tokens":32768,"neutralize_none":false,"isolated_levels":true,"temperature":1.0})
}
struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "decider-config-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("config.json"),json!({"model_type":"qwen3_5_text","hidden_size":2048,"num_hidden_layers":24,"tie_word_embeddings":true,"vocab_size":248320,"dtype":"bfloat16"}).to_string()).unwrap();
        Self(path)
    }
    fn calibration(&self, config: &Value) {
        std::fs::write(self.0.join("decider_config.json"), config.to_string()).unwrap();
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn public_validator_rejects_chat_metadata_on_plain_layout() {
    let mut config = plain();
    config["chat_template"] = json!(true);
    let error =
        Config::from_value(&config).expect_err("contradictory chat template must be rejected");
    assert!(error.to_string().contains("chat_template"));
}
#[test]
fn file_loader_rejects_chat_metadata_on_plain_layout() {
    let dir = Directory::new();
    let mut config = plain();
    config["chat_template"] = json!(true);
    dir.calibration(&config);
    let error = Config::load(&dir.0).expect_err("file-loading path must reject contradiction");
    assert!(error.to_string().contains("chat_template"));
}
#[test]
fn missing_and_disabled_chat_metadata_preserve_plain_calibration() {
    let dir = Directory::new();
    for config in [plain(), {
        let mut c = plain();
        c["chat_template"] = json!(false);
        c
    }] {
        assert_eq!(
            Config::from_value(&config)
                .unwrap()
                .temperature(Kind::Choice),
            1.0
        );
        dir.calibration(&config);
        assert_eq!(Config::load(&dir.0).unwrap().temperature(Kind::Score), 1.0);
    }
}
#[test]
fn unsupported_chat_metadata_types_are_rejected() {
    for value in [json!("true"), json!(1), json!(null), json!([]), json!({})] {
        let mut config = plain();
        config["chat_template"] = value;
        assert!(Config::from_value(&config).is_err());
    }
}
