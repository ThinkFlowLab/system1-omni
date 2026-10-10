use crate::{contract::Kind, json};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::path::Path;

/// Validated calibration for the released plain independent/isolated readout.
#[derive(Clone, Debug)]
pub struct Config {
    temperatures: [f32; 3],
}
impl Config {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let model = json::parse(&std::fs::read(dir.join("config.json"))?)?;
        json::object(&model)?;
        ensure!(
            model["model_type"] == "qwen3_5_text"
                && model["hidden_size"] == 2048
                && model["num_hidden_layers"] == 24
                && model["tie_word_embeddings"] == true
                && model["vocab_size"] == 248320
                && model["dtype"] == "bfloat16",
            "expected Decider-2B v11 text configuration"
        );
        Self::from_value(&json::parse(&std::fs::read(
            dir.join("decider_config.json"),
        )?)?)
    }
    /// Positive calibration overrides are supported, with missing types using fallback.
    /// Nonreleased prompt/readout modes and option-dependent temperatures are rejected.
    pub fn from_value(value: &Value) -> Result<Self> {
        let c = json::object(value)?;
        ensure!(
            c.get("version").and_then(Value::as_str) == Some("2b-v11"),
            "expected 2b-v11 configuration"
        );
        ensure!(
            c.get("layout").and_then(Value::as_str) == Some("plain"),
            "only plain layout supported"
        );
        for (key, want) in [
            ("chat_template", false),
            ("schema_first", false),
            ("schema_first_trained", false),
            ("neutralize_none", false),
            ("isolated_levels", true),
        ] {
            json::default_mode(c, key, want)?;
        }
        ensure!(
            c.get("max_options").and_then(Value::as_u64) == Some(255)
                && c.get("max_state_tokens").and_then(Value::as_u64) == Some(32768),
            "unsupported option/state limits"
        );
        for key in [
            "temperature_by_options",
            "temperature_schema_first",
            "temperature_schema_first_by_type",
        ] {
            ensure!(!c.contains_key(key), "unsupported calibration {key}");
        }
        for key in ["neutralize_none", "isolated_levels"] {
            json::required(c, key)?;
        }
        let raw = json::required(c, "temperature")?;
        let scalar = if let Some(text) = raw.as_str() {
            serde_json::json!(json::python_float(text).context("temperature must be numeric")?)
        } else {
            raw.clone()
        };
        let fallback = positive(&scalar)?;
        let mut temperatures = [fallback; 3];
        if let Some(m) = c.get("temperature_by_type").filter(|v| !v.is_null()) {
            for (key, value) in json::object(m)? {
                let kind = Kind::parse(key)?;
                ensure!(key != "bool", "unknown calibration type bool");
                temperatures[kind.index()] = positive(value)?;
            }
        }
        Ok(Self { temperatures })
    }
    pub fn temperature(&self, kind: Kind) -> f32 {
        self.temperatures[kind.index()]
    }
}
fn positive(value: &Value) -> Result<f32> {
    // The serving by-type path materializes its temperature list as FP32.
    let x = value.as_f64().context("temperature must be numeric")?;
    let rounded = x as f32;
    ensure!(
        x.is_finite() && x > 0.0 && rounded.is_finite() && rounded > 0.0,
        "temperature must be finite and positive in FP32"
    );
    Ok(rounded)
}
