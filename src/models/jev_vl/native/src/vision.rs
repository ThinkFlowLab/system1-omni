//! CPU validation of the pinned online vision export and its processor contract.

use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::vision::VisionConfig;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// The image source is fixed for the lifetime of a worker.
pub(crate) enum ImageSource {
    Prepared(Option<PathBuf>),
    Online(PathBuf),
}

impl ImageSource {
    pub(crate) fn from_paths(
        dir: &Path,
        vision: Option<PathBuf>,
        imgcache: Option<OsString>,
    ) -> Result<Self> {
        // Preserve the prepared worker's env::var fallback for non-UTF8 values.
        let imgcache = imgcache
            .and_then(|p| p.into_string().ok())
            .map(PathBuf::from);
        ensure!(
            vision.is_none() || imgcache.is_none(),
            "JEV_VL_VISION and JEV_VL_IMGCACHE are mutually exclusive"
        );
        if let Some(dir) = vision {
            ensure!(dir.is_dir(), "JEV_VL_VISION must be an existing directory");
            Ok(Self::Online(dir))
        } else {
            Ok(Self::Prepared(
                imgcache
                    .or_else(|| Some(dir.join("imgcache")))
                    .filter(|p| p.is_dir()),
            ))
        }
    }
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(crate::contract::hex_digest(&hash.finalize()))
}

fn hash_value(value: &Value) -> Result<&str> {
    let hash = value.as_str().context("expected SHA256 string")?;
    ensure!(
        hash.len() == 64
            && hash
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "expected lowercase SHA256"
    );
    Ok(hash)
}

fn validate_processor(processor: &Value) -> Result<()> {
    // Frozen upstream metadata. New fields can change preprocessing semantics;
    // accept them only after the shared processor implements those semantics.
    ensure!(
        processor
            == &json!({
                "size": {"longest_edge": 16777216, "shortest_edge": 65536},
                "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
                "processor_class": "Qwen3VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessorFast"
            }),
        "unsupported JEV image processor metadata"
    );
    Ok(())
}

fn validate_config(config: &Value, language: &Value) -> Result<()> {
    ensure!(
        config["model_type"] == "qwen3_5"
            && language["model_type"] == "qwen3_5"
            && config["vision_config"] == language["vision_config"]
            && config["text_config"] == language["text_config"],
        "vision and language configurations do not match"
    );
    let vision = VisionConfig::from_value(config["vision_config"].clone())?;
    let supported = serde_json::to_value(&vision)?;
    for key in config["vision_config"]
        .as_object()
        .context("vision_config")?
        .keys()
    {
        ensure!(
            supported.get(key).is_some() || key == "initializer_range",
            "unsupported vision configuration field: {key}"
        );
    }
    ensure!(
        vision.hidden_size == 1152 && vision.out_hidden_size == 5120,
        "online JEV vision requires the 27B layout"
    );
    ensure!(
        config["text_config"]["model_type"] == "qwen3_5_text"
            && config["text_config"]["hidden_size"] == 5120
            && config["text_config"]["intermediate_size"] == 17408
            && config["text_config"]["num_hidden_layers"] == 64
            && config["text_config"]["num_attention_heads"] == 24
            && config["text_config"]["num_key_value_heads"] == 4
            && config["image_token_id"].as_u64().is_some()
            && config["image_token_id"] == language["image_token_id"],
        "online JEV vision requires the matching 27B text configuration"
    );
    Ok(())
}

pub(crate) fn validate_export(dir: &Path, language_dir: &Path, language: &Value) -> Result<()> {
    let manifest: Value = serde_json::from_slice(&std::fs::read(dir.join("jev_vl_vision.json"))?)?;
    ensure!(
        manifest["format"] == "jev-vl-vision/1"
            && manifest["model_id"] == crate::contract::MODEL_ID,
        "expected a pinned JEV-27B-VL vision export"
    );
    for pin in [
        "model_index_sha256",
        "adapter_config_sha256",
        "adapter_model_sha256",
    ] {
        ensure!(
            hash_value(&manifest["pins"][pin])? == hash_value(&language["pins"][pin])?,
            "vision export source pin mismatch: {pin}"
        );
    }
    let files = manifest["files"]
        .as_object()
        .context("vision export files")?;
    let expected = [
        "config.json",
        "preprocessor_config.json",
        "vision.safetensors",
    ];
    ensure!(
        files.len() == expected.len() && expected.iter().all(|name| files.contains_key(*name)),
        "vision export must pin exactly config, processor and weights"
    );
    for name in expected {
        ensure!(
            hash_value(&files[name])? == sha256(&dir.join(name))?,
            "vision export file hash mismatch: {name}"
        );
    }
    let config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let language_config =
        serde_json::from_slice(&std::fs::read(language_dir.join("config.json"))?)?;
    validate_config(&config, &language_config)?;
    validate_processor(&serde_json::from_slice(&std::fs::read(
        dir.join("preprocessor_config.json"),
    )?)?)?;
    Ok(())
}

#[cfg(test)]
#[path = "../../../../../tests/jev_vl/vision.rs"]
mod tests;
