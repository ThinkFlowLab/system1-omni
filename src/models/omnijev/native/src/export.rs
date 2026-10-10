//! The export `recipe/omnijev/export.py` writes: its manifest, the hashes of its files
//! and the calibration saved with the heads.

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::contract::{Calibration, MODEL_ID};
use crate::processing::{MAX_PIXELS, OPTION_CLOSE, OPTION_OPEN};

pub const FORMAT: &str = "omnijev-merged/1";
pub const MANIFEST: &str = "omnijev_export.json";
pub const CHECKPOINT_REVISION: &str = "ffe5f436eaf22e20e2f041f8e74e121fd057a6cb";
pub const BASE_REVISION: &str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";
/// The tokenizer's length, to which the export resizes the tied embedding.
pub const VOCABULARY: usize = 248_079;
/// Files the worker reads besides the language weights; the manifest must list and
/// hash each.
const REQUIRED: [&str; 5] = [
    "config.json",
    "heads.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "vision.safetensors",
];

/// A checked export directory.
pub struct Export {
    pub calibration: Calibration,
}

/// Check `dir`'s manifest against the pinned release and every file it lists against
/// its SHA-256, then read the calibration.
pub fn load(dir: &Path) -> Result<Export> {
    let path = dir.join(MANIFEST);
    let manifest: Value =
        serde_json::from_slice(&fs::read(&path).with_context(|| {
            format!("reading {}; run recipe/omnijev/export.py", path.display())
        })?)
        .context("export manifest")?;
    let expected = [
        ("format", json!(FORMAT)),
        ("model_id", json!(MODEL_ID)),
        ("checkpoint_revision", json!(CHECKPOINT_REVISION)),
        ("base_revision", json!(BASE_REVISION)),
        ("vocabulary", json!(VOCABULARY)),
        (
            "option_tokens",
            json!({"<|opt|>": OPTION_OPEN, "<|/opt|>": OPTION_CLOSE}),
        ),
        ("max_pixels", json!(MAX_PIXELS)),
        (
            "heads",
            json!({"norm": "softmax", "lm_features": true, "ordinal": true}),
        ),
    ];
    for (key, value) in expected {
        ensure!(
            manifest[key] == value,
            "export manifest {key} is {}, expected {value}",
            manifest[key]
        );
    }
    let outputs = manifest["outputs"]
        .as_object()
        .context("export manifest lists no outputs")?;
    for name in REQUIRED {
        ensure!(
            outputs.contains_key(name),
            "export manifest does not list {name}"
        );
    }
    for (name, digest) in outputs {
        ensure!(
            !name.is_empty() && !name.contains(['/', '\\']) && name != ".." && name != MANIFEST,
            "export manifest lists an invalid file name {name:?}"
        );
        let digest = digest
            .as_str()
            .context("export manifest hashes must be strings")?;
        ensure!(
            sha256(&dir.join(name))? == digest,
            "{name} does not match the export manifest"
        );
    }
    // the language weights the shared loader reads through the index, all hashed above
    let index: Value = serde_json::from_slice(&fs::read(dir.join("model.safetensors.index.json"))?)
        .context("model.safetensors.index.json")?;
    let shards = index["weight_map"]
        .as_object()
        .context("model.safetensors.index.json has no weight_map")?;
    for shard in shards.values() {
        let shard = shard
            .as_str()
            .context("weight_map values must be file names")?;
        ensure!(
            outputs.contains_key(shard),
            "export manifest does not list {shard}"
        );
    }
    ensure!(
        !dir.join("model.safetensors").exists() || outputs.contains_key("model.safetensors"),
        "export has an unlisted model.safetensors"
    );
    let config: Value =
        serde_json::from_slice(&fs::read(dir.join("config.json"))?).context("config.json")?;
    ensure!(
        config["text_config"]["vocab_size"] == json!(VOCABULARY),
        "config.json vocabulary is {}, expected {VOCABULARY}",
        config["text_config"]["vocab_size"]
    );
    Ok(Export {
        calibration: calibration(&manifest["calibration"])?,
    })
}

/// The manifest's calibration: positive, finite temperatures and a finite Noul bias.
pub fn calibration(value: &Value) -> Result<Calibration> {
    let number = |v: &Value| v.as_f64().filter(|x| x.is_finite());
    let t = &value["temperatures"];
    let temperatures = ["noul", "choice", "score"].map(|k| number(&t[k]).filter(|&x| x > 0.0));
    ensure!(
        t.as_object().is_some_and(|o| o.len() == 3) && temperatures.iter().all(Option::is_some),
        "export calibration needs positive noul, choice and score temperatures"
    );
    let noul_bias =
        number(&value["noul_bias"]).context("export calibration needs a finite noul_bias")?;
    Ok(Calibration {
        temperatures: temperatures.map(Option::unwrap),
        noul_bias,
    })
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1 << 22];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
