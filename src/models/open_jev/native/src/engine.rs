//! Assemble the independent processor and model executor from one pinned export.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use omni_runtime::SerialScheduler;
use serde_json::Value;

use crate::contract::MODEL_ID;
use crate::executor::{DecisionHead, Executor};
use crate::processing::Processor;

pub use crate::contract::{BASE_REVISION, CHECKPOINT_REVISION};

pub struct Engine {
    pub processor: Processor,
    pub scheduler: SerialScheduler,
    pub executor: Executor,
}

impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        let scheduler = SerialScheduler::from_env()?;
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(dir.join("open_jev_export.json"))
                .context("export the merged checkpoint; see recipe/open_jev/native.md")?,
        )?;
        ensure!(
            manifest["format"] == "open-jev-text-merged/1"
                && manifest["model_id"] == MODEL_ID
                && manifest["base_revision"] == BASE_REVISION
                && manifest["checkpoint_revision"] == CHECKPOINT_REVISION,
            "expected a pinned Open-Jev-27B-v1.1 export"
        );
        let temperature = manifest["temperature"].as_f64().context("temperature")?;
        ensure!(
            temperature.is_finite() && temperature > 0.0,
            "invalid temperature"
        );
        let max_length = manifest["max_length"].as_u64().context("max_length")? as usize;
        ensure!(
            (1..=16384).contains(&max_length),
            "max_length must be within 1..=16384"
        );
        let prefix = manifest["chat_prefix"]
            .as_str()
            .context("chat_prefix")?
            .to_owned();
        let suffix = manifest["chat_suffix"]
            .as_str()
            .context("chat_suffix")?
            .to_owned();
        let head = DecisionHead::load(dir, &manifest)?;
        let processor = Processor::load(dir, prefix, suffix, temperature, max_length)?;
        let executor = Executor::load(dir, library, head).await?;
        Ok(Self {
            processor,
            scheduler,
            executor,
        })
    }
}
