//! Assemble the independent processor and model executor from one pinned export.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use omni_runtime::SerialScheduler;
use serde_json::Value;

use crate::contract::Checkpoint;
use crate::executor::{DecisionHead, Executor};
use crate::processing::Processor;

pub struct Engine {
    pub checkpoint: &'static Checkpoint,
    pub processor: Processor,
    pub scheduler: SerialScheduler,
    pub executor: Executor,
}

impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(dir.join("open_jev_export.json"))
                .context("export the merged checkpoint; see recipe/open_jev/native.md")?,
        )?;
        let checkpoint = Checkpoint::from_export(&manifest)?;
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
        let head = DecisionHead::load(dir, &manifest, checkpoint)?;
        let processor = Processor::load(dir, checkpoint, prefix, suffix, temperature, max_length)?;
        let executor = Executor::load(dir, library, head, checkpoint).await?;
        Ok(Self {
            checkpoint,
            processor,
            scheduler: SerialScheduler::default(),
            executor,
        })
    }
}
