//! Assemble the independent processor and model executor for the native worker.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use omni_runtime::SerialScheduler;

use crate::contract::LETTERS;
use crate::executor::Executor;
use crate::processing::Processor;

pub struct Engine {
    pub processor: Processor,
    pub scheduler: SerialScheduler,
    pub executor: Executor,
}

impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        // Written by export_text_merged.py; without it `dir` may hold the base model alone.
        ensure!(
            dir.join("cua_s1_export.json").exists(),
            "{} is not a merged text checkpoint; see recipe/cua_s1/native.md",
            dir.display()
        );
        let processor = Processor::load(dir)?;
        let ids = LETTERS
            .chars()
            .map(|c| {
                processor
                    .tokenizer
                    .token_to_id(&c.to_string())
                    .context("letter token")
            })
            .collect::<Result<Vec<_>>>()?;
        let executor = Executor::load(dir, library, &ids).await?;
        Ok(Self {
            processor,
            scheduler: SerialScheduler::default(),
            executor,
        })
    }
}
