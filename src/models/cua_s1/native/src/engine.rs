//! Assemble the independent processor and model executor for the native worker.

use std::path::Path;

use anyhow::{Context, Result};
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
        let export = crate::provenance::text_export(dir)?;
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
        let executor = Executor::load(dir, library, &ids, export).await?;
        Ok(Self {
            processor,
            scheduler: SerialScheduler::default(),
            executor,
        })
    }
}

#[cfg(test)]
#[path = "../../../../../tests/cua_s1/text_export.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../../../tests/cua_s1/legacy_merged_gpu.rs"]
mod legacy_merged_tests;
