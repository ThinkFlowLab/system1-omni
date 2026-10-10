//! Assemble the processor and model executor from one pinned JEV-27B-VL export.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use omni_runtime::SerialScheduler;
use serde_json::Value;

use crate::caches::{CacheCfg, Caches};
use crate::executor::{Executor, LabelHead};
use crate::processing::Processor;

pub struct Engine {
    pub manifest: Value,
    pub head: Arc<LabelHead>,
    pub processor: Processor,
    pub scheduler: SerialScheduler,
    pub executor: Executor,
    /// Shared caches: processors/executors consult them; the HTTP layer
    /// exposes its counters on /v1/cache/stats.
    pub caches: Arc<Caches>,
}

impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        let manifest: Value = serde_json::from_slice(
            &std::fs::read(dir.join("jev_vl_export.json"))
                .context("export the merged checkpoint; see recipe/jev_vl/export_merged.py")?,
        )?;
        ensure!(
            manifest["format"] == "jev-vl-text-merged/1"
                && manifest["model_id"] == crate::contract::MODEL_ID
                && manifest["protocol"] == crate::contract::PROTOCOL,
            "expected a pinned JEV-27B-VL text export"
        );
        let ranges: Vec<i64> = serde_json::from_value(manifest["slots"]["ranges"]["noul"].clone())?;
        let score: Vec<i64> = serde_json::from_value(manifest["slots"]["ranges"]["score"].clone())?;
        let choice: Vec<i64> =
            serde_json::from_value(manifest["slots"]["ranges"]["choice"].clone())?;
        ensure!(
            ranges == [0, 2] && score == [2, 8] && choice == [8, 24],
            "unexpected verbalizer slot ranges"
        );
        ensure!(
            manifest["verbalizer_ids"].as_array().map(Vec::len) == Some(24)
                && manifest["verbalizer_bias"].as_array().map(Vec::len) == Some(24),
            "24-slot verbalizer head"
        );
        let labels: Vec<String> = serde_json::from_value(manifest["labels"].clone())?;
        let label_ids: Vec<i64> = serde_json::from_value(manifest["label_ids"].clone())?;
        ensure!(
            labels.len() == label_ids.len() && !labels.is_empty(),
            "label table"
        );
        for t in ["noul", "score", "choice"] {
            let temp = manifest["temperatures"][t]
                .as_f64()
                .context("temperatures")?;
            ensure!(temp.is_finite() && temp > 0.0, "invalid temperature");
        }
        let max_length = manifest["max_length"].as_u64().context("max_length")? as usize;
        ensure!(
            (1..=32768).contains(&max_length),
            "max_length must be within 1..=32768"
        );
        let head = Arc::new(LabelHead::load(dir, &manifest)?);
        let caches = Caches::new(CacheCfg::from_env());
        let model_index_hash = manifest["pins"]["model_index_sha256"]
            .as_str()
            .context("model_index_sha256")?
            .to_owned();
        let processor = Processor::load(dir, labels, max_length, model_index_hash, caches.clone())?;
        let executor = Executor::load(dir, library, head.clone(), caches.clone()).await?;
        Ok(Self {
            manifest,
            head,
            processor,
            scheduler: SerialScheduler::default(),
            executor,
            caches,
        })
    }
}
