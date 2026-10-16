//! Assemble verified artifacts, model-owned processing, and serial eager execution.
use crate::{
    Config, Kind, Limits, Processor, batching::BatchLimits, checkpoint::Checkpoint,
    executor::Executor,
};
use anyhow::Result;
use omni_runtime::SerialScheduler;
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};

pub struct Engine {
    pub processor: Arc<Processor>,
    pub executor: Executor,
    pub scheduler: SerialScheduler,
    metadata: Value,
}
impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        let graph = crate::options::graph_env()?;
        Self::load_with_modes(
            dir,
            library,
            BatchLimits::from_env()?,
            graph,
            crate::options::prefix_env(graph)?,
        )
        .await
    }
    pub async fn load_with_batch(
        dir: &Path,
        library: &Path,
        batching: BatchLimits,
    ) -> Result<Self> {
        Self::load_with_options(dir, library, batching, false).await
    }
    pub async fn load_with_options(
        dir: &Path,
        library: &Path,
        batching: BatchLimits,
        graph: bool,
    ) -> Result<Self> {
        Self::load_with_modes(
            dir,
            library,
            batching,
            graph,
            crate::prefix::PrefixMode::Off,
        )
        .await
    }
    pub async fn load_with_modes(
        dir: &Path,
        library: &Path,
        batching: BatchLimits,
        graph: bool,
        prefix_mode: crate::prefix::PrefixMode,
    ) -> Result<Self> {
        anyhow::ensure!(
            !graph || prefix_mode == crate::prefix::PrefixMode::Off,
            "shared/fixed execution requires Graph off"
        );
        let directory = dir.to_owned();
        let (processor,checkpoint,mut metadata) = tokio::task::spawn_blocking(move || -> Result<_> {
            let processor = Processor::load(&directory,Limits::default())?;
            let labels: Vec<u32> = processor.labels().iter().map(|label| label.id).collect();
            let checkpoint = Checkpoint::load(&directory,&labels)?;
            let config = Config::load(&directory)?;
            let metadata = json!({"model":crate::MODEL_ID,"checkpoint_revision":crate::CHECKPOINT_REVISION,"reference_revision":crate::RUNTIME_REVISION,"execution":"eager","dtype":"bfloat16","temperatures":{"choice":config.temperature(Kind::Choice),"score":config.temperature(Kind::Score),"noul":config.temperature(Kind::Noul)}});
            Ok((processor,checkpoint,metadata))
        }).await??;
        let labels = processor.labels().iter().map(|label| label.id).collect();
        let executor = Executor::load(
            dir,
            library,
            checkpoint,
            labels,
            batching,
            graph,
            prefix_mode,
        )
        .await?;
        metadata["batch_limits"] =
            json!({"rows":batching.max_rows(),"tokens":batching.max_tokens()});
        metadata["prefix_mode"] = json!(prefix_mode.name());
        let engine = Self {
            processor: Arc::new(processor),
            executor,
            scheduler: SerialScheduler::default(),
            metadata,
        };
        // Loading returns only after a real complete model decision, before any socket binds.
        engine.predict(br#"{"state":"The worker is initialized.","questions":{"ready":{"type":"choice","instructions":"Choose the next action.","criteria":{"continue":"Continue","stop":"Stop"}}}}"#).await?;
        Ok(engine)
    }
    pub async fn predict(&self, raw: &[u8]) -> Result<Value> {
        let prepared = self.processor.prepare(raw)?;
        let logits = self
            .executor
            .execute(&self.scheduler, prepared.rows)
            .await?;
        prepared.context.finish(logits)
    }
    pub fn health(&self) -> Value {
        let mut metadata = self.metadata.clone();
        let stats = self.executor.graph_stats();
        metadata["execution"] = json!(if !self.executor.is_ready() {
            "unavailable"
        } else if stats.enabled {
            "graph"
        } else {
            "eager"
        });
        metadata["graph"] = json!(stats);
        metadata["prefix"] = json!(self.executor.prefix_stats());
        if self.executor.is_ready() && metadata["prefix_mode"] != "off" {
            metadata["execution"] = metadata["prefix_mode"].clone();
        }
        metadata["status"] = json!(if self.executor.is_ready() {
            "ready"
        } else {
            "unavailable"
        });
        metadata
    }
}
