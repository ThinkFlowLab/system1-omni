//! Assemble verified artifacts, model-owned processing, and serial eager execution.
use crate::{
    Config, Kind, Limits, Processor,
    batching::BatchLimits,
    checkpoint::Checkpoint,
    executor::Executor,
    runtime::{Runtime, RuntimeConfig},
};
use anyhow::Result;
use omni_runtime::SerialScheduler;
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};

pub struct Engine {
    pub processor: Arc<Processor>,
    pub executor: Executor,
    pub scheduler: SerialScheduler,
    pub runtime: Runtime,
    metadata: Value,
}
impl Engine {
    pub async fn load(dir: &Path, library: &Path) -> Result<Self> {
        Self::load_with_batch(dir, library, BatchLimits::from_env()?).await
    }
    pub async fn load_with_batch(
        dir: &Path,
        library: &Path,
        batching: BatchLimits,
    ) -> Result<Self> {
        Self::load_with_runtime(dir, library, batching, RuntimeConfig::from_env()?).await
    }
    pub async fn load_with_runtime(
        dir: &Path,
        library: &Path,
        batching: BatchLimits,
        config: RuntimeConfig,
    ) -> Result<Self> {
        let runtime = Runtime::new(config)?;
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
        let executor = Executor::load(dir, library, checkpoint, labels, batching).await?;
        metadata["batch_limits"] =
            json!({"rows":batching.max_rows(),"tokens":batching.max_tokens()});
        let engine = Self {
            processor: Arc::new(processor),
            executor,
            scheduler: SerialScheduler::default(),
            runtime,
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
        metadata["status"] = json!(if self.executor.is_ready() {
            "ready"
        } else {
            "unavailable"
        });
        let config = self.runtime.config();
        if config.max_requests.is_some() || config.max_tokens.is_some() || config.timeout.is_some()
        {
            metadata["runtime_limits"] = json!({"requests":config.max_requests,"processed_tokens":config.max_tokens,"timeout_ms":config.timeout.map(|timeout|timeout.as_millis())});
        }
        if let Some(report) = self.runtime.report() {
            metadata["depth"] = json!(report.depth);
            metadata["capacity"] = json!(report.capacity);
            metadata["rejected"] = json!(report.rejected);
        }
        metadata
    }
}
