//! Assemble verified artifacts, model-owned processing, and serial eager execution.
use crate::{Config, Kind, Limits, Processor, checkpoint::Checkpoint, executor::Executor};
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
        let directory = dir.to_owned();
        let (processor,checkpoint,metadata) = tokio::task::spawn_blocking(move || -> Result<_> {
            let processor = Processor::load(&directory,Limits::default())?;
            let labels: Vec<u32> = processor.labels().iter().map(|label| label.id).collect();
            let checkpoint = Checkpoint::load(&directory,&labels)?;
            let config = Config::load(&directory)?;
            let metadata = json!({"model":crate::MODEL_ID,"checkpoint_revision":crate::CHECKPOINT_REVISION,"reference_revision":crate::RUNTIME_REVISION,"execution":"eager","dtype":"bfloat16","temperatures":{"choice":config.temperature(Kind::Choice),"score":config.temperature(Kind::Score),"noul":config.temperature(Kind::Noul)}});
            Ok((processor,checkpoint,metadata))
        }).await??;
        let labels = processor.labels().iter().map(|label| label.id).collect();
        let executor = Executor::load(dir, library, checkpoint, labels).await?;
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
        metadata["status"] = json!(if self.executor.is_ready() {
            "ready"
        } else {
            "unavailable"
        });
        metadata
    }
}
