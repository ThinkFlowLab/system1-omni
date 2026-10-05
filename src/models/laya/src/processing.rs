//! Request preparation and response reconstruction; no device state or learned heads.
use std::{path::Path, sync::Arc};

use anyhow::Result;
use serde_json::{Value, json};

use crate::{
    config::{AgentConfig, Config},
    decision,
    packing::{self, Batch},
    preprocess::{Prepared, Preprocessor, Request},
};

pub struct Processor {
    preprocessor: Preprocessor,
    config: Arc<AgentConfig>,
}

pub struct PreparedRequest {
    pub inputs: Batch,
    pub context: ResponseContext,
}

pub struct ResponseContext {
    prepared: Prepared,
    config: Arc<AgentConfig>,
}

impl Processor {
    pub fn load(checkpoint: &Path) -> Result<Self> {
        Ok(Self {
            preprocessor: Preprocessor::load(&checkpoint.join("tokenizer/tokenizer.json"))?,
            config: Arc::new(Config::load(checkpoint)?.agent),
        })
    }

    pub fn prepare(&self, raw: &[u8]) -> Result<PreparedRequest> {
        let request = Request::from_json(std::str::from_utf8(raw)?)?;
        let prepared = self.preprocessor.prepare(&request)?;
        Ok(PreparedRequest {
            inputs: packing::pack(&prepared)?,
            context: ResponseContext {
                prepared,
                config: self.config.clone(),
            },
        })
    }
}

impl ResponseContext {
    pub fn finish(self, logits: Vec<Vec<f32>>, actions: Vec<[f32; 2]>) -> Result<Value> {
        let answers = decision::decode(&self.prepared, &self.config, &logits, &actions)?;
        Ok(json!({
            "model": "laya-rl-agent", "answers": answers,
            "usage": {"input_tokens": self.prepared.usage, "output_tokens": 0},
            "routing": {"model": "english", "repo": "convaiinnovations/laya",
                "reason": "explicit model='english'", "detection": null, "workflow": null}
        }))
    }
}
