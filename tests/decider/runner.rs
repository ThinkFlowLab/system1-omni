//! Diagnostic JSON-lines runner retaining exact prepared rows and raw logits.
use anyhow::{Context, Result};
use omni_decider_native::engine::Engine;
use serde_json::json;
use std::{
    io::{self, BufRead},
    path::Path,
};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model = args
        .get(1)
        .context("usage: decider-run MODEL_DIR CUDA_LIBRARY")?;
    let library = args
        .get(2)
        .context("usage: decider-run MODEL_DIR CUDA_LIBRARY")?;
    let engine = Engine::load(Path::new(model), Path::new(library)).await?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        let prepared = engine.processor.prepare(line.as_bytes())?;
        let rows:Vec<_> = prepared.rows.iter().map(|row| json!({"ids":row.ids,"candidate_ids":row.candidate_ids,"readout_position":row.readout_position,"question_id":row.question_id,"level_index":row.level_index,"temperature":row.temperature})).collect();
        let logits = engine
            .executor
            .execute(&engine.scheduler, prepared.rows)
            .await?;
        let response = prepared.context.finish(logits.clone())?;
        println!(
            "{}",
            json!({"rows":rows,"logits":logits,"response":response,"graph":engine.health()["graph"],"prefix":engine.health()["prefix"]})
        );
    }
    Ok(())
}
