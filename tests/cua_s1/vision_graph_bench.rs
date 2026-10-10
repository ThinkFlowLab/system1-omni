//! Maintained native benchmark: upload + forward + download, and complete decisions.
use anyhow::{Context, Result, ensure};
use omni_cua_s1_native::{contract::Question, vision_engine::VisionEngine};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 1,
        "usage: vision_graph_bench OUTPUT_JSON (CUA_S1_* environment)"
    );
    let output = PathBuf::from(&args[0]);
    ensure!(!output.exists(), "refusing to overwrite evidence");
    let path = |name| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .with_context(|| format!("set {name}"))
    };
    let started = Instant::now();
    let mut engine = VisionEngine::load(
        &path("CUA_S1_BASE")?,
        &path("CUA_S1_VISION_ADAPTER")?,
        &path("CUA_S1_MODEL")?,
        &path("CUA_S1_CUDA_LIB")?,
    )?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    let mut cold = Vec::new();
    let mut samples = Vec::new();
    for (width, height) in [(256, 256), (512, 256), (512, 512)] {
        let mut prepared = Vec::new();
        for sample in 0..10usize {
            let rgb: Vec<u8> = (0..width * height * 3)
                .map(|i| ((i * 17 + sample * 31 + (i / (width * 3)) * 7) % 251) as u8)
                .collect();
            let questions = [
                Question {
                    name: "move".into(),
                    goal: format!("Select the next move for screenshot {sample}."),
                    keys: vec![
                        "left".into(),
                        "right".into(),
                        "continue".into(),
                        "stop".into(),
                    ],
                    labels: vec![
                        "Move left".into(),
                        "Move right".into(),
                        "Continue".into(),
                        "Stop".into(),
                    ],
                },
                Question {
                    name: "visible".into(),
                    goal: "Is the scene visible?".into(),
                    keys: vec!["yes".into(), "no".into()],
                    labels: vec!["Yes".into(), "No".into()],
                },
            ];
            prepared.push(engine.prepare(width, height, &rgb, &questions)?);
        }
        let start = Instant::now();
        engine.vision.forward(&prepared[0].image)?;
        cold.push(json!({"width":width,"height":height,"grid":prepared[0].image.image_grid_thw,"vision_first_ms":start.elapsed().as_secs_f64()*1000.}));
        for request in prepared.iter().take(3) {
            engine.predict_prepared(request)?;
        }
        for (sample, request) in prepared.iter().enumerate() {
            let start = Instant::now();
            let features = engine.vision.forward(&request.image)?;
            let vision_ms = start.elapsed().as_secs_f64() * 1000.;
            let bytes: Vec<u8> = features
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect();
            let start = Instant::now();
            let response = engine.predict_prepared(request)?;
            let decision_ms = start.elapsed().as_secs_f64() * 1000.;
            samples.push(json!({"width":width,"height":height,"sample":sample,"grid":request.image.image_grid_thw,"features_sha256":format!("{:x}",Sha256::digest(bytes)),"vision_ms":vision_ms,"native_decision_ms":decision_ms,"response":response}));
        }
    }
    std::fs::write(
        output,
        serde_json::to_vec_pretty(
            &json!({"schema":"cua-s1-vision-graph-native-v1","load_ms":load_ms,"cold":cold,"samples":samples}),
        )?,
    )?;
    Ok(())
}
