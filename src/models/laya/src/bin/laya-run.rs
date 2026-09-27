#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use omni_laya::{
        decision,
        model::Model,
        preprocess::{Preprocessor, Request},
    };
    use std::{
        io::{self, BufRead},
        path::PathBuf,
        time::Instant,
    };
    let args: Vec<_> = std::env::args().collect();
    let checkpoint = PathBuf::from(
        args.get(1)
            .context("usage: laya-run CHECKPOINT CUDA_BUNDLE [--eager] [--original-rope]")?,
    );
    let bundle = PathBuf::from(args.get(2).context("missing CUDA bundle")?);
    let pre = Preprocessor::load(&checkpoint)?;
    let mut model = Model::load(
        &checkpoint,
        &bundle,
        !args.iter().any(|s| s == "--eager"),
        args.iter().any(|s| s == "--original-rope"),
    )?;
    eprintln!("READY native Laya (Rust + CUDA)");
    for line in io::stdin().lock().lines() {
        let line = line?;
        let start = Instant::now();
        let result = (|| -> anyhow::Result<_> {
            let request: Request = serde_json::from_str(&line)?;
            let batch = pre.prepare(&request)?;
            let (logits, actions) = model.infer(&batch)?;
            if std::env::var_os("LAYA_RAW_LOGITS").is_some() {
                eprintln!("raw_logits={logits:?} raw_actions={actions:?}");
            }
            decision::decode(&batch, &model.config.agent, &logits, &actions)
        })();
        match result {
            Ok(value) => println!("{}", serde_json::to_string(&value)?),
            Err(e) => println!("{}", serde_json::json!({"error":format!("{e:#}")})),
        }
        eprintln!(
            "engine_wall_ms={:.6}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("laya-run requires --features cuda; use laya-pack for CPU input validation");
    std::process::exit(2);
}
