#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use omni_laya::{model::Model, processing::Processor};
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
    let pre = Processor::load(&checkpoint)?;
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
        let prepared = pre.prepare(line.as_bytes());
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(e) => {
                println!("{}", serde_json::json!({"error":format!("{e:#}")}));
                eprintln!(
                    "engine_wall_ms={:.6}",
                    start.elapsed().as_secs_f64() * 1000.0
                );
                continue;
            }
        };
        // Only client input errors are recoverable. A native failure may poison
        // the CUDA context, so never submit another request after infer fails.
        let (logits, actions) = model
            .infer(&prepared.inputs)
            .context("native inference failed")?;
        if std::env::var_os("LAYA_RAW_LOGITS").is_some() {
            eprintln!("raw_logits={logits:?} raw_actions={actions:?}");
        }
        let value = prepared
            .context
            .finish(logits, actions)
            .context("native output decoding failed")?;
        println!("{}", serde_json::to_string(&value)?);
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
