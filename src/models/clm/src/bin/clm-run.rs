//! `clm-run`: a decision for a JSON request, so the engine can be checked by hand and
//! against the reference.
//!
//!     clm-run CHECKPOINT_DIR --emb-url http://127.0.0.1:8090/v1/embeddings [--model NAME]
//!         [--temperature T] [--max-tokens N|none]
//!
//! Reads one request object per line on stdin and writes one response object per line on
//! stdout, which is the shape the reference's own `laya-run`-style harnesses use.
use anyhow::{Context, Result, bail};
use omni_clm::{Engine, Heads, HttpEncoder, Request, Weights, serve};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut checkpoint: Option<PathBuf> = None;
    let mut emb_url = "http://127.0.0.1:8090/v1/embeddings".to_string();
    let mut model = "qwen3-8b".to_string();
    let mut temperature = 1.0f32;
    let mut max_tokens: Option<usize> = Some(2048);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--emb-url" => {
                emb_url = args.get(i + 1).context("--emb-url needs a value")?.clone();
                i += 2;
            }
            "--model" => {
                model = args.get(i + 1).context("--model needs a value")?.clone();
                i += 2;
            }
            "--temperature" => {
                temperature = args
                    .get(i + 1)
                    .context("--temperature needs a value")?
                    .parse()
                    .context("--temperature is not a number")?;
                i += 2;
            }
            "--max-tokens" => {
                max_tokens = match args
                    .get(i + 1)
                    .context("--max-tokens needs a value")?
                    .as_str()
                {
                    "none" | "off" => None,
                    value => Some(value.parse().context("--max-tokens is not a number")?),
                };
                i += 2;
            }
            other if !other.starts_with("--") => {
                checkpoint = Some(PathBuf::from(other));
                i += 1;
            }
            other => bail!("unknown flag {other}"),
        }
    }
    let checkpoint = checkpoint.context("usage: clm-run CHECKPOINT_DIR [--emb-url URL]")?;

    let weights = Weights::open(&checkpoint.join("model.safetensors"))
        .with_context(|| format!("open {}", checkpoint.display()))?;
    let heads = Heads::load(&weights)?;
    eprintln!(
        "clm-run: {} tensors, hidden {} -> projection {}, logit_scale {:.4}",
        omni_clm::head_tensors(&heads.config.head)?.len(),
        heads.config.head.hidden_size,
        heads.config.head.projection_dim,
        heads.config.logit_scale,
    );
    let encoder = HttpEncoder::new(emb_url.clone(), model, Duration::from_secs(90))?
        .with_max_tokens(max_tokens);
    let engine = Engine::new(heads, encoder);

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        // `parse_line`, not `parse`: the text is what keeps an integer literal's
        // digits, and it is gone once the line has been parsed.
        let request = Request::parse_line(trimmed)?;
        let started = Instant::now();
        let decision = engine
            .decide(&request, temperature)
            .with_context(|| format!("decide on {} bytes of request", trimmed.len()))?;
        let elapsed = started.elapsed();
        let response = serde_json::json!({
            "model": request.model.clone().unwrap_or_else(|| "clm-latest".into()),
            "answers": serve::answers_json(&decision.answers),
            "usage": {
                "billing_units": decision.answers.len(),
                "input_tokens": decision.encoder_tokens,
                "output_tokens": 0,
            },
            "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
        });
        writeln!(stdout, "{response}")?;
        stdout.flush()?;
    }
    Ok(())
}
