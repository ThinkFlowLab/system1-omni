use anyhow::{Context, Result};
use omni_laya::{
    config::Config,
    preprocess::{Preprocessor, Request},
};
use std::{
    io::{self, BufRead},
    path::PathBuf,
};
fn main() -> Result<()> {
    let path = PathBuf::from(
        std::env::args()
            .nth(1)
            .context("usage: laya-pack CHECKPOINT < requests.jsonl")?,
    );
    Config::load(&path)?;
    let p = Preprocessor::load(&path)?;
    for line in io::stdin().lock().lines() {
        let request: Request = serde_json::from_str(&line?)?;
        println!("{}", serde_json::to_string(&p.prepare(&request)?)?);
    }
    Ok(())
}
