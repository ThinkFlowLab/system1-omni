fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use omni_laya::preprocess::{Preprocessor, Request};
    use std::io::{self, BufRead};
    let args: Vec<_> = std::env::args().collect();
    let checkpoint = std::path::Path::new(args.get(1).context("usage: laya-pack CHECKPOINT")?);
    let preprocessor = Preprocessor::load(&checkpoint.join("tokenizer/tokenizer.json"))?;
    for line in io::stdin().lock().lines() {
        let request = Request::from_json(&line?)?;
        println!(
            "{}",
            serde_json::to_string(&preprocessor.prepare(&request)?)?
        );
    }
    Ok(())
}
