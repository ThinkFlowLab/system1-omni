//! The embeddings client: the engine's half of the split.
//!
//! CLM does not compute embeddings. A frozen Qwen3-8B encoder runs as its own process
//! behind an OpenAI-compatible `/v1/embeddings` endpoint, and this module is the client
//! to it. Everything the engine owns happens after the vectors come back.
//!
//! The wire shape follows `src/clm/embedder.py` in the CLM reference: a POST of
//! `{model, input, encoding_format, truncate_prompt_tokens}`, a base64 `f32` payload per
//! input in `index` order, and an `l2` normalisation applied on receipt — the encoder is
//! asked for raw vectors and the client normalises, so a server that already normalises
//! is harmless.
//!
//! `truncate_prompt_tokens` is not optional in practice: the reference sends its
//! `max_tokens` (2048 by default, matching the documented `--max-model-len 2048`), and a
//! text longer than that is truncated there but rejected by a server asked to embed it
//! whole.
use anyhow::{Context, Result, bail, ensure};
use base64::Engine as _;
use std::time::Duration;

use crate::scoring::normalize;

/// Sends texts to the encoder and returns one row per text.
///
/// A trait rather than a struct so the engine can be driven without an encoder; the
/// tests use [`HashingEncoder`], and a deployment uses [`HttpEncoder`].
pub trait Encoder: Send + Sync {
    /// One `hidden_size`-wide, L2-normalised vector per input, in the order given, and
    /// the tokens the encoder reported spending on them.
    fn embed(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u64)>;
}

/// A deterministic encoder with no server behind it.
///
/// The vector is derived from a SHA-256 of the text filled to `dim` and then normalised,
/// which is exactly what `recipe/clm/native/head_oracle.py` does — so a decision taken
/// against this encoder can be compared with the Python oracle, and the whole engine can
/// be exercised on a machine with no GPU and no weights. The values carry no meaning as
/// model output; the point is that both implementations see the same numbers.
pub struct HashingEncoder {
    dim: usize,
}

impl HashingEncoder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// The vector for one text, as both this encoder and the oracle compute it.
    pub fn vector(text: &str, dim: usize) -> Vec<f32> {
        use sha2::{Digest, Sha256};

        let mut out: Vec<f32> = Vec::with_capacity(dim);
        let mut counter = 0u32;
        while out.len() < dim {
            let digest = Sha256::digest(format!("{counter}:{text}").as_bytes());
            for chunk in digest.as_chunks::<4>().0 {
                if out.len() == dim {
                    break;
                }
                out.push(u32::from_be_bytes(*chunk) as f64 as f32 / 2f64.powi(31) as f32 - 1.0);
            }
            counter += 1;
        }
        normalize(&mut out);
        out
    }
}

impl Encoder for HashingEncoder {
    /// No server, so nothing was spent.
    fn embed(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u64)> {
        Ok((texts.iter().map(|t| Self::vector(t, self.dim)).collect(), 0))
    }
}

/// An OpenAI-compatible `/v1/embeddings` endpoint, which is what `vllm serve --runner
/// pooling` exposes.
pub struct HttpEncoder {
    url: String,
    model: String,
    client: reqwest::blocking::Client,
    /// Batching, as `embedder.py` does: a request carries at most this many inputs.
    batch: usize,
    /// `embedder.py`'s `max_tokens`, sent as `truncate_prompt_tokens`. `None` sends no
    /// limit at all, which is what a reference built with `max_tokens=None` does.
    max_tokens: Option<usize>,
}

impl HttpEncoder {
    pub fn new(
        url: impl Into<String>,
        model: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .context("build the embeddings HTTP client")?;
        Ok(Self {
            url: url.into(),
            model: model.into(),
            client,
            batch: 512,
            // The reference's default, and the deployment's `--max-model-len`.
            max_tokens: Some(2048),
        })
    }

    /// Override the truncation limit sent as `truncate_prompt_tokens`.
    pub fn with_max_tokens(mut self, max_tokens: Option<usize>) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// The request body, which is `embedder.py`'s own.
    fn body(&self, texts: &[String]) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "input": texts,
            "encoding_format": "base64",
        });
        if let Some(max_tokens) = self.max_tokens {
            body["truncate_prompt_tokens"] = max_tokens.into();
        }
        body
    }

    /// One request: its rows in `index` order, and the tokens the endpoint reported.
    fn fetch(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u64)> {
        let body = self.body(texts);
        let response = self
            .client
            .post(&self.url)
            .json(&body)
            .send()
            .with_context(|| format!("embeddings request to {}", self.url))?;
        let status = response.status();
        let payload: serde_json::Value = response
            .json()
            .with_context(|| format!("embeddings response from {} was not JSON", self.url))?;
        if !status.is_success() {
            bail!("embeddings endpoint returned {status}: {payload}");
        }

        let data = payload["data"]
            .as_array()
            .context("embeddings response has no data array")?;
        let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        for row in data {
            let index = row["index"].as_u64().context("a data row has no index")? as usize;
            ensure!(index < out.len(), "data index {index} is out of range");
            let encoded = row["embedding"]
                .as_str()
                .context("embedding is not a base64 string")?;
            let raw = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .context("embedding is not valid base64")?;
            ensure!(
                raw.len() % 4 == 0,
                "embedding payload is {} bytes, not a whole number of f32",
                raw.len()
            );
            let mut row: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            normalize(&mut row);
            out[index] = Some(row);
        }
        let tokens = payload["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
        out.into_iter()
            .enumerate()
            .map(|(i, v)| v.with_context(|| format!("no embedding for input {i}")))
            .collect::<Result<Vec<_>>>()
            .map(|v| (v, tokens))
    }
}

impl Encoder for HttpEncoder {
    fn embed(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u64)> {
        let mut out = Vec::with_capacity(texts.len());
        let mut tokens = 0;
        for chunk in texts.chunks(self.batch) {
            let (rows, spent) = self.fetch(chunk)?;
            out.extend(rows);
            tokens += spent;
        }
        Ok((out, tokens))
    }
}

#[cfg(test)]
#[path = "../../../../tests/clm/embedding.rs"]
mod tests;
