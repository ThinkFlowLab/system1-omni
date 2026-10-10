//! Request validation, tokenization, and response assembly.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::caches::{Caches, L1Meta, structure_key};
use crate::contract::{self, Compiled, Kind, Reject};
use crate::executor::{ContinueMm, ImgBlock, MmPlan, PreparedMm, Readout, SuffixBlock};
use crate::images::{self, ImageAsset};

pub struct Processor {
    tokenizer: Tokenizer,
    labels: Vec<String>,
    max_length: usize,
    imgcache: Option<PathBuf>,
    model_index_hash: String,
    image_pad: u32,
    vision_end: u32,
    caches: Arc<Caches>,
}

pub struct PreparedRequest {
    pub plan: MmPlan,
    pub readout: Readout,
    pub context: ResponseContext,
    /// x-jev-cache response-header marker: l1/l2/l3 hit flags + prefix length.
    pub cache_note: String,
}

/// The original request mapping, usage, and timing.
pub struct ResponseContext {
    kind: Kind,
    options: Vec<String>,
    input_tokens: usize,
    start: Instant,
}

impl Processor {
    pub(crate) fn load(
        dir: &Path,
        labels: Vec<String>,
        max_length: usize,
        model_index_hash: String,
        caches: Arc<Caches>,
    ) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let image_pad = tokenizer
            .token_to_id("<|image_pad|>")
            .context("tokenizer has no <|image_pad|>")?;
        let vision_end = tokenizer
            .token_to_id("<|vision_end|>")
            .context("tokenizer has no <|vision_end|>")?;
        let hub = std::env::var("JEV_VL_IMGCACHE")
            .ok()
            .map(PathBuf::from)
            .or_else(|| Some(dir.join("imgcache")))
            .filter(|p| p.is_dir());
        Ok(Self {
            tokenizer,
            labels,
            max_length,
            imgcache: hub,
            model_index_hash,
            image_pad,
            vision_end,
            caches,
        })
    }

    fn url_key(url: &str) -> String {
        use sha2::Digest;
        sha2::Sha256::digest(url.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// Load a prepared image asset, reusing the bounded L2 cache when enabled.
    /// With caches disabled, each request reads and parses the asset again.
    fn load_asset(&self, url: &str) -> Result<Arc<ImageAsset>, Reject> {
        let key = Self::url_key(url);
        if let Some(hit) = self.caches.l2_get(&key) {
            return Ok(hit);
        }
        let dir = match &self.imgcache {
            Some(d) => d.join(&key),
            None => {
                return Err(Reject::bad_request(
                    "image input: no imgcache is configured for this worker",
                ));
            }
        };
        let asset = ImageAsset::load(&dir, &key, &self.model_index_hash)
            .map_err(|e| Reject::bad_request(format!("image input is not preencoded: {e}")))?;
        self.caches.l2_insert(key, asset.clone());
        Ok(asset)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, Reject> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(|e| Reject::bad_request(format!("tokenization failed: {e}")))?
            .get_ids()
            .to_vec())
    }

    /// Validate, compile the raw prompt, and tokenize; add_special_tokens=false,
    /// like the official raw System-1 readout. Never truncates: oversize prompts
    /// get the upstream context-length rejection.
    pub fn prepare(
        &self,
        head: &crate::executor::LabelHead,
        manifest: &Value,
        raw: &[u8],
    ) -> Result<PreparedRequest, Reject> {
        let start = Instant::now();
        let compiled = contract::compile(raw, &self.labels)?;
        let readout = head
            .readout(manifest, compiled.kind.as_str(), compiled.options.len())
            .map_err(|e| Reject::bad_request(format!("invalid readout setup: {e:#}")))?;
        let cfg = &self.caches.cfg;
        if compiled.images.is_empty() || !cfg.enabled {
            // Text-only requests and disabled caches use a full forward.
            let text_ids = self.tokenize(&compiled.prompt)?;
            let (plan, input_tokens, note) = if compiled.images.is_empty() {
                (
                    MmPlan::Text { ids: text_ids },
                    0,
                    "l1=-,l3=-,p=-".to_string(),
                )
            } else {
                let assets: Vec<Arc<ImageAsset>> = compiled
                    .images
                    .iter()
                    .map(|url| self.load_asset(url))
                    .collect::<Result<Vec<_>, Reject>>()?;
                let e = images::expand(&text_ids, self.image_pad, &assets)
                    .map_err(|e| Reject::bad_request(format!("invalid image prompt: {e:#}")))?;
                let n = e.ids.len();
                (
                    MmPlan::Full(expanded_mm(e, &assets)),
                    n,
                    "l1=off,l3=off,p=-".to_string(),
                )
            };
            let input_tokens = match &plan {
                MmPlan::Text { ids } => ids.len(),
                _ => input_tokens,
            };
            return self.finish_c(input_tokens, plan, readout, &compiled, start, note);
        }
        self.prepare_cached(&compiled, readout, start)
    }

    /// L1 structure record + L3 continuation planning for one-image requests.
    /// The plan decides whether the model sees the full prompt (miss/populate) or
    /// only the continuation slice (record state present).
    fn prepare_cached(
        &self,
        compiled: &Compiled,
        readout: Readout,
        start: Instant,
    ) -> Result<PreparedRequest, Reject> {
        let cfg = &self.caches.cfg;
        let key = structure_key(compiled.kind.as_str(), &compiled.parts);
        let single_image = compiled.images.len() == 1;
        let tail = contract::tail_after_last_image(compiled, &self.labels);
        if let (Some(record), true) = (self.caches.record_get(key), single_image && tail.is_some())
        {
            // Scalars out of the record up front: the borrow ends so the record
            // can move into the plan arms below.
            let meta = &record.meta;
            let (p, pads_start, pads_end, base_pad, advance) = (
                meta.p,
                meta.pads_start,
                meta.pads_end,
                meta.base_pad,
                meta.advance,
            );
            let asset = self.load_asset(&compiled.images[0])?;
            // Suffix piece: pads(E-P) + <|vision_end|> + fresh tail tokenization.
            let tail = tail.unwrap();
            let tail_ids = self.tokenize(&tail)?;
            if tail_ids.contains(&self.image_pad) {
                return Err(Reject::bad_request(
                    "invalid image prompt: unexpected image placeholder in suffix",
                ));
            }
            let suffix_pads = pads_end - p;
            let mut ids = Vec::with_capacity(suffix_pads + 1 + tail_ids.len());
            ids.extend(std::iter::repeat_n(self.image_pad, suffix_pads));
            ids.push(self.vision_end);
            ids.extend_from_slice(&tail_ids);
            let mut pos =
                images::meshgrid_positions(asset.grid_thw, base_pad, p - pads_start, suffix_pads);
            let after = base_pad + advance;
            for k in 0..(ids.len() - suffix_pads) as i64 {
                pos[0].push(after + k);
                pos[1].push(after + k);
                pos[2].push(after + k);
            }
            let input_tokens = p + ids.len();
            match (cfg.l3, record.state()) {
                (true, Some(state)) => {
                    self.caches.l3_hit();
                    let block = Some(SuffixBlock {
                        asset,
                        row_offset: p - pads_start,
                        rows: suffix_pads,
                    });
                    return self.finish_c(
                        input_tokens,
                        MmPlan::Continue(ContinueMm {
                            ids,
                            positions: pos,
                            block,
                            state,
                        }),
                        readout,
                        compiled,
                        start,
                        format!("l1=hit,l3=hit,p={p}"),
                    );
                }
                (true, None) => {
                    self.caches.l3_miss();
                    let mm = rebuild_full(&ids, &pos, &record.meta, &asset);
                    return self.finish_c(
                        mm.ids.len(),
                        MmPlan::Populate { mm, record },
                        readout,
                        compiled,
                        start,
                        format!("l1=hit,l3=miss,p={p}"),
                    );
                }
                (false, _) => {
                    let mm = rebuild_full(&ids, &pos, &record.meta, &asset);
                    return self.finish_c(
                        mm.ids.len(),
                        MmPlan::Full(mm),
                        readout,
                        compiled,
                        start,
                        "l1=hit,l3=off,p=-".to_string(),
                    );
                }
            }
        }
        // L1 miss: full expansion (and the cache record when the shape qualifies).
        let text_ids = self.tokenize(&compiled.prompt)?;
        let assets: Vec<Arc<ImageAsset>> = compiled
            .images
            .iter()
            .map(|url| self.load_asset(url))
            .collect::<Result<Vec<_>, Reject>>()?;
        let e = images::expand(&text_ids, self.image_pad, &assets)
            .map_err(|e| Reject::bad_request(format!("invalid image prompt: {e:#}")))?;
        let input_tokens = e.ids.len();
        let meta = (single_image && tail.is_some() && cfg.l1)
            .then(|| prefix_meta(&e, input_tokens))
            .flatten();
        if let Some(meta) = meta {
            let p = meta.p;
            let record = self.caches.record_insert(key, meta);
            if cfg.l3 {
                self.caches.l3_miss();
                return self.finish_c(
                    input_tokens,
                    MmPlan::Populate {
                        mm: expanded_mm(e, &assets),
                        record,
                    },
                    readout,
                    compiled,
                    start,
                    format!("l1=miss,l3=miss,p={p}"),
                );
            }
            return self.finish_c(
                input_tokens,
                MmPlan::Full(expanded_mm(e, &assets)),
                readout,
                compiled,
                start,
                "l1=miss,l3=off,p=-".to_string(),
            );
        }
        self.finish_c(
            input_tokens,
            MmPlan::Full(expanded_mm(e, &assets)),
            readout,
            compiled,
            start,
            "l1=miss,l3=off,p=-".to_string(),
        )
    }

    fn finish_c(
        &self,
        input_tokens: usize,
        plan: MmPlan,
        readout: Readout,
        compiled: &Compiled,
        start: Instant,
        cache_note: String,
    ) -> Result<PreparedRequest, Reject> {
        let t = input_tokens;
        if t > self.max_length {
            return Err(Reject::bad_request(format!(
                "This model's maximum context length is {} tokens. However, you requested 1 output tokens and your prompt contains at least {t} input tokens, for a total of at least {} tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value={t})",
                self.max_length,
                t + 1
            )));
        }
        Ok(PreparedRequest {
            plan,
            readout,
            cache_note,
            context: ResponseContext {
                kind: compiled.kind,
                options: compiled.options.clone(),
                input_tokens: t,
                start,
            },
        })
    }
}

/// Cache-anchor geometry for one expanded single-image prompt: prefix = rows
/// [0, floor(pads_end/64)*64) — a multiple of the 64-token GDN chunk, containing
/// the whole text head and all but a tail slice of the image block.
fn prefix_meta(e: &images::Expanded, input_tokens: usize) -> Option<L1Meta> {
    let block = e.blocks.last()?;
    let p = block.end / 64 * 64;
    if p < 64 || p < block.start || p >= input_tokens {
        return None;
    }
    Some(L1Meta {
        pads_start: block.start,
        pads_end: block.end,
        p,
        base_pad: block.base,
        advance: block.advance,
        ids_prefix: e.ids[..p].to_vec(),
        positions_prefix: [
            e.positions[0][..p].to_vec(),
            e.positions[1][..p].to_vec(),
            e.positions[2][..p].to_vec(),
        ],
    })
}

/// Full expanded ids/positions with image blocks borrowed from their assets.
fn expanded_mm(e: images::Expanded, assets: &[Arc<ImageAsset>]) -> PreparedMm {
    let blocks = e
        .blocks
        .iter()
        .map(|b| ImgBlock {
            start: b.start,
            end: b.end,
            asset: assets[b.asset].clone(),
        })
        .collect();
    PreparedMm {
        ids: e.ids,
        positions: e.positions,
        blocks,
    }
}

/// Rebuild the full expanded ids/positions from the cached prefix plus the
/// freshly built suffix — exactly the bytes the miss path would have produced.
fn rebuild_full(
    ids_suffix: &[u32],
    pos_suffix: &[Vec<i64>; 3],
    meta: &L1Meta,
    asset: &Arc<ImageAsset>,
) -> PreparedMm {
    let mut ids = meta.ids_prefix.clone();
    ids.extend_from_slice(ids_suffix);
    let mut positions = meta.positions_prefix.clone();
    for a in 0..3 {
        positions[a].extend_from_slice(&pos_suffix[a]);
    }
    let blocks = vec![ImgBlock {
        start: meta.pads_start,
        end: meta.pads_end,
        asset: asset.clone(),
    }];
    PreparedMm {
        ids,
        positions,
        blocks,
    }
}

impl ResponseContext {
    pub fn finish(self, probabilities: Vec<f64>) -> Value {
        contract::answer(
            self.kind,
            &self.options,
            &probabilities,
            self.input_tokens,
            self.start.elapsed().as_secs_f64(),
        )
    }
}

#[cfg(test)]
#[path = "../../../../../tests/jev_vl/processing.rs"]
mod tests;
