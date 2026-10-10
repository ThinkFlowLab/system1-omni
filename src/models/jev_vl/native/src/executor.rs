//! Verbalizer readout: merged label-head rows on the CPU in f64 plus slot bias
//! and per-kind calibration temperature.
//!
//! The official readout uses full-vocabulary logprobs; the per-kind softmax only
//! sees `(logprob(t) + bias_slot) / T_kind`, where `logprob = logit - logZ` shares
//! one constant per request — so softmax((logit + bias)/T) is identical. The
//! export stores the merged lm_head rows for every exported label id, hence one
//! dot product per candidate token instead of a vocabulary GEMM.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use omni_qwen3_5_native::inputs::MultimodalInput;

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::model::{Config, Model};
use omni_runtime::SerialScheduler;
use serde_json::Value;

use crate::images::ImageAsset;

/// Per-request readout setup: which label tokens, with which bias, divided by
/// which temperature, matching decision_head.json's slots + calibration.json.
#[derive(Clone)]
pub struct Readout {
    pub token_ids: Vec<u32>,
    pub bias: Vec<f64>,
    pub temperature: f64,
}

/// The merged lm_head rows for the exported label ids, row-major float32.
pub struct LabelHead {
    width: usize,
    rows: Vec<f32>,
    index: HashMap<u32, usize>,
}

impl LabelHead {
    pub fn load(dir: &Path, manifest: &Value) -> Result<Self> {
        let cfg = Config::load(dir)?;
        ensure!(
            (
                cfg.hidden,
                cfg.intermediate,
                cfg.full_attention.len(),
                cfg.heads,
                cfg.kv_heads,
                cfg.lin_k_heads,
                cfg.lin_v_heads
            ) == (5120, 17408, 64, 24, 4, 16, 48),
            "expected the Qwen3.8-27B backbone dimensions"
        );
        let spec = manifest["label_head"].as_object().context("label_head")?["file"]
            .as_str()
            .context("label_head.file")?;
        let data = std::fs::read(dir.join(spec)).context("read label_head safetensors")?;
        let st = safetensors::SafeTensors::deserialize(&data)?;
        let rows = st.tensor("rows")?;
        ensure!(
            rows.dtype() == safetensors::Dtype::F32
                && rows.shape().len() == 2
                && rows.shape()[1] == cfg.hidden,
            "label_head rows must be float32 [n, {}]",
            cfg.hidden
        );
        let ids_t = st.tensor("ids")?;
        ensure!(
            ids_t.dtype() == safetensors::Dtype::I64 && ids_t.shape() == [rows.shape()[0]],
            "label_head ids must be int64 [n]"
        );
        let rows_f: Vec<f32> = rows
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let ids: Vec<i64> = ids_t
            .data()
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect();
        let exported: Vec<i64> = serde_json::from_value(manifest["label_head"]["ids"].clone())?;
        ensure!(
            ids == exported,
            "label_head ids do not match the export manifest"
        );
        ensure!(
            rows_f.iter().all(|v| v.is_finite()),
            "non-finite label head"
        );
        Ok(Self {
            width: cfg.hidden,
            rows: rows_f,
            index: ids
                .iter()
                .enumerate()
                .map(|(i, &t)| (t as u32, i))
                .collect(),
        })
    }

    /// Slot bias from the trained 24-slot head: [slot_lo, slot_hi), then zeros.
    pub fn readout(&self, manifest: &Value, kind_token: &str, count: usize) -> Result<Readout> {
        let temps = manifest["temperatures"]
            .as_object()
            .context("temperatures")?;
        let temperature = temps[kind_token].as_f64().context("temperature")?;
        ensure!(
            temperature.is_finite() && temperature > 0.0,
            "invalid temperature"
        );
        let bias: Vec<f64> = manifest["verbalizer_bias"]
            .as_array()
            .context("verbalizer_bias")?
            .iter()
            .map(|v| v.as_f64().context("numeric bias"))
            .collect::<Result<_>>()?;
        let verbalizer: Vec<i64> = serde_json::from_value(manifest["verbalizer_ids"].clone())?;
        ensure!(bias.len() == 24 && verbalizer.len() == 24, "24-slot head");
        let label_ids: Vec<i64> = serde_json::from_value(manifest["label_ids"].clone())?;
        let (token_ids, slot_bias): (Vec<i64>, Vec<f64>) = match kind_token {
            "noul" => (verbalizer[0..2].to_vec(), bias[0..2].to_vec()),
            "score" => (verbalizer[2..8].to_vec(), bias[2..8].to_vec()),
            "choice" => {
                ensure!(count <= label_ids.len(), "choice options above label table");
                let b: Vec<f64> = (0..count)
                    .map(|i| if i < 16 { bias[8 + i] } else { 0.0 })
                    .collect();
                (label_ids[..count].to_vec(), b)
            }
            other => anyhow::bail!("unknown kind {other}"),
        };
        ensure!(token_ids.len() == slot_bias.len(), "readout shape");
        for &t in &token_ids {
            ensure!(
                self.index.contains_key(&(t as u32)),
                "token {t} missing from the label head"
            );
        }
        Ok(Readout {
            token_ids: token_ids.iter().map(|&t| t as u32).collect(),
            bias: slot_bias,
            temperature,
        })
    }
}

impl LabelHead {
    /// One candidate distribution: `(logit + bias) / T` per slot then a stable
    /// softmax. `last` is the final-norm hidden state as float32.
    pub fn probabilities(&self, last: &[f32], readout: &Readout) -> Result<Vec<f64>> {
        ensure!(last.len() == self.width, "hidden width");
        let w = self.width;
        let mut z = Vec::with_capacity(readout.token_ids.len());
        for (t, b) in readout.token_ids.iter().zip(&readout.bias) {
            let i = *self
                .index
                .get(t)
                .with_context(|| format!("token {t} missing from the label head"))?;
            let row = &self.rows[i * w..(i + 1) * w];
            z.push(
                (row.iter()
                    .zip(last)
                    .map(|(&w, &h)| w as f64 * h as f64)
                    .sum::<f64>()
                    + b)
                    / readout.temperature,
            );
        }
        let base = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let wgt: Vec<f64> = z.iter().map(|&v| (v - base).exp()).collect();
        let total: f64 = wgt.iter().sum();
        ensure!(total > 0.0 && total.is_finite(), "degenerate readout");
        Ok(wgt.iter().map(|v| v / total).collect())
    }
}

/// One image block in the expanded request (adapted rows borrowed from L2).
pub struct ImgBlock {
    /// Global expanded-id row of the first pad.
    pub start: usize,
    /// Global expanded-id row one past the last pad.
    pub end: usize,
    /// Parsed adapter output (grid + rows), shared with the L2 cache.
    pub asset: Arc<ImageAsset>,
}

/// Full expanded ids + absolute positions + image blocks (global coordinates).
pub struct PreparedMm {
    pub ids: Vec<u32>,
    pub positions: [Vec<i64>; 3],
    pub blocks: Vec<ImgBlock>,
}

/// The cached-prefix continuation slice of one request.
pub struct ContinueMm {
    /// Rows [p, T): (pads_end - p) pads, <|vision_end|>, fresh tail tokens.
    pub ids: Vec<u32>,
    /// Absolute positions for those rows.
    pub positions: [Vec<i64>; 3],
    /// The image rows within the slice (the last block's tail), if any.
    pub block: Option<SuffixBlock>,
    /// L3 device state of the prefix.
    pub state: Arc<omni_qwen3_5_native::model::PrefixState>,
}

/// The image rows of one continuation slice: an offset range of one asset's rows.
pub struct SuffixBlock {
    pub asset: Arc<ImageAsset>,
    /// First adapted-row index feeding the slice (= p - pads_start).
    pub row_offset: usize,
    /// Number of rows = number of pads leading the slice.
    pub rows: usize,
}

/// The inputs and cached state needed to execute one request.
pub enum MmPlan {
    /// Text-only prompt.
    Text { ids: Vec<u32> },
    /// Full multimodal prefill.
    Full(PreparedMm),
    /// Structure recorded but its device state is not built yet: run the capture
    /// phase over rows [0, p) followed by the continuation over [p, T), then
    /// publish the state so later requests go straight to `Continue`.
    Populate {
        mm: PreparedMm,
        record: Arc<crate::caches::PrefixRecord>,
    },
    /// Cached-prefix hit: only the continuation slice runs.
    Continue(ContinueMm),
}

pub struct Executor {
    model: Arc<Mutex<Model>>,
    head: Arc<LabelHead>,
    caches: Arc<crate::caches::Caches>,
}

impl Executor {
    pub(crate) async fn load(
        dir: &Path,
        library: &Path,
        head: Arc<LabelHead>,
        caches: Arc<crate::caches::Caches>,
    ) -> Result<Self> {
        let (d, lib) = (dir.to_path_buf(), library.to_path_buf());
        let model = tokio::task::spawn_blocking(move || Model::load(&d, &lib)).await??;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            head,
            caches,
        })
    }

    /// One forward, then `(logit + bias) / T` per slot and a stable softmax,
    /// matching serve_decide.py's s1_pass ordering.
    pub async fn execute(
        &self,
        scheduler: &SerialScheduler,
        plan: MmPlan,
        readout: Readout,
    ) -> Result<Vec<f64>> {
        let model = self.model.clone();
        let head = self.head.clone();
        let caches = self.caches.clone();
        scheduler
            .run(move || {
                let mut model = model
                    .lock()
                    .map_err(|_| anyhow::anyhow!("poisoned model"))?;
                let result = (|| {
                    let last = match &plan {
                        MmPlan::Text { ids } => model.forward(ids)?,
                        MmPlan::Full(mm) => {
                            let one = slice_mm(mm, 0, mm.ids.len());
                            let input = one.input();
                            model.forward_multimodal(&input)?
                        }
                        MmPlan::Populate { mm, record } => {
                            let p = record.meta.p;
                            let (populated, state) = match model.alloc_prefix(p) {
                                Ok(mut state) => {
                                    let result = (|| -> Result<Vec<f32>> {
                                        let pf = slice_mm(mm, 0, p);
                                        model
                                            .forward_multimodal_capture(&pf.input(), &mut state)?;
                                        let cf = slice_mm(mm, p, mm.ids.len());
                                        model.forward_multimodal_continue(&cf.input(), &state)
                                    })();
                                    // The captured buffers must stay alive until this drain,
                                    // including a failed capture or continuation.
                                    model.synchronize()?;
                                    (result, Some(state))
                                }
                                Err(e) => (Err(e), None),
                            };
                            match populated {
                                Ok(last) => {
                                    caches.record_publish_state(record.key, state.unwrap());
                                    caches.l3_populate();
                                    last
                                }
                                Err(e) => {
                                    // Device allocation or capture failed: serve the
                                    // request on the proven one-shot path instead.
                                    eprintln!("prefix populate failed; one-shot fallback: {e:#}");
                                    caches.l3_fallback_full();
                                    let one = slice_mm(mm, 0, mm.ids.len());
                                    let input = one.input();
                                    model.forward_multimodal(&input)?
                                }
                            }
                        }
                        MmPlan::Continue(c) => {
                            let indices: Vec<usize> = match &c.block {
                                Some(b) => (0..b.rows).collect(),
                                None => Vec::new(),
                            };
                            let embeddings: &[half::bf16] = match &c.block {
                                Some(b) => {
                                    &b.asset.embeddings
                                        [b.row_offset * 5120..(b.row_offset + b.rows) * 5120]
                                }
                                None => &[],
                            };
                            let input = MultimodalInput {
                                token_ids: &c.ids,
                                image_token_indices: &indices,
                                image_embeddings: embeddings,
                                position_ids: [
                                    c.positions[0].as_slice(),
                                    c.positions[1].as_slice(),
                                    c.positions[2].as_slice(),
                                ],
                            };
                            model.forward_multimodal_continue(&input, &c.state)?
                        }
                    };
                    head.probabilities(&last, &readout)
                })();
                // Keep admission until all queued device work has drained, including errors.
                model.synchronize()?;
                result
            })
            .await
    }
}

/// A slice of one request's expanded data for rows [from, to) in slice-local
/// coordinates: borrowed ids/positions, local pad indices, and the adapted rows
/// those pads need — borrowed straight out of the L2 asset when they form one
/// contiguous run (the usual single-image case), else concatenated once.
struct SlicedMm<'a> {
    ids: &'a [u32],
    positions: [&'a [i64]; 3],
    indices: Vec<usize>,
    borrowed: Option<&'a [half::bf16]>,
    owned: Option<Vec<half::bf16>>,
}

impl<'a> SlicedMm<'a> {
    fn input(&'a self) -> MultimodalInput<'a> {
        MultimodalInput {
            token_ids: self.ids,
            image_token_indices: &self.indices,
            image_embeddings: self.borrowed.or(self.owned.as_deref()).unwrap_or(&[]),
            position_ids: self.positions,
        }
    }
}

fn slice_mm<'a>(mm: &'a PreparedMm, from: usize, to: usize) -> SlicedMm<'a> {
    let indices: Vec<usize> = mm
        .blocks
        .iter()
        .flat_map(|b| (b.start.max(from)..b.end.min(to)).map(move |g| g - from))
        .collect();
    let overlapping: Vec<&ImgBlock> = mm
        .blocks
        .iter()
        .filter(|b| b.end > from && b.start < to)
        .collect();
    let (borrowed, owned) = match overlapping.as_slice() {
        [] => (None, None),
        [one] => {
            let rows = (one.start.max(from) - one.start)..(one.end.min(to) - one.start);
            (
                Some(&one.asset.embeddings[rows.start * 5120..rows.end * 5120]),
                None,
            )
        }
        _ => (
            None,
            Some(
                overlapping
                    .iter()
                    .flat_map(|b| {
                        let rows = (b.start.max(from) - b.start)..(b.end.min(to) - b.start);
                        &b.asset.embeddings[rows.start * 5120..rows.end * 5120]
                    })
                    .copied()
                    .collect::<Vec<_>>(),
            ),
        ),
    };
    SlicedMm {
        ids: &mm.ids[from..to],
        positions: [
            &mm.positions[0][from..to],
            &mm.positions[1][from..to],
            &mm.positions[2][from..to],
        ],
        indices,
        borrowed,
        owned,
    }
}

#[cfg(test)]
#[path = "../../../../../tests/jev_vl/readout.rs"]
mod readout_tests;
