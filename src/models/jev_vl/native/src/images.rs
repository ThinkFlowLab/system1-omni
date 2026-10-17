//! Adapted image assets (pre-encoded by recipe/jev_vl/preencode.py) and the
//! language-side position expansion for Qwen mrope, ported from HF transformers
//! modeling_qwen3_5 get_rope_index / get_vision_position_ids.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::image_preprocess::ProcessedImage;

/// CPU preparation retains pixels until the serialized executor runs vision.
#[derive(Clone)]
pub enum ImageInput {
    Ready(Arc<ImageAsset>),
    Inline {
        key: String,
        pixels: Arc<ProcessedImage>,
    },
}

impl ImageInput {
    pub(crate) fn n_tokens(&self) -> usize {
        match self {
            Self::Ready(asset) => asset.n_tokens(),
            Self::Inline { pixels, .. } => pixels.image_tokens(),
        }
    }

    pub fn grid_thw(&self) -> [i64; 3] {
        match self {
            Self::Ready(asset) => asset.grid_thw,
            Self::Inline { pixels, .. } => pixels.image_grid_thw.map(|n| n as i64),
        }
    }
}

/// One adapted image: merger output rows in placeholder order + the patch grid.
pub struct ImageAsset {
    pub grid_thw: [i64; 3],
    pub embeddings: Vec<half::bf16>,
}

impl ImageAsset {
    pub fn n_tokens(&self) -> usize {
        let [t, h, w] = self.grid_thw;
        (t * h * w / 4) as usize
    }

    pub fn load(dir: &Path, url_hash: &str, model_index_hash: &str) -> Result<Arc<Self>> {
        let grid: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("grid.json")).context("imgcache asset grid.json")?,
        )?;
        ensure!(
            grid["url_sha256"].as_str() == Some(url_hash)
                && grid["model_index_sha256"].as_str() == Some(model_index_hash),
            "image asset source does not match the request and model index"
        );
        let thw: [i64; 3] = serde_json::from_value(grid["grid_thw"].clone())?;
        ensure!(
            thw[0] == 1 && thw[1] > 0 && thw[2] > 0 && thw[1] % 2 == 0 && thw[2] % 2 == 0,
            "expected one image with an even, positive spatial grid"
        );
        let asset_n = thw[1]
            .checked_mul(thw[2])
            .and_then(|v| usize::try_from(v / 4).ok())
            .context("image grid is too large")?;
        ensure!(asset_n <= 32768, "image exceeds the supported token limit");
        ensure!(
            grid["n_tokens"].as_u64() == Some(asset_n as u64),
            "grid.json n_tokens mismatch"
        );
        let data = std::fs::read(dir.join("emb.safetensors")).context("imgcache asset emb")?;
        let st = safetensors::SafeTensors::deserialize(&data)?;
        let rows = st.tensor("rows")?;
        ensure!(
            rows.dtype() == safetensors::Dtype::BF16 && rows.shape() == [asset_n, 5120],
            "image rows vs grid mismatch"
        );
        let embeddings: Vec<half::bf16> = rows
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half::bf16::from_le_bytes(*b))
            .collect();
        ensure!(
            embeddings.iter().all(|x| x.is_finite()),
            "non-finite image embedding"
        );
        Ok(Arc::new(Self {
            grid_thw: [thw[0], thw[1], thw[2]],
            embeddings,
        }))
    }
}

/// The token-expanded multimodal prompt for the language-side boundary.
pub struct Expanded {
    pub ids: Vec<u32>,
    pub positions: [Vec<i64>; 3],
    /// Each image's pad run in expanded-id coordinates plus its position base
    /// and mrope advance, used to split a cached prefix from the question.
    pub blocks: Vec<ImageBlock>,
}

/// One image block's pad run in expanded coordinates.
#[derive(Clone, Debug)]
pub struct ImageBlock {
    /// ids row of the block's first pad.
    pub start: usize,
    /// ids row one past the block's last pad.
    pub end: usize,
    /// The meshgrid position base of the block (equals `start` for the first
    /// image; earlier images' mrope advances make it smaller for later ones).
    pub base: i64,
    /// mrope advance after the image: max(grid_h, grid_w) / 2.
    pub advance: i64,
    /// Index of the image in the request's image list.
    pub asset: usize,
}

/// mrope meshgrid triples for the image rows [from, from + rows) of a grid,
/// offset by `base` — a slice of what `expand` would build for the full block.
pub fn meshgrid_positions(grid: [i64; 3], base: i64, from: usize, rows: usize) -> [Vec<i64>; 3] {
    let [gt, gh, gw] = grid;
    let (lh, lw) = ((gh / 2) as usize, (gw / 2) as usize);
    let _ = gt;
    let mut out = [
        Vec::with_capacity(rows),
        Vec::with_capacity(rows),
        Vec::with_capacity(rows),
    ];
    for l in from..from + rows {
        let w = l % lw;
        let h = (l / lw) % lh;
        let t = l / (lh * lw);
        out[0].push(base + t as i64);
        out[1].push(base + h as i64);
        out[2].push(base + w as i64);
    }
    out
}

/// Tokens for one image block: the raw prompt text carries one literal
/// `<|vision_start|><|image_pad|><|vision_end|>` per image; the token sequence
/// repeats image_pad n = prod(grid)/merge² (merge=2) times. The three mrope
/// axes follow HF's meshgrid (t outer, h mid, w fastest), each offset by the
/// running start position; the next text token continues at
/// start + max(grid_h, grid_w) / merge.
pub fn expand(text_ids: &[u32], image_pad: u32, assets: &[Arc<ImageAsset>]) -> Result<Expanded> {
    expand_grids(
        text_ids,
        image_pad,
        &assets.iter().map(|a| a.grid_thw).collect::<Vec<_>>(),
    )
}

/// Expand from CPU-validated geometry, before any online vision forward.
pub fn expand_grids(text_ids: &[u32], image_pad: u32, grids: &[[i64; 3]]) -> Result<Expanded> {
    let marks: Vec<usize> = text_ids
        .iter()
        .enumerate()
        .filter_map(|(i, &id)| (id == image_pad).then_some(i))
        .collect();
    ensure!(
        marks.len() == grids.len(),
        "prompt has {} image placeholders but the request carries {} images",
        marks.len(),
        grids.len()
    );
    let mut ids: Vec<u32> = Vec::new();
    let mut positions: [Vec<i64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut blocks: Vec<ImageBlock> = Vec::with_capacity(marks.len());
    let mut cursor = 0usize;
    let mut current = 0i64;
    for (k, &mark) in marks.iter().enumerate() {
        for rel in 0..(mark - cursor) as i64 {
            positions[0].push(current + rel);
            positions[1].push(current + rel);
            positions[2].push(current + rel);
        }
        ids.extend_from_slice(&text_ids[cursor..mark]);
        current += (mark - cursor) as i64;
        cursor = mark + 1;
        let [gt, gh, gw] = grids[k];
        let tokens = gh.checked_mul(gw).and_then(|n| n.checked_div(4));
        ensure!(
            gt == 1
                && gh > 0
                && gw > 0
                && gh % 2 == 0
                && gw % 2 == 0
                && tokens.is_some_and(|n| n <= 32768),
            "expected one image with a bounded, even, positive spatial grid"
        );
        let (lg_t, lg_h, lg_w) = (gt, gh / 2, gw / 2); // temp_merge=1, spatial_merge=2
        let start = ids.len();
        for _ in 0..tokens.unwrap() {
            ids.push(image_pad);
        }
        blocks.push(ImageBlock {
            start,
            end: ids.len(),
            base: current,
            advance: gh.max(gw) / 2,
            asset: k,
        });
        for ti in 0..lg_t {
            for hi in 0..lg_h {
                for wi in 0..lg_w {
                    positions[0].push(current + ti);
                    positions[1].push(current + hi);
                    positions[2].push(current + wi);
                }
            }
        }
        current += gh.max(gw) / 2;
    }
    for rel in 0..(text_ids.len() - cursor) as i64 {
        positions[0].push(current + rel);
        positions[1].push(current + rel);
        positions[2].push(current + rel);
    }
    ids.extend_from_slice(&text_ids[cursor..]);
    Ok(Expanded {
        ids,
        positions,
        blocks,
    })
}
