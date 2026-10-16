//! GPU checks of `Model::forward_multimodal_readout`. They need a GPU, CUA_S1_CUDA_LIB
//! pointing at libqwen3_5_cuda.so and QWEN3_5_CHECKPOINT pointing at a merged
//! multimodal Qwen3.5 export with `image_token_id`, such as an OmniJev export:
//!
//!     CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so QWEN3_5_CHECKPOINT=<export> \
//!       cargo test --release -p omni-qwen3-5-native --test readout -- --ignored --test-threads=1

use std::fs::File;
use std::path::PathBuf;

use half::bf16;
use omni_qwen3_5_native::inputs::{MultimodalInput, image_positions};
use omni_qwen3_5_native::model::Model;
use safetensors::SafeTensors;
use serde_json::Value;

fn checkpoint() -> PathBuf {
    PathBuf::from(std::env::var_os("QWEN3_5_CHECKPOINT").expect("set QWEN3_5_CHECKPOINT"))
}

fn model() -> Model {
    let library = PathBuf::from(std::env::var_os("CUA_S1_CUDA_LIB").expect("set CUA_S1_CUDA_LIB"));
    Model::load(&checkpoint(), &library).unwrap()
}

/// Token ids in [1000, 100000) from a fixed seed.
fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1000 + (x >> 33) as u32 % 99_000
        })
        .collect()
}

fn bits(rows: &[Vec<f32>]) -> Vec<u32> {
    rows.iter().flatten().map(|v| v.to_bits()).collect()
}

/// A prompt of `before` text tokens, an 8×12 grid of image tokens and `after` text tokens,
/// with random image rows.
struct Prompt {
    ids: Vec<u32>,
    images: Vec<usize>,
    features: Vec<bf16>,
    positions: [Vec<i64>; 3],
}

impl Prompt {
    fn new(model: &Model, before: usize, after: usize) -> Self {
        let image = model.cfg.image_token_id.expect("a multimodal checkpoint");
        let grid = [1, 8, 12];
        let count = 4 * 6;
        let mut ids = tokens(before, 1);
        ids.extend(std::iter::repeat_n(image, count));
        ids.extend(tokens(after, 2));
        let positions = image_positions(&ids, image, grid).unwrap();
        let mut x = 3u64;
        let features = (0..count * model.cfg.hidden)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                bf16::from_f32(((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.2)
            })
            .collect();
        Self {
            images: (before..before + count).collect(),
            ids,
            features,
            positions,
        }
    }

    fn input(&self) -> MultimodalInput<'_> {
        MultimodalInput {
            token_ids: &self.ids,
            image_token_indices: &self.images,
            image_embeddings: &self.features,
            position_ids: [&self.positions[0], &self.positions[1], &self.positions[2]],
        }
    }
}

/// log p(token) after the final-norm `hidden` state on the CPU: float64 dot products with
/// the tied embedding read from the checkpoint, rounded to bfloat16 as the GPU's logits,
/// then a float64 log-softmax. Also the token's logit.
fn cpu_logprob(hidden: &[f32], token: u32) -> (f64, f64) {
    let dir = checkpoint();
    let index: Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let name = [
        "embed_tokens.weight",
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    ]
    .into_iter()
    .find(|n| index["weight_map"][n].is_string())
    .unwrap();
    let file = File::open(dir.join(index["weight_map"][name].as_str().unwrap())).unwrap();
    // SAFETY: the checkpoint is not modified while the test runs.
    let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&map).unwrap();
    let tensor = tensors.tensor(name).unwrap();
    let width = hidden.len();
    let logits: Vec<f64> = tensor
        .data()
        .chunks_exact(width * 2)
        .map(|row| {
            let dot: f64 = row
                .as_chunks::<2>()
                .0
                .iter()
                .zip(hidden)
                .map(|(b, &h)| bf16::from_le_bytes(*b).to_f64() * h as f64)
                .sum();
            bf16::from_f64(dot).to_f64()
        })
        .collect();
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
    (logits[token as usize] - lse, logits[token as usize])
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB and a multimodal QWEN3_5_CHECKPOINT"]
fn text_readout_equals_forward_fixed() {
    let mut model = model();
    for n in [1usize, 63, 64, 65, 300] {
        let ids = tokens(n, n as u64);
        let positions: Vec<i64> = (0..n as i64).collect();
        let readout = model
            .forward_multimodal_readout(
                &MultimodalInput {
                    token_ids: &ids,
                    image_token_indices: &[],
                    image_embeddings: &[],
                    position_ids: [&positions; 3],
                },
                &[n - 1],
                &[],
            )
            .unwrap();
        let fixed = model.forward_fixed(&ids).unwrap();
        assert_eq!(bits(&readout.hidden), bits(&[fixed]), "{n} tokens");
    }
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB and a multimodal QWEN3_5_CHECKPOINT"]
fn readouts_do_not_depend_on_what_else_is_read() {
    let mut model = model();
    let prompt = Prompt::new(&model, 10, 140);
    let t = prompt.ids.len();
    let vocab_ids = [0u32, 13, 2000, 151_000, 248_000];
    // 130 targets: more than one 64-row chunk of logits
    let targets: Vec<(usize, u32)> = (0..130)
        .map(|i| (t - 130 + i, vocab_ids[i % vocab_ids.len()]))
        .collect();
    let all = model
        .forward_multimodal_readout(&prompt.input(), &[0, 40, t - 1], &targets)
        .unwrap();
    assert!(all.logprobs.iter().all(|&p| p.is_finite() && p <= 0.0));
    let few = model
        .forward_multimodal_readout(&prompt.input(), &[t - 1, 40], &targets[65..67])
        .unwrap();
    assert_eq!(
        bits(&few.hidden),
        bits(&[all.hidden[2].clone(), all.hidden[1].clone()])
    );
    assert_eq!(few.logprobs, all.logprobs[65..67]);
    // against the CPU, from the downloaded hidden state: the GPU may round a logit to the
    // neighbouring bfloat16 value, which moves the log-probability by one unit of it
    let (row, token) = targets[129];
    assert_eq!(row, t - 1);
    let (want, logit) = cpu_logprob(&all.hidden[2], token);
    let ulp = bf16::from_f64(logit.abs()).to_f64() * 2f64.powi(-7);
    assert!(
        (all.logprobs[129] as f64 - want).abs() <= ulp + 1e-3,
        "{} vs {want}",
        all.logprobs[129]
    );
}
