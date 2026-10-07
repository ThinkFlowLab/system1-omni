//! Batch-one, unpadded inputs at the adapted-vision / language-model boundary.

use anyhow::{Result, ensure};
use half::bf16;

/// Image rows are already adapted to the language hidden size, in placeholder
/// order. Positions are the T/H/W slices of an int64 `[3, 1, sequence]` tensor.
/// The caller owns preprocessing, vision execution and the position calculation.
pub struct MultimodalInput<'a> {
    pub token_ids: &'a [u32],
    pub image_token_indices: &'a [usize],
    pub image_embeddings: &'a [bf16],
    pub position_ids: [&'a [i64]; 3],
}

impl MultimodalInput<'_> {
    /// Check the entire boundary before allocating buffers or launching CUDA.
    pub fn validate(
        &self,
        hidden: usize,
        vocab: usize,
        image_token: u32,
        max_position: usize,
    ) -> Result<()> {
        let t = self.token_ids.len();
        ensure!(t > 0 && t <= max_position, "empty or oversized prompt");
        ensure!(
            self.token_ids.iter().all(|&id| (id as usize) < vocab),
            "token id outside the vocabulary"
        );
        let expected: Vec<usize> = self
            .token_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == image_token).then_some(i))
            .collect();
        ensure!(
            self.image_token_indices == expected,
            "image indices must exactly match the ordered placeholders"
        );
        ensure!(
            Some(self.image_embeddings.len()) == expected.len().checked_mul(hidden),
            "image embedding shape mismatch"
        );
        ensure!(
            self.image_embeddings.iter().all(|x| x.is_finite()),
            "non-finite image embedding"
        );
        ensure!(
            self.position_ids.iter().all(|axis| axis.len() == t),
            "position_ids must have shape [3, 1, sequence]"
        );
        ensure!(
            self.position_ids
                .iter()
                .flat_map(|axis| axis.iter())
                .all(|&p| p >= 0 && (p as u64) < max_position as u64),
            "position outside the configured range"
        );
        Ok(())
    }
}

/// Qwen3.5's interleaved recomposition: overwrite H at 1::3 and W at 2::3 up
/// to section[axis] * 3, retaining T elsewhere. The second rotary half repeats
/// these frequencies, which the existing attention-prep kernel handles.
/// Float32 inverse frequencies/products and host float64 trig preserve the
/// original native text table rounding when all three axes are equal.
pub(crate) fn rotary_tables(
    positions: [&[i64]; 3],
    half: usize,
    theta: f64,
    sections: [usize; 3],
) -> (Vec<u8>, Vec<u8>) {
    let inv: Vec<f32> = (0..half)
        .map(|i| 1f32 / (theta as f32).powf((2 * i) as f32 / (2 * half) as f32))
        .collect();
    let mut cos = Vec::with_capacity(positions[0].len() * half * 2);
    let mut sin = Vec::with_capacity(cos.capacity());
    for (t, _) in positions[0].iter().enumerate() {
        for (i, &f) in inv.iter().enumerate() {
            let axis = if i % 3 == 1 && i < sections[1] * 3 {
                1
            } else if i % 3 == 2 && i < sections[2] * 3 {
                2
            } else {
                0
            };
            let angle = (f * positions[axis][t] as f32) as f64;
            cos.extend(bf16::from_f32(angle.cos() as f32).to_le_bytes());
            sin.extend(bf16::from_f32(angle.sin() as f32).to_le_bytes());
        }
    }
    (cos, sin)
}
