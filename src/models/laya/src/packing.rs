//! Assemble one request into the padded layout consumed by the CUDA executor.
use anyhow::{Result, ensure};
use serde::Serialize;

use crate::preprocess::Prepared;

#[derive(Debug, Serialize)]
pub struct Batch {
    pub b: usize,
    pub l: usize,
    pub input_ids: Vec<i64>,
    pub lens: Vec<i32>,
    pub qtypes: Vec<i64>,
    pub markers: Vec<Vec<usize>>,
}

pub fn pack(prepared: &Prepared) -> Result<Batch> {
    let n = prepared.questions.len();
    ensure!(n <= 16, "at most 16 questions per CUDA request");
    ensure!(
        prepared
            .questions
            .iter()
            .map(|q| q.markers.len())
            .sum::<usize>()
            <= 2048,
        "at most 2048 option markers per CUDA request"
    );
    let b = if n == 0 { 0 } else { n.next_power_of_two() };
    let max_l = prepared
        .questions
        .iter()
        .map(|q| q.ids.len())
        .max()
        .unwrap_or(0);
    let alignment = if max_l <= 256 { 16 } else { 64 };
    let l = max_l.div_ceil(alignment) * alignment;
    ensure!(l <= 512, "CUDA sequence length exceeds 512");
    let mut batch = Batch {
        b,
        l,
        input_ids: vec![0; b * l],
        lens: vec![0; b],
        qtypes: vec![0; b],
        markers: Vec::with_capacity(n),
    };
    for (i, q) in prepared.questions.iter().enumerate() {
        ensure!(
            !q.ids.is_empty()
                && q.ids.iter().all(|id| *id < 50368)
                && (0..=2).contains(&q.qtype)
                && !q.markers.is_empty()
                && q.markers.iter().all(|m| *m < q.ids.len()),
            "invalid prepared sequence"
        );
        batch.input_ids[i * l..i * l + max_l].fill(50283);
        for (dst, id) in batch.input_ids[i * l..].iter_mut().zip(&q.ids) {
            *dst = i64::from(*id);
        }
        batch.lens[i] = q.ids.len() as i32;
        batch.qtypes[i] = q.qtype;
        batch.markers.push(q.markers.clone());
    }
    Ok(batch)
}
