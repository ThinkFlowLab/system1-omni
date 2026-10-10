use anyhow::{Result, ensure};
/// Divide, subtract, exponentiate and normalize in FP32, then expose those values to Python-style FP64 finishing.
/// CPU exp/summation order can differ from torch's vectorized FP32 softmax by a few ulps.
pub(crate) fn softmax(logits: &[f32], temperature: f32) -> Result<Vec<f64>> {
    ensure!(
        logits.len() >= 2 && logits.iter().all(|v| v.is_finite()),
        "invalid candidate logits"
    );
    let scaled: Vec<f32> = logits.iter().map(|v| v / temperature).collect();
    ensure!(
        scaled.iter().all(|v| v.is_finite()),
        "temperature-scaled logits overflow FP32"
    );
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = scaled.iter().map(|v| (v - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.iter_mut().for_each(|v| *v /= total);
    Ok(exps.into_iter().map(f64::from).collect())
}
/// Fixed decimal formatting rounds the original binary64 value to even.
/// Multiplying by 10^d before rounding changes cases such as Python round(2.675,2).
pub(crate) fn round(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}")
        .parse()
        .expect("finite decimal result")
}
