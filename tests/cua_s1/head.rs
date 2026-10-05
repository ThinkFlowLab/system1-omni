use super::*;

#[test]
fn letter_projection_preserves_fp64_accumulation_and_fp32_readout() {
    let letters = [
        16_777_216.0,
        1.0,
        -16_777_216.0,
        16_777_216.0,
        1.0,
        0.0,
        f32::NAN,
        f32::NAN,
        f32::NAN,
    ];
    // The first dot product would be zero with FP32 accumulation. The second
    // must round before postprocessing; unused letter rows must not be read.
    assert_eq!(
        letter_logits(&letters, &[1.0, 1.0, 1.0], 2).unwrap(),
        [1.0, 16_777_216.0]
    );
    assert_eq!(letter_logits(&letters, &[1.0, 1.0, 1.0], 1).unwrap(), [1.0]);
}

#[test]
fn letter_projection_rejects_nonfinite_scores_before_later_prompts() {
    assert!(letter_logits(&[f32::MAX], &[2.0], 1).is_err());
    assert!(letter_logits(&[f32::NAN], &[1.0], 1).is_err());
}
