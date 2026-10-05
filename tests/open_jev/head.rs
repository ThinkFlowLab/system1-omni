use super::*;

#[test]
fn scalar_projection_preserves_accumulation_rounding_and_bias_order() {
    let head = DecisionHead {
        weights: vec![16_777_216.0, 1.0, -16_777_216.0],
        bias: 0.25,
    };
    assert_eq!(head.score(vec![1.0, 1.0, 1.0]).unwrap(), 1.25);
    // Bias is added after rounding the dot product to FP32, not in FP64.
    let head = DecisionHead {
        weights: vec![16_777_216.0, 1.0],
        bias: -16_777_216.0,
    };
    assert_eq!(head.score(vec![1.0, 1.0]).unwrap(), 0.0);
}

#[test]
fn scalar_projection_rejects_nonfinite_scores() {
    let head = DecisionHead {
        weights: vec![f32::MAX],
        bias: 0.0,
    };
    assert!(head.score(vec![2.0]).is_err());
    assert!(head.score(vec![f32::NAN]).is_err());
}
