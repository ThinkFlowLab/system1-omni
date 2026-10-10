use super::*;
use crate::Kind;
fn row(ids: Vec<u32>, candidates: Vec<u32>, position: usize) -> RowInput {
    RowInput {
        ids,
        candidate_ids: candidates,
        readout_position: position,
        question_id: "q".into(),
        level_index: None,
        kind: Kind::Choice,
        temperature: 1.164,
    }
}
#[test]
fn validates_every_row_before_dispatch() {
    let labels = [32, 33, 34];
    assert!(validate_rows(&[row(vec![1, 2], vec![32, 33], 1)], &labels).is_ok());
    for input in [
        row(vec![], vec![32, 33], 0),
        row(vec![1, 2], vec![32, 33], 0),
        row(vec![VOCAB as u32], vec![32, 33], 0),
        row(vec![1], vec![33, 32], 0),
        row(vec![1], vec![], 0),
        row(vec![1], vec![32], 0),
        row(vec![1; 36865], vec![32, 33], 36864),
    ] {
        assert!(validate_rows(&[row(vec![1], vec![32, 33], 0), input], &labels).is_err());
    }
    assert!(validate_rows(&[], &labels).is_ok());
}

#[test]
#[ignore = "requires the built Qwen CUDA library and an available NVIDIA GPU"]
fn bf16_head_rounds_before_fp32_calibration() {
    let library = std::env::var("DECIDER_CUDA_LIB").unwrap();
    cuda::load(Path::new(&library)).unwrap();
    cuda::set_device(0).unwrap();
    let mut weights = vec![0; PADDED_LABELS * HIDDEN * 2];
    for (offset, value) in [
        (0, 1.0),
        (2, 0.00390625),
        (HIDDEN * 2, 1.0),
        (HIDDEN * 2 + 2, 0.015625),
    ] {
        weights[offset..offset + 2].copy_from_slice(&bf16::from_f32(value).to_le_bytes());
    }
    let head = Head::new(&weights, 1).unwrap();
    let mut hidden = vec![0.0; HIDDEN];
    hidden[..2].copy_from_slice(&[1.0, 1.0]);
    assert_eq!(head.project(&hidden, 2).unwrap(), vec![1.0, 1.015625]);
    assert_eq!(head.project(&hidden, 255).unwrap()[254], 0.0);
    hidden[0] = f32::NAN;
    assert!(head.project(&hidden, 2).is_err());
}

#[test]
#[ignore = "requires the built Qwen CUDA library and an available NVIDIA GPU"]
fn batched_head_preserves_rows_counts_and_bf16_rounding() {
    cuda::load(Path::new(&std::env::var("DECIDER_CUDA_LIB").unwrap())).unwrap();
    cuda::set_device(0).unwrap();
    let mut weights = vec![0; PADDED_LABELS * HIDDEN * 2];
    for (offset, value) in [
        (0, 1.0),
        (2, 0.00390625),
        (HIDDEN * 2, 1.0),
        (HIDDEN * 2 + 2, 0.015625),
    ] {
        weights[offset..offset + 2].copy_from_slice(&bf16::from_f32(value).to_le_bytes());
    }
    let head = Head::new(&weights, 2).unwrap();
    let mut a = vec![0.0; HIDDEN];
    a[..2].copy_from_slice(&[1.0, 1.0]);
    let mut b = vec![0.0; HIDDEN];
    b[..2].copy_from_slice(&[2.0, 1.0]);
    let first = head.project_batch(&[&a, &b], &[2, 255]).unwrap();
    assert_eq!(first[0], vec![1.0, 1.015625]);
    assert_eq!(&first[1][..2], &[2.0, 2.015625]);
    assert_eq!(first[1].len(), 255);
    assert!(first[1][2..].iter().all(|&x| x == 0.0));
    let reversed = head.project_batch(&[&b, &a], &[255, 2]).unwrap();
    assert_eq!(reversed[0], first[1]);
    assert_eq!(reversed[1], first[0]);
    assert!(head.project_batch(&[&a, &b, &a], &[2, 2, 2]).is_err());
    assert!(head.project_batch(&[&a, &b], &[2]).is_err());
    b[0] = f32::NAN;
    assert!(head.project_batch(&[&a, &b], &[2, 2]).is_err());
}
