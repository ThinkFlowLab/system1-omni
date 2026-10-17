//! Weight-free checks of metadata shared by host and resident input validation.
//! No synthetic device pointer or CUDA allocation is needed for these checks.
use super::*;

fn input() -> MultimodalInput<'static> {
    MultimodalInput {
        token_ids: &[1, 9, 9, 2],
        image_token_indices: &[1, 2],
        image_embeddings: &[],
        position_ids: [&[0, 1, 1, 2], &[0, 1, 2, 3], &[0, 1, 2, 3]],
    }
}

#[test]
fn resident_metadata_accepts_valid_placeholder_rows_and_positions() {
    let input = input();
    input.validate_metadata(2, 10, 9, 8, 4).unwrap();
    input.validate_positions(8).unwrap();
}

#[test]
fn resident_metadata_rejects_missing_reordered_and_extra_placeholders() {
    for indices in [&[1][..], &[2, 1], &[1, 2, 3], &[1, 1]] {
        let mut input = input();
        input.image_token_indices = indices;
        let error = input
            .validate_metadata(2, 10, 9, 8, indices.len() * 2)
            .unwrap_err();
        assert!(error.to_string().contains("ordered placeholders"));
    }
}

#[test]
fn resident_metadata_rejects_bad_prompt_bounds_vocabulary_and_extent() {
    let mut invalid = input();
    invalid.token_ids = &[];
    assert!(invalid.validate_metadata(2, 10, 9, 8, 4).is_err());
    assert!(input().validate_metadata(2, 10, 9, 3, 4).is_err());
    assert!(input().validate_metadata(2, 9, 9, 8, 4).is_err());
    for elements in [0, 3, 5] {
        assert!(input().validate_metadata(2, 10, 9, 8, elements).is_err());
    }
    assert!(input().validate_metadata(usize::MAX, 10, 9, 8, 0).is_err());
    input().validate_metadata(2, 10, 9, 8, 4).unwrap();
}

#[test]
fn resident_positions_reject_bad_axis_length_negative_and_limit() {
    for axis in [&[0, 1][..], &[0, -1, 2, 3], &[0, 1, 2, 8]] {
        for changed_axis in 0..3 {
            let mut input = input();
            input.position_ids[changed_axis] = axis;
            assert!(input.validate_positions(8).is_err());
        }
    }
    input().validate_positions(8).unwrap();
}
