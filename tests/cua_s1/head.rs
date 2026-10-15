use super::*;

#[test]
fn legacy_merged_projection_preserves_fp32_close_scores() {
    // Real BF16-representable weights and hidden values, with two distinct
    // FP32 dot products in the same BF16 bin. Legacy merged exports must not
    // acquire the new reference profile's ordered tie when the worker upgrades.
    assert_eq!(
        letter_logits(
            &[20.0, 0.03125, 20.0, 0.046875],
            &[1.0, 1.0],
            2,
            TextExport::Merged
        )
        .unwrap(),
        [20.03125, 20.046875]
    );
}

#[test]
fn letter_projection_matches_the_reference_bf16_output_boundary() {
    assert_eq!(
        letter_logits(&[1.0078125], &[1.0078125], 1, TextExport::Unmerged).unwrap(),
        [1.015625]
    );
    // Both exact dot products belong to the same BF16 bin. Retaining FP32
    // invents a winning second option where the reference has an ordered tie.
    assert_eq!(
        letter_logits(
            &[20.0, 0.03125, 20.0, 0.046875],
            &[1.0, 1.0],
            2,
            TextExport::Unmerged
        )
        .unwrap(),
        [20.0, 20.0]
    );
}

#[test]
fn letter_projection_preserves_fp64_accumulation_and_bf16_readout() {
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
        letter_logits(&letters, &[1.0, 1.0, 1.0], 2, TextExport::Unmerged).unwrap(),
        [1.0, 16_777_216.0]
    );
    assert_eq!(
        letter_logits(&letters, &[1.0, 1.0, 1.0], 1, TextExport::Unmerged).unwrap(),
        [1.0]
    );
    assert_eq!(
        letter_logits(&letters, &[1.0, 1.0, 1.0], 2, TextExport::Merged).unwrap(),
        [1.0, 16_777_216.0]
    );
}

#[test]
fn export_profiles_keep_the_four_observed_near_tie_orders_separate() {
    use serde_json::{Value, json};
    use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/legacy_merged_readout.json")).unwrap();
    // Tokenization is incidental here: run the real response finishing on the
    // frozen request's option order, but use synthetic BF16 projection inputs.
    // These vectors reproduce the readout boundary, not the model's hidden state.
    let tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab([("[UNK]".into(), 0)].into_iter().collect())
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    let processor = crate::processing::Processor { tokenizer };
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let keys: Vec<_> = case["request"]["questions"]["q"]["criteria"]
            .as_object()
            .unwrap()
            .keys()
            .collect();
        let legacy = case["expected_response"]["answers"]["q"]["choice"]
            .as_str()
            .unwrap();
        let rounded = case["rounded_choice"].as_str().unwrap();
        let legacy_index = keys.iter().position(|key| key.as_str() == legacy).unwrap();
        let rounded_index = keys.iter().position(|key| key.as_str() == rounded).unwrap();
        assert!(rounded_index < legacy_index);
        let mut letters = [10.0, 0.0].repeat(keys.len());
        letters[rounded_index * 2..rounded_index * 2 + 2].copy_from_slice(&[20.0, 0.03125]);
        letters[legacy_index * 2..legacy_index * 2 + 2].copy_from_slice(&[20.0, 0.046875]);
        // Alternating profiles must not leak a readout policy between exports.
        for export in [TextExport::Merged, TextExport::Unmerged, TextExport::Merged] {
            let prepared = processor
                .prepare(&serde_json::to_vec(&case["request"]).unwrap())
                .ok()
                .unwrap();
            let logits = letter_logits(&letters, &[1.0, 1.0], keys.len(), export).unwrap();
            let response = prepared.context.finish(vec![logits]).unwrap();
            let choice = match export {
                TextExport::Merged => legacy,
                TextExport::Unmerged => rounded,
            };
            assert_eq!(
                response["answers"]["q"]["choice"],
                json!(choice),
                "{}",
                case["id"]
            );
            let probabilities = response["answers"]["q"]["probabilities"]
                .as_object()
                .unwrap();
            assert_eq!(probabilities.keys().collect::<Vec<_>>(), keys);
            match export {
                TextExport::Merged => {
                    assert!(probabilities[legacy].as_f64() > probabilities[rounded].as_f64())
                }
                TextExport::Unmerged => assert_eq!(probabilities[legacy], probabilities[rounded]),
            }
        }
    }
}

#[test]
fn finite_fp32_legacy_scores_are_not_overflowed_by_bf16_conversion() {
    assert_eq!(
        letter_logits(&[f32::MAX], &[1.0], 1, TextExport::Merged).unwrap(),
        [f32::MAX]
    );
    assert!(letter_logits(&[f32::MAX], &[1.0], 1, TextExport::Unmerged).is_err());
}

#[test]
fn letter_projection_rejects_nonfinite_scores_before_later_prompts() {
    for export in [TextExport::Merged, TextExport::Unmerged] {
        assert!(letter_logits(&[f32::MAX], &[2.0], 1, export).is_err());
        assert!(letter_logits(&[f32::NAN], &[1.0], 1, export).is_err());
    }
}
