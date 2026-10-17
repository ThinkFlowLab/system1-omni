//! GPU parity and recovery checks for the borrowed vision/language boundary.
use super::*;
use crate::image_preprocess::ProcessedImage;

pub(super) fn load() -> VisionEngine {
    let path = |name| std::path::PathBuf::from(std::env::var_os(name).expect(name));
    VisionEngine::load(
        &path("CUA_S1_BASE"),
        &path("CUA_S1_VISION_ADAPTER"),
        &path("CUA_S1_MODEL"),
        &path("CUA_S1_CUDA_LIB"),
    )
    .unwrap()
}
pub(super) fn questions(seed: usize) -> Vec<Question> {
    vec![
        Question {
            name: "move".into(),
            goal: format!("Select the next move {seed}."),
            keys: vec!["left".into(), "right".into()],
            labels: vec!["Move left".into(), "Move right".into()],
        },
        Question {
            name: "visible".into(),
            goal: "Is the scene visible?".into(),
            keys: vec!["yes".into(), "no".into()],
            labels: vec!["Yes".into(), "No".into()],
        },
    ]
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_* weights; run serially"]
fn device_features_host_parity_reuse_geometry_growth_and_recovery() {
    let mut engine = load();
    // Run this test in separate processes with CUA_S1_VISION_GRAPH=0 and =1.
    let mut previous = None;
    for (width, height, seed, text_seed, count) in [
        (256, 256, 1, 1, 1),
        (256, 256, 2, 1, 2),
        (256, 256, 2, 2, 2),
        (512, 256, 2, 2, 2),
        (256, 512, 3, 3, 2),
        (512, 512, 4, 4, 2),
        (256, 256, 1, 1, 8),
    ] {
        let rgb: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 17 + seed * 31) % 251) as u8)
            .collect();
        let qs: Vec<_> = (0..count)
            .map(|i| {
                let mut q = questions(text_seed).remove(i % 2);
                q.name = format!("q{i}");
                q
            })
            .collect();
        let prepared = engine.prepare(width, height, &rgb, &qs).unwrap();
        let host = engine.vision.forward(&prepared.image).unwrap();
        let expected: Vec<_> = prepared
            .prompts
            .iter()
            .map(|p| engine.score(p, &host, 2).unwrap())
            .collect();
        if width == 256 && height == 256 && count <= 2 {
            let feature_bits: Vec<_> = host.iter().map(|v| v.to_bits()).collect();
            let hidden_bits = bits(&expected[0].hidden);
            if let Some((old_seed, old_text, old_features, old_hidden)) = previous {
                if old_seed != seed {
                    assert_ne!(feature_bits, old_features, "changed pixels oracle");
                }
                if old_seed == seed && old_text != text_seed {
                    assert_eq!(feature_bits, old_features);
                    assert_ne!(hidden_bits, old_hidden, "changed text oracle");
                }
            }
            previous = Some((seed, text_seed, feature_bits, hidden_bits));
        }
        let features = engine.vision.forward_device(&prepared.image).unwrap();
        assert_eq!(
            features
                .download()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            host.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        for (prompt, expected) in prepared
            .prompts
            .iter()
            .zip(&expected)
            .cycle()
            .take(expected.len() + 2)
        {
            let (hidden, logits) = VisionEngine::raw_score_device(
                &mut engine.language,
                &engine.letters,
                prompt,
                &features,
                2,
            )
            .unwrap();
            assert_eq!(bits(&hidden), bits(&expected.hidden));
            assert_eq!(bits(&logits), bits(&expected.logits));
            assert_eq!(
                bits(&probabilities(&logits).unwrap()),
                bits(&expected.probabilities)
            );
        }
        let p = &prepared.prompts[0];
        let mut invalid_prompt = ImagePrompt {
            token_ids: p.token_ids.clone(),
            image_token_indices: p.image_token_indices.clone(),
            position_ids: p.position_ids.clone(),
        };
        invalid_prompt.image_token_indices.pop();
        assert!(
            VisionEngine::raw_score_device(
                &mut engine.language,
                &engine.letters,
                &invalid_prompt,
                &features,
                2
            )
            .is_err()
        );
        assert!(
            VisionEngine::raw_score_device(
                &mut engine.language,
                &engine.letters,
                &prepared.prompts[0],
                &features,
                2
            )
            .is_ok()
        );
        engine.device_features = false;
        let host_response = engine.predict_prepared(&prepared).unwrap();
        engine.device_features = true;
        assert_eq!(engine.predict_prepared(&prepared).unwrap(), host_response);
        let mut invalid = ProcessedImage {
            image_grid_thw: prepared.image.image_grid_thw,
            resized_height: prepared.image.resized_height,
            resized_width: prepared.image.resized_width,
            pixel_values: prepared.image.pixel_values.clone(),
        };
        invalid.pixel_values[0] = f32::NAN;
        assert!(engine.vision.forward_device(&invalid).is_err());
        assert_eq!(engine.predict_prepared(&prepared).unwrap(), host_response);
    }
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_* weights; run serially"]
fn device_features_reject_nonfinite_and_recover() {
    let mut engine = load();
    let rgb = vec![32; 256 * 256 * 3];
    let prepared = engine.prepare(256, 256, &rgb, &questions(1)).unwrap();
    let host = engine.vision.forward(&prepared.image).unwrap();
    let expected = engine.score(&prepared.prompts[0], &host, 2).unwrap();
    let stream = crate::cuda::new_stream().unwrap();
    let buffer = crate::cuda::DeviceBuffer::new(host.len() * 2).unwrap();
    for value in [bf16::NAN, bf16::INFINITY, bf16::NEG_INFINITY] {
        let mut invalid = host.clone();
        invalid[0] = value;
        let bytes: Vec<_> = invalid.iter().flat_map(|v| v.to_le_bytes()).collect();
        unsafe {
            crate::cuda::upload(buffer.at(0), &bytes, stream).unwrap();
        }
        let view = buffer.bf16_matrix(host.len() / 2560, 2560).unwrap();
        assert!(engine.score_device(&prepared.prompts[0], &view, 2).is_err());
        let recovered = engine.score(&prepared.prompts[0], &host, 2).unwrap();
        assert_eq!(bits(&recovered.hidden), bits(&expected.hidden));
    }
    crate::cuda::synchronize(stream).unwrap();
    unsafe {
        (crate::cuda::api().cs1_stream_destroy)(stream);
    }
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_* weights; run serially"]
fn device_features_language_scratch_growth_and_short_prompt_recovery() {
    let mut engine = load();
    let rgb = vec![80; 256 * 256 * 3];
    for words in [1, 1400, 1] {
        let mut qs = questions(1);
        qs.truncate(1);
        qs[0].goal = "Continue ".repeat(words);
        let request = engine.prepare(256, 256, &rgb, &qs).unwrap();
        if words > 1 {
            assert!(request.prompts[0].token_ids.len() > 1024);
        }
        engine.device_features = false;
        let expected = engine.predict_prepared(&request).unwrap();
        engine.device_features = true;
        assert_eq!(engine.predict_prepared(&request).unwrap(), expected);
    }
}
