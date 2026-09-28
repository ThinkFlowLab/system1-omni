//! CPU checks for the CLM head loader. No GPU and no encoder, but the frozen export.
use omni_clm::{
    Kind, Question, Weights, answer, confidence, distribution, head_tensors, weights::Heads,
};
use std::path::PathBuf;
use tempfile::tempdir;

fn export_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("CLM_EXPORT").expect("set CLM_EXPORT to the export directory"))
}

fn load() -> Heads {
    let weights = Weights::open(&export_dir().join("model.safetensors")).unwrap();
    Heads::load(&weights).unwrap()
}

#[test]
#[ignore = "requires CLM_EXPORT at a converted checkpoint; CPU only"]
fn every_tensor_conversion_matches_the_oracle() {
    use sha2::{Digest, Sha256};

    let dir = export_dir();
    let oracle: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(dir.join("oracle.json")).unwrap()).unwrap();
    let weights = Weights::open(&dir.join("model.safetensors")).unwrap();

    assert_eq!(oracle.len(), 16, "oracle must cover both heads");
    for row in oracle {
        let name = row["name"].as_str().unwrap();
        let shape: Vec<usize> = serde_json::from_value(row["shape"].clone()).unwrap();
        let values = weights.f32(name, &shape).unwrap();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            row["f32"].as_str().unwrap(),
            "{name} f32"
        );
    }
}

#[test]
#[ignore = "requires CLM_EXPORT at a converted checkpoint; CPU only"]
fn decisions_match_the_reference_implementation() {
    let heads = load();
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(export_dir().join("head-oracle.json")).unwrap())
            .or_else(|_| serde_json::from_slice(&std::fs::read("/tmp/clm-oracle.json").unwrap()))
            .unwrap();

    for case in oracle["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let keys: Vec<String> = serde_json::from_value(case["keys"].clone()).unwrap();
        let temperature = case["temperature"].as_f64().unwrap() as f32;
        let expected: Vec<f32> = serde_json::from_value(case["probabilities"].clone()).unwrap();

        let state = embedding(&format!("state::{name}"), heads.config.head.hidden_size);
        let candidates: Vec<Vec<f32>> = keys
            .iter()
            .map(|k| embedding(&format!("cand::{name}::{k}"), heads.config.head.hidden_size))
            .collect();

        let probs = distribution(&heads, &state, &candidates, temperature).unwrap();
        assert_eq!(probs.len(), expected.len(), "{name}: length");
        for (got, want) in probs.iter().zip(&expected) {
            assert!(
                (got - want).abs() < 1e-4,
                "{name}: probability {got} vs {want}"
            );
        }

        let kind = match case["kind"].as_str().unwrap() {
            "choice" => Kind::Choice,
            "noul" => Kind::Noul,
            _ => Kind::Score,
        };
        let question = Question {
            id: name.to_string(),
            kind,
            keys,
        };
        let answer = answer(&question, &probs).unwrap();
        match (&answer, case.get("choice")) {
            (
                omni_clm::Answer::Choice {
                    choice,
                    confidence: c,
                    ..
                },
                Some(want),
            ) => {
                assert_eq!(choice, want.as_str().unwrap(), "{name}: choice");
                let want_c = case["confidence"].as_f64().unwrap() as f32;
                assert!(
                    (c - want_c).abs() < 1e-4,
                    "{name}: confidence {c} vs {want_c}"
                );
            }
            (
                omni_clm::Answer::Score {
                    score,
                    confidence: c,
                    ..
                },
                None,
            ) => {
                let want_s = case["score"].as_f64().unwrap() as f32;
                assert!(
                    (score - want_s).abs() < 1e-4,
                    "{name}: score {score} vs {want_s}"
                );
                let want_c = case["confidence"].as_f64().unwrap() as f32;
                assert!(
                    (c - want_c).abs() < 1e-4,
                    "{name}: confidence {c} vs {want_c}"
                );
            }
            (omni_clm::Answer::Noul { noul }, None) => {
                let want = case["noul"].as_f64().unwrap() as f32;
                assert!((noul - want).abs() < 1e-4, "{name}: noul {noul} vs {want}");
            }
            (other, _) => panic!("{name}: unexpected answer {other:?}"),
        }
    }
}

/// The same synthesised embedding the oracle uses, so both sides see identical vectors.
fn embedding(text: &str, dim: usize) -> Vec<f32> {
    use sha2::{Digest, Sha256};

    let mut out: Vec<f32> = Vec::with_capacity(dim);
    let mut counter = 0u32;
    while out.len() < dim {
        let digest = Sha256::digest(format!("{counter}:{text}").as_bytes());
        for chunk in digest.as_chunks::<4>().0 {
            if out.len() == dim {
                break;
            }
            out.push(u32::from_be_bytes(*chunk) as f64 as f32 / 2f64.powi(31) as f32 - 1.0);
        }
        counter += 1;
    }
    let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
    out.iter().map(|v| v / norm).collect()
}

#[test]
fn confidence_matches_the_reference_definition() {
    // Top minus the mean of the rest, clamped; a single candidate is fully decided.
    assert_eq!(confidence(&[1.0]), 1.0);
    assert!((confidence(&[0.75, 0.25]) - 0.5).abs() < 1e-6);
    assert!((confidence(&[0.5, 0.3, 0.2]) - 0.25).abs() < 1e-6);
    assert_eq!(confidence(&[0.4, 0.4, 0.4]), 0.0);
}

#[test]
fn a_short_inventory_is_reported_with_both_sides() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    safetensors::tensor::serialize_to_file(
        std::collections::HashMap::<String, safetensors::tensor::TensorView>::new(),
        None,
        &path,
    )
    .unwrap();
    let weights = Weights::open(&path).unwrap();
    let message = weights.validate_names(["absent"]).unwrap_err().to_string();
    assert!(message.contains("missing [absent]"), "{message}");
}

#[test]
#[ignore = "requires CLM_EXPORT at a converted checkpoint; CPU only"]
fn the_inventory_follows_the_head_configuration() {
    let heads = load();
    let cfg = &heads.config.head;
    let expected = head_tensors(cfg).unwrap();
    assert_eq!(expected.len(), 16, "two heads, eight tensors each");
    let weights = Weights::open(&export_dir().join("model.safetensors")).unwrap();
    weights
        .validate_names(expected.iter().map(|(n, _)| n.as_str()))
        .unwrap();
}
