//! CPU checks for the CLM head loader. No GPU and no encoder, but the frozen export.
use omni_clm::{
    Kind, Question, Weights, answer, confidence, distribution, head_tensors, weights::Heads,
};
use std::path::PathBuf;

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

/// `layernorm: false` is a configuration `head_tensors` honours, so the loader must too:
/// a checkpoint that declares no LayerNorm is complete without `norms.*`, and one that
/// declares it is still rejected when the tensors are absent.
#[test]
fn the_loader_follows_the_layernorm_flag() {
    let dir = std::env::temp_dir().join("omni-clm-synthetic-heads");
    std::fs::create_dir_all(&dir).unwrap();

    let plain = synthetic(&dir.join("no-layernorm.safetensors"), false, false);
    let heads = Heads::load(&Weights::open(&plain).unwrap()).unwrap();
    assert!(heads.state.norm_weight.is_none());
    assert!(heads.action.norm_weight.is_none());
    assert_eq!(head_tensors(&heads.config.head).unwrap().len(), 12);

    let normed = synthetic(&dir.join("layernorm.safetensors"), true, true);
    let heads = Heads::load(&Weights::open(&normed).unwrap()).unwrap();
    assert!(heads.state.norm_weight.is_some());
    assert_eq!(head_tensors(&heads.config.head).unwrap().len(), 16);

    let claiming = synthetic(&dir.join("claims-layernorm.safetensors"), true, false);
    let Err(err) = Heads::load(&Weights::open(&claiming).unwrap()) else {
        panic!("a checkpoint that declares layernorm but omits norms.* was accepted");
    };
    assert!(err.to_string().contains("norms.0.weight"), "{err}");
}

/// A two-head checkpoint small enough to write here, with the LayerNorm tensors present
/// or absent independently of what the configuration declares.
fn synthetic(path: &std::path::Path, layernorm: bool, include_norms: bool) -> PathBuf {
    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize_to_file};
    use std::collections::HashMap;

    let metadata: HashMap<String, String> = HashMap::from([
        ("format".to_string(), "clm-heads".to_string()),
        (
            "cfg".to_string(),
            format!(
                r#"{{"hidden_size":4,"projection_dim":2,"width":3,"depth":3,"activation":"gelu","layernorm":{layernorm},"residual":false}}"#
            ),
        ),
        ("hidden_size".to_string(), "4".to_string()),
        ("projection_dim".to_string(), "2".to_string()),
        ("logit_scale".to_string(), "1.0".to_string()),
    ]);

    let (mut names, mut shapes, mut buffers) = (Vec::new(), Vec::new(), Vec::new());
    for head in ["state_head", "action_head"] {
        let mut spec = vec![
            (format!("{head}.inp.weight"), vec![3, 4]),
            (format!("{head}.inp.bias"), vec![3]),
            (format!("{head}.hidden.0.weight"), vec![3, 3]),
            (format!("{head}.hidden.0.bias"), vec![3]),
            (format!("{head}.out.weight"), vec![2, 3]),
            (format!("{head}.out.bias"), vec![2]),
        ];
        if include_norms {
            spec.push((format!("{head}.norms.0.weight"), vec![3]));
            spec.push((format!("{head}.norms.0.bias"), vec![3]));
        }
        for (name, shape) in spec {
            buffers.push(vec![0u8; shape.iter().product::<usize>() * 4]);
            shapes.push(shape);
            names.push(name);
        }
    }
    let tensors: Vec<(String, TensorView)> = names
        .iter()
        .zip(&shapes)
        .zip(&buffers)
        .map(|((name, shape), buffer)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), buffer).unwrap(),
            )
        })
        .collect();

    serialize_to_file(tensors, Some(metadata), path).unwrap();
    path.to_path_buf()
}
