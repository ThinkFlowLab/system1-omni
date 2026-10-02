use omni_laya::{config::Config, weights::Weights};
use serde_json::{Value, json};
use std::fs;
use tempfile::{TempDir, tempdir};

fn configs() -> (Value, Value) {
    let encoder = json!({
        "model_type": "modernbert", "hidden_activation": "gelu",
        "attention_bias": false, "mlp_bias": false, "norm_bias": false,
        "hidden_size": 1024, "intermediate_size": 2624,
        "num_attention_heads": 16, "num_hidden_layers": 28,
        "vocab_size": 50368, "norm_eps": 0.00001, "local_attention": 128,
        "layer_types": (["full_attention", "sliding_attention", "sliding_attention"]
            .repeat(10)[..28]),
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
        }
    });
    let agent = json!({
        "max_len": 512, "head_max_len": 192, "head_layers": 2,
        "temperature": [0.5, 1.0, 2.0], "temperature_by_options": {"4": 1.5}
    });
    (encoder, agent)
}

fn load_config(encoder: &Value, agent: &Value) -> anyhow::Result<Config> {
    let dir = tempdir()?;
    fs::create_dir(dir.path().join("encoder"))?;
    fs::write(dir.path().join("encoder/config.json"), encoder.to_string())?;
    fs::write(dir.path().join("rl_agent_config.json"), agent.to_string())?;
    Config::load(dir.path())
}

#[test]
fn config_accepts_supported_model_and_rejects_incompatible_inputs() {
    let (encoder, agent) = configs();
    let loaded = load_config(&encoder, &agent).unwrap();
    assert_eq!(loaded.agent.temperature, [0.5, 1.0, 2.0]);
    assert_eq!(loaded.encoder.hidden_size, 1024);
    for (pointer, value) in [
        ("/model_type", json!("bert")),
        ("/hidden_activation", json!("relu")),
        ("/attention_bias", json!(true)),
        ("/hidden_size", json!(768)),
        ("/num_hidden_layers", json!(27)),
        ("/layer_types/1", json!("full_attention")),
        ("/rope_parameters/full_attention/rope_theta", json!(10000)),
        (
            "/rope_parameters/sliding_attention/rope_type",
            json!("linear"),
        ),
    ] {
        let mut bad = encoder.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(load_config(&bad, &agent).is_err(), "accepted {pointer}");
    }
    let mut missing = encoder.clone();
    missing.as_object_mut().unwrap().remove("vocab_size");
    assert!(load_config(&missing, &agent).is_err());
    for (pointer, value) in [
        ("/max_len", json!(513)),
        ("/head_layers", json!(3)),
        ("/temperature", json!([1.0, 1.0])),
        ("/temperature/0", json!(0.0)),
        ("/temperature/1", json!(-1.0)),
        ("/temperature/2", json!(1e100)),
        ("/temperature_by_options/4", json!(0.0)),
    ] {
        let mut bad = agent.clone();
        *bad.pointer_mut(pointer).unwrap() = value;
        assert!(load_config(&encoder, &bad).is_err(), "accepted {pointer}");
    }
}

fn tensor(dtype: &str, shape: &[usize], data: &[u8]) -> (TempDir, Weights) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let mut header = json!({"w": {
        "dtype": dtype, "shape": shape, "data_offsets": [0, data.len()]
    }})
    .to_string();
    header.extend(std::iter::repeat_n(' ', (8 - header.len() % 8) % 8));
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(data);
    fs::write(&path, bytes).unwrap();
    let weights = Weights::open(&path).unwrap();
    (dir, weights)
}

#[test]
fn f32_conversion_preserves_zero_and_rounds_ties_to_even() {
    // Independent IEEE encodings: signed zero, subnormals and half-way values.
    let bits: [u32; 12] = [
        0x00000000, 0x80000000, 0x33800000, 0x33000000, 0x33c00000, 0x3f801000, 0x3f803000,
        0x3f808000, 0x3f818000, 0x00010000, 0x00008000, 0x00018000,
    ];
    let bytes: Vec<_> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
    let (_dir, weights) = tensor("F32", &[12], &bytes);
    let actual: Vec<_> = weights
        .f32("w", &[12])
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect();
    assert_eq!(actual, bits);
    assert_eq!(
        weights.f16("w", &[12]).unwrap(),
        [0, 0x8000, 1, 0, 2, 0x3c00, 0x3c02, 0x3c04, 0x3c0c, 0, 0, 0]
    );
    assert_eq!(
        weights.bf16("w", &[12]).unwrap(),
        [
            0, 0x8000, 0x3380, 0x3300, 0x33c0, 0x3f80, 0x3f80, 0x3f80, 0x3f82, 1, 0, 2
        ]
    );
}

#[test]
fn half_precision_inputs_decode_exactly() {
    for (dtype, bits, expected) in [
        (
            "F16",
            [0_u16, 0x8000, 1, 0x03ff, 0x0400, 0x3c00, 0xbc00],
            [
                0, 0x80000000, 0x33800000, 0x387fc000, 0x38800000, 0x3f800000, 0xbf800000,
            ],
        ),
        (
            "BF16",
            [0_u16, 0x8000, 1, 0x007f, 0x0080, 0x3f80, 0xbf80],
            [
                0, 0x80000000, 0x00010000, 0x007f0000, 0x00800000, 0x3f800000, 0xbf800000,
            ],
        ),
    ] {
        let bytes: Vec<_> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
        let (_dir, weights) = tensor(dtype, &[7], &bytes);
        let actual: Vec<_> = weights
            .f32("w", &[7])
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(actual, expected, "{dtype}");
    }
}

#[test]
fn tensor_shape_names_and_dtype_are_checked() {
    let (_dir, weights) = tensor("F32", &[2], &[0; 8]);
    weights.validate_names(["w"]).unwrap();
    assert!(weights.validate_names([]).is_err());
    assert!(weights.validate_names(["w", "extra"]).is_err());
    assert!(weights.validate_names(["w", "w"]).is_err());
    assert!(weights.f32("w", &[1, 2]).is_err());
    assert!(weights.f32("missing", &[2]).is_err());
    let (_dir, integers) = tensor("I64", &[1], &[0; 8]);
    assert!(integers.f32("w", &[1]).is_err());
    let dir = tempdir().unwrap();
    let path = dir.path().join("broken.safetensors");
    fs::write(&path, b"not a safetensors file").unwrap();
    assert!(Weights::open(&path).is_err());
}

#[test]
fn inventory_mismatch_names_the_tensors_that_differ() {
    let (_dir, weights) = tensor("F32", &[2], &[0; 8]);
    // The checkpoint holds exactly "w". expected = what the caller asked for,
    // actual = what the file has, so each direction has its own side.
    let message = weights
        .validate_names(["w", "absent"])
        .unwrap_err()
        .to_string();
    assert!(message.contains("missing [absent]"), "{message}");
    assert!(message.contains("unexpected []"), "{message}");
    // Asking for nothing leaves the file's tensor unaccounted for.
    let message = weights.validate_names([]).unwrap_err().to_string();
    assert!(message.contains("missing []"), "{message}");
    assert!(message.contains("unexpected [w]"), "{message}");
    let message = weights
        .validate_names(["w", "absent", "also_absent"])
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("missing [absent, also_absent]"),
        "{message}"
    );
}

#[test]
fn inventory_includes_gated_projection_and_legacy_buffer() {
    let tensors = omni_laya::weights::checkpoint_tensors();
    let names: std::collections::HashSet<_> = tensors.iter().map(|t| &t.name).collect();
    assert_eq!(tensors.len(), 206);
    assert_eq!(names.len(), tensors.len());
    let gated = tensors
        .iter()
        .find(|t| t.name == "encoder.layers.0.mlp.Wi.weight")
        .unwrap();
    assert_eq!(gated.shape, [5248, 1024]);
    let legacy = tensors.iter().find(|t| t.name == "temperature").unwrap();
    assert_eq!(legacy.shape, [3]);
}
