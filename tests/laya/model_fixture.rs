use super::*;
use serde_json::{Value, json};
use tempfile::tempdir;
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

pub(super) fn fixture_model_at(original_rope: bool, path: &Path) -> Model {
    let cuda = unsafe { Cuda::load(path) }.unwrap();
    let mut weights = HashMap::new();
    for spec in crate::weights::checkpoint_tensors() {
        let width = if storage_dtype(&spec.name) == "f32" {
            4
        } else {
            2
        };
        weights.insert(
            spec.name,
            cuda.alloc(spec.shape.iter().product::<usize>() * width)
                .unwrap(),
        );
    }
    for n in [D, 3 * D, 4 * D] {
        weights.insert(format!("zeros.{n}"), cuda.alloc(n * 4).unwrap());
    }
    for kind in ["full", "local"] {
        for part in ["cos", "sin"] {
            weights.insert(
                format!("rope_{kind}_{part}"),
                cuda.alloc(512 * 32 * 4).unwrap(),
            );
        }
    }
    let (encoder, agent) = configs();
    let plan = prepare_encoder(&cuda, &weights, original_rope).unwrap();
    Model {
        config: load_config(&encoder, &agent).unwrap(),
        blas: Blas::new(&cuda).unwrap(),
        cuda,
        weights,
        plan,
        cache: VecDeque::new(),
        cached_bytes: 0,
        cache_config: CacheConfig::default(),
        eager: None,
        graphs: false,
    }
}
