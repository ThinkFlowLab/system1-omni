use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Deserialize)]
pub struct AgentConfig {
    pub max_len: usize,
    pub head_max_len: usize,
    pub head_layers: usize,
    pub temperature: Vec<f32>,
    pub temperature_by_options: HashMap<String, f32>,
}

#[derive(Debug, Deserialize)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub norm_eps: f32,
    pub local_attention: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: serde_json::Value,
}

pub struct Config {
    pub agent: AgentConfig,
    pub encoder: EncoderConfig,
}
impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name| fs::read(dir.join(name)).with_context(|| format!("read {name}"));
        let agent: AgentConfig = serde_json::from_slice(&read("rl_agent_config.json")?)?;
        let encoder_bytes = read("encoder/config.json")?;
        let raw: serde_json::Value = serde_json::from_slice(&encoder_bytes)?;
        ensure!(
            raw["model_type"] == "modernbert"
                && raw["hidden_activation"] == "gelu"
                && raw["attention_bias"] == false
                && raw["mlp_bias"] == false
                && raw["norm_bias"] == false,
            "unsupported encoder activation, bias or model type"
        );
        let encoder: EncoderConfig = serde_json::from_slice(&encoder_bytes)?;
        ensure!(
            agent.max_len == 512 && agent.head_max_len == 192 && agent.head_layers == 2,
            "native Laya supports max_len=512, head_max_len=192, head_layers=2"
        );
        ensure!(
            encoder.hidden_size == 1024
                && encoder.intermediate_size == 2624
                && encoder.num_attention_heads == 16
                && encoder.num_hidden_layers == 28
                && encoder.vocab_size == 50368
                && encoder.local_attention == 128
                && encoder.norm_eps == 1e-5,
            "unsupported encoder configuration"
        );
        let expected: Vec<_> = (0..28)
            .map(|i| {
                if i % 3 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        ensure!(
            encoder.layer_types == expected,
            "unsupported attention schedule"
        );
        for (kind, theta) in [("full_attention", 160000.0), ("sliding_attention", 10000.0)] {
            let r = &encoder.rope_parameters[kind];
            ensure!(
                r["rope_type"] == "default" && r["rope_theta"].as_f64() == Some(theta),
                "unsupported RoPE configuration"
            );
        }
        ensure!(
            agent.temperature.len() == 3
                && agent
                    .temperature
                    .iter()
                    .chain(agent.temperature_by_options.values())
                    .all(|t| t.is_finite() && *t > 0.0),
            "invalid temperatures"
        );
        Ok(Self { agent, encoder })
    }
}
