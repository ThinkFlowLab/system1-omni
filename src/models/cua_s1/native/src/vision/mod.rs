//! Structurally validated vision checkpoints and native CUDA vision execution.
//! `VisionCheckpoint` loads on CPU; `VisionModel` uploads separate base/LoRA tensors.
//!
//! Callers must keep checkpoint files immutable (including no truncation) for the
//! lifetime of the checkpoint and its borrowed views. Structural checks do not
//! verify upstream hashes or tensor values.

use anyhow::{Context, Result, ensure};
use omni_qwen3_5_native::vision::{TensorStore, UniqueMap, inside, is_visual};
use safetensors::{Dtype, tensor::TensorView};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const BASE: &str = "model.visual.";
const ADAPTER: &str = "base_model.model.model.visual.";
type Inventory = BTreeMap<String, Vec<usize>>;

/// The supported Qwen3.5-4B vision architecture; all fields are validated at load.
#[derive(Debug, Deserialize, Serialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub hidden_act: String,
    pub deepstack_visual_indexes: Vec<usize>,
    pub model_type: String,
}
impl VisionConfig {
    fn load(dir: &Path) -> Result<Self> {
        let config: Value = serde_json::from_slice(&fs::read(inside(dir, "config.json")?)?)?;
        let vision: Self =
            serde_json::from_value(config["vision_config"].clone()).context("vision config")?;
        let sizes = [
            vision.depth,
            vision.hidden_size,
            vision.intermediate_size,
            vision.num_heads,
            vision.num_position_embeddings,
            vision.out_hidden_size,
            vision.in_channels,
            vision.patch_size,
            vision.temporal_patch_size,
            vision.spatial_merge_size,
        ];
        ensure!(
            sizes == [24, 1024, 4096, 16, 2304, 2560, 3, 16, 2, 2]
                && vision.hidden_act == "gelu_pytorch_tanh"
                && vision.deepstack_visual_indexes.is_empty()
                && vision.model_type == "qwen3_5"
                && config["model_type"] == "qwen3_5"
                && config["text_config"]["hidden_size"].as_u64()
                    == Some(vision.out_hidden_size as u64),
            "unsupported vision/text config: expected pinned Qwen3.5-4B layout"
        );
        Ok(vision)
    }
    fn inventory(&self) -> Inventory {
        let mut tensors = Inventory::new();
        let h = self.hidden_size;
        let i = self.intermediate_size;
        let mut linear = |name: String, output: usize, input: Option<usize>| {
            tensors.insert(
                format!("{BASE}{name}.weight"),
                input.map_or_else(|| vec![output], |input| vec![output, input]),
            );
            tensors.insert(format!("{BASE}{name}.bias"), vec![output]);
        };
        for block in 0..self.depth {
            for norm in ["norm1", "norm2"] {
                linear(format!("blocks.{block}.{norm}"), h, None);
            }
            for (name, output, input) in [
                ("attn.qkv", 3 * h, h),
                ("attn.proj", h, h),
                ("mlp.linear_fc1", i, h),
                ("mlp.linear_fc2", h, i),
            ] {
                linear(format!("blocks.{block}.{name}"), output, Some(input));
            }
        }
        let merged = h * self.spatial_merge_size * self.spatial_merge_size;
        linear("merger.norm".into(), h, None);
        linear("merger.linear_fc1".into(), merged, Some(merged));
        linear(
            "merger.linear_fc2".into(),
            self.out_hidden_size,
            Some(merged),
        );
        tensors.insert(
            format!("{BASE}patch_embed.proj.weight"),
            vec![
                h,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ],
        );
        tensors.insert(format!("{BASE}patch_embed.proj.bias"), vec![h]);
        tensors.insert(
            format!("{BASE}pos_embed.weight"),
            vec![self.num_position_embeddings, h],
        );
        tensors
    }
}

/// Inference LoRA parameters; base and adapter bytes remain separate.
#[derive(Debug)]
pub struct AdapterConfig {
    pub rank: usize,
    pub alpha: usize,
}
impl AdapterConfig {
    pub fn scale(&self) -> f64 {
        self.alpha as f64 / self.rank as f64
    }
    fn load(dir: &Path) -> Result<Self> {
        let config: Value =
            serde_json::from_slice(&fs::read(inside(dir, "adapter_config.json")?)?)?;
        let c = config
            .as_object()
            .context("adapter config must be an object")?;
        for (key, value) in c {
            let supported = match key.as_str() {
                "r" => value == 16,
                "lora_alpha" => value == 32,
                "peft_type" => value == "LORA",
                "bias" => value == "none",
                "base_model_name_or_path" => value == "Qwen/Qwen3.5-4B",
                "task_type" => value == "CAUSAL_LM",
                "lora_bias"
                | "use_dora"
                | "use_rslora"
                | "use_qalora"
                | "fan_in_fan_out"
                | "ensure_weight_tying" => value == false,
                "rank_pattern" | "alpha_pattern" | "loftq_config" => {
                    value.as_object().is_some_and(|v| v.is_empty())
                }
                "exclude_modules"
                | "modules_to_save"
                | "layers_to_transform"
                | "layers_pattern"
                | "layer_replication"
                | "target_parameters"
                | "trainable_token_indices"
                | "alora_invocation_tokens"
                | "arrow_config"
                | "corda_config"
                | "eva_config"
                | "megatron_config" => value.is_null(),
                // Training/serialization metadata does not change ordinary inference LoRA.
                "auto_mapping" | "inference_mode" | "init_lora_weights" | "lora_dropout"
                | "megatron_core" | "peft_version" | "qalora_group_size" | "revision"
                | "target_modules" => true,
                _ => false,
            };
            ensure!(
                supported,
                "unsupported adapter config option {key}: {value}"
            );
        }
        for (key, value) in [
            ("r", Value::from(16)),
            ("lora_alpha", Value::from(32)),
            ("peft_type", Value::from("LORA")),
            ("bias", Value::from("none")),
        ] {
            ensure!(config[key] == value, "unsupported adapter config {key}");
        }
        let targets = config["target_modules"]
            .as_array()
            .context("adapter config target_modules must be an array")?;
        let expected = BTreeSet::from([
            "up_proj",
            "k_proj",
            "linear_fc1",
            "q_proj",
            "linear_fc2",
            "down_proj",
            "gate_proj",
            "o_proj",
            "v_proj",
        ]);
        let actual: BTreeSet<_> = targets.iter().filter_map(Value::as_str).collect();
        ensure!(
            actual == expected && targets.len() == expected.len(),
            "adapter config requires the full multimodal target_modules"
        );
        Ok(Self {
            rank: 16,
            alpha: 32,
        })
    }
    fn inventory(&self, base: &Inventory) -> Inventory {
        let mut tensors = Inventory::new();
        for (name, shape) in base {
            if name.ends_with(".weight")
                && (name.contains(".linear_fc1.") || name.contains(".linear_fc2."))
            {
                let module = name
                    .strip_prefix(BASE)
                    .unwrap()
                    .strip_suffix(".weight")
                    .unwrap();
                tensors.insert(
                    format!("{ADAPTER}{module}.lora_A.weight"),
                    vec![self.rank, shape[1]],
                );
                tensors.insert(
                    format!("{ADAPTER}{module}.lora_B.weight"),
                    vec![shape[0], self.rank],
                );
            }
        }
        tensors
    }
}

/// Immutable mmap storage for one base checkpoint and its multimodal adapter.
pub struct VisionCheckpoint {
    config: VisionConfig,
    adapter: AdapterConfig,
    base: TensorStore,
    lora: TensorStore,
}
impl VisionCheckpoint {
    /// Loads and validates headers on CPU. Keep the files immutable while mapped.
    pub fn load(base_dir: impl AsRef<Path>, adapter_dir: impl AsRef<Path>) -> Result<Self> {
        let base_dir = fs::canonicalize(base_dir).context("base checkpoint directory")?;
        let adapter_dir = fs::canonicalize(adapter_dir).context("adapter checkpoint directory")?;
        let config = VisionConfig::load(&base_dir).context("base config")?;
        let adapter = AdapterConfig::load(&adapter_dir).context("adapter config")?;
        let expected = config.inventory();
        let lora_expected = adapter.inventory(&expected);
        let index_path = base_dir.join("model.safetensors.index.json");
        let index = if index_path.try_exists()? {
            #[derive(Deserialize)]
            struct Index {
                weight_map: UniqueMap<String>,
            }
            let index: Index = serde_json::from_slice(&fs::read(inside(
                &base_dir,
                "model.safetensors.index.json",
            )?)?)
            .context("safetensors index")?;
            let map = index.weight_map.0;
            for name in map.keys().filter(|n| is_visual(n)) {
                ensure!(
                    expected.contains_key(name),
                    "unexpected visual tensor in index: {name}"
                );
            }
            for name in expected.keys() {
                ensure!(
                    map.contains_key(name),
                    "missing visual tensor in index: {name}"
                );
            }
            Some(map)
        } else {
            None
        };
        let files: BTreeSet<String> = match &index {
            Some(index) => expected.keys().map(|n| index[n].clone()).collect(),
            None => BTreeSet::from(["model.safetensors".into()]),
        };
        let base = TensorStore::load(&base_dir, files, &expected, Dtype::BF16, index.as_ref())?;
        let lora = TensorStore::load(
            &adapter_dir,
            BTreeSet::from(["adapter_model.safetensors".into()]),
            &lora_expected,
            Dtype::F32,
            None,
        )?;
        Ok(Self {
            config,
            adapter,
            base,
            lora,
        })
    }
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }
    pub fn adapter(&self) -> &AdapterConfig {
        &self.adapter
    }
    pub fn base_names(&self) -> impl Iterator<Item = &str> {
        self.base.names()
    }
    pub fn adapter_names(&self) -> impl Iterator<Item = &str> {
        self.lora.names()
    }
    pub fn base_tensor(&self, name: &str) -> Result<TensorView<'_>> {
        self.base.tensor(name)
    }
    pub fn adapter_tensor(&self, name: &str) -> Result<TensorView<'_>> {
        self.lora.tensor(name)
    }
}

mod model;
pub use model::VisionModel;

#[cfg(test)]
#[path = "../../../../../../tests/cua_s1/vision_checkpoint.rs"]
mod tests;
