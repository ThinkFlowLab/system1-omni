//! Structurally validated vision checkpoints and native CUDA vision execution.
//! `VisionCheckpoint` loads on CPU; `VisionModel` uploads separate base/LoRA tensors.
//!
//! Callers must keep checkpoint files immutable (including no truncation) for the
//! lifetime of the checkpoint and its borrowed views. Structural checks do not
//! verify upstream hashes or tensor values.

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorInfo, TensorView},
};
use serde::{
    Deserialize,
    de::{self, MapAccess, Visitor},
};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File},
    path::{Component, Path, PathBuf},
};

const BASE: &str = "model.visual.";
const ADAPTER: &str = "base_model.model.model.visual.";
type Inventory = BTreeMap<String, Vec<usize>>;

/// The supported Qwen3.5-4B vision architecture; all fields are validated at load.
#[derive(Debug, Deserialize)]
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
        self.base.tensors.keys().map(String::as_str)
    }
    pub fn adapter_names(&self) -> impl Iterator<Item = &str> {
        self.lora.tensors.keys().map(String::as_str)
    }
    pub fn base_tensor(&self, name: &str) -> Result<TensorView<'_>> {
        self.base.tensor(name)
    }
    pub fn adapter_tensor(&self, name: &str) -> Result<TensorView<'_>> {
        self.lora.tensor(name)
    }
}

struct TensorStore {
    maps: Vec<Mmap>,
    tensors: BTreeMap<String, (usize, TensorInfo)>,
}
impl TensorStore {
    fn load(
        dir: &Path,
        files: BTreeSet<String>,
        expected: &Inventory,
        dtype: Dtype,
        index: Option<&BTreeMap<String, String>>,
    ) -> Result<Self> {
        let mut store = Self {
            maps: Vec::new(),
            tensors: BTreeMap::new(),
        };
        for filename in files {
            let path = inside(dir, &filename)?;
            let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
            // SAFETY: callers must not mutate or truncate checkpoint files while mapped.
            let map = unsafe { Mmap::map(&file) }
                .with_context(|| format!("mmap safetensors {}", path.display()))?;
            validate_header(&map)
                .with_context(|| format!("safetensors header {}", path.display()))?;
            let (header_len, metadata) = SafeTensors::read_metadata(&map)
                .with_context(|| format!("safetensors {}", path.display()))?;
            for (name, info) in metadata.tensors() {
                if !is_visual(&name) {
                    continue;
                }
                let shape = expected
                    .get(&name)
                    .with_context(|| format!("unexpected visual tensor {name} in {filename}"))?;
                if let Some(index) = index {
                    ensure!(
                        index.get(&name) == Some(&filename),
                        "index mismatch for {name} in {filename}"
                    );
                }
                ensure!(
                    &info.shape == shape,
                    "shape mismatch for {name}: {:?}, expected {shape:?}",
                    info.shape
                );
                ensure!(
                    info.dtype == dtype,
                    "dtype mismatch for {name}: {:?}, expected {dtype:?}",
                    info.dtype
                );
                let mut info = info.clone();
                info.data_offsets.0 += 8 + header_len;
                info.data_offsets.1 += 8 + header_len;
                ensure!(
                    store
                        .tensors
                        .insert(name.clone(), (store.maps.len(), info))
                        .is_none(),
                    "duplicate visual tensor {name}"
                );
            }
            store.maps.push(map);
        }
        for name in expected.keys() {
            ensure!(
                store.tensors.contains_key(name),
                "missing visual tensor {name}"
            );
        }
        Ok(store)
    }
    fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let (shard, info) = self
            .tensors
            .get(name)
            .with_context(|| format!("unknown visual tensor {name}"))?;
        Ok(TensorView::new(
            info.dtype,
            info.shape.clone(),
            &self.maps[*shard][info.data_offsets.0..info.data_offsets.1],
        )?)
    }
}
// Bound all offsets before safetensors 0.8 adds payload size to header size:
// its final length check uses unchecked addition, even for ignored language tensors.
fn validate_header(bytes: &[u8]) -> Result<()> {
    let length_bytes = bytes.get(..8).context("missing header length")?;
    let header_len = usize::try_from(u64::from_le_bytes(length_bytes.try_into()?))?;
    // Match safetensors 0.8's header allocation limit.
    ensure!(header_len <= 100_000_000, "header too large");
    let data_start = header_len
        .checked_add(8)
        .context("header length overflow")?;
    let header = bytes.get(8..data_start).context("truncated header")?;
    let payload_len = bytes.len() - data_start;
    let entries: UniqueMap<Value> = serde_json::from_slice(header)?;
    for (name, entry) in entries.0 {
        if name == "__metadata__" {
            continue;
        }
        let (start, end): (usize, usize) = serde_json::from_value(entry["data_offsets"].clone())
            .with_context(|| format!("invalid offsets for {name}"))?;
        ensure!(
            start <= end && end <= payload_len,
            "tensor offsets exceed payload for {name}"
        );
    }
    Ok(())
}
fn is_visual(name: &str) -> bool {
    name.split('.').any(|part| part == "visual")
}
fn inside(dir: &Path, name: &str) -> Result<PathBuf> {
    ensure!(
        !name.is_empty()
            && Path::new(name)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
        "path must remain inside checkpoint directory: {name}"
    );
    let resolved =
        fs::canonicalize(dir.join(name)).with_context(|| format!("checkpoint file {name}"))?;
    ensure!(
        resolved.starts_with(dir),
        "path escapes checkpoint directory: {name}"
    );
    Ok(resolved)
}

// serde_json's ordinary maps overwrite repeated keys. Reject ambiguous headers/indexes.
struct UniqueMap<T>(BTreeMap<String, T>);
impl<'de, T: Deserialize<'de>> Deserialize<'de> for UniqueMap<T> {
    fn deserialize<D: de::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct UniqueVisitor<T>(std::marker::PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for UniqueVisitor<T> {
            type Value = UniqueMap<T>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object with unique names")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut entries = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, T>()? {
                    if entries.insert(key.clone(), value).is_some() {
                        return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
                    }
                }
                Ok(UniqueMap(entries))
            }
        }
        deserializer.deserialize_map(UniqueVisitor(std::marker::PhantomData))
    }
}

mod geometry;
mod model;
pub use model::VisionModel;
