//! Configurable shared Qwen vision execution. Checkpoint validation is CPU-only.
use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorInfo, TensorView},
};
use serde::{
    Deserialize, Serialize,
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
type Inventory = BTreeMap<String, Vec<usize>>;

/// Exact supported Qwen3.5-4B and Qwen3.8-27B vision configurations.
#[derive(Clone, Debug, Deserialize, Serialize)]
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
    pub fn from_value(value: Value) -> Result<Self> {
        let vision: Self = serde_json::from_value(value).context("vision config")?;
        vision.validate()?;
        Ok(vision)
    }
    pub fn validate(&self) -> Result<()> {
        let sizes = [
            self.depth,
            self.hidden_size,
            self.intermediate_size,
            self.num_heads,
            self.num_position_embeddings,
            self.out_hidden_size,
            self.in_channels,
            self.patch_size,
            self.temporal_patch_size,
            self.spatial_merge_size,
        ];
        ensure!(
            (sizes == [24, 1024, 4096, 16, 2304, 2560, 3, 16, 2, 2]
                || sizes == [27, 1152, 4304, 16, 2304, 5120, 3, 16, 2, 2])
                && self.hidden_act == "gelu_pytorch_tanh"
                && self.deepstack_visual_indexes.is_empty()
                && self.model_type == "qwen3_5",
            "unsupported Qwen vision layout"
        );
        Ok(())
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
    pub fn inventory(&self) -> Inventory {
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

/// Structurally checked normalized base vision export. Files must stay immutable while mapped.
pub struct VisionCheckpoint {
    config: VisionConfig,
    base: TensorStore,
}
impl VisionCheckpoint {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = fs::canonicalize(dir).context("vision export directory")?;
        let config: Value = serde_json::from_slice(&fs::read(inside(&dir, "config.json")?)?)?;
        let vision = VisionConfig::from_value(config["vision_config"].clone())?;
        ensure!(
            config["model_type"] == "qwen3_5"
                && config["text_config"]["hidden_size"].as_u64()
                    == Some(vision.out_hidden_size as u64),
            "vision/text config mismatch"
        );
        let base = TensorStore::load(
            &dir,
            BTreeSet::from(["vision.safetensors".into()]),
            &vision.inventory(),
            Dtype::BF16,
            None,
        )?;
        Ok(Self {
            config: vision,
            base,
        })
    }
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }
    pub fn base_names(&self) -> impl Iterator<Item = &str> {
        self.base.tensors.keys().map(String::as_str)
    }
    pub fn base_tensor(&self, name: &str) -> Result<TensorView<'_>> {
        self.base.tensor(name)
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
pub use geometry::VisionGeometry;
pub use model::VisionModel;
