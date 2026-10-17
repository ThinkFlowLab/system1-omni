use super::*;
use omni_qwen3_5_native::vision::VisionCheckpoint as QwenCheckpoint;
use safetensors::tensor::TensorInfo;
use serde_json::json;
use std::{
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

type Tensors = BTreeMap<String, (Dtype, Vec<usize>)>;
static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    root: PathBuf,
    base: PathBuf,
    adapter: PathBuf,
    tensors: Tensors,
    lora: Tensors,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "omni-vision-checkpoint-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let base = root.join("base");
        let adapter = root.join("adapter");
        fs::create_dir_all(&base).unwrap();
        fs::create_dir_all(&adapter).unwrap();
        let root = fs::canonicalize(root).unwrap();
        let base = fs::canonicalize(base).unwrap();
        let adapter = fs::canonicalize(adapter).unwrap();
        let vision = json!({
            "depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
            "num_heads": 16, "num_position_embeddings": 2304, "out_hidden_size": 2560,
            "in_channels": 3, "patch_size": 16, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "hidden_act": "gelu_pytorch_tanh",
            "deepstack_visual_indexes": [], "model_type": "qwen3_5"
        });
        fs::write(
            base.join("config.json"),
            json!({"model_type": "qwen3_5", "text_config": {"hidden_size": 2560},
                "vision_config": vision})
            .to_string(),
        )
        .unwrap();
        fs::write(
            adapter.join("adapter_config.json"),
            json!({"r": 16, "lora_alpha": 32, "peft_type": "LORA", "bias": "none",
                "target_modules": ["up_proj", "k_proj", "linear_fc1", "q_proj", "linear_fc2",
                    "down_proj", "gate_proj", "o_proj", "v_proj"]})
            .to_string(),
        )
        .unwrap();
        let inventory = VisionConfig::load(&base).unwrap().inventory();
        let mut tensors = tensor_specs(&inventory, Dtype::BF16);
        tensors.insert("model.language.weight".into(), (Dtype::U8, vec![1]));
        let mut lora = tensor_specs(
            &AdapterConfig::load(&adapter).unwrap().inventory(&inventory),
            Dtype::F32,
        );
        lora.insert(
            "base_model.model.model.language.lora_A.weight".into(),
            (Dtype::U8, vec![1]),
        );
        write_tensors(&base.join("model.safetensors"), &tensors);
        fs::hard_link(
            base.join("model.safetensors"),
            base.join("vision.safetensors"),
        )
        .unwrap();
        write_tensors(&adapter.join("adapter_model.safetensors"), &lora);
        Self {
            root,
            base,
            adapter,
            tensors,
            lora,
        }
    }
    fn both_reject(&self, message: &str) {
        assert_error(QwenCheckpoint::load(&self.base), message);
        self.cua_reject(message);
    }
    fn cua_reject(&self, message: &str) {
        assert_error(VisionCheckpoint::load(&self.base, &self.adapter), message);
    }
    fn index(&self, map: &BTreeMap<String, String>) {
        fs::write(
            self.base.join("model.safetensors.index.json"),
            json!({"weight_map": map}).to_string(),
        )
        .unwrap();
    }
    fn weight_map(&self) -> BTreeMap<String, String> {
        self.tensors
            .keys()
            .filter(|name| is_visual(name))
            .map(|name| (name.clone(), "model.safetensors".into()))
            .collect()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}
fn tensor_specs(inventory: &Inventory, dtype: Dtype) -> Tensors {
    inventory
        .iter()
        .map(|(name, shape)| (name.clone(), (dtype, shape.clone())))
        .collect()
}
fn write_tensors(path: &Path, tensors: &Tensors) {
    let mut offset = 0;
    let header: BTreeMap<_, _> = tensors
        .iter()
        .map(|(name, (dtype, shape))| {
            let start = offset;
            offset += shape.iter().product::<usize>() * dtype.bitsize() / 8;
            (
                name,
                TensorInfo {
                    dtype: *dtype,
                    shape: shape.clone(),
                    data_offsets: (start, offset),
                },
            )
        })
        .collect();
    write_header(path, &serde_json::to_string(&header).unwrap(), offset);
}
fn write_header(path: &Path, header: &str, payload_len: usize) {
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    // Sparse zero payloads exercise full inventories without allocating model weights.
    file.set_len((8 + header.len() + payload_len) as u64)
        .unwrap();
}
fn write_marker(path: &Path, name: &str, marker: &[u8]) {
    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut length = [0; 8];
    file.read_exact(&mut length).unwrap();
    let mut header = vec![0; u64::from_le_bytes(length) as usize];
    file.read_exact(&mut header).unwrap();
    let tensors: BTreeMap<String, TensorInfo> = serde_json::from_slice(&header).unwrap();
    file.seek(SeekFrom::Start(
        (8 + header.len() + tensors[name].data_offsets.0) as u64,
    ))
    .unwrap();
    file.write_all(marker).unwrap();
}
fn assert_error<T>(result: Result<T>, message: &str) {
    let error = match result {
        Ok(_) => panic!("expected {message}"),
        Err(error) => format!("{error:#}"),
    };
    assert!(error.contains(message), "expected {message:?}, got {error}");
}

#[test]
fn both_checkpoint_entries_keep_base_and_visual_lora_views_separate() {
    let f = Fixture::new();
    let name = "model.visual.blocks.0.mlp.linear_fc1.weight";
    let second_name = "model.visual.patch_embed.proj.bias";
    let lora_name = "base_model.model.model.visual.blocks.0.mlp.linear_fc1.lora_A.weight";
    write_marker(&f.base.join("model.safetensors"), name, &[1, 2]);
    write_marker(&f.base.join("model.safetensors"), second_name, &[3, 4]);
    write_marker(
        &f.adapter.join("adapter_model.safetensors"),
        lora_name,
        &[5, 6, 7, 8],
    );
    let qwen = QwenCheckpoint::load(&f.base).unwrap();
    let unindexed = VisionCheckpoint::load(&f.base, &f.adapter).unwrap();
    let mut second = f.tensors.clone();
    let first = BTreeMap::from([(name.into(), second.remove(name).unwrap())]);
    write_tensors(&f.base.join("first.safetensors"), &first);
    write_tensors(&f.base.join("second.safetensors"), &second);
    write_marker(&f.base.join("first.safetensors"), name, &[1, 2]);
    write_marker(&f.base.join("second.safetensors"), second_name, &[3, 4]);
    let map = f
        .weight_map()
        .into_keys()
        .map(|tensor| {
            let shard = if tensor == name {
                "first.safetensors"
            } else {
                "second.safetensors"
            };
            (tensor, shard.into())
        })
        .collect();
    f.index(&map);
    let cua = VisionCheckpoint::load(&f.base, &f.adapter).unwrap();
    assert_eq!(
        qwen.base_names().collect::<Vec<_>>(),
        cua.base_names().collect::<Vec<_>>()
    );
    assert_eq!(cua.base_names().count(), 297);
    assert_eq!(cua.adapter_names().count(), 100);
    for view in [
        qwen.base_tensor(name).unwrap(),
        unindexed.base_tensor(name).unwrap(),
        cua.base_tensor(name).unwrap(),
    ] {
        assert_eq!(view.dtype(), Dtype::BF16);
        assert_eq!(view.shape(), [4096, 1024]);
        assert_eq!(&view.data()[..2], [1, 2]);
    }
    assert_eq!(&cua.base_tensor(second_name).unwrap().data()[..2], [3, 4]);
    let view = cua.adapter_tensor(lora_name).unwrap();
    assert_eq!(view.dtype(), Dtype::F32);
    assert_eq!(view.shape(), [16, 1024]);
    assert_eq!(&view.data()[..4], [5, 6, 7, 8]);
    assert_eq!(cua.adapter().scale(), 2.0);
    assert_error(
        qwen.base_tensor("model.language.weight"),
        "unknown visual tensor",
    );
    assert_error(
        cua.adapter_tensor("base_model.model.model.language.lora_A.weight"),
        "unknown visual tensor",
    );
}

#[test]
fn common_header_checks_reject_malformed_lengths_and_json() {
    let f = Fixture::new();
    for (bytes, message) in [
        (vec![0; 7], "missing header length"),
        (100_000_001_u64.to_le_bytes().to_vec(), "header too large"),
        (9_u64.to_le_bytes().to_vec(), "truncated header"),
    ] {
        fs::write(f.base.join("model.safetensors"), bytes).unwrap();
        f.both_reject(message);
    }
    write_header(&f.base.join("model.safetensors"), "{", 0);
    f.both_reject("EOF while parsing");
    write_header(
        &f.base.join("model.safetensors"),
        r#"{"language":{"data_offsets":[0,0]},"language":{"data_offsets":[0,0]}}"#,
        0,
    );
    f.both_reject("duplicate JSON key: language");
}

#[test]
fn ignored_language_offsets_are_checked_before_safetensors_arithmetic() {
    let f = Fixture::new();
    for offsets in [[1, 0], [0, 2], [0, usize::MAX]] {
        let header = json!({"model.language.weight": {"dtype": "U8", "shape": [1], "data_offsets": offsets}});
        write_header(&f.base.join("model.safetensors"), &header.to_string(), 1);
        f.both_reject("tensor offsets exceed payload for model.language.weight");
    }
    write_header(&f.adapter.join("adapter_model.safetensors"),
        &json!({"base_model.model.model.language.lora_A.weight": {"data_offsets": [0, usize::MAX]}}).to_string(), 0);
    write_tensors(&f.base.join("model.safetensors"), &f.tensors);
    f.cua_reject("tensor offsets exceed payload");
}

#[test]
fn both_base_entries_require_exact_dtype_shape_and_visual_inventory() {
    let f = Fixture::new();
    let name = "model.visual.patch_embed.proj.bias";
    let mut wrong = f.tensors.clone();
    wrong.get_mut(name).unwrap().0 = Dtype::F32;
    write_tensors(&f.base.join("model.safetensors"), &wrong);
    f.both_reject("dtype mismatch");
    wrong = f.tensors.clone();
    wrong.get_mut(name).unwrap().1 = vec![1023];
    write_tensors(&f.base.join("model.safetensors"), &wrong);
    f.both_reject("shape mismatch");
    wrong = f.tensors.clone();
    wrong.remove(name);
    write_tensors(&f.base.join("model.safetensors"), &wrong);
    f.both_reject("missing visual tensor");
    wrong = f.tensors.clone();
    wrong.insert("other.visual.weight".into(), (Dtype::BF16, vec![1]));
    write_tensors(&f.base.join("model.safetensors"), &wrong);
    f.both_reject("unexpected visual tensor");
}

#[test]
fn cua_lora_requires_exact_f32_shape_and_visual_inventory() {
    let f = Fixture::new();
    let name = "base_model.model.model.visual.blocks.0.mlp.linear_fc1.lora_A.weight";
    let mut wrong = f.lora.clone();
    wrong.get_mut(name).unwrap().0 = Dtype::BF16;
    write_tensors(&f.adapter.join("adapter_model.safetensors"), &wrong);
    f.cua_reject("dtype mismatch");
    wrong = f.lora.clone();
    wrong.get_mut(name).unwrap().1 = vec![16, 1023];
    write_tensors(&f.adapter.join("adapter_model.safetensors"), &wrong);
    f.cua_reject("shape mismatch");
    wrong = f.lora.clone();
    wrong.remove(name);
    write_tensors(&f.adapter.join("adapter_model.safetensors"), &wrong);
    f.cua_reject("missing visual tensor");
    wrong = f.lora.clone();
    wrong.insert("other.visual.lora_A.weight".into(), (Dtype::F32, vec![1]));
    write_tensors(&f.adapter.join("adapter_model.safetensors"), &wrong);
    f.cua_reject("unexpected visual tensor");
}

#[test]
fn cua_index_rejects_duplicates_inventory_mismatch_and_escaping_shards() {
    let f = Fixture::new();
    let name = "model.visual.patch_embed.proj.bias";
    fs::write(
        f.base.join("model.safetensors.index.json"),
        format!(
            r#"{{"weight_map":{{"{name}":"model.safetensors","{name}":"other.safetensors"}}}}"#
        ),
    )
    .unwrap();
    f.cua_reject("duplicate JSON key");
    let mut map = f.weight_map();
    map.remove(name);
    f.index(&map);
    f.cua_reject("missing visual tensor in index");
    map = f.weight_map();
    map.insert("other.visual.weight".into(), "model.safetensors".into());
    f.index(&map);
    f.cua_reject("unexpected visual tensor in index");
    map = f.weight_map();
    map.insert(name.into(), "other.safetensors".into());
    write_tensors(&f.base.join("other.safetensors"), &f.tensors);
    f.index(&map);
    f.cua_reject("index mismatch");
    for filename in ["../outside.safetensors", "/outside.safetensors"] {
        let map = f
            .weight_map()
            .into_keys()
            .map(|name| (name, filename.into()))
            .collect();
        f.index(&map);
        f.cua_reject("path must remain inside checkpoint directory");
    }
}

#[cfg(unix)]
#[test]
fn both_entries_reject_config_and_tensor_symlinks_outside_checkpoint() {
    use std::os::unix::fs::symlink;
    for config in [true, false] {
        let f = Fixture::new();
        let filename = if config {
            "config.json"
        } else {
            "model.safetensors"
        };
        let outside = f.root.join("outside");
        fs::rename(f.base.join(filename), &outside).unwrap();
        symlink(&outside, f.base.join(filename)).unwrap();
        if !config {
            fs::remove_file(f.base.join("vision.safetensors")).unwrap();
            symlink(&outside, f.base.join("vision.safetensors")).unwrap();
        }
        f.both_reject("path escapes checkpoint directory");
    }
    let f = Fixture::new();
    let outside = f.root.join("outside");
    fs::rename(f.adapter.join("adapter_model.safetensors"), &outside).unwrap();
    symlink(&outside, f.adapter.join("adapter_model.safetensors")).unwrap();
    f.cua_reject("path escapes checkpoint directory");
}

#[test]
fn qwen_27b_layout_does_not_expand_cua_4b_support() {
    let f = Fixture::new();
    let config_path = f.base.join("config.json");
    let mut config: Value = serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
    for (key, value) in [
        ("depth", 27),
        ("hidden_size", 1152),
        ("intermediate_size", 4304),
        ("out_hidden_size", 5120),
    ] {
        config["vision_config"][key] = json!(value);
    }
    config["text_config"]["hidden_size"] = json!(5120);
    let vision =
        omni_qwen3_5_native::vision::VisionConfig::from_value(config["vision_config"].clone())
            .unwrap();
    fs::write(config_path, config.to_string()).unwrap();
    write_tensors(
        &f.base.join("vision.safetensors"),
        &tensor_specs(&vision.inventory(), Dtype::BF16),
    );
    assert_eq!(
        QwenCheckpoint::load(&f.base).unwrap().config().head_dim(),
        72
    );
    f.cua_reject("expected pinned Qwen3.5-4B layout");
}
