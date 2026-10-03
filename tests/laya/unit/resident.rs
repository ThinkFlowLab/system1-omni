use super::*;
use libloading::Library;
use safetensors::{Dtype, tensor::TensorView};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf, process::Command};
use tempfile::{TempDir, tempdir};

const EMBED: &str = "encoder.embeddings.tok_embeddings.weight";

struct Fixture {
    dir: TempDir,
    library: Library,
    cuda: Cuda,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempdir().unwrap();
        let path = dir
            .path()
            .join(format!("runtime{}", std::env::consts::DLL_SUFFIX));
        let output = Command::new("cc")
            .args([
                if cfg!(target_os = "macos") {
                    "-dynamiclib"
                } else {
                    "-shared"
                },
                "-fPIC",
            ])
            .arg(
                PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../tests/backends/cuda/fixtures/runtime.c"),
            )
            .arg("-o")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let library = unsafe { Library::new(&path) }.unwrap();
        let cuda = unsafe { Cuda::load(&path, 0) }.unwrap();
        Self { dir, library, cuda }
    }

    fn live(&self) -> i32 {
        unsafe {
            self.library
                .get::<unsafe extern "C" fn() -> i32>(b"laya_test_live\0")
                .unwrap()()
        }
    }

    fn source(&self, specs: &[TensorSpec]) -> Weights {
        let values: Vec<u8> = [1.0f32, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let tensors: Vec<_> = specs
            .iter()
            .map(|s| {
                (
                    s.name.as_str(),
                    TensorView::new(Dtype::F32, s.shape.clone(), &values).unwrap(),
                )
            })
            .collect();
        let path = self.dir.path().join("weights.safetensors");
        safetensors::tensor::serialize_to_file(tensors, None, &path).unwrap();
        Weights::open(&path).unwrap()
    }
}

fn spec(name: &str, shape: &[usize]) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        shape: shape.into(),
    }
}

#[test]
fn precision_and_residency_preserve_expected_bytes() {
    let fixture = Fixture::new();
    let specs = [
        spec(EMBED, &[1, 2]),
        spec("encoder.layers.0.mlp_norm.weight", &[2]),
        spec("encoder.layers.0.attn.Wqkv.weight", &[1, 2]),
        spec("head.layers.0.norm1.weight", &[2]),
        spec("head.layers.0.self_attn.in_proj_bias", &[2]),
        spec("head.layers.0.self_attn.in_proj_weight", &[1, 2]),
        spec("scorer.0.bias", &[2]),
        spec("scorer.1.bias", &[2]),
        spec("act_head.0.bias", &[2]),
        spec("type_emb.weight", &[1, 2]),
    ];
    let source = fixture.source(&specs);
    let weights = ResidentWeights::upload_tensors(&fixture.cuda, &source, &specs).unwrap();
    drop(source);
    let f16 = vec![0x00, 0x3c, 0x00, 0xc0];
    let bf16 = vec![0x80, 0x3f, 0x00, 0xc0];
    let f32 = vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0];
    let expected = [
        &f16, &f32, &bf16, &f32, &f32, &bf16, &f32, &bf16, &bf16, &bf16,
    ];
    for (s, bytes) in specs.iter().zip(expected) {
        let buffer = weights.get(&s.name).unwrap();
        assert_eq!(&buffer.read(buffer.bytes()).unwrap(), bytes, "{}", s.name);
    }
    assert_eq!(weights.bytes(), 56);
    assert!(weights.get("missing").is_err());
    assert_eq!(fixture.live(), 11);
    drop(fixture.cuda);
    assert_eq!(weights.get(EMBED).unwrap().read(4).unwrap(), f16);
    drop(weights);
    assert_eq!(
        unsafe {
            fixture
                .library
                .get::<unsafe extern "C" fn() -> i32>(b"laya_test_live\0")
                .unwrap()()
        },
        0
    );
}

#[test]
fn invalid_checkpoint_and_partial_load_release_allocations() {
    let fixture = Fixture::new();
    let mut specs = [
        spec(EMBED, &[1, 2]),
        spec("head.layers.0.norm1.weight", &[2]),
    ];
    let source = fixture.source(&specs);
    assert!(ResidentWeights::upload(&fixture.cuda, &source).is_err());
    assert_eq!(fixture.live(), 1);
    specs[1].shape = vec![3];
    let error = ResidentWeights::upload_tensors(&fixture.cuda, &source, &specs)
        .err()
        .unwrap();
    assert!(error.to_string().contains("head.layers.0.norm1.weight"));
    assert_eq!(fixture.live(), 1);
    specs[1].shape = vec![2];
    unsafe {
        fixture
            .library
            .get::<unsafe extern "C" fn(i32)>(b"laya_test_mode\0")
            .unwrap()(2);
    }
    assert!(ResidentWeights::upload_tensors(&fixture.cuda, &source, &specs).is_err());
    assert_eq!(fixture.live(), 1);
}

#[test]
#[ignore = "requires approved GPU, LAYA_CUDA_LIBRARY, LAYA_CUDA_DEVICE, LAYA_CHECKPOINT and LAYA_WEIGHT_ORACLE"]
fn real_checkpoint_residency_matches_torch() {
    let library = PathBuf::from(std::env::var_os("LAYA_CUDA_LIBRARY").unwrap());
    let device = std::env::var("LAYA_CUDA_DEVICE").unwrap().parse().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os("LAYA_CHECKPOINT").unwrap());
    let rows: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(std::env::var_os("LAYA_WEIGHT_ORACLE").unwrap()).unwrap())
            .unwrap();
    let cuda = unsafe { Cuda::load(&library, device) }.unwrap();
    let source = Weights::open(&checkpoint.join("model.safetensors")).unwrap();
    let weights = ResidentWeights::upload(&cuda, &source).unwrap();
    drop(source);
    drop(cuda);
    assert_eq!(rows.len(), 206);
    assert_eq!(weights.buffers.len(), 205);
    assert!(weights.get("temperature").is_err());
    let mut names = std::collections::HashSet::new();
    let mut bytes = 0;
    for row in rows {
        let name = row["name"].as_str().unwrap();
        assert!(names.insert(name.to_owned()));
        if name == "temperature" {
            continue;
        }
        let buffer = weights.get(name).unwrap();
        let hash = format!("{:x}", Sha256::digest(buffer.read(buffer.bytes()).unwrap()));
        // The oracle contains independent Torch conversions at all three precisions.
        let shape = row["shape"].as_array().unwrap();
        let dtype = if name == EMBED {
            "f16"
        } else if name.starts_with("scorer.0.")
            || (shape.len() == 1 && (name.starts_with("encoder.") || name.starts_with("head.")))
        {
            "f32"
        } else {
            "bf16"
        };
        assert_eq!(hash, row[dtype].as_str().unwrap(), "{name} {dtype}");
        bytes += buffer.bytes();
    }
    assert_eq!(weights.bytes(), bytes);
    println!("verified_resident_tensors=205 weight_allocation_bytes={bytes}");
}
