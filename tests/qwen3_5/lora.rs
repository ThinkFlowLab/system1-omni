use super::*;
use safetensors::{
    Dtype,
    tensor::{TensorView, serialize},
};

fn config() -> Config {
    Config::load(Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../tests/qwen3_5/data"
    )))
    .unwrap()
}
fn tensors(values: &[(&str, Vec<usize>, Dtype, f32)]) -> Vec<u8> {
    let data: Vec<Vec<u8>> = values
        .iter()
        .map(|(_, shape, dtype, value)| {
            let count: usize = shape.iter().product();
            match dtype {
                Dtype::F32 => value.to_le_bytes().repeat(count),
                Dtype::BF16 => half::bf16::from_f32(*value).to_le_bytes().repeat(count),
                _ => unreachable!(),
            }
        })
        .collect();
    let views: Vec<_> = values
        .iter()
        .zip(&data)
        .map(|((name, shape, dtype, _), bytes)| {
            (
                *name,
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    serialize(views, None).unwrap()
}
fn pair(target: &str, rows: usize, cols: usize, rank: usize) -> Vec<u8> {
    tensors(&[
        (
            &format!("{PREFIX}{target}.lora_A.weight"),
            vec![rank, cols],
            Dtype::F32,
            0.125,
        ),
        (
            &format!("{PREFIX}{target}.lora_B.weight"),
            vec![rows, rank],
            Dtype::F32,
            -0.25,
        ),
    ])
}

#[test]
fn layouts_cover_both_mixers_and_the_small_gate_projections() {
    for (hidden, intermediate, heads, values) in [(5120, 17408, 24, 48), (2560, 9216, 16, 32)] {
        let mut cfg = config();
        cfg.hidden = hidden;
        cfg.intermediate = intermediate;
        cfg.heads = heads;
        cfg.lin_v_heads = values;
        let layouts = targets(&cfg).unwrap();
        let find = |name: &str| layouts.iter().find(|t| t.name == name).unwrap();
        let vd = values * 128;
        let kd = cfg.lin_k_heads * 128;
        let small = find("layers.0.linear_attn.in_proj_a");
        assert_eq!(
            (small.rows, small.cols, small.offset, small.stride),
            (
                values,
                hidden,
                2 * kd + 2 * vd + values,
                2 * kd + 2 * vd + 2 * values
            )
        );
        let q = find("layers.3.self_attn.q_proj");
        let k = find("layers.3.self_attn.k_proj");
        assert_eq!((q.rows, q.cols, q.offset), (heads * 256 * 2, hidden, 0));
        assert_eq!(k.offset, q.rows);
        let up = find("layers.0.mlp.up_proj");
        assert_eq!(
            (up.rows, up.cols, up.offset, up.stride),
            (intermediate, hidden, intermediate, 2 * intermediate)
        );
        assert_eq!(layouts.len(), 496);
        assert!(
            !layouts
                .iter()
                .any(|t| t.name == "layers.0.self_attn.q_proj")
        );
    }
}
#[test]
fn partial_pairs_keep_fp32_bytes_and_match_native_component_layouts() {
    let cfg = config();
    for (target, rows, cols) in [
        ("layers.0.linear_attn.in_proj_a", 48, 5120),
        ("layers.3.self_attn.q_proj", 12288, 5120),
        ("layers.0.mlp.down_proj", 5120, 17408),
    ] {
        let bytes = pair(target, rows, cols, 16);
        let parsed = parse(&cfg, &bytes, 2.0).unwrap();
        assert_eq!(
            parsed.base_width, 17408,
            "partial adapters must retain all base projection widths"
        );
        assert_eq!(parsed.pairs.len(), 1);
        let p = &parsed.pairs[0];
        assert_eq!((p.rank, p.target.rows, p.target.cols), (16, rows, cols));
        assert_eq!(&bytes[p.a.clone()][..4], &0.125f32.to_le_bytes());
        assert_eq!(&bytes[p.b.clone()][..4], &(-0.25f32).to_le_bytes());
    }
}
#[test]
fn rank_dtype_shapes_pairing_targets_and_nonfinite_values_are_rejected() {
    let cfg = config();
    for rank in [0, 129] {
        assert!(
            parse(
                &cfg,
                &pair("layers.0.linear_attn.in_proj_a", 48, 5120, rank),
                2.
            )
            .is_err()
        );
    }
    for scale in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(
            parse(
                &cfg,
                &pair("layers.0.linear_attn.in_proj_a", 48, 5120, 1),
                scale
            )
            .is_err()
        );
    }
    for (target, rows, cols) in [
        ("layers.0.linear_attn.in_proj_a", 49, 5120),
        ("layers.0.linear_attn.in_proj_a", 48, 5121),
        ("layers.0.self_attn.q_proj", 12288, 5120),
        ("layers.64.mlp.down_proj", 5120, 17408),
        ("layers.0.input_layernorm", 5120, 1),
    ] {
        assert!(parse(&cfg, &pair(target, rows, cols, 1), 2.).is_err());
    }
    let a = format!("{PREFIX}layers.0.linear_attn.in_proj_a.lora_A.weight");
    let b = format!("{PREFIX}layers.0.linear_attn.in_proj_a.lora_B.weight");
    assert!(parse(&cfg, &tensors(&[(&a, vec![1, 5120], Dtype::F32, 0.)]), 2.).is_err());
    assert!(parse(&cfg, &tensors(&[(&b, vec![48, 1], Dtype::F32, 0.)]), 2.).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for letter in [0, 1] {
            let vals = [
                (
                    &*a,
                    vec![1, 5120],
                    Dtype::F32,
                    if letter == 0 { bad } else { 0. },
                ),
                (
                    &*b,
                    vec![48, 1],
                    Dtype::F32,
                    if letter == 1 { bad } else { 0. },
                ),
            ];
            assert!(parse(&cfg, &tensors(&vals), 2.).is_err());
        }
    }
    assert!(
        parse(
            &cfg,
            &tensors(&[
                (&a, vec![1, 5120], Dtype::BF16, 0.),
                (&b, vec![48, 1], Dtype::F32, 0.)
            ]),
            2.
        )
        .is_err()
    );
    assert!(parse(&cfg, &tensors(&[]), 2.).is_err());
}
#[test]
fn optional_work_is_absent_without_adapters_and_checked_for_overflow() {
    assert!(WorkLayout::for_pairs(8192, &[]).unwrap().is_none());
    let bytes = pair("layers.0.linear_attn.in_proj_a", 48, 5120, 16);
    let parsed = parse(&config(), &bytes, 2.).unwrap();
    let work = WorkLayout::for_pairs(7, &parsed.pairs).unwrap().unwrap();
    assert_eq!(
        (work.input, work.rank, work.delta, work.gather),
        (7 * 5120 * 4, 7 * 16 * 4, 7 * 48 * 4, 7 * 48 * 2)
    );
    assert!(WorkLayout::for_pairs(usize::MAX, &parsed.pairs).is_err());
}

#[test]
fn base_components_match_original_27b_and_4b_projection_shapes() {
    for (hidden, intermediate, heads, values, linear, attention) in [
        (
            5120,
            17408,
            24,
            48,
            [10240, 6144, 48, 48],
            [12288, 1024, 1024, 0],
        ),
        (
            2560,
            9216,
            16,
            32,
            [8192, 4096, 32, 32],
            [8192, 1024, 1024, 0],
        ),
    ] {
        let mut cfg = config();
        cfg.hidden = hidden;
        cfg.intermediate = intermediate;
        cfg.heads = heads;
        cfg.lin_v_heads = values;
        assert_eq!(component_rows(&cfg, Group::LinearInput).unwrap(), linear);
        assert_eq!(
            component_rows(&cfg, Group::AttentionInput).unwrap(),
            attention
        );
        assert_eq!(
            component_rows(&cfg, Group::GateUp).unwrap(),
            [intermediate, intermediate, 0, 0]
        );
        assert!(component_rows(&cfg, Group::Down).is_err());
    }
}

struct Temporary(std::path::PathBuf);
impl Temporary {
    fn new() -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("qwen-lora-{}-{unique}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn malformed_adapter_is_rejected_before_loading_cuda() {
    let temporary = Temporary::new();
    let path = temporary.0.join("adapter.safetensors");
    let bytes = pair("layers.0.linear_attn.in_proj_a", 49, 5120, 1);
    std::fs::write(&path, bytes).unwrap();
    let config = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../tests/qwen3_5/data"
    ));
    let error =
        crate::model::Model::load_with_lora(config, Path::new("/missing/CUDA-library"), &path, 2.)
            .err()
            .unwrap();
    assert!(
        error.to_string().contains("LoRA shape mismatch"),
        "{error:#}"
    );
}

#[cfg(unix)]
#[test]
fn cpu_cuda_boundary_checks_fp32_sums_fused_offsets_sequences_and_retirement() {
    let temporary = Temporary::new();
    let source = temporary.0.join("fake.c");
    let library = temporary.0.join(if cfg!(target_os = "macos") {
        "fake.dylib"
    } else {
        "fake.so"
    });
    let mut code = include_str!("lora_fake.c").to_owned();
    let implemented = [
        "cs1_abi_version",
        "cs1_error_string",
        "cs1_malloc",
        "cs1_free",
        "cs1_stream_create",
        "cs1_stream_sync",
        "cs1_stream_destroy",
        "cs1_upload",
        "cs1_download",
        "cs1_copy2d",
        "cs1_vision_to_float",
        "cs1_gemm_f32",
        "cs1_gemm",
        "cs1_vision_lora_add",
    ];
    // Required but unused functions return an error, so an unexpected dispatch fails.
    let cuda_source = include_str!("../../src/models/qwen3_5/native/src/cuda.rs");
    let abi = cuda_source
        .lines()
        .find(|line| line.starts_with("const ABI_VERSION:"))
        .unwrap()
        .split('=')
        .nth(1)
        .unwrap()
        .trim()
        .trim_end_matches(';');
    let api = cuda_source
        .split("api! {")
        .nth(1)
        .unwrap()
        .split("\n}\n")
        .next()
        .unwrap();
    for line in api.lines() {
        let line = line.trim();
        if !line.starts_with("cs1_") {
            continue;
        }
        let name = line.split('(').next().unwrap();
        if !implemented.contains(&name) {
            code.push_str(&format!("\nint {name}(void) {{ return 77; }}\n"));
        }
    }
    std::fs::write(&source, code).unwrap();
    let output = std::process::Command::new("cc")
        .args(["-shared", "-fPIC", "-ffp-contract=off"])
        .arg(format!("-DLORA_FAKE_ABI={abi}"))
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    cuda::load(&library).unwrap();
    // SAFETY: these symbols are defined immediately above and live for this test.
    let metrics = unsafe { libloading::Library::new(&library).unwrap() };
    let live = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"lora_fake_live\0")
            .unwrap()
    };
    let gemms = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"lora_fake_gemms\0")
            .unwrap()
    };
    let get_m = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32) -> i32>(b"lora_fake_m\0")
            .unwrap()
    };
    let copies = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"lora_fake_copies\0")
            .unwrap()
    };
    let height = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32) -> i32>(b"lora_fake_height\0")
            .unwrap()
    };
    let reset = unsafe {
        *metrics
            .get::<unsafe extern "C" fn()>(b"lora_fake_reset\0")
            .unwrap()
    };
    let stream = cuda::new_stream().unwrap();
    let mut cfg = config();
    cfg.hidden = 8;
    cfg.intermediate = 8;
    cfg.heads = 1;
    cfg.kv_heads = 1;
    cfg.lin_k_heads = 1;
    cfg.lin_v_heads = 1;
    cfg.full_attention = vec![false, true];
    for (name, layer, group) in [
        ("layers.0.linear_attn.in_proj_b", 0, Group::LinearInput),
        ("layers.1.self_attn.k_proj", 1, Group::AttentionInput),
        ("layers.0.mlp.up_proj", 0, Group::GateUp),
        ("layers.0.mlp.down_proj", 0, Group::Down),
    ] {
        let target = targets(&cfg)
            .unwrap()
            .into_iter()
            .find(|t| t.name == name)
            .unwrap();
        let path = temporary.0.join("adapter.safetensors");
        std::fs::write(&path, pair(name, target.rows, target.cols, 1)).unwrap();
        let checkpoint = Checkpoint::load(&cfg, &path, 2.).unwrap();
        unsafe {
            reset();
        }
        {
            let adapter = checkpoint.upload(2, stream).unwrap();
            let work = adapter.work(2).unwrap();
            let input: Vec<_> = (1..=5)
                .flat_map(|value| {
                    half::bf16::from_f32(value as f32)
                        .to_le_bytes()
                        .repeat(target.cols)
                })
                .collect();
            let original = half::bf16::ONE.to_le_bytes().repeat(5 * target.stride);
            let x = DeviceBuffer::new(input.len()).unwrap();
            let y = DeviceBuffer::new(original.len()).unwrap();
            // Start at absolute row 1 and preserve the first/last rows and every neighboring component.
            unsafe {
                cuda::upload(x.at(0), &input, stream).unwrap();
                cuda::upload(y.at(0), &original, stream).unwrap();
                adapter
                    .apply(
                        layer,
                        group,
                        x.at(target.cols * 2),
                        y.at(target.stride * 2),
                        &[1, 2],
                        &work,
                        std::ptr::null_mut(),
                        stream,
                    )
                    .unwrap();
            }
            let mut actual = vec![0; original.len()];
            unsafe {
                cuda::download(&mut actual, y.at(0), stream).unwrap();
            }
            for row in 0..5 {
                for col in 0..target.stride {
                    let offset = (row * target.stride + col) * 2;
                    let value =
                        half::bf16::from_le_bytes(actual[offset..offset + 2].try_into().unwrap());
                    let expected = if (1..4).contains(&row)
                        && (target.offset..target.offset + target.rows).contains(&col)
                    {
                        half::bf16::from_f32(
                            1. - (row + 1) as f32 * target.cols as f32 * 0.125 * 0.25 * 2.,
                        )
                    } else {
                        half::bf16::ONE
                    };
                    assert_eq!(value, expected, "{name}, row {row}, col {col}");
                }
            }
            assert_eq!(unsafe { gemms() }, 4);
            assert_eq!(
                (0..4).map(|i| unsafe { get_m(i) }).collect::<Vec<_>>(),
                [1, 1, 2, 2]
            );
            assert_eq!(
                unsafe { copies() },
                if target.rows == target.stride { 0 } else { 4 }
            );
            if target.rows != target.stride {
                assert_eq!(
                    (0..4).map(|i| unsafe { height(i) }).collect::<Vec<_>>(),
                    [1, 1, 2, 2]
                );
            }
        }
        assert_eq!(
            unsafe { live() },
            0,
            "adapter/work/input/output allocations must retire"
        );
    }
    let base_calls = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"lora_fake_base_calls\0")
            .unwrap()
    };
    let base_arg = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32, i32) -> i32>(b"lora_fake_base_arg\0")
            .unwrap()
    };
    for (group, components) in [
        (Group::LinearInput, [384, 128, 1, 1]),
        (Group::AttentionInput, [512, 256, 256, 0]),
        (Group::GateUp, [8, 8, 0, 0]),
    ] {
        assert_eq!(component_rows(&cfg, group).unwrap(), components);
        let cols = cfg.hidden;
        let stride: usize = components.iter().sum();
        let input: Vec<_> = (1..=5)
            .flat_map(|value| {
                half::bf16::from_f32(value as f32)
                    .to_le_bytes()
                    .repeat(cols)
            })
            .collect();
        let weights: Vec<_> = (1..=stride)
            .flat_map(|value| {
                half::bf16::from_f32(value as f32)
                    .to_le_bytes()
                    .repeat(cols)
            })
            .collect();
        let original = half::bf16::ONE.to_le_bytes().repeat(5 * stride);
        unsafe {
            reset();
        }
        {
            let path = temporary.0.join("base-partial.safetensors");
            std::fs::write(&path, pair("layers.0.linear_attn.in_proj_b", 1, 8, 1)).unwrap();
            let adapter = Checkpoint::load(&cfg, &path, 2.)
                .unwrap()
                .upload(2, stream)
                .unwrap();
            let work = adapter.work(2).unwrap();
            let x = DeviceBuffer::new(input.len()).unwrap();
            let w = DeviceBuffer::new(weights.len()).unwrap();
            let y = DeviceBuffer::new(original.len()).unwrap();
            unsafe {
                cuda::upload(x.at(0), &input, stream).unwrap();
                cuda::upload(w.at(0), &weights, stream).unwrap();
                cuda::upload(y.at(0), &original, stream).unwrap();
                project_components(
                    x.at(cols * 2),
                    w.at(0),
                    y.at(stride * 2),
                    &[1, 2],
                    cols,
                    &components,
                    &work,
                    std::ptr::null_mut(),
                    stream,
                )
                .unwrap();
            }
            let mut actual = vec![0; original.len()];
            unsafe {
                cuda::download(&mut actual, y.at(0), stream).unwrap();
            }
            for row in 0..5 {
                for col in 0..stride {
                    let offset = (row * stride + col) * 2;
                    let value =
                        half::bf16::from_le_bytes(actual[offset..offset + 2].try_into().unwrap());
                    let expected = if (1..4).contains(&row) {
                        half::bf16::from_f32(
                            (row + 1) as f32
                                * half::bf16::from_f32((col + 1) as f32).to_f32()
                                * cols as f32,
                        )
                    } else {
                        half::bf16::ONE
                    };
                    assert_eq!(value, expected, "{group:?}, row{row}, output{col}");
                }
            }
            let parts: Vec<_> = components.iter().copied().filter(|&n| n > 0).collect();
            assert_eq!(unsafe { base_calls() }, parts.len() as i32 * 2);
            let mut call = 0;
            for m in [1, 2] {
                let mut row = 0;
                for &n in &parts {
                    assert_eq!(
                        (0..4)
                            .map(|arg| unsafe { base_arg(call, arg) })
                            .collect::<Vec<_>>(),
                        [m, n as i32, cols as i32, n as i32]
                    );
                    assert_eq!(
                        unsafe { base_arg(call, 4) },
                        half::bf16::from_f32((row + 1) as f32).to_bits() as i32,
                        "component weight offset"
                    );
                    row += n;
                    call += 1;
                }
            }
            assert_eq!(unsafe { copies() }, parts.len() as i32 * 2);
            assert_eq!(
                unsafe { gemms() },
                0,
                "base split must not call FP32 adapter GEMM"
            );
        }
        assert_eq!(unsafe { live() }, 0);
    }
}
