use super::*;

fn config() -> VisionConfig {
    VisionConfig::from_value(serde_json::json!({
        "depth":27,"hidden_size":1152,"intermediate_size":4304,"num_heads":16,
        "num_position_embeddings":2304,"out_hidden_size":5120,"in_channels":3,"patch_size":16,
        "temporal_patch_size":2,"spatial_merge_size":2,"hidden_act":"gelu_pytorch_tanh",
        "deepstack_visual_indexes":[],"model_type":"qwen3_5"
    }))
    .unwrap()
}
fn image(grid: [usize; 3]) -> ProcessedImage {
    ProcessedImage {
        pixel_values: vec![0.; grid[1] * grid[2] * 1536],
        image_grid_thw: grid,
        resized_height: grid[1] * 16,
        resized_width: grid[2] * 16,
    }
}
#[cfg(unix)]
#[test]
fn cpu_batch_uses_aggregate_gemms_and_image_local_attention() {
    if std::env::var_os("QWEN_TEST_VISION_BATCH_CHILD").is_none() {
        let name = format!(
            "{}::cpu_batch_uses_aggregate_gemms_and_image_local_attention",
            module_path!().split_once("::").unwrap().1
        );
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture"])
            .env("QWEN_TEST_VISION_BATCH_CHILD", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let directory = std::env::temp_dir().join(format!("qwen-vision-batch-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let library = directory.join(if cfg!(target_os = "macos") {
        "fake.dylib"
    } else {
        "fake.so"
    });
    let source = directory.join("fake.c");
    let mut code = include_str!("vision_batch_fake.c").to_owned();
    let implemented = [
        "cs1_abi_version",
        "cs1_error_string",
        "cs1_set_device",
        "cs1_malloc",
        "cs1_free",
        "cs1_stream_create",
        "cs1_stream_sync",
        "cs1_stream_destroy",
        "cs1_upload",
        "cs1_download",
        "cs1_gemm_create",
        "cs1_gemm_destroy",
        "cs1_vision_linear",
        "cs1_vision_norm",
        "cs1_vision_bias",
        "cs1_vision_gelu",
        "cs1_vision_add",
        "cs1_graph_begin",
        "cs1_graph_end",
        "cs1_graph_launch",
        "cs1_graph_destroy",
    ];
    let api_source = include_str!("../../src/models/qwen3_5/native/src/cuda.rs");
    for line in api_source
        .split("api! {")
        .nth(1)
        .unwrap()
        .split("\n}\n")
        .next()
        .unwrap()
        .lines()
    {
        let line = line.trim();
        if line.starts_with("cs1_") {
            let name = line.split('(').next().unwrap();
            if !implemented.contains(&name) {
                code.push_str(&format!("\nint {name}(void){{return 77;}}\n"));
            }
        }
    }
    std::fs::write(&source, code).unwrap();
    let compiled = std::process::Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(&source)
        .arg("-o")
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    cuda::load(&library).unwrap();
    // SAFETY: these signatures are defined by the test-only C library above.
    let metrics = unsafe { libloading::Library::new(&library).unwrap() };
    let count = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"batch_linear_count\0")
            .unwrap()
    };
    let rows = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32) -> i32>(b"batch_linear_rows\0")
            .unwrap()
    };
    let out = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32) -> i32>(b"batch_linear_out\0")
            .unwrap()
    };
    let attention_count = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"batch_attention_count\0")
            .unwrap()
    };
    let attention_rows = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32) -> i32>(b"batch_attention_rows\0")
            .unwrap()
    };
    let q = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32) -> usize>(b"batch_q\0")
            .unwrap()
    };
    let k = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32) -> usize>(b"batch_k\0")
            .unwrap()
    };
    let v = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32) -> usize>(b"batch_v\0")
            .unwrap()
    };
    let output = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32) -> usize>(b"batch_out\0")
            .unwrap()
    };
    let qkv_base = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> usize>(b"batch_qkv_base\0")
            .unwrap()
    };
    let captures = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"batch_captures\0")
            .unwrap()
    };
    let launches = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"batch_launches\0")
            .unwrap()
    };
    let destroyed = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"batch_destroyed\0")
            .unwrap()
    };
    let fail_capture = unsafe {
        *metrics
            .get::<unsafe extern "C" fn(i32)>(b"batch_fail_capture\0")
            .unwrap()
    };
    let live = unsafe {
        *metrics
            .get::<unsafe extern "C" fn() -> i32>(b"batch_live\0")
            .unwrap()
    };
    let config = config();
    let mut base = BTreeMap::new();
    for name in config.inventory().keys() {
        base.insert(
            name.strip_prefix("model.visual.").unwrap().to_owned(),
            DeviceBuffer::new(0).unwrap(),
        );
    }
    base.insert("__zero_bias".into(), DeviceBuffer::new(0).unwrap());
    let mut model = VisionModel {
        config,
        scratch: VecDeque::new(),
        batch_scratch: VecDeque::new(),
        graph_enabled: false,
        reference: None,
        base,
        lora: BTreeMap::new(),
        gemm: Gemm(unsafe { (api().cs1_gemm_create)(32 << 20) }),
        stream: OwnedStream(cuda::new_stream().unwrap()),
    };
    let images = [image([1, 2, 2]), image([1, 2, 4])];
    let result = model.forward_images(&images).unwrap();
    assert_eq!(result.len(), 3 * 5120);
    for index in 0..unsafe { count() } {
        let expected = if unsafe { out(index) } == 4608 || unsafe { out(index) } == 5120 {
            3
        } else {
            12
        };
        assert_eq!(
            unsafe { rows(index) },
            expected,
            "all-row vision GEMM must preserve aggregate M (call {index})"
        );
    }
    assert_eq!(unsafe { attention_count() }, 27 * 2);
    for index in 0..54 {
        assert_eq!(
            unsafe { attention_rows(index) },
            if index % 2 == 0 { 4 } else { 8 }
        );
    }
    assert_eq!(unsafe { q(1) - q(0) }, 4 * 1152 * 2);
    assert_eq!(unsafe { k(1) - k(0) }, 4 * 1152 * 2);
    assert_eq!(unsafe { v(1) - v(0) }, 4 * 1152 * 3 * 2);
    assert_eq!(unsafe { v(0) - qkv_base() }, 2 * 1152 * 2);
    assert_eq!(unsafe { output(1) - output(0) }, 4 * 1152 * 2);
    assert_eq!(model.batch_scratch.len(), 1);
    assert!(model.scratch.is_empty());
    model.graph_enabled = true;
    model.forward_images(&images).unwrap();
    assert_eq!(unsafe { captures() }, 1);
    let mut changed = [image([1, 2, 2]), image([1, 2, 4])];
    changed[0].pixel_values[0] = 1.;
    model.forward_images(&changed).unwrap();
    assert_eq!(unsafe { launches() }, 1);
    let mut pixel = [0u8; 2];
    unsafe {
        cuda::download(
            &mut pixel,
            model.batch_scratch[0].buffers.pixels.at(0),
            model.stream.0,
        )
        .unwrap();
    }
    assert_eq!(bf16::from_le_bytes(pixel), bf16::ONE);
    // Equal aggregate rows with a different order require a distinct capture.
    model
        .forward_images(&[image([1, 2, 4]), image([1, 2, 2])])
        .unwrap();
    assert_eq!(unsafe { captures() }, 2);
    assert_eq!(model.batch_scratch.len(), 2);
    let allocations = unsafe { live() };
    let dispatched = unsafe { count() };
    let mut invalid = [image([1, 2, 2]), image([1, 2, 4])];
    invalid[1].pixel_values[0] = f32::NAN;
    assert!(model.forward_images(&invalid).is_err());
    assert_eq!(unsafe { live() }, allocations);
    assert_eq!(unsafe { count() }, dispatched);
    assert_eq!(model.batch_scratch.len(), 2);
    for grids in [
        [[1, 2, 2], [1, 4, 2]],
        [[1, 4, 2], [1, 2, 2]],
        [[1, 2, 2], [1, 4, 4]],
    ] {
        model.forward_images(&grids.map(image)).unwrap();
    }
    assert_eq!(model.batch_scratch.len(), 4);
    assert_eq!(unsafe { destroyed() }, 1);
    assert!(
        !model
            .batch_scratch
            .iter()
            .any(|s| s.grids == [[1, 2, 2], [1, 2, 4]])
    );
    // Single-image dispatch still uses the original separate cache/pipeline.
    model.forward_images(&[image([1, 2, 2])]).unwrap();
    assert_eq!(model.scratch.len(), 1);
    assert_eq!(model.batch_scratch.len(), 4);
    assert!(model.forward_images(&[]).unwrap().is_empty());
    unsafe {
        fail_capture(1);
    }
    let fallback = model
        .forward_images(&[image([1, 4, 4]), image([1, 2, 4])])
        .unwrap();
    assert_eq!(fallback.len(), 6 * 5120);
    assert!(!model.graph_enabled);
    assert!(model.scratch.iter().all(|s| s.graph.is_none()));
    assert!(model.batch_scratch.iter().all(|s| s.graph.is_none()));
    unsafe {
        fail_capture(0);
    }
    let capture_count = unsafe { captures() };
    model.forward_images(&images).unwrap();
    assert_eq!(unsafe { captures() }, capture_count);

    drop(model);
    assert_eq!(unsafe { live() }, 0);
    drop(metrics);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn batch_geometry_preserves_each_image_coordinates_and_merge_boundaries() {
    let config = config();
    let grids = [[1, 2, 2], [1, 2, 4], [1, 4, 2]];
    assert_eq!(BatchGeometry::lengths(&grids, &config).unwrap(), [4, 8, 8]);
    let batch = BatchGeometry::new(&grids, &config).unwrap().geometry;
    let images: Vec<_> = grids
        .iter()
        .map(|&g| VisionGeometry::new(g, &config).unwrap())
        .collect();
    assert_eq!(
        batch.indices,
        images
            .iter()
            .flat_map(|g| g.indices.iter().copied())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        batch.weights,
        images
            .iter()
            .flat_map(|g| g.weights.iter().copied())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        batch.cos,
        images
            .iter()
            .flat_map(|g| g.cos.iter().copied())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        batch.sin,
        images
            .iter()
            .flat_map(|g| g.sin.iter().copied())
            .collect::<Vec<_>>()
    );
    assert_eq!(batch.cos.len(), 20 * 36);
    assert!(BatchGeometry::lengths(&[[1, 256, 256], [1, 2, 2]], &config).is_err());
    assert!(BatchGeometry::lengths(&[[1, 2, 2]; 5], &config).is_err());
    assert!(BatchGeometry::lengths(&[[1, 2, 2], [2, 2, 2]], &config).is_err());
    assert!(BatchGeometry::lengths(&[[1, usize::MAX - 1, 2], [1, 2, 2]], &config).is_err());
}
