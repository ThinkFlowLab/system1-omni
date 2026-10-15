//! GPU regression bodies live outside production source. Run serially with pinned weights.
use super::*;

fn load() -> VisionModel {
    load_with_cache(None)
}
fn load_cached() -> VisionModel {
    load_with_cache(Some(CacheLimits {
        entries: std::env::var("CUA_S1_VISION_CACHE_ENTRIES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1)
            .min(16),
        bytes: std::env::var("CUA_S1_VISION_CACHE_BYTES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256 << 20),
    }))
}
fn load_with_cache(cache: Option<CacheLimits>) -> VisionModel {
    use memmap2::Mmap;
    use safetensors::SafeTensors;
    use std::{collections::BTreeSet, fs::File};
    let path = |name| std::path::PathBuf::from(std::env::var_os(name).expect(name));
    let base_dir = path("CUA_S1_BASE");
    let adapter_dir = path("CUA_S1_VISION_ADAPTER");
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(base_dir.join("config.json")).unwrap()).unwrap();
    let config = VisionConfig::from_value(config["vision_config"].clone()).unwrap();
    let inventory = config.inventory();
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(base_dir.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let files: BTreeSet<_> = inventory
        .keys()
        .map(|name| index["weight_map"][name].as_str().unwrap().to_owned())
        .collect();
    let maps: Vec<Mmap> = files
        .iter()
        .map(|name| {
            // SAFETY: the pinned checkpoints remain immutable throughout this GPU test.
            unsafe { Mmap::map(&File::open(base_dir.join(name)).unwrap()).unwrap() }
        })
        .collect();
    let stores: Vec<_> = maps
        .iter()
        .map(|map| SafeTensors::deserialize(map).unwrap())
        .collect();
    let tensors = inventory
        .keys()
        .map(|name| {
            let tensor = stores
                .iter()
                .find_map(|store| store.tensor(name).ok())
                .unwrap();
            (name.clone(), tensor)
        })
        .collect();
    // SAFETY: the pinned adapter remains immutable throughout this GPU test.
    let adapter_map = unsafe {
        Mmap::map(&File::open(adapter_dir.join("adapter_model.safetensors")).unwrap()).unwrap()
    };
    let adapter = SafeTensors::deserialize(&adapter_map).unwrap();
    let lora = adapter
        .names()
        .iter()
        .filter(|name| name.starts_with("base_model.model.model.visual."))
        .map(|name| ((*name).to_owned(), adapter.tensor(name).unwrap()))
        .collect();
    match cache {
        Some(limits) => VisionModel::from_tensors_with_cache(
            config,
            tensors,
            lora,
            &path("CUA_S1_CUDA_LIB"),
            limits,
        ),
        None => VisionModel::from_tensors(config, tensors, lora, &path("CUA_S1_CUDA_LIB")),
    }
    .unwrap()
}
fn image(grid: [usize; 3], seed: usize) -> ProcessedImage {
    let n = grid[1] * grid[2] * 1536;
    ProcessedImage {
        image_grid_thw: grid,
        resized_height: grid[1] * 16,
        resized_width: grid[2] * 16,
        pixel_values: (0..n)
            .map(|i| ((i * 17 + seed * 31) % 251) as f32 / 125. - 1.)
            .collect(),
    }
}
fn bits(values: &[bf16]) -> Vec<u16> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_BASE/CUA_S1_VISION_ADAPTER/CUA_S1_CUDA_LIB"]
fn vision_graph_changed_pixels_and_trace() {
    let mut model = load();
    model.graph_enabled = false;
    let inputs = [
        image([1, 16, 16], 1),
        image([1, 16, 16], 2),
        image([1, 16, 16], 3),
    ];
    let expected: Vec<_> = inputs
        .iter()
        .map(|i| bits(&model.forward(i).unwrap()))
        .collect();
    assert_ne!(
        expected[0], expected[1],
        "changed pixels must change the feature oracle"
    );
    model.graph_enabled = true;
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    assert!(
        model
            .scratch
            .iter()
            .any(|s| s.grid == inputs[0].image_grid_thw && s.graph.is_some()),
        "missing vision capture"
    );
    for (input, reference) in inputs.iter().zip(&expected).cycle().take(9) {
        assert_eq!(bits(&model.forward(input).unwrap()), *reference);
    }
    let mut stages = Vec::new();
    let traced = model
        .forward_with_trace(&inputs[1], |name, _| {
            stages.push(name.to_string());
            Ok(())
        })
        .unwrap();
    assert_eq!(bits(&traced), expected[1]);
    assert_eq!(stages.len(), 29);
    assert!(
        model
            .scratch
            .iter()
            .any(|s| s.grid == inputs[0].image_grid_thw && s.graph.is_some())
    );
    assert_eq!(bits(&model.forward(&inputs[2]).unwrap()), expected[2]);
    let mut invalid = image([1, 16, 16], 4);
    invalid.pixel_values[0] = f32::NAN;
    assert!(model.forward(&invalid).is_err());
    assert!(
        model
            .scratch
            .iter()
            .any(|s| s.grid == inputs[0].image_grid_thw && s.graph.is_some())
    );
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_BASE/CUA_S1_VISION_ADAPTER/CUA_S1_CUDA_LIB"]
fn vision_graph_equal_patch_count_geometry_and_retirement() {
    let mut model = load();
    model.graph_enabled = false;
    let inputs = [
        image([1, 8, 32], 7),
        image([1, 16, 16], 7),
        image([1, 32, 8], 7),
        image([1, 16, 32], 8),
        image([1, 32, 16], 9),
    ];
    let expected: Vec<_> = inputs
        .iter()
        .map(|i| bits(&model.forward(i).unwrap()))
        .collect();
    assert_ne!(
        expected[0], expected[1],
        "geometry must affect the feature oracle"
    );
    model.graph_enabled = true;
    for index in [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0] {
        assert_eq!(
            bits(&model.forward(&inputs[index]).unwrap()),
            expected[index]
        );
        let scratch = model
            .scratch
            .iter()
            .find(|s| s.grid == inputs[index].image_grid_thw)
            .unwrap();
        assert_eq!(scratch.grid, inputs[index].image_grid_thw);
        assert!(scratch.graph.is_some());
        assert_eq!(model.scratch.len(), 4);
        if index == 4 {
            assert!(
                model
                    .scratch
                    .iter()
                    .all(|s| s.grid != inputs[0].image_grid_thw),
                "the fifth distinct grid must evict the oldest FIFO entry"
            );
        }
    }
    drop(model); // graph must retire before scratch/weights/GEMM/stream
    let mut fresh = load();
    fresh.graph_enabled = true;
    assert_eq!(bits(&fresh.forward(&inputs[1]).unwrap()), expected[1]);
    assert_eq!(bits(&fresh.forward(&inputs[1]).unwrap()), expected[1]);
}

#[test]
#[ignore = "requires the test-only failing CUDA wrapper; see native_vision_graph.md"]
fn vision_capture_failure_keeps_eager_result_and_disables_replay() {
    check_capture_failure(load());
}
#[test]
#[ignore = "requires the test-only failing CUDA wrapper; see native_vision_graph.md"]
fn vision_capture_failure_cached_keeps_eager_result_and_disables_replay() {
    check_capture_failure(load_cached());
}
fn check_capture_failure(mut model: VisionModel) {
    assert_eq!(
        std::env::var("CUA_S1_TEST_CAPTURE_FAILURE").as_deref(),
        Ok("1"),
        "use the test-only failing CUDA wrapper"
    );
    model.graph_enabled = false;
    let input = image([1, 16, 16], 9);
    let changed = image([1, 8, 32], 10);
    let expected = bits(&model.forward(&input).unwrap());
    let changed_expected = bits(&model.forward(&changed).unwrap());
    model.graph_enabled = true;
    // The wrapper fails graph instantiation after the real eager forward and
    // capture have finished. This must return that same forward's eager result.
    assert_eq!(bits(&model.forward(&input).unwrap()), expected);
    assert!(!model.graph_enabled);
    assert!(model.scratch.iter().all(|s| s.graph.is_none()));
    assert_eq!(bits(&model.forward(&changed).unwrap()), changed_expected);
    assert!(!model.graph_enabled);
    assert!(model.scratch.iter().all(|s| s.graph.is_none()));
    assert_eq!(bits(&model.forward(&input).unwrap()), expected);
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_BASE/CUA_S1_VISION_ADAPTER/CUA_S1_CUDA_LIB"]
fn vision_cache_aba_probe() {
    let mut model = load_cached();
    model.graph_enabled = false;
    let inputs = [image([1, 8, 32], 21), image([1, 16, 16], 22)];
    let expected: Vec<_> = inputs
        .iter()
        .map(|i| bits(&model.forward(i).unwrap()))
        .collect();
    model.graph_enabled = true;
    eprintln!("VISION_CACHE_ABA_BEGIN");
    for index in [0, 1, 0] {
        assert_eq!(
            bits(&model.forward(&inputs[index]).unwrap()),
            expected[index]
        );
    }
    eprintln!("VISION_CACHE_ABA_END");
}

fn config(hidden: usize) -> VisionConfig {
    VisionConfig::from_value(serde_json::json!({"depth": if hidden==1024 {24} else {27},
        "hidden_size":hidden,"intermediate_size":if hidden==1024 {4096} else {4304},
        "num_heads":16,"num_position_embeddings":2304,"out_hidden_size":if hidden==1024 {2560} else {5120},
        "in_channels":3,"patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,
        "hidden_act":"gelu_pytorch_tanh","deepstack_visual_indexes":[],"model_type":"qwen3_5"})).unwrap()
}

#[test]
fn vision_cache_dimension_storage() {
    for (hidden, lora, per_patch) in [
        (1024, true, 64_096),
        (1024, false, 31_264),
        (1152, false, 47_168),
    ] {
        let config = config(hidden);
        for grid in [[1, 2, 2], [1, 8, 32], [1, 32, 8]] {
            let geometry = VisionGeometry::new(grid, &config).unwrap();
            let sizes = Scratch::layout(&config, lora, grid);
            assert_eq!(
                [
                    geometry.indices.len() * 4,
                    geometry.weights.len() * 4,
                    geometry.cos.len() * 4,
                    geometry.sin.len() * 4
                ],
                sizes[1..5]
            );
            assert_eq!(
                Scratch::bytes_for_grid(&config, lora, grid),
                grid[1] * grid[2] * per_patch
            );
        }
    }
}

#[test]
fn vision_cache_budget_policy() {
    let patch_bytes = Scratch::bytes_for_grid(&config(1024), true, [1, 2, 2]) / 4;
    assert_eq!(patch_bytes, 64_096);
    let geometry = VisionGeometry::new([1, 2, 2], &config(1024)).unwrap();
    let geometry_bytes =
        (geometry.indices.len() + geometry.weights.len() + geometry.cos.len() + geometry.sin.len())
            * 4;
    assert_eq!(geometry_bytes / 4, 288);
    assert_eq!(
        Scratch::bytes_for_grid(&config(1024), true, [1, 32, 72]),
        147_677_184
    );
    let limits = CacheLimits {
        entries: 2,
        bytes: 100,
    };
    assert!(limits.retains(100));
    assert!(!limits.retains(101));
    assert!(
        !CacheLimits {
            entries: 0,
            bytes: 100
        }
        .retains(1)
    );
    assert!(!limits.needs_eviction(1, 60, 40));
    assert!(limits.needs_eviction(2, 0, 1));
    assert!(limits.needs_eviction(1, 61, 40));
}

fn clear_scratch(model: &mut VisionModel) {
    model.synchronize().unwrap();
    model.scratch.clear();
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_BASE/CUA_S1_VISION_ADAPTER/CUA_S1_CUDA_LIB"]
fn vision_cache_capacity_lru_and_invalid_input() {
    let mut model = load_cached();
    model.cache_limits = Some(CacheLimits {
        entries: 2,
        bytes: 256 << 20,
    });
    model.graph_enabled = false;
    let inputs = [
        image([1, 8, 32], 30),
        image([1, 16, 16], 31),
        image([1, 32, 8], 32),
    ];
    let expected: Vec<_> = inputs
        .iter()
        .map(|i| bits(&model.forward(i).unwrap()))
        .collect();
    clear_scratch(&mut model);
    model.graph_enabled = true;
    for index in [0, 1, 0] {
        assert_eq!(
            bits(&model.forward(&inputs[index]).unwrap()),
            expected[index]
        );
    }
    let a = model.scratch.back().unwrap().pixels.at(0);
    assert!(model.scratch.back().unwrap().graph.is_some());
    assert_eq!(
        model.scratch.front().unwrap().grid,
        inputs[1].image_grid_thw
    );
    let b = model.scratch.front().unwrap().pixels.at(0);
    let mut invalid = image([1, 16, 16], 31);
    invalid.pixel_values.pop();
    assert!(model.forward(&invalid).is_err());
    invalid = image([2, 8, 32], 30);
    assert!(model.forward(&invalid).is_err());
    assert_eq!(model.scratch.back().unwrap().pixels.at(0), a);
    assert_eq!(model.scratch.front().unwrap().pixels.at(0), b);
    assert_eq!(bits(&model.forward(&inputs[2]).unwrap()), expected[2]);
    assert_eq!(model.scratch.len(), 2);
    assert_eq!(
        model.scratch.front().unwrap().grid,
        inputs[0].image_grid_thw,
        "B must be LRU victim after A/B/A/C"
    );
    assert_eq!(model.scratch.front().unwrap().pixels.at(0), a);
    assert_eq!(model.retained_bytes(), 512 * 64_096);
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    assert_eq!(model.scratch.back().unwrap().pixels.at(0), a);
    drop(model);
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_BASE/CUA_S1_VISION_ADAPTER/CUA_S1_CUDA_LIB"]
fn vision_cache_byte_eviction_and_transient_oversize() {
    let mut model = load_cached();
    model.cache_limits = Some(CacheLimits {
        entries: 4,
        bytes: 512 * 64_096,
    });
    model.graph_enabled = false;
    let inputs = [
        image([1, 8, 32], 40),
        image([1, 16, 32], 41),
        image([1, 32, 8], 42),
        image([1, 32, 32], 43),
    ];
    let expected: Vec<_> = inputs
        .iter()
        .map(|i| bits(&model.forward(i).unwrap()))
        .collect();
    clear_scratch(&mut model);
    model.graph_enabled = true;
    for index in [0, 1, 2, 0] {
        assert_eq!(
            bits(&model.forward(&inputs[index]).unwrap()),
            expected[index]
        );
        assert!(model.retained_bytes() <= model.cache_limits.as_ref().unwrap().bytes);
    }
    assert_eq!(model.scratch.len(), 2);
    assert_eq!(
        model.scratch.front().unwrap().grid,
        inputs[2].image_grid_thw
    );
    let a = model.scratch.back().unwrap().pixels.at(0);
    let c = model.scratch.front().unwrap().pixels.at(0);
    for _ in 0..2 {
        assert_eq!(bits(&model.forward(&inputs[3]).unwrap()), expected[3]);
        assert_eq!(model.scratch.back().unwrap().pixels.at(0), a);
        assert_eq!(model.scratch.front().unwrap().pixels.at(0), c);
        assert_eq!(
            model.retained_bytes(),
            model.cache_limits.as_ref().unwrap().bytes
        );
        assert!(model.graph_enabled);
    }
    assert!(
        model
            .forward_with_trace(&inputs[3], |_, _| anyhow::bail!("test trace failure"))
            .is_err()
    );
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    model.cache_limits.as_mut().unwrap().bytes = 1;
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    // Internal test-only budget changes may encounter an already resident same grid.
    assert_eq!(model.scratch.back().unwrap().pixels.at(0), a);
    clear_scratch(&mut model);
    model.cache_limits.as_mut().unwrap().entries = 0;
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    assert_eq!(model.retained_bytes(), 0);
    assert!(model.scratch.is_empty());
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_BASE/CUA_S1_VISION_ADAPTER/CUA_S1_CUDA_LIB"]
fn device_vision_graph_replays_changed_pixels_and_replaces_geometry() {
    let mut model = load();
    let inputs = [
        image([1, 16, 16], 1),
        image([1, 16, 16], 2),
        image([1, 8, 32], 2),
        image([1, 16, 32], 3),
        image([1, 16, 16], 1),
    ];
    model.graph_enabled = false;
    let expected: Vec<_> = inputs
        .iter()
        .map(|i| bits(&model.forward(i).unwrap()))
        .collect();
    model.graph_enabled = true;
    for (input, expected) in inputs.iter().zip(&expected) {
        for _ in 0..2 {
            let features = model.forward_device(input).unwrap();
            assert_eq!(bits(&features.download().unwrap()), *expected);
        }
        let scratch = model
            .scratch
            .iter()
            .find(|s| s.grid == input.image_grid_thw)
            .unwrap();
        assert_eq!(scratch.grid, input.image_grid_thw);
        assert!(
            scratch.graph.is_some(),
            "resident path must capture/replay vision"
        );
    }
}

#[test]
#[ignore = "requires the test-only failing CUDA wrapper; see native_vision_graph.md"]
fn device_vision_capture_failure_preserves_resident_eager_result_and_recovery() {
    assert_eq!(
        std::env::var("CUA_S1_TEST_CAPTURE_FAILURE").as_deref(),
        Ok("1")
    );
    let mut model = load();
    model.graph_enabled = false;
    let input = image([1, 16, 16], 9);
    let changed = image([1, 8, 32], 10);
    let expected = bits(&model.forward(&input).unwrap());
    let expected_changed = bits(&model.forward(&changed).unwrap());
    model.graph_enabled = true;
    assert_eq!(
        bits(&model.forward_device(&input).unwrap().download().unwrap()),
        expected
    );
    assert!(!model.graph_enabled);
    assert!(model.scratch.iter().all(|s| s.graph.is_none()));
    assert_eq!(
        bits(&model.forward_device(&changed).unwrap().download().unwrap()),
        expected_changed
    );
    assert_eq!(
        bits(&model.forward_device(&input).unwrap().download().unwrap()),
        expected
    );
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_BASE/CUA_S1_VISION_ADAPTER/CUA_S1_CUDA_LIB"]
fn device_cache_transient_output_keeps_retained_entries_and_releases_after_consumption() {
    let mut model = load_cached();
    model.cache_limits = Some(CacheLimits {
        entries: 1,
        bytes: 256 * 64_096,
    });
    model.graph_enabled = false;
    let small = image([1, 16, 16], 61);
    let large = image([1, 16, 32], 62);
    let expected_small = bits(&model.forward(&small).unwrap());
    let expected_large = bits(&model.forward(&large).unwrap());
    assert!(
        model.transient.is_none(),
        "host transient must be released before return"
    );
    let retained = model.scratch[0].pixels.at(0);
    model.graph_enabled = true;
    for input in [&small, &large, &large, &small] {
        let observed = {
            let output = model.forward_device(input).unwrap();
            bits(&output.download().unwrap())
        };
        assert_eq!(
            observed,
            if input.image_grid_thw == small.image_grid_thw {
                &expected_small
            } else {
                &expected_large
            }
            .clone()
        );
        assert_eq!(model.scratch[0].pixels.at(0), retained);
        assert_eq!(model.retained_bytes(), 256 * 64_096);
        if input.image_grid_thw == large.image_grid_thw {
            assert!(model.transient.as_ref().unwrap().graph.is_none());
        }
        model.release_transient().unwrap();
        assert!(model.transient.is_none());
    }
    clear_scratch(&mut model);
    model.cache_limits.as_mut().unwrap().entries = 0;
    let observed = {
        let output = model.forward_device(&small).unwrap();
        bits(&output.download().unwrap())
    };
    assert_eq!(observed, expected_small);
    assert!(model.scratch.is_empty());
    assert!(model.transient.is_some());
    model.release_transient().unwrap();
    assert!(model.transient.is_none());
}
