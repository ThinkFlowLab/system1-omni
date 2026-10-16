//! GPU regression bodies live outside production source. Run serially with pinned weights.
use super::*;

fn load() -> VisionModel {
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
    VisionModel::from_tensors(config, tensors, lora, &path("CUA_S1_CUDA_LIB")).unwrap()
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
    assert_eq!(
        std::env::var("CUA_S1_TEST_CAPTURE_FAILURE").as_deref(),
        Ok("1"),
        "use the test-only failing CUDA wrapper"
    );
    let mut model = load();
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
