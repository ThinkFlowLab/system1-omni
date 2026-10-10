//! GPU regression bodies live outside production source. Run serially with pinned weights.
use super::*;

fn load() -> VisionModel {
    let path = |name| std::path::PathBuf::from(std::env::var_os(name).expect(name));
    VisionModel::load(
        path("CUA_S1_BASE"),
        path("CUA_S1_VISION_ADAPTER"),
        &path("CUA_S1_CUDA_LIB"),
    )
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
        model.scratch.as_ref().unwrap().graph.is_some(),
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
    assert!(model.scratch.as_ref().unwrap().graph.is_some());
    assert_eq!(bits(&model.forward(&inputs[2]).unwrap()), expected[2]);
    let mut invalid = image([1, 16, 16], 4);
    invalid.pixel_values[0] = f32::NAN;
    assert!(model.forward(&invalid).is_err());
    assert!(model.scratch.as_ref().unwrap().graph.is_some());
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
    for index in [0, 0, 1, 1, 2, 2, 3, 3, 0, 0] {
        assert_eq!(
            bits(&model.forward(&inputs[index]).unwrap()),
            expected[index]
        );
        let scratch = model.scratch.as_ref().unwrap();
        assert_eq!(scratch.grid, inputs[index].image_grid_thw);
        assert!(scratch.graph.is_some());
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
    assert!(model.scratch.as_ref().unwrap().graph.is_none());
    assert_eq!(bits(&model.forward(&changed).unwrap()), changed_expected);
    assert!(!model.graph_enabled);
    assert!(model.scratch.as_ref().unwrap().graph.is_none());
    assert_eq!(bits(&model.forward(&input).unwrap()), expected);
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_BASE/CUA_S1_VISION_ADAPTER/CUA_S1_CUDA_LIB"]
fn vision_cache_aba_probe() {
    let mut model = load();
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

#[test]
fn vision_cache_limits_from_env() {
    // Each probe gets a private environment; parallel tests never mutate global env.
    if std::env::var_os("CUA_S1_TEST_CACHE_LIMITS_PROBE").is_some() {
        let limits = CacheLimits::from_env();
        eprintln!("CACHE_LIMITS {} {}", limits.entries, limits.bytes);
        return;
    }
    let test = format!(
        "{}::vision_cache_limits_from_env",
        module_path!().split_once("::").unwrap().1
    );
    let probe = |entries: Option<&str>,
                 bytes: Option<&str>,
                 expected: (usize, usize),
                 warning: Option<&str>| {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", &test, "--nocapture"])
            .env("CUA_S1_TEST_CACHE_LIMITS_PROBE", "1")
            .env_remove("CUA_S1_VISION_CACHE_ENTRIES")
            .env_remove("CUA_S1_VISION_CACHE_BYTES");
        if let Some(value) = entries {
            command.env("CUA_S1_VISION_CACHE_ENTRIES", value);
        }
        if let Some(value) = bytes {
            command.env("CUA_S1_VISION_CACHE_BYTES", value);
        }
        let output = command.output().unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(output.status.success(), "{stderr}");
        assert!(
            stderr.contains(&format!("CACHE_LIMITS {} {}", expected.0, expected.1)),
            "{stderr}"
        );
        assert_eq!(
            stderr.matches("Invalid CUA_S1_VISION_CACHE_").count(),
            usize::from(warning.is_some()),
            "{stderr}"
        );
        if let Some(name) = warning {
            let default = if name.ends_with("ENTRIES") {
                1
            } else {
                256 << 20
            };
            assert!(stderr.contains(&format!("Invalid {name}=")), "{stderr}");
            assert!(
                stderr.contains(&format!("using default {default}")),
                "{stderr}"
            );
        }
    };
    probe(None, None, (1, 256 << 20), None);
    probe(Some("0"), Some("0"), (0, 0), None);
    probe(Some("4"), Some("1024"), (4, 1024), None);
    probe(Some("17"), None, (16, 256 << 20), None);
    for invalid in ["4 ", "-1", "four"] {
        probe(
            Some(invalid),
            None,
            (1, 256 << 20),
            Some("CUA_S1_VISION_CACHE_ENTRIES"),
        );
        probe(
            None,
            Some(invalid),
            (1, 256 << 20),
            Some("CUA_S1_VISION_CACHE_BYTES"),
        );
    }
}

#[test]
fn vision_cache_budget_policy() {
    let patch_bytes = Scratch::bytes_for_grid([1, 2, 2]) / 4;
    assert_eq!(patch_bytes, 64_096);
    let geometry = Geometry::new([1, 2, 2]).unwrap();
    let geometry_bytes =
        (geometry.indices.len() + geometry.weights.len() + geometry.cos.len() + geometry.sin.len())
            * 4;
    assert_eq!(geometry_bytes / 4, 288);
    assert_eq!(Scratch::bytes_for_grid([1, 32, 72]), 147_677_184);
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
    model.scratch = None;
    model.cached.clear();
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_BASE/CUA_S1_VISION_ADAPTER/CUA_S1_CUDA_LIB"]
fn vision_cache_capacity_lru_and_invalid_input() {
    let mut model = load();
    model.cache_limits = CacheLimits {
        entries: 2,
        bytes: 256 << 20,
    };
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
    let a = model.scratch.as_ref().unwrap().pixels.at(0);
    assert!(model.scratch.as_ref().unwrap().graph.is_some());
    assert_eq!(model.cached[0].grid, inputs[1].image_grid_thw);
    let b = model.cached[0].pixels.at(0);
    let mut invalid = image([1, 16, 16], 31);
    invalid.pixel_values.pop();
    assert!(model.forward(&invalid).is_err());
    invalid = image([2, 8, 32], 30);
    assert!(model.forward(&invalid).is_err());
    assert_eq!(model.scratch.as_ref().unwrap().pixels.at(0), a);
    assert_eq!(model.cached[0].pixels.at(0), b);
    assert_eq!(bits(&model.forward(&inputs[2]).unwrap()), expected[2]);
    assert_eq!(model.cached.len(), 1);
    assert_eq!(
        model.cached[0].grid, inputs[0].image_grid_thw,
        "B must be LRU victim after A/B/A/C"
    );
    assert_eq!(model.cached[0].pixels.at(0), a);
    assert_eq!(model.retained_bytes(), 512 * 64_096);
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    assert_eq!(model.scratch.as_ref().unwrap().pixels.at(0), a);
    drop(model);
}

#[test]
#[ignore = "requires CUDA and pinned CUA_S1_BASE/CUA_S1_VISION_ADAPTER/CUA_S1_CUDA_LIB"]
fn vision_cache_byte_eviction_and_transient_oversize() {
    let mut model = load();
    model.cache_limits = CacheLimits {
        entries: 4,
        bytes: 512 * 64_096,
    };
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
        assert!(model.retained_bytes() <= model.cache_limits.bytes);
    }
    assert_eq!(model.cached.len(), 1);
    assert_eq!(model.cached[0].grid, inputs[2].image_grid_thw);
    let a = model.scratch.as_ref().unwrap().pixels.at(0);
    let c = model.cached[0].pixels.at(0);
    for _ in 0..2 {
        assert_eq!(bits(&model.forward(&inputs[3]).unwrap()), expected[3]);
        assert_eq!(model.scratch.as_ref().unwrap().pixels.at(0), a);
        assert_eq!(model.cached[0].pixels.at(0), c);
        assert_eq!(model.retained_bytes(), model.cache_limits.bytes);
        assert!(model.graph_enabled);
    }
    assert!(
        model
            .forward_with_trace(&inputs[3], |_, _| anyhow::bail!("test trace failure"))
            .is_err()
    );
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    model.cache_limits.bytes = 1;
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    // Internal test-only budget changes may encounter an already resident same grid.
    assert_eq!(model.scratch.as_ref().unwrap().pixels.at(0), a);
    clear_scratch(&mut model);
    model.cache_limits.entries = 0;
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    assert_eq!(bits(&model.forward(&inputs[0]).unwrap()), expected[0]);
    assert_eq!(model.retained_bytes(), 0);
    assert!(model.scratch.is_none() && model.cached.is_empty());
}
