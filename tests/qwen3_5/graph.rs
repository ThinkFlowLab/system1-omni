//! Full-checkpoint graph lifetime, shape and current-token checks.
use omni_qwen3_5_native::model::Model;
use std::path::Path;

#[test]
#[ignore = "requires Decider-2B checkpoint and NVIDIA CUDA"]
fn ordered_shape_replay_current_ids_growth_and_bounded_cache() {
    let dir = std::env::var("DECIDER_MODEL").unwrap();
    let library = std::env::var("DECIDER_CUDA_LIB").unwrap();
    let mut eager = Model::load_with_graph(Path::new(&dir), Path::new(&library), false).unwrap();
    assert!(!eager.graph_stats().requested);
    let mut graph = Model::load_with_graph(Path::new(&dir), Path::new(&library), true).unwrap();
    let a = vec![100; 64];
    let b = vec![101; 65];
    let first = graph.forward_batch(&[&a, &b]).unwrap();
    assert_eq!(first, eager.forward_batch(&[&a, &b]).unwrap());
    assert_eq!(graph.graph_stats().captures, 1);
    assert_eq!(graph.forward_batch(&[&a, &b]).unwrap(), first);
    assert_eq!(graph.graph_stats().replays, 1);
    assert_eq!(
        graph.forward_batch(&[&b, &a]).unwrap(),
        eager.forward_batch(&[&b, &a]).unwrap()
    );
    assert_eq!(graph.graph_stats().captures, 2); // same total, different ordered vector
    let c = vec![102; 64];
    assert_eq!(
        graph.forward_batch(&[&c, &b]).unwrap(),
        eager.forward_batch(&[&c, &b]).unwrap()
    );
    assert_eq!(graph.graph_stats().replays, 2); // fresh IDs uploaded before replay
    let long = vec![100; 2049];
    assert_eq!(graph.forward(&long).unwrap(), eager.forward(&long).unwrap());
    assert_eq!(graph.graph_stats().invalidations, 1);
    assert_eq!(graph.graph_stats().cached_shapes, 1);
    assert_eq!(graph.forward_batch(&[&a, &b]).unwrap(), first);
    for n in 1..=65 {
        let ids = vec![100; n];
        graph.forward(&ids).unwrap();
    }
    assert_eq!(graph.graph_stats().cached_shapes, 64);
    let captures = graph.graph_stats().captures;
    graph.forward(&[100]).unwrap();
    assert_eq!(graph.graph_stats().captures, captures + 1); // FIFO eviction
    assert_eq!(graph.graph_stats().cached_shapes, 64);
    assert_eq!(graph.graph_stats().fallbacks, 0);
    graph.synchronize().unwrap();
}

#[test]
#[ignore = "requires Qwen3.5 checkpoint, CUA_S1_CUDA_LIB and NVIDIA CUDA"]
fn legacy_loader_retains_cua_graph_switch() {
    let dir = std::env::var("QWEN3_5_MODEL").unwrap();
    let library = std::env::var("CUA_S1_CUDA_LIB").unwrap();
    let enabled = std::env::var("CUA_S1_GRAPH").as_deref() == Ok("1");
    let mut legacy = Model::load(Path::new(&dir), Path::new(&library)).unwrap();
    let mut explicit =
        Model::load_with_graph(Path::new(&dir), Path::new(&library), enabled).unwrap();
    let ids = vec![100; 128];
    let first = legacy.forward(&ids).unwrap();
    assert_eq!(first, explicit.forward(&ids).unwrap());
    assert_eq!(legacy.forward(&ids).unwrap(), first);
    assert_eq!(legacy.graph_stats().requested, enabled);
    assert_eq!(legacy.graph_stats().replays, u64::from(enabled));
}
