use super::*;

fn batch_for(b: usize, l: usize, value: i64) -> Batch {
    Batch {
        b,
        l,
        input_ids: vec![value; b * l],
        lens: vec![l as i32; b],
        qtypes: vec![0; b],
        markers: vec![vec![0]; b],
    }
}

#[test]
fn cache_limits_evict_before_allocation_and_allow_exact_fits() {
    let fixture = fixture::Fixture::new(&[]);
    let mut model = fixture_model_at(false, fixture.path());
    model.graphs = true;
    let a = Workspace::required_bytes(1, 16).unwrap();
    let b = Workspace::required_bytes(1, 32).unwrap();
    model.cache_config = CacheConfig {
        max_shapes: 1,
        max_bytes: a.max(b),
    };
    let live = unsafe {
        model
            .cuda
            .symbol::<unsafe extern "C" fn() -> usize>(b"test_live_bytes\0")
            .unwrap()
    };
    let peak = unsafe {
        model
            .cuda
            .symbol::<unsafe extern "C" fn() -> usize>(b"test_peak_bytes\0")
            .unwrap()
    };
    let reset = unsafe {
        model
            .cuda
            .symbol::<unsafe extern "C" fn()>(b"test_reset_peak\0")
            .unwrap()
    };
    let weights = unsafe { live() };
    model.infer(&batch_for(1, 16, 1)).unwrap();
    assert_eq!(model.cached_bytes, a);
    unsafe { reset() };
    model.infer(&batch_for(1, 32, 2)).unwrap();
    assert_eq!(model.cache.len(), 1);
    assert_eq!(model.cached_bytes, b);
    assert_eq!(model.cache[0].bytes, b);
    assert!(
        unsafe { peak() } <= weights + b,
        "eviction must happen before new allocation"
    );
    let mut invalid = batch_for(1, 16, 3);
    invalid.input_ids[0] = 50368;
    let pointer = model.cache[0].ids.ptr();
    assert!(model.infer(&invalid).is_err());
    assert_eq!(model.cached_bytes, b);
    assert_eq!(model.cache[0].ids.ptr(), pointer);
    model.infer(&batch_for(1, 32, 7)).unwrap();
    assert_eq!(model.cache[0].ids.ptr(), pointer);
    assert_eq!(
        model.cache[0].ids.read(32 * 8).unwrap(),
        [7i64.to_le_bytes(); 32].concat()
    );
    model.infer(&batch_for(1, 16, 1)).unwrap();
    assert_eq!(model.cached_bytes, a);
}

#[test]
fn byte_budget_lru_and_disabled_or_oversized_fallback_reuse() {
    let fixture = fixture::Fixture::new(&[]);
    let mut model = fixture_model_at(false, fixture.path());
    model.graphs = true;
    let a = Workspace::required_bytes(1, 16).unwrap();
    let b = Workspace::required_bytes(1, 32).unwrap();
    model.cache_config = CacheConfig {
        max_shapes: 8,
        max_bytes: a + b,
    };
    model.infer(&batch_for(1, 16, 1)).unwrap();
    model.infer(&batch_for(1, 32, 2)).unwrap();
    assert_eq!(model.cache.len(), 2);
    assert_eq!(model.cached_bytes, a + b);
    model.infer(&batch_for(1, 16, 3)).unwrap();
    assert_eq!(model.cache.back().unwrap().l, 16);
    model.infer(&batch_for(1, 48, 4)).unwrap();
    assert_eq!(model.cache.len(), 1);
    assert_eq!(model.cache[0].l, 48);
    for config in [
        CacheConfig {
            max_shapes: 0,
            max_bytes: usize::MAX,
        },
        CacheConfig {
            max_shapes: 8,
            max_bytes: 0,
        },
        CacheConfig {
            max_shapes: 8,
            max_bytes: a - 1,
        },
    ] {
        model.clear_cache();
        model.cache_config = config;
        let begins = unsafe {
            model
                .cuda
                .symbol::<unsafe extern "C" fn() -> i32>(b"test_begins\0")
                .unwrap()
        };
        let before = unsafe { begins() };
        model.infer(&batch_for(1, 16, 5)).unwrap();
        assert_eq!(model.cache.len(), 0);
        assert_eq!(model.cached_bytes, 0);
        let pointer = model.eager.as_ref().unwrap().ids.ptr();
        model.infer(&batch_for(1, 16, 6)).unwrap();
        assert_eq!(model.eager.as_ref().unwrap().ids.ptr(), pointer);
        assert!(model.eager.as_ref().unwrap().graph.is_none());
        assert_eq!(unsafe { begins() }, before);
        assert_eq!(
            model.eager.as_ref().unwrap().ids.read(16 * 8).unwrap(),
            [6i64.to_le_bytes(); 16].concat()
        );
        model.infer(&batch_for(1, 32, 7)).unwrap();
        assert_eq!(model.eager.as_ref().unwrap().l, 32);
    }
}

#[test]
fn workspace_estimate_matches_allocation_for_all_supported_shapes() {
    let fixture = fixture::Fixture::new(&[]);
    let model = fixture_model_at(false, fixture.path());
    let live = unsafe {
        model
            .cuda
            .symbol::<unsafe extern "C" fn() -> usize>(b"test_live_bytes\0")
            .unwrap()
    };
    let weights = unsafe { live() };
    for b in [1, 2, 4, 8, 16] {
        for l in (16..=512).step_by(16) {
            let workspace = Workspace::new(&model.cuda, b, l).unwrap();
            let allocated = unsafe { live() } - weights;
            assert_eq!(
                Workspace::required_bytes(b, l).unwrap(),
                allocated,
                "B={b}, L={l}"
            );
            assert_eq!(workspace.bytes, allocated);
            drop(workspace);
            assert_eq!(unsafe { live() }, weights);
        }
    }
    for (b, l) in [(0, 16), (3, 16), (32, 16), (1, 0), (1, 17), (1, 528)] {
        assert!(Workspace::required_bytes(b, l).is_err());
    }
}

#[test]
fn clear_releases_graph_and_eager_workspaces_and_allows_reuse() {
    let fixture = fixture::Fixture::new(&[]);
    let mut model = fixture_model_at(false, fixture.path());
    model.graphs = true;
    model.cache_config = CacheConfig {
        max_shapes: 1,
        max_bytes: Workspace::required_bytes(1, 16).unwrap(),
    };
    let live = unsafe {
        model
            .cuda
            .symbol::<unsafe extern "C" fn() -> usize>(b"test_live_bytes\0")
            .unwrap()
    };
    let weights = unsafe { live() };
    model.infer(&batch_for(1, 16, 1)).unwrap();
    model.infer(&batch_for(1, 32, 2)).unwrap();
    assert_eq!(model.cache.len(), 1);
    assert!(model.eager.is_some());
    model.clear_cache();
    assert_eq!(model.cached_bytes, 0);
    assert!(model.cache.is_empty());
    assert!(model.eager.is_none());
    assert_eq!(unsafe { live() }, weights);
    model.clear_cache();
    model.infer(&batch_for(1, 16, 3)).unwrap();
    assert_eq!(model.cache.len(), 1);
}

#[test]
fn failed_capture_admission_releases_partial_workspace_and_can_retry() {
    let fixture = fixture::Fixture::new(&[]);
    let mut model = fixture_model_at(false, fixture.path());
    model.graphs = true;
    model.cache_config = CacheConfig {
        max_shapes: 1,
        max_bytes: usize::MAX,
    };
    let live = unsafe {
        model
            .cuda
            .symbol::<unsafe extern "C" fn() -> usize>(b"test_live_bytes\0")
            .unwrap()
    };
    let weights = unsafe { live() };
    let mode = unsafe {
        model
            .cuda
            .symbol::<unsafe extern "C" fn(i32)>(b"test_graph_mode\0")
            .unwrap()
    };
    model.infer(&batch_for(1, 16, 1)).unwrap();
    assert_eq!(model.cache.len(), 1);
    unsafe { mode(1) };
    assert!(model.infer(&batch_for(1, 32, 2)).is_err());
    unsafe { mode(0) };
    assert!(model.cache.is_empty());
    assert_eq!(model.cached_bytes, 0);
    assert_eq!(unsafe { live() }, weights);
    model.infer(&batch_for(1, 16, 3)).unwrap();
    assert_eq!(model.cache.len(), 1);
    assert_eq!(
        model.cached_bytes,
        Workspace::required_bytes(1, 16).unwrap()
    );
}
