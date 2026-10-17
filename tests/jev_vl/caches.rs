use std::sync::Arc;

use omni_jev_vl_native::caches::{CacheCfg, Caches};
use omni_jev_vl_native::images::ImageAsset;

const ASSET_BYTES: usize = 5120 * 2 + 64;

fn asset() -> Arc<ImageAsset> {
    Arc::new(ImageAsset {
        grid_thw: [1, 2, 2],
        embeddings: vec![half::bf16::ONE; 5120],
    })
}

fn caches(enabled: bool, l2: bool, budget: usize) -> Arc<Caches> {
    Caches::new(CacheCfg {
        enabled,
        l1: false,
        l2,
        l3: false,
        l1_max: 0,
        l2_bytes: budget,
        l3_bytes: 0,
    })
}

#[test]
fn l2_cold_rechecks_count_each_lookup() {
    let caches = caches(true, true, ASSET_BYTES);
    assert!(caches.l2_get("image").is_none());
    assert!(caches.l2_get("image").is_none());
    let stats = caches.snapshot();
    assert_eq!(stats.l2_hit, 0);
    assert_eq!(stats.l2_miss, 2);
}

#[test]
fn l2_recheck_sees_a_queued_publication_and_refreshes_lru() {
    let caches = caches(true, true, ASSET_BYTES * 2);
    // Preparation misses; another admitted request then publishes this image.
    assert!(caches.l2_get("queued").is_none());
    let queued = asset();
    caches.l2_insert("queued".into(), queued.clone());
    caches.l2_insert("older".into(), asset());
    assert!(Arc::ptr_eq(&caches.l2_get("queued").unwrap(), &queued));
    caches.l2_insert("newer".into(), asset());
    assert!(caches.l2_get("older").is_none());
    assert!(Arc::ptr_eq(&caches.l2_get("queued").unwrap(), &queued));
    let stats = caches.snapshot();
    assert_eq!(stats.l2_hit, 2);
    assert_eq!(stats.l2_miss, 2);
    assert_eq!(stats.l2_records, 2);
    assert_eq!(stats.l2_bytes, (ASSET_BYTES * 2) as u64);
}

#[test]
fn l2_disabled_or_over_budget_rechecks_remain_counted_misses() {
    for (enabled, l2, budget) in [
        (false, true, ASSET_BYTES),
        (true, false, ASSET_BYTES),
        (true, true, ASSET_BYTES - 1),
    ] {
        let caches = caches(enabled, l2, budget);
        caches.l2_insert("image".into(), asset());
        assert!(caches.l2_get("image").is_none());
        assert!(caches.l2_get("image").is_none());
        let stats = caches.snapshot();
        assert_eq!(stats.l2_hit, 0);
        assert_eq!(stats.l2_miss, 2);
        assert_eq!(stats.l2_records, 0);
        assert_eq!(stats.l2_bytes, 0);
    }
}
