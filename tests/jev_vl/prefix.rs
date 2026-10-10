//! R2d cache-structure tests: key stability, LRU/budget accounting, and the
//! manifest-faithful suffix-split equivalence (bit-exact vs full expansion on
//! the FROZEN R1 same-image prompts — the L1 hit path's correctness contract).

use std::path::PathBuf;
use std::sync::Arc;

use omni_jev_vl_native::caches::{CacheCfg, Caches, structure_key};
use omni_jev_vl_native::contract::{self, Part};
use omni_jev_vl_native::images::{ImageAsset, expand};

// Compile the real cache implementation against a CPU-only prefix allocation.
// PrefixState's device buffers are private, and these tests exercise cache
// ownership/accounting, not CUDA operations or prefix numerical equivalence.
extern crate self as omni_qwen3_5_native;

pub mod model {
    pub struct PrefixState {
        pub bytes: usize,
    }

    impl PrefixState {
        pub fn bytes(&self) -> usize {
            self.bytes
        }
    }
}

pub mod images {
    pub use omni_jev_vl_native::images::ImageAsset;
}

#[allow(dead_code)]
#[path = "../../src/models/jev_vl/native/src/caches.rs"]
mod cache_accounting;

fn accounting_hub(budget: usize) -> Arc<cache_accounting::Caches> {
    cache_accounting::Caches::new(cache_accounting::CacheCfg {
        enabled: true,
        l1: true,
        l2: true,
        l3: true,
        l1_max: 64,
        l2_bytes: 1 << 20,
        l3_bytes: budget,
    })
}

fn accounting_meta() -> cache_accounting::L1Meta {
    cache_accounting::L1Meta {
        pads_start: 10,
        pads_end: 106,
        p: 64,
        base_pad: 10,
        advance: 12,
        ids_prefix: vec![0; 64],
        positions_prefix: [vec![0; 64], vec![0; 64], vec![0; 64]],
    }
}

#[test]
fn publishing_prefix_enforces_budget_immediately() {
    let cache = accounting_hub(2048);
    for key in 1..=3 {
        cache.record_insert(key, accounting_meta());
        cache.record_publish_state(key, model::PrefixState { bytes: 1024 });
    }
    assert_eq!(cache.snapshot().l3_bytes, 2048);
    assert!(cache.record_get(1).is_none());
    assert!(cache.record_get(2).unwrap().state().is_some());
    assert!(cache.record_get(3).unwrap().state().is_some());
}

#[test]
fn oversized_prefix_is_not_retained() {
    let cache = accounting_hub(1024);
    let record = cache.record_insert(1, accounting_meta());
    cache.record_publish_state(1, model::PrefixState { bytes: 2048 });
    assert!(record.state().is_none());
    assert_eq!(cache.snapshot().l3_bytes, 0);
}

#[test]
fn zero_budget_disables_state_retention() {
    let cache = accounting_hub(0);
    let record = cache.record_insert(1, accounting_meta());
    cache.record_publish_state(1, model::PrefixState { bytes: 1024 });
    assert!(record.state().is_none());
    assert_eq!(cache.snapshot().l3_bytes, 0);
}

#[test]
fn zero_record_limit_retains_no_structure() {
    let cache = hub(|cfg| cfg.l1_max = 0);
    cache.record_insert(
        1,
        omni_jev_vl_native::caches::L1Meta {
            pads_start: 10,
            pads_end: 106,
            p: 64,
            base_pad: 10,
            advance: 12,
            ids_prefix: vec![0; 64],
            positions_prefix: [vec![0; 64], vec![0; 64], vec![0; 64]],
        },
    );
    assert!(cache.record_get(1).is_none());
    assert_eq!(cache.snapshot().l1_records, 0);
}

#[test]
fn repeated_publication_and_eviction_preserve_live_state() {
    let cache = accounting_hub(1024);
    let record = cache.record_insert(1, accounting_meta());
    cache.record_publish_state(1, model::PrefixState { bytes: 1024 });
    let in_flight = record.state().unwrap();
    cache.record_publish_state(1, model::PrefixState { bytes: 1024 });
    assert_eq!(cache.snapshot().l3_bytes, 1024);
    assert!(Arc::ptr_eq(&in_flight, &record.state().unwrap()));

    cache.record_insert(2, accounting_meta());
    cache.record_publish_state(2, model::PrefixState { bytes: 1024 });
    assert!(cache.record_get(1).is_none());
    assert_eq!(cache.snapshot().l3_bytes, 1024);
    assert_eq!(in_flight.bytes(), 1024);
}

#[test]
fn stats_and_prefix_publication_finish_concurrently() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Barrier, mpsc};
    use std::time::Duration;

    let cache = accounting_hub(64 * 1024);
    let start = Arc::new(Barrier::new(2));
    let done = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel();
    let publisher = cache.clone();
    let publisher_start = start.clone();
    let publisher_done = done.clone();
    let publisher_tx = tx.clone();
    std::thread::spawn(move || {
        publisher_start.wait();
        for key in 0..20_000 {
            publisher.record_insert(key, accounting_meta());
            publisher.record_publish_state(key, model::PrefixState { bytes: 1024 });
        }
        publisher_done.store(true, Ordering::Release);
        publisher_tx.send(()).unwrap();
    });
    std::thread::spawn(move || {
        start.wait();
        while !done.load(Ordering::Acquire) {
            cache.snapshot();
        }
        tx.send(()).unwrap();
    });
    for _ in 0..2 {
        rx.recv_timeout(Duration::from_secs(10))
            .expect("cache publication and stats must not deadlock");
    }
}

fn asset(grid: [i64; 3]) -> Arc<ImageAsset> {
    let [t, h, w] = grid;
    let n = (t * h * w / 4) as usize;
    Arc::new(ImageAsset {
        grid_thw: grid,
        embeddings: vec![half::bf16::ONE; n * 5120],
    })
}

#[test]
fn structure_key_is_sensitive_and_stable() {
    let parts_a = vec![Part::Text("state text".into()), Part::Image("u://a".into())];
    let parts_b = vec![Part::Text("state text".into()), Part::Image("u://b".into())];
    let parts_c = vec![Part::Image("u://a".into()), Part::Text("state text".into())];
    assert_ne!(
        structure_key("noul", &parts_a),
        structure_key("score", &parts_a)
    );
    assert_ne!(
        structure_key("noul", &parts_a),
        structure_key("noul", &parts_b)
    );
    assert_ne!(
        structure_key("noul", &parts_a),
        structure_key("noul", &parts_c)
    );
    assert_eq!(
        structure_key("noul", &parts_a),
        structure_key("noul", &parts_a)
    );
    assert_ne!(
        structure_key("noul", &parts_a),
        structure_key("noul", &parts_a[..1])
    );
}

fn hub(f: impl FnOnce(&mut CacheCfg)) -> Arc<Caches> {
    let mut cfg = CacheCfg {
        enabled: true,
        l1: true,
        l2: true,
        l3: true,
        l1_max: 64,
        l2_bytes: 1 << 20,
        l3_bytes: 1 << 20,
    };
    f(&mut cfg);
    Caches::new(cfg)
}

#[test]
fn l2_lru_hits_misses_and_evicition() {
    let c = hub(|cfg| cfg.l2_bytes = (2 * 5120 * 2 + 64) * 2 + 128);
    let (a, b, d) = (asset([1, 2, 4]), asset([1, 2, 4]), asset([1, 2, 4]));
    assert!(c.l2_get("k-a").is_none());
    c.l2_insert("k-a".into(), a.clone());
    assert!(c.l2_get("k-a").is_some());
    c.l2_insert("k-b".into(), b);
    c.l2_insert("k-d".into(), d); // a evicted: budget fits two
    assert!(c.l2_get("k-a").is_none());
    assert!(c.l2_get("k-b").is_some());
    let s = c.snapshot();
    assert_eq!(s.l2_hit, 2);
    assert_eq!(s.l2_miss, 2);
    assert_eq!(s.l2_records, 2);
    // Disabled: no reuse, all misses counted.
    let off = hub(|cfg| cfg.enabled = false);
    off.l2_insert("k-a".into(), asset([1, 2, 4]));
    assert!(off.l2_get("k-a").is_none());
    assert_eq!(off.snapshot().l2_miss, 1);
}

#[test]
fn records_are_lru_and_budgeted() {
    let c = hub(|cfg| cfg.l1_max = 2);
    let meta = |pads: usize| omni_jev_vl_native::caches::L1Meta {
        pads_start: pads,
        pads_end: pads + 96,
        p: 64,
        base_pad: pads as i64,
        advance: 12,
        ids_prefix: vec![0u32; 64],
        positions_prefix: [vec![0i64; 64], vec![0i64; 64], vec![0i64; 64]],
    };
    let r1 = c.record_insert(1, meta(10));
    c.record_insert(2, meta(20));
    assert!(c.record_get(1).is_some());
    assert!(Arc::ptr_eq(&c.record_get(1).unwrap(), &r1));
    c.record_insert(3, meta(30)); // l1_max=2: one eviction of the oldest (2)
    assert!(c.record_get(2).is_none());
    assert!(c.record_get(1).is_some());
    let s = c.snapshot();
    assert!(s.l1_hit >= 3 && s.l1_miss >= 1 && s.l1_records <= 2);
}

/// The manifest-faithful equivalence: for every img entry of the FROZEN R1
/// manifest the suffix split (pads-tail + vision_end + fresh tail tokenization)
/// must equal `expand` on the whole prompt bit for bit — that's the L1 hit path.
#[test]
#[ignore = "needs JEV_VL_EXPORT (export dir with tokenizer + manifest)"]
fn manifest_suffix_split_matches_full_expand() {
    let dir = PathBuf::from(std::env::var_os("JEV_VL_EXPORT").expect("set JEV_VL_EXPORT"));
    let manifest_dir =
        PathBuf::from(std::env::var_os("JEV_VL_MANIFEST").expect("set JEV_VL_MANIFEST"));
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("jev_vl_export.json")).unwrap()).unwrap();
    let labels: Vec<String> = serde_json::from_value(manifest["labels"].clone()).unwrap();
    let image_pad = tokenizer.token_to_id("<|image_pad|>").unwrap();
    let vision_end = tokenizer.token_to_id("<|vision_end|>").unwrap();
    let mut checked = 0;
    for line in std::fs::read_to_string(manifest_dir).unwrap().lines() {
        let entry: serde_json::Value = serde_json::from_str(line).unwrap();
        if !entry["id"].as_str().unwrap_or_default().starts_with("img-") {
            continue;
        }
        let raw = serde_json::to_vec(&entry["request"]).unwrap();
        let compiled = contract::compile(&raw, &labels).unwrap();
        let tail = contract::tail_after_last_image(&compiled, &labels).unwrap();
        let grid = [1, 60, 60];
        let e = expand(
            tokenizer
                .encode(compiled.prompt.as_str(), false)
                .unwrap()
                .get_ids(),
            image_pad,
            &[asset(grid)],
        )
        .unwrap();
        assert_eq!(e.blocks.len(), 1, "{}", entry["id"]);
        let p = e.blocks[0].end / 64 * 64;
        // Rebuild as the L1 hit path does.
        let b = &e.blocks[0];
        let suffix_pads = b.end - p;
        let tail_ids: Vec<u32> = tokenizer
            .encode(tail.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        let mut ids: Vec<u32> = std::iter::repeat_n(image_pad, suffix_pads).collect();
        ids.push(vision_end);
        ids.extend_from_slice(&tail_ids);
        assert_eq!(&ids[..], &e.ids[p..], "suffix ids diverge: {}", entry["id"]);
        let pos =
            omni_jev_vl_native::images::meshgrid_positions(grid, b.base, p - b.start, suffix_pads);
        for (a, axis) in pos.iter().enumerate() {
            assert_eq!(
                &axis[..],
                &e.positions[a][p..p + suffix_pads],
                "suffix meshgrid diverges: {}",
                entry["id"]
            );
        }
        assert_eq!(
            e.positions[0][p + suffix_pads],
            b.base + b.advance,
            "after-image base: {}",
            entry["id"]
        );
        // The structure key is shared across the 12 same-image entries per kind.
        checked += 1;
    }
    assert_eq!(checked, 12, "the frozen manifest carries 12 img entries");
}
