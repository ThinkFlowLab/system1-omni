//! Process-local caches for prepared images and reusable language prefixes.
//!
//! - L1 processor: a per-prefix-structure record. Key = sha256 over the exact
//!   request structure (kind + every state part: text verbatim / image url).
//!   Value = the compiled prefix geometry (pad run, cached-prefix length P,
//!   position bases, the expanded prefix ids/positions), so a hit skips the
//!   prompt-head assembly, tokenization, expansion and meshgrid work; the suffix
//!   (`<|vision_end|>` onward) is always tokenized fresh per question.
//! - L2 vision: parsed image assets (adapter output rows + grid) keyed by
//!   sha256(url) — the same key form as the offline imgcache directory. A hit
//!   skips the disk read, safetensors parsing and validation. Vision encoding
//!   runs offline and is not part of this cache lookup.
//! - L3 KV prefix: the device-side prefix state (per-full-attention-layer
//!   post-prep K and raw V rows; per-GDN-layer float32 recurrent state + conv
//!   tail). Holders are `Arc<PrefixState>`; a continuation reads it while the
//!   device buffers stay alive. Full-attention KV is per-prefix; the GDN/conv
//!   states are per-prefix too — identical token prefix ⇒ identical state, and
//!   they never blend across different prefixes (hybrid constraint).
//!
//! All collections are process-local, LRU + budgeted; everything is disabled by
//! JEV_VL_CACHE=0 and per level by JEV_VL_L1/L2/L3=0 (A/B decomposition).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use omni_qwen3_5_native::model::PrefixState;

use crate::images::ImageAsset;

/// Env-read switches + budgets.
pub struct CacheCfg {
    pub enabled: bool,
    pub l1: bool,
    pub l2: bool,
    pub l3: bool,
    /// Max structural records.
    pub l1_max: usize,
    /// L2 parsed-asset budget in bytes.
    pub l2_bytes: usize,
    /// L3 retained device-state budget in bytes; zero disables retention.
    pub l3_bytes: usize,
}

impl CacheCfg {
    pub fn from_env() -> Self {
        let flag = |name: &str, default: bool| {
            std::env::var(name).map_or(default, |v| v != "0" && !v.eq_ignore_ascii_case("false"))
        };
        let num = |name: &str, default: usize| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        Self {
            enabled: flag("JEV_VL_CACHE", true),
            l1: flag("JEV_VL_L1", true),
            l2: flag("JEV_VL_L2", true),
            l3: flag("JEV_VL_L3", true),
            l1_max: num("JEV_VL_L1_MAX", 256),
            l2_bytes: num("JEV_VL_L2_BYTES", 1 << 30),
            l3_bytes: num("JEV_VL_L3_BYTES", 2 << 30),
        }
    }
}

#[derive(Clone, Default)]
pub struct CacheStats {
    pub l1_hit: u64,
    pub l1_miss: u64,
    pub l2_hit: u64,
    pub l2_miss: u64,
    pub l3_hit: u64,
    pub l3_miss: u64,
    pub l3_populate: u64,
    pub l3_fallback_full: u64,
    pub l1_records: u64,
    pub l1_bytes: u64,
    pub l2_records: u64,
    pub l2_bytes: u64,
    pub l3_records: u64,
    pub l3_bytes: u64,
}

/// The compiled geometry of one structural prefix (small; see module doc).
pub struct L1Meta {
    /// Global row of the last image block's first pad; also its position base.
    pub pads_start: usize,
    /// Global row one past the last image block's last pad.
    pub pads_end: usize,
    /// Cached-prefix length, a multiple of 64: floor(pads_end / 64) * 64.
    pub p: usize,
    /// Position base of the pads (text-position base for the meshgrid offset).
    pub base_pad: i64,
    /// mrope advance after the image: max(grid_h, grid_w) / 2.
    pub advance: i64,
    /// Expanded ids of rows [0, p) (needed by the L1-only mode's full rebuild).
    pub ids_prefix: Vec<u32>,
    /// Absolute positions of rows [0, p).
    pub positions_prefix: [Vec<i64>; 3],
}

impl L1Meta {
    fn bytes(&self) -> usize {
        self.ids_prefix.len() * 4 + 3 * self.positions_prefix[0].len() * 8 + 96
    }
}

struct Registry {
    map: HashMap<u64, Arc<PrefixRecord>>,
    lru: VecDeque<u64>,
    meta_bytes: usize,
    state_bytes: usize,
    max_records: usize,
    state_budget: usize,
}

/// One structural record: compiled prefix geometry + (lazily) its device state.
pub struct PrefixRecord {
    pub key: u64,
    pub meta: L1Meta,
    state: OnceLock<Arc<PrefixState>>,
}

impl PrefixRecord {
    pub fn state(&self) -> Option<Arc<PrefixState>> {
        self.state.get().cloned()
    }
}

/// Parsed image assets (L2).
struct L2Store {
    map: HashMap<String, Arc<ImageAsset>>,
    lru: VecDeque<String>,
    bytes: usize,
    budget: usize,
}

/// Cache state shared by the processor and executor.
pub struct Caches {
    pub cfg: CacheCfg,
    l2: Mutex<L2Store>,
    registry: Mutex<Registry>,
    stats: Mutex<CacheStats>,
}

impl Caches {
    pub fn new(cfg: CacheCfg) -> Arc<Self> {
        Arc::new(Self {
            l2: Mutex::new(L2Store {
                map: HashMap::new(),
                lru: VecDeque::new(),
                bytes: 0,
                budget: cfg.l2_bytes,
            }),
            registry: Mutex::new(Registry {
                map: HashMap::new(),
                lru: VecDeque::new(),
                meta_bytes: 0,
                state_bytes: 0,
                max_records: cfg.l1_max,
                state_budget: cfg.l3_bytes,
            }),
            cfg,
            stats: Mutex::new(CacheStats::default()),
        })
    }

    fn bump(&self, f: impl FnOnce(&mut CacheStats)) {
        if let Ok(mut s) = self.stats.lock() {
            f(&mut s)
        }
    }

    pub fn snapshot(&self) -> CacheStats {
        let mut s = self.stats.lock().map(|s| s.clone()).unwrap_or_default();
        if let Ok(r) = self.registry.lock() {
            s.l1_records = r.map.len() as u64;
            s.l1_bytes = r.meta_bytes as u64;
            s.l3_records = r.map.values().filter(|rec| rec.state().is_some()).count() as u64;
            s.l3_bytes = r.state_bytes as u64;
        }
        if let Ok(l2) = self.l2.lock() {
            s.l2_records = l2.map.len() as u64;
            s.l2_bytes = l2.bytes as u64;
        }
        s
    }

    pub fn reset_stats(&self) {
        if let Ok(mut s) = self.stats.lock() {
            *s = CacheStats::default();
        }
    }

    // ---- L2: parsed image assets ----

    pub fn l2_get(&self, key: &str) -> Option<Arc<ImageAsset>> {
        if !(self.cfg.enabled && self.cfg.l2) {
            self.bump(|s| s.l2_miss += 1);
            return None;
        }
        let hit = self.l2.lock().ok().and_then(|mut st| {
            let asset = st.map.get(key).cloned();
            if asset.is_some() {
                st.lru.retain(|k| k != key);
                st.lru.push_back(key.to_owned());
            }
            asset
        });
        match &hit {
            Some(_) => self.bump(|s| s.l2_hit += 1),
            None => self.bump(|s| s.l2_miss += 1),
        }
        hit
    }

    /// Insert a freshly loaded asset, evicting LRU entries over the byte budget.
    pub fn l2_insert(&self, key: String, asset: Arc<ImageAsset>) {
        if !(self.cfg.enabled && self.cfg.l2) {
            return;
        }
        let bytes = asset.embeddings.len() * 2 + 64;
        if bytes <= self.cfg.l2_bytes
            && let Ok(mut st) = self.l2.lock()
        {
            st.lru.retain(|k| k != &key);
            st.lru.push_back(key.clone());
            if let Some(old) = st.map.insert(key, asset) {
                st.bytes = st.bytes.saturating_sub(old.embeddings.len() * 2 + 64);
            }
            st.bytes += bytes;
            while st.bytes > st.budget && st.lru.len() > 1 {
                let evict = st.lru.pop_front().unwrap();
                if let Some(old) = st.map.remove(&evict) {
                    st.bytes = st.bytes.saturating_sub(old.embeddings.len() * 2 + 64);
                }
            }
        }
    }

    // ---- L1 + L3: structural records ----

    pub fn record_get(&self, key: u64) -> Option<Arc<PrefixRecord>> {
        if !(self.cfg.enabled && self.cfg.l1) {
            self.bump(|s| s.l1_miss += 1);
            return None;
        }
        let hit = self.registry.lock().ok().and_then(|mut r| {
            let rec = r.map.get(&key).cloned();
            if rec.is_some() {
                r.lru.retain(|k| *k != key);
                r.lru.push_back(key);
            }
            rec
        });
        match &hit {
            Some(_) => self.bump(|s| s.l1_hit += 1),
            None => self.bump(|s| s.l1_miss += 1),
        }
        hit
    }

    /// Insert a new structural record, evicting LRU records over the record
    /// count (evicted device states drop with their Arc).
    pub fn record_insert(&self, key: u64, meta: L1Meta) -> Arc<PrefixRecord> {
        let rec = Arc::new(PrefixRecord {
            key,
            state: OnceLock::new(),
            meta,
        });
        if !(self.cfg.enabled && self.cfg.l1) || self.cfg.l1_max == 0 {
            return rec;
        }
        if let Ok(mut r) = self.registry.lock() {
            r.meta_bytes = r.meta_bytes.saturating_add(rec.meta.bytes());
            r.lru.retain(|k| *k != key);
            r.lru.push_back(key);
            if let Some(old) = r.map.insert(key, rec.clone()) {
                r.meta_bytes = r.meta_bytes.saturating_sub(old.meta.bytes());
                if let Some(s) = old.state() {
                    r.state_bytes = r.state_bytes.saturating_sub(s.bytes());
                }
            }
            loop {
                let over = r.map.len() > r.max_records;
                if !over || r.lru.len() <= 1 {
                    break;
                }
                let evict = r.lru.pop_front().unwrap();
                if evict == key {
                    r.lru.push_front(evict);
                    break;
                }
                if let Some(old) = r.map.remove(&evict) {
                    r.meta_bytes = r.meta_bytes.saturating_sub(old.meta.bytes());
                    if let Some(s) = old.state() {
                        r.state_bytes = r.state_bytes.saturating_sub(s.bytes());
                    }
                }
            }
        }
        rec
    }

    /// Publish the device state for a record the executor just populated;
    /// Enforce the retained-state budget before publication. No-op when the
    /// record was already populated, evicted, or larger than the budget.
    pub fn record_publish_state(&self, key: u64, state: PrefixState) {
        if !(self.cfg.enabled && self.cfg.l3) {
            return;
        }
        let Ok(mut r) = self.registry.lock() else {
            return;
        };
        let Some(rec) = r.map.get(&key).cloned() else {
            return;
        };
        let bytes = state.bytes();
        if rec.state.get().is_some() || bytes > r.state_budget || r.state_budget == 0 {
            return;
        }
        // Publication and eviction share the registry lock. A state is immutable
        // once published, so readers never need a second, oppositely ordered lock.
        r.lru.retain(|k| *k != key);
        r.lru.push_back(key);
        while r.state_bytes > r.state_budget - bytes {
            let Some(evict) = r.lru.pop_front() else {
                return;
            };
            if let Some(old) = r.map.remove(&evict) {
                r.meta_bytes = r.meta_bytes.saturating_sub(old.meta.bytes());
                if let Some(s) = old.state() {
                    r.state_bytes = r.state_bytes.saturating_sub(s.bytes());
                }
            }
        }
        let _ = rec.state.set(Arc::new(state));
        r.state_bytes += bytes;
    }

    pub fn l3_hit(&self) {
        self.bump(|s| s.l3_hit += 1);
    }

    pub fn l3_miss(&self) {
        self.bump(|s| s.l3_miss += 1);
    }

    pub fn l3_populate(&self) {
        self.bump(|s| s.l3_populate += 1);
    }

    pub fn l3_fallback_full(&self) {
        self.bump(|s| s.l3_fallback_full += 1);
    }
}

/// Key of one structural prefix: kind, then every state part verbatim
/// (text body / image url). Two requests share L1..L3 state only through an
/// identical key — the "same image, many questions" anchor.
pub fn structure_key(kind: &str, parts: &[crate::contract::Part]) -> u64 {
    use sha2::Digest;
    let mut h = sha2::Sha256::new();
    h.update(b"jev27-prefix-v1\0");
    h.update((kind.len() as u32).to_le_bytes());
    h.update(kind.as_bytes());
    for part in parts {
        match part {
            crate::contract::Part::Text(t) => {
                h.update([1u8]);
                h.update((t.len() as u64).to_le_bytes());
                h.update(t.as_bytes());
            }
            crate::contract::Part::Image(url) => {
                h.update([2u8]);
                h.update((url.len() as u64).to_le_bytes());
                h.update(url.as_bytes());
            }
        }
    }
    let digest = h.finalize();
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}
