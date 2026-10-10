//! GPU checks of `Model::forward_shared` against `forward_fixed` on each full prompt. They
//! need a GPU, CUA_S1_CUDA_LIB pointing at libqwen3_5_cuda.so and QWEN3_5_MODEL pointing at
//! a merged Qwen3.5/3.8 checkpoint, such as a Cua-S1 or Open-Jev export:
//!
//!     CUA_S1_CUDA_LIB=$PWD/target/release/libqwen3_5_cuda.so QWEN3_5_MODEL=<export> \
//!       cargo test --release -p omni-qwen3-5-native --test prefix -- --ignored --test-threads=1
//!
//! With CUA_S1_GRAPH=1, repeated plain `forward` calls of one length replay a CUDA Graph,
//! also between shared runs.

use std::path::PathBuf;

use omni_qwen3_5_native::model::{Model, PromptGroup, SharedPrompts};

fn model() -> Model {
    let var = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .unwrap_or_else(|| panic!("{name} must be set"))
    };
    Model::load(&var("QWEN3_5_MODEL"), &var("CUA_S1_CUDA_LIB")).unwrap()
}

/// Token ids in [1000, 100000) from a fixed seed, inside every Qwen3.5 vocabulary.
fn tokens(n: usize, seed: u64) -> Vec<u32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1000 + (x >> 33) as u32 % 99_000
        })
        .collect()
}

struct Case {
    prefix: Vec<u32>,
    groups: Vec<(Vec<u32>, Vec<Vec<u32>>)>,
}

impl Case {
    /// A request prefix of `p` tokens and groups of (prefix length, branch lengths).
    fn new(p: usize, groups: &[(usize, &[usize])], seed: u64) -> Self {
        let mut seed = seed * 1000;
        let mut next = |n| {
            seed += 1;
            tokens(n, seed)
        };
        Self {
            prefix: next(p),
            groups: groups
                .iter()
                .map(|&(q, branches)| (next(q), branches.iter().map(|&b| next(b)).collect()))
                .collect(),
        }
    }

    fn prompts(&self) -> SharedPrompts<'_> {
        SharedPrompts {
            prefix: &self.prefix,
            groups: self
                .groups
                .iter()
                .map(|(prefix, branches)| PromptGroup {
                    prefix,
                    branches: branches.iter().map(Vec::as_slice).collect(),
                })
                .collect(),
        }
    }

    fn full(&self, g: usize, b: usize) -> Vec<u32> {
        let (prefix, branches) = &self.groups[g];
        [&self.prefix[..], prefix, &branches[b]].concat()
    }
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let diff: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    let norm: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (diff / norm).sqrt()
}

fn bits(rows: &[Vec<Vec<f32>>]) -> Vec<u32> {
    rows.iter()
        .flatten()
        .flatten()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB and QWEN3_5_MODEL"]
fn shared_prefixes_match_full_prompts() {
    let mut model = model();
    // Request prefixes around and inside 64-token chunks; groups without a prefix of their
    // own, with one token, 70 and 128; branches of 1 to 200 tokens.
    let groups: &[(usize, &[usize])] =
        &[(0, &[1, 64]), (1, &[2, 65]), (70, &[1, 200]), (128, &[3])];
    for (i, p) in [0usize, 1, 63, 64, 65, 300, 1000].into_iter().enumerate() {
        let case = Case::new(p, groups, i as u64 + 1);
        let shared = model.forward_shared(&case.prompts()).unwrap();
        let mut to_forward = 0f64;
        for (g, rows) in shared.iter().enumerate() {
            for (b, row) in rows.iter().enumerate() {
                let full = case.full(g, b);
                let want = model.forward_fixed(&full).unwrap();
                assert!(row.iter().all(|v| v.is_finite()));
                assert!(
                    row.iter()
                        .map(|v| v.to_bits())
                        .eq(want.iter().map(|v| v.to_bits())),
                    "prefix {p}, group {g}, branch {b}"
                );
                // the GEMM algorithms differ from `forward`'s, not the math
                let plain = model.forward(&full).unwrap();
                to_forward = to_forward.max(rel_l2(row, &plain));
            }
        }
        eprintln!(
            "prefix {p}: bit-identical to forward_fixed; largest relative L2 difference from forward {to_forward:.2e}"
        );
        // A sanity bound for the change of GEMM algorithms, not a tolerance for reuse.
        assert!(to_forward <= 3e-2, "prefix {p}: {to_forward}");
    }
}

#[test]
#[ignore = "needs a GPU, CUA_S1_CUDA_LIB and QWEN3_5_MODEL"]
fn shared_prefix_state_stays_inside_one_call() {
    let mut model = model();
    let a = Case::new(200, &[(10, &[3, 40, 1]), (0, &[64]), (65, &[2, 7])], 11);
    // longer than any prompt before it, so the buffers grow in between
    let b = Case::new(1100, &[(30, &[9, 100])], 12);
    let plain = model.forward(&a.full(0, 1)).unwrap();
    // with CUA_S1_GRAPH=1, the first call captured a graph and this one replays it
    assert_eq!(model.forward(&a.full(0, 1)).unwrap(), plain);
    let first = model.forward_shared(&a.prompts()).unwrap();
    // the same request again, and after a longer one
    let again = model.forward_shared(&a.prompts()).unwrap();
    assert_eq!(bits(&first), bits(&again));
    // the plain forward between shared runs, replayed with CUA_S1_GRAPH=1
    assert_eq!(model.forward(&a.full(0, 1)).unwrap(), plain);
    model.forward_shared(&b.prompts()).unwrap();
    let after = model.forward_shared(&a.prompts()).unwrap();
    assert_eq!(bits(&first), bits(&after));
    // groups and branches in reverse order: every branch keeps its result
    let reversed = Case {
        prefix: a.prefix.clone(),
        groups: a
            .groups
            .iter()
            .rev()
            .map(|(prefix, branches)| (prefix.clone(), branches.iter().rev().cloned().collect()))
            .collect(),
    };
    let back = model.forward_shared(&reversed.prompts()).unwrap();
    let mut back: Vec<Vec<Vec<f32>>> = back.into_iter().rev().collect();
    back.iter_mut().for_each(|branches| branches.reverse());
    assert_eq!(bits(&first), bits(&back));
    // the plain forward is unaffected (eager here: the longer request dropped the graphs)
    assert_eq!(model.forward(&a.full(0, 1)).unwrap(), plain);
    // Invalid requests fail without breaking the model. All of these are refused before
    // any GPU work is queued.
    let empty: &[u32] = &[];
    let outside: &[u32] = &[5, u32::MAX];
    let group = |prefix, branches| PromptGroup { prefix, branches };
    let bad = [
        SharedPrompts {
            prefix: &a.prefix,
            groups: vec![],
        },
        SharedPrompts {
            prefix: &a.prefix,
            groups: vec![group(empty, vec![])],
        },
        SharedPrompts {
            prefix: &a.prefix,
            groups: vec![group(empty, vec![empty])],
        },
        SharedPrompts {
            prefix: &a.prefix,
            groups: vec![group(empty, vec![outside])],
        },
        SharedPrompts {
            prefix: &a.prefix,
            groups: vec![group(outside, vec![&a.prefix])],
        },
        SharedPrompts {
            prefix: outside,
            groups: vec![group(empty, vec![&a.prefix])],
        },
    ];
    for prompts in &bad {
        assert!(model.forward_shared(prompts).is_err());
    }
    assert!(model.forward_fixed(outside).is_err());
    let last = model.forward_shared(&a.prompts()).unwrap();
    assert_eq!(bits(&first), bits(&last));
}
