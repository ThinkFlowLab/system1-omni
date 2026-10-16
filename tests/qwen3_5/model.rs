use super::*;

fn seq(n: usize, base: u32) -> Vec<u32> {
    (0..n as u32).map(|i| base + i).collect()
}

#[test]
fn runs_keep_every_prompt_and_end_prefixes_on_chunks() {
    for p in [0usize, 1, 63, 64, 65, 130] {
        let prefix = seq(p, 0);
        let groups = [
            (seq(0, 1000), vec![seq(1, 2000), seq(70, 3000)]),
            (seq(1, 4000), vec![seq(2, 5000)]),
            (seq(64, 6000), vec![seq(5, 7000), seq(1, 8000)]),
        ];
        let prompts = SharedPrompts {
            prefix: &prefix,
            groups: groups
                .iter()
                .map(|(prefix, branches)| PromptGroup {
                    prefix,
                    branches: branches.iter().map(Vec::as_slice).collect(),
                })
                .collect(),
        };
        let r = runs(&prompts);
        let chunk = |n: usize| n - n % GDN_CHUNK;
        assert_eq!(r.prefix.len(), chunk(p), "p = {p}");
        for ((group, branches), run) in groups.iter().zip(&r.groups) {
            assert_eq!(r.prefix.len() + run.prefix.len(), chunk(p + group.len()));
            assert_eq!(branches.len(), run.branches.len());
            for (branch, run_branch) in branches.iter().zip(&run.branches) {
                assert!(!run_branch.is_empty());
                assert_eq!(
                    [&r.prefix[..], &run.prefix, run_branch].concat(),
                    [&prefix[..], group, branch].concat(),
                    "p = {p}"
                );
            }
        }
    }
}

#[test]
fn token_logprob_is_a_log_softmax_over_the_vocabulary_only() {
    let vocab = 1001;
    // logits in [-20, 20) and, past the vocabulary, padding that must not count
    let mut x = 7u64;
    let mut values: Vec<f32> = (0..vocab)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 40.0 - 20.0
        })
        .collect();
    values.extend([1e4f32; 7]);
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    let rounded: Vec<f64> = values[..vocab]
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f64())
        .collect();
    let lse = rounded.iter().map(|v| v.exp()).sum::<f64>().ln();
    for token in [0u32, 1, 500, 1000] {
        let want = rounded[token as usize] - lse;
        let got = token_logprob(&bytes, vocab, token) as f64;
        assert!((got - want).abs() <= 1e-5, "token {token}: {got} vs {want}");
    }
    let total: f64 = (0..vocab as u32)
        .map(|t| (token_logprob(&bytes, vocab, t) as f64).exp())
        .sum();
    assert!((total - 1.0).abs() <= 1e-5, "{total}");
}
