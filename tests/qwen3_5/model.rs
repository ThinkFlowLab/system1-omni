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
