use omni_decider_native::{Kind, RowInput, prefix::PrefixPlan};
fn row(ids: Vec<u32>, q: &str) -> RowInput {
    let n = ids.len();
    RowInput {
        ids,
        readout_position: n - 1,
        candidate_ids: vec![32, 33],
        question_id: q.into(),
        level_index: None,
        temperature: 1.,
        kind: Kind::Choice,
    }
}
#[test]
fn exact_token_prefixes_keep_origins_and_nonempty_branches() {
    let mut a = vec![100; 130];
    a.push(101);
    let mut b = vec![100; 130];
    b.push(102);
    let rows = vec![row(a, "q"), row(b, "q")];
    let plan = PrefixPlan::new(&rows).unwrap();
    assert_eq!(plan.saved_tokens, 128);
    assert_eq!(plan.prompts.groups.len(), 1);
    for (row, branch) in rows.iter().zip(&plan.prompts.groups[0].branches) {
        assert!(!branch.is_empty());
        assert_eq!(
            [plan.prompts.prefix, plan.prompts.groups[0].prefix, *branch].concat(),
            row.ids
        );
    }
}
#[test]
fn question_prefix_and_request_prefix_are_independent_and_ordered() {
    let mut a = vec![100; 64];
    a.extend(vec![101; 65]);
    a.push(102);
    let mut b = a.clone();
    *b.last_mut().unwrap() = 103;
    let mut c = vec![100; 64];
    c.push(104);
    let rows = vec![row(a, "score"), row(b, "score"), row(c, "last")];
    let plan = PrefixPlan::new(&rows).unwrap();
    assert_eq!(plan.saved_tokens, 192);
    assert_eq!(plan.prompts.groups.len(), 2);
    let rebuilt: Vec<_> = plan
        .prompts
        .groups
        .iter()
        .flat_map(|group| {
            group
                .branches
                .iter()
                .map(|branch| [plan.prompts.prefix, group.prefix, *branch].concat())
        })
        .collect();
    assert_eq!(
        rebuilt,
        rows.iter().map(|row| row.ids.clone()).collect::<Vec<_>>()
    );
}
#[test]
fn short_unique_single_and_identical_rows_fallback_safely() {
    assert!(PrefixPlan::new(&[]).is_none());
    assert!(PrefixPlan::new(&[row(vec![100; 100], "q")]).is_none());
    assert!(PrefixPlan::new(&[row(vec![100; 63], "q"), row(vec![100; 63], "q")]).is_none());
    assert!(PrefixPlan::new(&[row(vec![100; 100], "q"), row(vec![101; 100], "q")]).is_none());
    let rows = vec![row(vec![100; 65], "q"), row(vec![100; 65], "q")];
    let plan = PrefixPlan::new(&rows).unwrap();
    assert_eq!(plan.saved_tokens, 64);
    assert!(
        plan.prompts.groups[0]
            .branches
            .iter()
            .all(|b| !b.is_empty())
    );
}

#[test]
fn auto_reuse_requires_saved_work_and_a_large_shared_fraction() {
    let pair = |prefix: usize, suffix: usize| {
        let mut a = vec![100; prefix];
        a.extend(vec![101; suffix]);
        let mut b = vec![100; prefix];
        b.extend(vec![102; suffix]);
        vec![row(a, "q"), row(b, "q")]
    };
    // Exactly 4096 aligned tokens saved and at least one third of the original work.
    let fits = pair(4096, 1);
    assert!(PrefixPlan::new(&fits).unwrap().worth_auto(&fits));
    let too_small = pair(4032, 1);
    assert!(!PrefixPlan::new(&too_small).unwrap().worth_auto(&too_small));
    let exact_fraction = pair(4096, 2048);
    assert!(
        PrefixPlan::new(&exact_fraction)
            .unwrap()
            .worth_auto(&exact_fraction)
    );
    let below_fraction = pair(4096, 2049);
    assert!(
        !PrefixPlan::new(&below_fraction)
            .unwrap()
            .worth_auto(&below_fraction)
    );
    let too_much_suffix = pair(4096, 8192);
    assert!(
        !PrefixPlan::new(&too_much_suffix)
            .unwrap()
            .worth_auto(&too_much_suffix)
    );
}
