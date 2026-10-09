use super::ranges;

#[test]
fn preserves_all_candidates_across_token_and_sequence_limits() {
    let prompts: Vec<Vec<u32>> = [2048, 2048, 1, 5000, 3]
        .into_iter()
        .chain(std::iter::repeat_n(4, 17))
        .map(|length| vec![0; length])
        .collect();
    let inputs: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
    let batches = ranges(&inputs);
    assert_eq!(batches, vec![0..2, 2..3, 3..4, 4..20, 20..22]);
    assert_eq!(
        batches.into_iter().flatten().collect::<Vec<_>>(),
        (0..inputs.len()).collect::<Vec<_>>()
    );
}
