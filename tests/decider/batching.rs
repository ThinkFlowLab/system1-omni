use omni_decider_native::batching::BatchLimits;
#[test]
fn bounded_contiguous_ranges_preserve_order_and_long_rows() {
    let limits = BatchLimits::new(3, 8).unwrap();
    assert_eq!(
        limits.ranges(&[2, 3, 3, 1, 9, 2, 2, 2, 2]),
        vec![0..3, 3..4, 4..5, 5..8, 8..9]
    );
    assert!(limits.ranges(&[]).is_empty());
    assert_eq!(limits.ranges(&[9, 9]), vec![0..1, 1..2]);
}
#[test]
fn row_limit_and_exact_token_boundary_are_independent() {
    assert_eq!(
        BatchLimits::new(2, 4096).unwrap().ranges(&[1, 1, 1]),
        vec![0..2, 2..3]
    );
    assert_eq!(
        BatchLimits::new(4, 4096).unwrap().ranges(&[2048, 2048, 1]),
        vec![0..2, 2..3]
    );
    assert_eq!(
        BatchLimits::default().ranges(&[1, 2, 3]),
        vec![0..1, 1..2, 2..3]
    );
}
#[test]
fn startup_limits_reject_unbounded_or_malformed_settings() {
    for (rows, tokens) in [(0, 1), (5, 1), (8, 1), (16, 1), (1, 0), (1, 4097)] {
        assert!(BatchLimits::new(rows, tokens).is_err());
    }
    assert!(BatchLimits::from_values(Some("wrong"), None).is_err());
    assert!(BatchLimits::from_values(None, Some("-1")).is_err());
    let limits = BatchLimits::from_values(Some("4"), Some("4096")).unwrap();
    assert_eq!(limits.max_rows(), 4);
    assert_eq!(limits.max_tokens(), 4096);
}
