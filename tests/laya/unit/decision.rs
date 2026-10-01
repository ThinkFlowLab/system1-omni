use super::round4;

#[test]
fn rounding_matches_python_at_decimal_boundaries() {
    for (value, expected) in [(0.00035, 0.0003), (0.12345, 0.1235), (-0.00035, -0.0003)] {
        assert_eq!(round4(value), expected);
    }
}
