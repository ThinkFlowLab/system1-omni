use omni_decider_native::options::graph_value;
#[test]
fn graph_switch_defaults_off_and_accepts_only_zero_or_one() {
    assert!(!graph_value(None).unwrap());
    assert!(!graph_value(Some("0")).unwrap());
    assert!(graph_value(Some("1")).unwrap());
    for value in ["", "true", "false", "2", "-1", "01", " 1"] {
        assert!(graph_value(Some(value)).is_err(), "{value}");
    }
}
