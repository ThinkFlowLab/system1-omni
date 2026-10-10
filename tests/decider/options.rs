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

#[test]
fn prefix_and_fixed_controls_reject_graph_or_each_other() {
    use omni_decider_native::{options::prefix_values, prefix::PrefixMode};
    assert_eq!(prefix_values(None, None, false).unwrap(), PrefixMode::Off);
    assert_eq!(
        prefix_values(Some("1"), None, false).unwrap(),
        PrefixMode::Shared
    );
    assert_eq!(
        prefix_values(None, Some("1"), false).unwrap(),
        PrefixMode::Fixed
    );
    assert!(prefix_values(Some("1"), Some("1"), false).is_err());
    assert!(prefix_values(Some("1"), None, true).is_err());
    assert!(prefix_values(None, Some("1"), true).is_err());
    assert!(prefix_values(Some("true"), None, false).is_err());
}

#[test]
fn auto_prefix_mode_is_opt_in_and_rejects_fixed_or_graph() {
    use omni_decider_native::{options::prefix_values, prefix::PrefixMode};
    assert_eq!(
        prefix_values(Some("auto"), None, false).unwrap(),
        PrefixMode::Auto
    );
    assert!(prefix_values(Some("auto"), Some("1"), false).is_err());
    assert!(prefix_values(Some("auto"), None, true).is_err());
    for invalid in ["AUTO", " auto", "auto ", "2"] {
        assert!(prefix_values(Some(invalid), None, false).is_err());
    }
}
