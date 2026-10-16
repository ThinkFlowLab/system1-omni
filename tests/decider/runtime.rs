use omni_decider_native::runtime::{Runtime, RuntimeConfig};
use std::time::{Duration, Instant};

#[test]
fn default_policy_is_unbounded_and_has_no_deadline() {
    let runtime = Runtime::new(RuntimeConfig::from_values(None, None, None).unwrap()).unwrap();
    let request = runtime.admit(Instant::now()).unwrap();
    assert!(request.deadline().is_none());
    assert!(runtime.report().is_none());
}
#[test]
fn invalid_limits_are_configuration_errors() {
    for values in [
        (Some("0"), None, None),
        (None, Some("0"), None),
        (None, None, Some("0")),
        (Some("-1"), None, None),
        (None, None, Some("nan")),
        (None, None, Some("18446744073709551616")),
    ] {
        assert!(
            RuntimeConfig::from_values(values.0, values.1, values.2)
                .and_then(Runtime::new)
                .is_err()
        );
    }
}
#[test]
fn request_and_processed_token_limits_are_independent_and_recover() {
    let runtime =
        Runtime::new(RuntimeConfig::from_values(Some("2"), Some("100"), Some("1000")).unwrap())
            .unwrap();
    let now = Instant::now();
    let first = runtime.admit(now).unwrap();
    assert_eq!(first.deadline(), Some(now + Duration::from_secs(1)));
    let first = runtime.reserve_tokens(first, 60).unwrap();
    let second = runtime.admit(now).unwrap();
    assert!(runtime.admit(now).is_err());
    assert!(runtime.reserve_tokens(second, 41).is_err());
    assert_eq!(runtime.report().unwrap().depth, 1);
    assert_eq!(runtime.report().unwrap().rejected, 2);
    let second = runtime
        .reserve_tokens(runtime.admit(now).unwrap(), 40)
        .unwrap();
    drop(first);
    drop(second);
    assert_eq!(runtime.report().unwrap().depth, 0);
    let all = runtime
        .reserve_tokens(runtime.admit(now).unwrap(), 100)
        .unwrap();
    drop(all);
    assert!(
        runtime
            .reserve_tokens(runtime.admit(now).unwrap(), 101)
            .is_err()
    );
    assert_eq!(runtime.report().unwrap().depth, 0);
}
#[test]
fn deadline_before_preparation_is_explicit_and_does_not_hold_capacity() {
    let runtime =
        Runtime::new(RuntimeConfig::from_values(Some("1"), None, Some("1")).unwrap()).unwrap();
    let context = runtime
        .admit(Instant::now() - Duration::from_secs(1))
        .unwrap();
    assert!(context.check().is_err());
    drop(context);
    assert_eq!(runtime.report().unwrap().depth, 0);
}
