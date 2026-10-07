//! Optional capacity limits: what they admit, what they refuse, and what a permit means.
//!
//! Everything goes through the public API, because that is all a worker sees.

use omni_runtime::admission::BoundedAdmission;

#[test]
fn a_request_bound_refuses_once_it_is_full() {
    let admission = BoundedAdmission::new(2, None);
    let first = admission.try_admit(1).expect("the first of two");
    let second = admission.try_admit(1).expect("the second of two");
    assert!(
        admission.try_admit(1).is_none(),
        "the third has nowhere to go"
    );

    let report = admission.report();
    assert_eq!(report.depth, 2, "a refused request must not be counted");
    assert_eq!(report.capacity, 2);
    assert_eq!(report.rejected, 1);

    // A refusal is not a lost slot: releasing one makes room for the next.
    drop(first);
    assert_eq!(admission.report().depth, 1);
    assert!(admission.try_admit(1).is_some());
    drop(second);
}

/// The point of a permit: capacity means "accepted and not yet finished with". A worker that
/// hands work to a thread and forgets to release would be bounded forever, so release is tied
/// to the permit's lifetime rather than to a call someone has to remember.
#[test]
fn capacity_is_held_until_the_permit_is_dropped() {
    let admission = BoundedAdmission::new(1, None);
    let permit = admission.try_admit(1).expect("one of one");
    assert!(admission.try_admit(1).is_none(), "still held");
    drop(permit);
    assert_eq!(admission.report().depth, 0);
    assert!(admission.try_admit(1).is_some(), "released");
}

/// Units are the caller's, which is what lets a prefill worker bound tokens and a worker with
/// large payloads bound bytes without this type knowing which it is looking at.
#[test]
fn a_unit_bound_admits_by_cost_rather_than_by_count() {
    // Room for three requests, or 100 units, whichever runs out first.
    let admission = BoundedAdmission::new(3, Some(100));

    let big = admission.try_admit(60).expect("60 of 100 units");
    assert_eq!(admission.report().depth, 1, "one request, 60 units");

    let small = admission.try_admit(40).expect("the remaining 40 units");
    assert!(
        admission.try_admit(1).is_none(),
        "the unit ceiling is reached even though only two requests are outstanding"
    );

    // Releasing the large one frees its units, not just its request slot.
    drop(big);
    assert!(admission.try_admit(50).is_some(), "60 units came back");
    drop(small);
}

/// A request that costs more than the whole ceiling is refused rather than admitted and
/// overrunning it — the case a naive `depth < capacity` check gets wrong.
#[test]
fn one_request_larger_than_the_ceiling_is_refused() {
    let admission = BoundedAdmission::new(10, Some(100));
    assert!(admission.try_admit(101).is_none());
    assert_eq!(admission.report().depth, 0, "nothing was reserved");
    assert!(
        admission.try_admit(100).is_some(),
        "exactly the ceiling fits"
    );
}

#[test]
fn the_request_count_is_reported_as_capacity() {
    let admission = BoundedAdmission::new(7, Some(1_000));
    assert_eq!(admission.report().capacity, 7);
    assert_eq!(admission.report().depth, 0);
    assert_eq!(admission.report().rejected, 0);
}

/// A zero bound admits nothing, which is a configuration mistake rather than a limit. It is
/// refused where it is written instead of showing up as every request being turned away.
#[test]
#[should_panic(expected = "admits nothing")]
fn a_zero_request_bound_is_refused_at_construction() {
    let _ = BoundedAdmission::new(0, None);
}

#[test]
#[should_panic(expected = "admits nothing")]
fn a_zero_unit_bound_is_refused_at_construction() {
    let _ = BoundedAdmission::new(1, Some(0));
}

/// Clones share the bound, so the queue and the handlers feeding it cannot disagree about how
/// much room is left.
#[test]
fn clones_share_one_bound() {
    let admission = BoundedAdmission::new(1, None);
    let shared = admission.clone();
    let permit = shared.try_admit(1).expect("one of one");
    assert!(admission.try_admit(1).is_none(), "the clone held the slot");
    drop(permit);
    assert!(admission.try_admit(1).is_some());
    assert_eq!(admission.report().rejected, 1, "both see the same count");
}
