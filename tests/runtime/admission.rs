//! Optional capacity limits: what they admit, what they refuse, and what a permit means.
//!
//! Everything goes through the public API, because that is all a worker sees.

use std::{
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

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

/// Regression: every admission is released by exactly what it reserved, in any order. The
/// permit stores the units it actually reserved rather than the units it was asked for, so the
/// counter cannot be driven negative — an earlier version stored the requested cost and, where
/// the two differed, released past zero and panicked on a later release.
#[test]
fn a_reservation_is_released_by_what_was_reserved_not_by_what_was_asked() {
    // Room for many requests and exactly 100 units, so the unit ceiling is what binds.
    let admission = BoundedAdmission::new(10, Some(100));

    let a = admission.try_admit(10).expect("10 of 100");
    let b = admission.try_admit(60).expect("60 more");
    let c = admission.try_admit(30).expect("the remaining 30");
    assert!(
        admission.try_admit(1).is_none(),
        "the ceiling is exactly reached"
    );

    // Release order is irrelevant when what goes in is what comes out.
    drop(b);
    let again = admission.try_admit(60).expect("b's units came back");
    drop(a);
    drop(c);
    assert_eq!(
        admission.report().depth,
        1,
        "only the re-admitted 60 is held"
    );
    drop(again);
    assert_eq!(admission.report().depth, 0, "and it released its own units");
    assert!(
        admission.try_admit(100).is_some(),
        "the ceiling is whole again"
    );
}

/// A request costing more than the ceiling is refused outright, which is what keeps the held
/// total below `usize::MAX`: a reservation larger than the ceiling it was admitted under would
/// be released against the others and unbalance the counter.
#[test]
fn a_request_larger_than_the_unit_ceiling_is_refused_however_large_it_is() {
    let admission = BoundedAdmission::new(10, Some(64));

    for cost in [65, 1_000, usize::MAX] {
        assert!(
            admission.try_admit(cost).is_none(),
            "a request costing {cost} must be refused, not clamped"
        );
    }
    assert_eq!(admission.report().depth, 0, "nothing was reserved");
    assert_eq!(admission.report().rejected, 3, "all three were counted");

    // The largest request that fits is still admitted, and released with its own units.
    let full = admission.try_admit(64).expect("exactly the ceiling");
    assert_eq!(full.units(), 64);
    assert!(
        admission.try_admit(1).is_none(),
        "and it took the whole ceiling"
    );
    drop(full);
    assert_eq!(admission.report().depth, 0);
    assert!(admission.try_admit(64).is_some(), "the ceiling is reusable");
}

/// The ceiling is the whole budget, so one request may take all of it and no more.
#[test]
fn the_ceiling_is_the_largest_single_reservation() {
    let admission = BoundedAdmission::new(10, Some(usize::MAX));

    let all = admission.try_admit(usize::MAX).expect("the whole ceiling");
    assert!(
        admission.try_admit(1).is_none(),
        "even one unit is refused: the ceiling is taken"
    );
    drop(all);
    assert_eq!(admission.report().depth, 0);
    assert!(
        admission.try_admit(usize::MAX).is_some(),
        "and the whole ceiling comes back"
    );
}

/// Concurrent admitters can never together hold more than the request bound, and every
/// admission is released exactly once. A counter that drifted one way would let the live count
/// pass the bound; drifting the other way would leave permits outstanding at the end.
#[test]
fn concurrent_admissions_never_exceed_the_bound_and_all_come_back() {
    const THREADS: usize = 32;
    const ROUNDS: usize = 100;
    const CAPACITY: usize = 4;

    for _ in 0..ROUNDS {
        let admission = BoundedAdmission::new(CAPACITY, None);
        let barrier = Arc::new(Barrier::new(THREADS));
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let admission = admission.clone();
                let barrier = Arc::clone(&barrier);
                let live = Arc::clone(&live);
                let peak = Arc::clone(&peak);
                thread::spawn(move || {
                    barrier.wait();
                    if let Some(_permit) = admission.try_admit(1) {
                        let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        live.fetch_sub(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().expect("no admitting thread may panic");
        }

        assert_eq!(
            live.load(Ordering::SeqCst),
            0,
            "every admission was released"
        );
        assert!(
            peak.load(Ordering::SeqCst) <= CAPACITY,
            "at most the capacity is ever held at once"
        );
        assert_eq!(admission.report().depth, 0, "every admission was released");
        assert_eq!(
            admission.report().capacity,
            CAPACITY,
            "the bound outlives the contention"
        );
    }
}

/// A panic while the counters are locked must not refuse every later request: the counters are
/// still consistent, so the lock is recovered rather than poisoned for good.
#[test]
fn a_poisoned_lock_keeps_admitting() {
    let admission = BoundedAdmission::new(1, None);

    let held = admission.clone();
    let panicked = thread::spawn(move || {
        let _permit = held.try_admit(1).expect("one of one");
        panic!("admit and then die holding the lock");
    })
    .join();
    assert!(panicked.is_err(), "the helper thread was meant to panic");

    let report = admission.report();
    assert_eq!(
        report.depth, 0,
        "the panicking thread's permit was dropped as it unwound, so its slot came back"
    );
    assert!(
        admission.try_admit(1).is_some(),
        "the recovered lock admits"
    );
}

/// A permit is owned, so it can be moved into the work it was granted for instead of being tied
/// to the task that asked for it. This is the shape the runtime's executor uses, and the reason
/// the permit is not a borrow of the admission.
#[test]
fn a_permit_can_be_moved_into_the_work_it_was_granted_for() {
    let admission = BoundedAdmission::new(1, None);
    let permit = admission.try_admit(1).expect("one of one");
    assert!(admission.try_admit(1).is_none(), "held by the permit");

    let held = admission.clone();
    thread::spawn(move || {
        // The work owns the permit and releases the capacity when it finishes with it.
        let _permit = permit;
        assert!(
            held.try_admit(1).is_none(),
            "still held while the work runs"
        );
    })
    .join()
    .expect("the work thread may not panic");

    assert_eq!(admission.report().depth, 0, "the work released its slot");
    assert!(admission.try_admit(1).is_some(), "and it can be reused");
}
