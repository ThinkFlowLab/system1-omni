//! Optional capacity limits for a worker that wants them.
//!
//! `SerialScheduler` admits one unit of work at a time and never refuses: a hundred requests
//! queue behind each other and the only bound is the caller's patience. That is the right
//! default for the workers here today, and this type is what a worker opts into when it wants a
//! bound instead.
//!
//! **Units are the caller's, and they are explicit.** Admission counts two things: how many
//! requests are outstanding, and how much each one costs. A request counts as one; the cost is
//! whatever the caller passes, so a prefill worker can admit by token count, a worker with
//! large payloads by bytes, and a worker that cares about neither by `1`. Nothing here guesses
//! which of those a model means.
//!
//! **The permit is owned, not borrowed.** It holds a share of the admission, so it can be moved
//! into the blocking work it was granted for — which is what the runtime's executor already
//! does — and the capacity is released when that work finishes with it rather than when the
//! request that asked for it goes away.
//!
//! This is admission only. It decides whether work may start and hands back a permit that
//! releases when the work is done with; it does not own a thread, run anything, or move a
//! request between states. Readiness, worker-thread ownership and drain are separate.

use std::sync::{Arc, Mutex};

use crate::engine::Report;

/// A bound a worker may opt into. Clones share it, so a queue can hold one and so can the
/// handlers that feed it.
///
/// ```
/// use omni_runtime::admission::BoundedAdmission;
///
/// let admission = BoundedAdmission::new(2, None);
/// let first = admission.try_admit(1).expect("one of two");
/// let second = admission.try_admit(1).expect("two of two");
/// assert!(admission.try_admit(1).is_none(), "the third has nowhere to go");
///
/// // A permit releases its capacity when the work is done with.
/// drop(first);
/// assert!(admission.try_admit(1).is_some());
/// ```
///
/// It wraps a shared inner rather than *being* an [`Arc`] alias: an alias would inherit `Arc`'s
/// associated functions, so `BoundedAdmission::new` would resolve to `Arc::new`.
#[derive(Clone, Debug)]
pub struct BoundedAdmission(Arc<Admission>);

/// The bound and the counters behind one [`BoundedAdmission`].
#[derive(Debug)]
struct Admission {
    /// Requests that may be outstanding at once. Zero would admit nothing, so it is refused at
    /// construction rather than silently rejecting every request.
    requests: usize,
    /// Total cost that may be outstanding, when the caller wants to bound something other than
    /// a request count. `None` bounds requests only.
    units: Option<usize>,
    held: Mutex<Held>,
}

/// What admission is currently holding, and what it has refused.
#[derive(Debug, Default)]
struct Held {
    outstanding: usize,
    units: usize,
    rejected: u64,
}

/// One admitted piece of work's reservation, released when its [`Permit`] drops.
///
/// A permit stores the units it *actually* reserved rather than the units its caller asked for.
/// The two differ only when the request was clamped to the ceiling, and storing the clamped
/// value is what makes the counter self-balancing: what goes in is exactly what comes out, so no
/// sequence of admissions and releases can drive the counter below zero.
#[derive(Clone, Copy, Debug)]
pub struct Reservation {
    units: usize,
}

impl BoundedAdmission {
    /// A bound of `requests` at a time, optionally also of `units` at a time.
    ///
    /// # Panics
    ///
    /// If `requests` is zero, or if `units` is `Some(0)`. Both would admit nothing at all,
    /// which is a configuration mistake rather than a limit worth enforcing, and it would show
    /// up as every request refused rather than as the mistake it is.
    pub fn new(requests: usize, units: Option<usize>) -> Self {
        assert!(
            requests > 0,
            "admission with no request capacity admits nothing"
        );
        assert!(
            units != Some(0),
            "admission with no unit capacity admits nothing"
        );
        Self(Arc::new(Admission {
            requests,
            units,
            held: Mutex::new(Held::default()),
        }))
    }

    /// Admits one piece of work costing `units`, or refuses it because there is no room.
    ///
    /// The permit is the reservation: capacity is held until it is dropped, which is what
    /// makes "outstanding" mean "accepted and not yet finished with" rather than "accepted".
    /// A refusal is counted and nothing is reserved.
    ///
    /// A request costing more than the unit ceiling is refused rather than admitted and
    /// overrunning it, so a reservation is never larger than the ceiling and the held total
    /// cannot reach `usize::MAX` by accumulation.
    ///
    /// The permit owns a share of this admission, so it can outlive the handle that issued it
    /// and be moved into the blocking work it was granted for.
    pub fn try_admit(&self, units: usize) -> Option<Permit> {
        let mut held = self.0.lock();
        if held.outstanding >= self.0.requests {
            held.rejected += 1;
            return None;
        }
        let reserved = match self.0.units {
            Some(ceiling) => {
                // A request larger than the whole ceiling is refused rather than clamped, so a
                // reservation is never bigger than the budget it was admitted under. What is
                // left of the ceiling is then known to fit, because `held.units <= ceiling`
                // holds as an invariant once every reservation is bounded this way.
                if units > ceiling || held.units > ceiling - units {
                    held.rejected += 1;
                    return None;
                }
                units
            }
            None => units,
        };
        held.outstanding += 1;
        held.units += reserved;
        Some(Permit {
            admission: Arc::clone(&self.0),
            reserved: Reservation { units: reserved },
        })
    }

    /// What a worker would publish as its queue state. `capacity` is the request bound, which
    /// is the one every caller of this type has.
    pub fn report(&self) -> Report {
        let held = self.0.lock();
        Report {
            depth: held.outstanding,
            capacity: self.0.requests,
            rejected: held.rejected,
        }
    }
}

impl Admission {
    /// Releases one reservation. Called when its [`Permit`] drops.
    ///
    /// Guarded rather than saturating: a reservation larger than what is held would mean two
    /// permits for one admission, and quietly underflowing would hide that. The public API
    /// cannot produce one, because a reservation is refused unless it fits under the ceiling.
    fn release(&self, reservation: Reservation) {
        let mut held = self.lock();
        held.outstanding = held
            .outstanding
            .checked_sub(1)
            .expect("a permit releases exactly one admission");
        held.units = held
            .units
            .checked_sub(reservation.units)
            .expect("a permit releases the units it was admitted with");
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Held> {
        // A panic while holding this leaves the counters readable and consistent, so a
        // poisoned lock is not a reason to refuse every later request.
        self.held.lock().unwrap_or_else(|error| error.into_inner())
    }
}

/// One admitted piece of work. Holding it holds its capacity; dropping it releases both.
///
/// It owns a share of its admission rather than borrowing it, so it can be moved into the
/// blocking work it was granted for. The runtime's own executor does exactly that; a borrow
/// would not compile there, because that work has to be `'static`.
///
/// This is deliberately not `Clone` and not `Copy`: two permits for one admission would release
/// capacity twice, which is the whole failure this type exists to prevent.
#[derive(Debug)]
pub struct Permit {
    admission: Arc<Admission>,
    reserved: Reservation,
}

impl Permit {
    /// What this permit reserved, for a caller that wants to report or account it.
    ///
    /// This is the capacity actually held, which is the cost the work was admitted at. Nothing
    /// is clamped, so it is also the cost the caller asked for.
    pub fn units(&self) -> usize {
        self.reserved.units
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.admission.release(self.reserved);
    }
}
