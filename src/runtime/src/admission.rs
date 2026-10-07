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
//! This is admission only. It decides whether work may start and hands back a permit that
//! releases when the work is done with; it does not own a thread, run anything, or move a
//! request between states. Readiness, worker-thread ownership and drain are separate.

use std::sync::{Arc, Mutex};

use crate::engine::Report;

/// What admission is currently holding, and what it has refused.
#[derive(Debug, Default)]
struct Held {
    outstanding: usize,
    units: usize,
    rejected: u64,
}

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
#[derive(Clone, Debug)]
pub struct BoundedAdmission {
    /// Requests that may be outstanding at once. Zero would admit nothing, so it is refused at
    /// construction rather than silently rejecting every request.
    requests: usize,
    /// Total cost that may be outstanding, when the caller wants to bound something other than
    /// a request count. `None` bounds requests only.
    units: Option<usize>,
    held: Arc<Mutex<Held>>,
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
        Self {
            requests,
            units,
            held: Arc::new(Mutex::new(Held::default())),
        }
    }

    /// Admits one piece of work costing `units`, or refuses it because there is no room.
    ///
    /// The permit is the reservation: capacity is held until it is dropped, which is what
    /// makes "outstanding" mean "accepted and not yet finished with" rather than "accepted".
    /// A refusal is counted and nothing is reserved.
    pub fn try_admit(&self, units: usize) -> Option<Permit<'_>> {
        let mut held = self.lock();
        let fits_requests = held.outstanding < self.requests;
        let fits_units = match self.units {
            Some(ceiling) => held.units.saturating_add(units) <= ceiling,
            None => true,
        };
        if !fits_requests || !fits_units {
            held.rejected += 1;
            return None;
        }
        held.outstanding += 1;
        held.units += units;
        Some(Permit {
            admission: self,
            units,
        })
    }

    /// What a worker would publish as its queue state. `capacity` is the request bound, which
    /// is the one every caller of this type has.
    pub fn report(&self) -> Report {
        let held = self.lock();
        Report {
            depth: held.outstanding,
            capacity: self.requests,
            rejected: held.rejected,
        }
    }

    /// Releases one piece of work's capacity. Called when its [`Permit`] drops.
    fn release(&self, units: usize) {
        let mut held = self.lock();
        // Guarded rather than saturating: a permit that released more than it held would mean
        // two permits for one admission, and quietly underflowing would hide that.
        held.outstanding = held
            .outstanding
            .checked_sub(1)
            .expect("a permit releases exactly one admission");
        held.units = held
            .units
            .checked_sub(units)
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
/// This is deliberately not `Clone` and not `Copy`: two permits for one admission would release
/// capacity twice, which is the whole failure this type exists to prevent.
#[derive(Debug)]
pub struct Permit<'a> {
    admission: &'a BoundedAdmission,
    units: usize,
}

impl Permit<'_> {
    /// What this permit was admitted for, for a caller that wants to report or account it.
    pub fn units(&self) -> usize {
        self.units
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.admission.release(self.units);
    }
}
