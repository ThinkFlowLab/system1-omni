//! Request deadlines, cancellation and owned admission across blocking work.
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use crate::admission::Permit;

#[derive(Clone, Default)]
pub struct ExecutionContext {
    deadline: Option<Instant>,
    cancelled: Arc<AtomicBool>,
    permits: Vec<Arc<Permit>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ExecutionStopped {
    Deadline,
    Cancelled,
}

impl std::fmt::Display for ExecutionStopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Deadline => "request deadline expired",
            Self::Cancelled => "request cancelled",
        })
    }
}
impl std::error::Error for ExecutionStopped {}

impl ExecutionContext {
    pub fn new(deadline: Option<Instant>) -> Self {
        Self {
            deadline,
            ..Self::default()
        }
    }
    /// Keep capacity alive in every clone, including detached blocking tasks.
    pub fn retain(mut self, permit: Permit) -> Self {
        self.permits.push(Arc::new(permit));
        self
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    pub fn check(&self) -> Result<(), ExecutionStopped> {
        if self
            .deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Err(ExecutionStopped::Deadline);
        }
        if self.cancelled.load(Ordering::Acquire) {
            return Err(ExecutionStopped::Cancelled);
        }
        Ok(())
    }
    /// The awaiting caller owns this guard; blocking work owns context clones.
    pub fn cancel_on_drop(&self) -> CancellationGuard {
        CancellationGuard(self.cancelled.clone())
    }
}

pub struct CancellationGuard(Arc<AtomicBool>);
impl Drop for CancellationGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
