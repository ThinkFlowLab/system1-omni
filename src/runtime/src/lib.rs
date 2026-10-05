//! Shared admission and blocking dispatch for a loaded native executor.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use tokio::sync::Semaphore;

/// All pending execution slots are occupied; the submitted work was not dispatched.
#[derive(Debug)]
pub struct Overloaded;

impl std::fmt::Display for Overloaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("native execution capacity exhausted")
    }
}

impl std::error::Error for Overloaded {}

/// FIFO admission of one execution unit at a time. Clones share the same queue.
/// Each loaded executor gets its own scheduler; models define the unit of work.
#[derive(Clone)]
pub struct SerialScheduler {
    admission: Arc<Semaphore>,
    capacity: Arc<Semaphore>,
}

impl Default for SerialScheduler {
    fn default() -> Self {
        Self::new(64).expect("default pending limit is valid")
    }
}

impl SerialScheduler {
    /// Bound waiting and dispatched units together; execution remains serial.
    pub fn new(max_pending: usize) -> Result<Self> {
        ensure!(
            (1..=Semaphore::MAX_PERMITS).contains(&max_pending),
            "pending limit must be within 1..={}",
            Semaphore::MAX_PERMITS
        );
        Ok(Self {
            admission: Arc::new(Semaphore::new(1)),
            capacity: Arc::new(Semaphore::new(max_pending)),
        })
    }

    /// Configure a worker with OMNI_NATIVE_MAX_PENDING (default: 64 units).
    /// Invalid values fail startup before loading the model.
    pub fn from_env() -> Result<Self> {
        match std::env::var("OMNI_NATIVE_MAX_PENDING") {
            Ok(value) => Self::new(value.parse().context("OMNI_NATIVE_MAX_PENDING")?)
                .context("OMNI_NATIVE_MAX_PENDING"),
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            Err(e) => Err(e).context("OMNI_NATIVE_MAX_PENDING"),
        }
    }

    /// Wait asynchronously before dispatching blocking model work.
    /// Full capacity returns `Overloaded` immediately without waiting or dispatch.
    /// Cancellation while waiting removes the work from admission. Once dispatched,
    /// the work retains both permits and captured resources until the closure returns,
    /// even when its caller stops waiting. Device work must complete before return.
    pub async fn run<F, T>(&self, work: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let capacity = self
            .capacity
            .clone()
            .try_acquire_owned()
            .map_err(|_| Overloaded)?;
        let permit = self
            .admission
            .clone()
            .acquire_owned()
            .await
            .expect("private admission semaphore is never closed");
        tokio::task::spawn_blocking(move || {
            let _capacity = capacity;
            let _permit = permit;
            work()
        })
        .await
        .context("execution task failed")?
    }
}
