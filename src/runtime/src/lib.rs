//! Shared admission, dispatch and the worker-side engine contract for a loaded executor.

pub mod engine;

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::Semaphore;

/// FIFO admission of one execution unit at a time. Clones share the same queue.
/// Each loaded executor gets its own scheduler; models define the unit of work.
#[derive(Clone)]
pub struct SerialScheduler {
    admission: Arc<Semaphore>,
}

impl Default for SerialScheduler {
    fn default() -> Self {
        Self {
            admission: Arc::new(Semaphore::new(1)),
        }
    }
}

impl SerialScheduler {
    /// Wait asynchronously before dispatching blocking model work.
    /// Cancellation while waiting removes the work from admission. Once dispatched,
    /// the work retains its permit and captured resources until the closure returns,
    /// even when its caller stops waiting. Device work must complete before return.
    pub async fn run<F, T>(&self, work: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let permit = self
            .admission
            .clone()
            .acquire_owned()
            .await
            .expect("private admission semaphore is never closed");
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            work()
        })
        .await
        .context("execution task failed")?
    }
}
