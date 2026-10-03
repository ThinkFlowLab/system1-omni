//! One thread owns the engine, and a bounded queue owns admission.
//!
//! This is the half of the native path that does not depend on a model: the transport
//! hands over request bytes, the worker runs them one at a time, and everything a
//! deployment has to get right lives here — readiness that follows warmup, a queue that
//! refuses instead of growing, a request budget that can expire before work starts, and a
//! shutdown that stops admission without cutting a running inference.
//!
//! A model supplies three things and nothing else:
//!
//! ```no_run
//! # use std::{io, path::PathBuf};
//! # use omni_jev::{engine::Answer, worker};
//! # struct Model;
//! # impl Model {
//! #   fn load(_: PathBuf) -> io::Result<Self> { Ok(Model) }
//! #   fn warmup(&mut self) -> io::Result<()> { Ok(()) }
//! #   fn infer(&mut self, _: &[u8]) -> Result<Vec<u8>, String> { Ok(Vec::new()) }
//! # }
//! let running = worker::spawn(
//!     PathBuf::from("/checkpoints/laya"),
//!     worker::Options::default(),
//!     // 1. Build the engine. Runs on the worker thread, so a runtime that binds to its
//!     //    creating thread is created by the thread that keeps it.
//!     Model::load,
//!     // 2. Pay for compilation and allocation here, so readiness means something.
//!     |model| model.warmup(),
//!     // 3. Answer one request. The value becomes the response body.
//!     |request| match request.engine.infer(request.body) {
//!         Ok(body) => Ok(Answer { body, ..request.answer() }),
//!         Err(detail) => Err(worker::Failure::InvalidRequest(detail)),
//!     },
//! )
//! .expect("engine failed to load");
//! ```

use std::{
    io,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use tokio::sync::{mpsc, oneshot};

use crate::engine::{Answer, Engine, EngineError, Readiness, Reply, Report};

/// How the worker was asked to run.
#[derive(Clone, Debug)]
pub struct Options {
    /// Requests accepted and not yet completed, running or queued. [`Engine::submit`]
    /// refuses rather than waiting once this many are outstanding, which is what turns
    /// overload into a fast `503` instead of an unbounded queue. The queue itself is only a
    /// buffer for these jobs, so it is sized the same and enforces nothing on its own.
    pub queue_capacity: usize,
    /// How long a shutdown may spend waiting for accepted work. It bounds the wait, it does
    /// not cancel anything: a kernel already submitted runs to completion.
    pub drain: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            queue_capacity: 32,
            drain: Duration::from_secs(10),
        }
    }
}

/// Why a request could not be executed. The worker decides between a caller's mistake and
/// its own failure; the transport decides which status code that becomes.
#[derive(Debug)]
pub enum Failure {
    /// The body is not a request this engine accepts. The worker keeps serving.
    InvalidRequest(String),
    /// The engine failed at work it accepted, so it is retired: see [`run`].
    Inference,
}

/// One request the worker has taken off the queue, with the state a model's own code
/// cannot see: its budget and whether anyone is still waiting for it.
pub struct Request<'a, E> {
    pub engine: &'a mut E,
    pub body: &'a [u8],
    /// The caller's budget, covering upload and queueing as well as this inference.
    pub deadline: Instant,
    /// False once the client has gone, which happens when the transport gave up on the
    /// deadline or the connection dropped. Work not yet submitted should be dropped; work
    /// already submitted is never cancelled.
    pub waiting: bool,
    /// Queue state measured when this request was dequeued, for the answer to carry back.
    depth: usize,
}

impl<E> Request<'_, E> {
    /// An answer carrying the queue depth this request was dequeued at. Fill in `body`.
    pub fn answer(&self) -> Answer {
        Answer {
            body: Vec::new(),
            depth: self.depth,
        }
    }
}

struct Job {
    body: Vec<u8>,
    deadline: Instant,
    reply: oneshot::Sender<Result<Answer, EngineError>>,
}

/// The engine side of a running worker: what [`crate::engine::app`] talks to.
pub struct Handle {
    /// Taken by shutdown to close the queue. `None` means admission is over.
    sender: Mutex<Option<mpsc::Sender<Job>>>,
    /// Serializes admission against shutdown, so a request cannot be accepted after the
    /// queue has been closed. Held only for the length of a non-blocking send.
    admission: Mutex<()>,
    ready: AtomicBool,
    stopped: AtomicBool,
    depth: AtomicUsize,
    /// The outstanding-request limit. Fixed for the life of the worker.
    capacity: usize,
    rejected: AtomicU64,
    failure: Mutex<Option<String>>,
}

/// A started worker: the engine handle and the thread that owns the engine.
pub struct Running {
    /// The handle to give to [`crate::engine::app`].
    pub handle: Arc<Handle>,
    /// `Some` until [`Running::run_until_drained`] takes it to join.
    worker: Option<thread::JoinHandle<()>>,
    drain: Duration,
}

impl Running {
    /// Closes the queue without waiting for accepted work. Idempotent, and called for you
    /// by [`Running::run_until_drained`] and by this type's drop.
    pub fn stop_accepting(&self) {
        self.handle.stop_accepting();
    }

    /// How long a caller is willing to spend draining, so a shutdown sequence can compute
    /// one deadline for all of its steps instead of giving each step its own.
    pub fn drain_budget(&self) -> Duration {
        self.drain
    }

    /// Waits for accepted work to finish and for the worker thread to exit, no later than
    /// `deadline`.
    ///
    /// The deadline is absolute and passed in rather than measured from here, because the
    /// budget belongs to the shutdown sequence: a caller that has already spent part of it
    /// stopping admission and closing the listener must not get a fresh budget for the
    /// drain. Returns false if the deadline passed first, which is the honest answer when a
    /// kernel or a device call is wedged — nothing here can interrupt it.
    pub fn run_until_drained(self, deadline: Instant) -> bool {
        // Taking a field out of a type that has a `Drop` impl needs the drop taken over
        // first; the sequence below is what that `Drop` would have done anyway.
        let mut this = std::mem::ManuallyDrop::new(self);
        // The take cannot be empty: this is the only place the thread handle is moved out.
        let worker = this
            .worker
            .take()
            .expect("worker thread handle was already taken");
        this.handle.stop_accepting();
        while !worker.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        if !worker.is_finished() {
            eprintln!("omni-jev: engine did not drain within {:?}", this.drain);
            return false;
        }
        match worker.join() {
            Ok(()) => true,
            Err(_) => {
                eprintln!("omni-jev: engine thread panicked");
                false
            }
        }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        // A caller that never reached shutdown must not leave the engine accepting work.
        self.handle.stop_accepting();
    }
}

impl Engine for Handle {
    fn readiness(&self) -> Readiness {
        if self.ready.load(Ordering::Acquire) {
            Readiness::Ready
        } else if lock(&self.failure).is_some() {
            Readiness::Failed
        } else {
            Readiness::Starting
        }
    }

    fn submit(&self, body: Vec<u8>, deadline: Instant) -> Result<Reply, EngineError> {
        // One lock orders admission against shutdown: without it a request can pass the
        // readiness check and then be accepted after the queue was closed, leaving it
        // accepted with no worker left to answer it.
        let admission = lock(&self.admission);
        if !self.ready.load(Ordering::Acquire) || self.stopped.load(Ordering::Acquire) {
            return Err(EngineError::Unavailable);
        }
        let sender = lock(&self.sender);
        let Some(sender) = sender.as_ref() else {
            return Err(EngineError::Unavailable);
        };
        // The chain is what makes the check a decision rather than a read: `depth` may only
        // increase while the admission lock is held, so the count that was compared is still
        // the count when the job is handed over.
        let reserved = self.depth.fetch_add(1, Ordering::AcqRel) < self.capacity;
        let refusal = if !reserved {
            Some(EngineError::Busy)
        } else {
            let (reply, receiver) = oneshot::channel();
            match sender.try_send(Job {
                body,
                deadline,
                reply,
            }) {
                // The worker is free to dequeue, answer and release immediately: it never
                // takes the admission lock. Counting after the send would therefore release a
                // slot that had not been taken yet, so the slot is taken before the handover.
                Ok(()) => {
                    drop(admission);
                    return Ok(receiver);
                }
                Err(mpsc::error::TrySendError::Full(_)) => Some(EngineError::Busy),
                Err(mpsc::error::TrySendError::Closed(_)) => Some(EngineError::Unavailable),
            }
        };
        // Nothing was accepted, so the reservation goes back; the refusal is then reported
        // as such rather than left counted against the queue.
        self.depth.fetch_sub(1, Ordering::AcqRel);
        if let Some(EngineError::Busy) = refusal {
            self.rejected.fetch_add(1, Ordering::Relaxed);
        }
        let refusal = refusal.expect("a refusal was decided above");
        drop(admission);
        Err(refusal)
    }

    fn report(&self) -> Option<Report> {
        Some(Report {
            depth: self.depth.load(Ordering::Acquire),
            capacity: self.capacity,
            rejected: self.rejected.load(Ordering::Relaxed),
        })
    }

    fn failure(&self) -> Option<String> {
        lock(&self.failure).clone()
    }
}

impl Handle {
    /// Closes the queue and clears readiness, without waiting for accepted work.
    pub fn stop_accepting(&self) {
        let admission = lock(&self.admission);
        self.stopped.store(true, Ordering::Release);
        self.ready.store(false, Ordering::Release);
        // Dropped while the admission lock is held, so no request can slip a send between
        // the close and the check in `submit`.
        let sender = lock(&self.sender).take();
        drop(sender);
        drop(admission);
    }

    fn fail(&self, detail: String) {
        eprintln!("omni-jev: engine failed: {detail}");
        *lock(&self.failure) = Some(detail);
        self.ready.store(false, Ordering::Release);
    }
}

/// Starts the worker thread and returns once the engine has loaded and warmed up.
///
/// `load` runs on the worker thread, which is the thread that keeps the engine, so a
/// runtime that binds to its creating thread is created in the right place. `warmup` runs
/// before readiness is reported, so the first request after `/health` is ready is not the
/// one that pays for compilation. A load failure returns `Err` and leaves no thread behind;
/// a failure after startup marks the handle [`Readiness::Failed`].
pub fn spawn<E, L, W, P>(
    path: PathBuf,
    options: Options,
    load: L,
    warmup: W,
    process: P,
) -> io::Result<Running>
where
    L: FnOnce(PathBuf) -> io::Result<E> + Send + 'static,
    W: FnOnce(&mut E) -> io::Result<()> + Send + 'static,
    P: FnMut(Request<'_, E>) -> Result<Answer, Failure> + Send + 'static,
    E: Send + 'static,
{
    if options.queue_capacity == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "queue capacity must be at least 1",
        ));
    }

    let handle = Arc::new(Handle {
        sender: Mutex::new(None),
        admission: Mutex::new(()),
        ready: AtomicBool::new(false),
        stopped: AtomicBool::new(false),
        depth: AtomicUsize::new(0),
        capacity: options.queue_capacity,
        rejected: AtomicU64::new(0),
        failure: Mutex::new(None),
    });
    let (jobs, queue) = mpsc::channel::<Job>(options.queue_capacity);
    let (started, loading) = oneshot::channel::<io::Result<()>>();

    let worker = {
        let handle = Arc::clone(&handle);
        thread::Builder::new()
            .name("omni-engine".into())
            .spawn(move || {
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    load(path).and_then(|mut engine| warmup(&mut engine).map(|()| engine))
                }));
                let engine = match outcome {
                    Ok(Ok(engine)) => engine,
                    Ok(Err(error)) => {
                        handle.fail(error.to_string());
                        let _ = started.send(Err(error));
                        return;
                    }
                    Err(panic) => {
                        let error = io::Error::other(format!(
                            "engine panicked while loading: {}",
                            panic_text(&panic)
                        ));
                        handle.fail(error.to_string());
                        let _ = started.send(Err(error));
                        return;
                    }
                };
                // The queue is open before readiness is set, so a request that observes
                // Ready always finds somewhere to go.
                *lock(&handle.sender) = Some(jobs);
                *lock(&handle.failure) = None;
                handle.ready.store(true, Ordering::Release);
                if started.send(Ok(())).is_err() {
                    return;
                }
                run(engine, queue, &handle, process);
                // Reached on shutdown or on engine failure. Either way the process must
                // stop reporting itself ready.
                handle.ready.store(false, Ordering::Release);
                *lock(&handle.sender) = None;
            })?
    };

    // Waiting for the load result has to happen off the caller's thread. A caller that is
    // already inside an async runtime cannot block, and blocking there starves the runtime
    // that will later serve requests — which is how this is reached in practice, since the
    // process's `main` is an async entry point.
    let startup: io::Result<io::Result<()>> = thread::Builder::new()
        .name("omni-engine-startup".into())
        .spawn(move || loading.blocking_recv())?
        .join()
        .map(|outcome| {
            outcome.map_err(|_| io::Error::other("engine thread stopped during startup"))
        })
        .unwrap_or_else(|_| Err(io::Error::other("engine thread stopped during startup")));

    match startup {
        Ok(Ok(())) => Ok(Running {
            handle,
            worker: Some(worker),
            drain: options.drain,
        }),
        // The engine reported why it could not load, which is the message a caller wants.
        Ok(Err(error)) | Err(error) => {
            let _ = worker.join();
            Err(error)
        }
    }
}

/// The worker loop: one job at a time, in arrival order, until the queue closes.
fn run<E, P>(mut engine: E, mut queue: mpsc::Receiver<Job>, handle: &Handle, mut process: P)
where
    P: FnMut(Request<'_, E>) -> Result<Answer, Failure>,
{
    while let Some(job) = queue.blocking_recv() {
        // The depth slot is released on every path out of this iteration, including the
        // skip and panic paths, so a refused request cannot leave depth permanently high.
        let slot = Slot(&handle.depth);
        if job.reply.is_closed() || Instant::now() >= job.deadline {
            // A cancelled or expired request never reaches the engine, which is the point
            // of carrying a deadline this far: no GPU work for an answer nobody will read.
            handle.rejected.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let mut stop = false;
        let body = &job.body;
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            process(Request {
                engine: &mut engine,
                body,
                deadline: job.deadline,
                waiting: !job.reply.is_closed(),
                depth: handle.depth.load(Ordering::Acquire),
            })
        }));
        let response = match outcome {
            Ok(Ok(answer)) => Ok(answer),
            // Bad input, not a broken engine. The worker keeps serving, so one caller's
            // malformed body cannot take the deployment down.
            Ok(Err(Failure::InvalidRequest(detail))) => Err(EngineError::InvalidRequest(detail)),
            Ok(Err(Failure::Inference)) => {
                // The engine failed at work it accepted: a device error, a lost context, an
                // allocation that did not come back. The worker retires rather than offering
                // the next caller an engine it can no longer vouch for, and `/health` carries
                // the reason until a restart.
                handle.fail("inference failed; the engine was retired".into());
                stop = true;
                Err(EngineError::InferenceFailed)
            }
            Err(panic) => {
                // A panic leaves the engine's state unknown for the same reason and is
                // reported the same way. The payload is kept because it says more than
                // "inference failed" does.
                handle.fail(format!("engine panicked: {}", panic_text(&panic)));
                stop = true;
                Err(EngineError::InferenceFailed)
            }
        };
        let _ = job.reply.send(response);
        // Released before the next dequeue, so a fresh answer never reports this request.
        drop(slot);
        if stop {
            break;
        }
    }
    handle.ready.store(false, Ordering::Release);
}

/// Releases one unit of queue depth when it goes out of scope.
struct Slot<'a>(&'a AtomicUsize);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

fn panic_text(panic: &Box<dyn std::any::Any + Send>) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic payload")
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}
