//! Worker lifecycle tests: readiness, queue admission, request budget, panic isolation
//! and shutdown. All of it runs against a fake engine, so it needs no model and no GPU —
//! which is the point of keeping this half free of model code.

use std::{
    io,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

use omni_jev::{
    engine::{Answer, Engine, EngineError, Readiness, Reply},
    worker::{self, Failure, Options, Request},
};
use tokio::runtime::Runtime;

/// Waits for a readiness state to settle. The worker answers a request and only then
/// retires, so a client can see the failure answer before the engine reads as failed.
fn wait_for(handle: &worker::Handle, wanted: Readiness) -> Readiness {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let state = handle.readiness();
        if state == wanted || Instant::now() >= deadline {
            return state;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

/// One gate per inference a test intends to hold: `open` is signalled by the test, `held`
/// is what the fake inference waits on.
type Gate = (mpsc::Sender<()>, mpsc::Receiver<()>);

/// The deadline a caller computes from its drain budget when shutdown begins.
fn drain_now(budget: Duration) -> Instant {
    Instant::now() + budget
}

fn checkpoint() -> PathBuf {
    PathBuf::from("/nonexistent/checkpoint")
}

/// The engine side of a running worker, plus the runtime needed to await its replies.
struct Harness {
    handle: Arc<worker::Handle>,
    runtime: Runtime,
    /// The budget the worker under test was started with, used to end its shutdown.
    drain: Duration,
}

impl Harness {
    fn submit(&self, body: &[u8]) -> Result<Reply, EngineError> {
        self.handle
            .submit(body.to_vec(), Instant::now() + Duration::from_secs(10))
    }

    /// Submits and waits, which is what an HTTP request does. A submission that is refused —
    /// because the queue is full, say — comes back as a value rather than panicking, so a
    /// caller that expects back-pressure can assert on it.
    fn try_reply(&self, body: &[u8]) -> Result<Result<Vec<u8>, EngineError>, EngineError> {
        let reply = self.submit(body)?;
        Ok(self
            .runtime
            .block_on(reply)
            .expect("worker dropped an accepted request")
            .map(|answer| answer.body))
    }

    /// As [`Harness::try_reply`], for the requests a test expects to be served.
    fn reply(&self, body: &[u8]) -> Result<Vec<u8>, EngineError> {
        self.try_reply(body).expect("submission refused")
    }
}

/// A fake engine, standing in for the model. Loading is instant; inference blocks until the
/// test opens a gate, so queue state can be observed instead of raced.
struct Fake {
    /// Set by warmup, so a test can prove readiness follows it and not merely loading.
    warmed: Arc<AtomicBool>,
    /// Requests that reached inference.
    served: Arc<AtomicUsize>,
    /// One per in-flight inference. The receiver is kept, not dropped: a dropped receiver
    /// would open the gate immediately, and the point is to hold an inference open.
    gates: Arc<Mutex<Vec<Gate>>>,
    /// Makes inference fail instead of answering.
    fail: bool,
    /// Makes inference panic instead of answering.
    panic: bool,
    /// Signalled when an inference starts, so a test can wait for the worker to be inside
    /// one instead of sleeping and hoping.
    entered: Arc<Mutex<Option<mpsc::Sender<()>>>>,
}

impl Fake {
    fn new() -> Self {
        Self {
            warmed: Arc::new(AtomicBool::new(false)),
            served: Arc::new(AtomicUsize::new(0)),
            gates: Arc::new(Mutex::new(Vec::new())),
            fail: false,
            panic: false,
            entered: Arc::new(Mutex::new(None)),
        }
    }

    /// Issues a gate for one inference the test intends to hold open.
    fn expect_in_flight(&self) {
        let pair = mpsc::channel();
        self.gates.lock().unwrap().push(pair);
    }

    /// Lets every held inference finish.
    fn open_gates(&self) {
        for (open, _held) in self.gates.lock().unwrap().drain(..) {
            let _ = open.send(());
        }
    }
}

fn start(
    fake: &Fake,
    queue_capacity: usize,
    drain: Duration,
) -> io::Result<(worker::Running, Harness)> {
    let for_load = Arc::clone(&fake.warmed);
    let for_warmup = Arc::clone(&fake.warmed);
    let for_process = Arc::clone(&fake.served);
    let gates = Arc::clone(&fake.gates);
    let entered = Arc::clone(&fake.entered);
    let (fail, panic) = (fake.fail, fake.panic);
    let running = worker::spawn(
        checkpoint(),
        Options {
            queue_capacity,
            drain,
        },
        // The engine value is a unit: these tests exercise the worker, not a model.
        move |_path| {
            let _ = &for_load;
            Ok(())
        },
        move |_engine| {
            for_warmup.store(true, Ordering::Release);
            Ok(())
        },
        move |request: Request<'_, ()>| {
            process(request, &for_process, &gates, &entered, fail, panic)
        },
    )?;
    let harness = Harness {
        handle: Arc::clone(&running.handle),
        runtime: Runtime::new().unwrap(),
        drain,
    };
    Ok((running, harness))
}

#[allow(clippy::too_many_arguments)]
fn process(
    request: Request<'_, ()>,
    served: &AtomicUsize,
    gates: &Mutex<Vec<Gate>>,
    entered: &Mutex<Option<mpsc::Sender<()>>>,
    fail: bool,
    panic: bool,
) -> Result<Answer, Failure> {
    served.fetch_add(1, Ordering::AcqRel);
    if let Some(signal) = entered.lock().unwrap().take() {
        let _ = signal.send(());
    }
    if panic {
        panic!("engine panicked on purpose");
    }
    if fail {
        return Err(Failure::Inference);
    }
    // Blocking is deliberate: a real inference occupies the worker for its duration, and
    // these tests need one that stays occupied. Gates are consumed in the order the test
    // declared them, which is the order the requests arrive in.
    if !gates.lock().unwrap().is_empty() {
        let (open, held) = gates.lock().unwrap().remove(0);
        // Waits for the test to open this gate. Tests that hold an inference are explicit
        // about it by calling `expect_in_flight`; one that does not is not held.
        let _ = held.recv_timeout(Duration::from_secs(10));
        let _ = open.send(());
    }
    Ok(Answer {
        body: request.body.to_vec(),
        ..request.answer()
    })
}

#[test]
fn spawn_returns_only_after_loading_and_warmup() {
    let fake = Fake::new();
    let (running, harness) = start(&fake, 1, Duration::from_secs(1)).unwrap();

    // Both happened on the worker thread before `spawn` returned, so a caller holding a
    // handle cannot observe a half-loaded engine.
    assert!(fake.warmed.load(Ordering::Acquire));
    assert_eq!(harness.handle.readiness(), Readiness::Ready);
    assert_eq!(harness.reply(b"hello").unwrap(), b"hello");

    assert!(running.run_until_drained(drain_now(harness.drain)));
    assert_eq!(harness.handle.readiness(), Readiness::Starting);
}

#[test]
fn startup_failure_is_reported() {
    let error = worker::spawn(
        checkpoint(),
        Options {
            queue_capacity: 1,
            drain: Duration::from_secs(1),
        },
        |_path| Err(io::Error::other("no such checkpoint")),
        |_engine| Ok(()),
        |request: Request<'_, ()>| {
            Ok(Answer {
                body: Vec::new(),
                ..request.answer()
            })
        },
    )
    .err()
    .expect("a failed load must not return a running worker");
    assert!(error.to_string().contains("no such checkpoint"), "{error}");
}

#[test]
fn queue_capacity_refuses_rather_than_grows() {
    let fake = Fake::new();
    let (running, harness) = start(&fake, 2, Duration::from_secs(2)).unwrap();
    // The blocker occupies inference; the next request takes the second slot; the third has
    // nowhere to go.
    fake.expect_in_flight();
    fake.expect_in_flight();
    let first = harness.submit(b"first").expect("first refused");
    let second = harness.submit(b"second").expect("second refused");
    match harness.submit(b"third") {
        Err(EngineError::Busy) => {}
        Err(other) => panic!("expected Busy, got {other:?}"),
        Ok(_) => panic!("expected Busy, got a reply channel"),
    }

    let report = harness
        .handle
        .report()
        .expect("the worker reports its queue");
    assert_eq!(report.depth, 2, "a refused request must not count");
    assert_eq!(report.rejected, 1);

    fake.open_gates();
    assert!(harness.runtime.block_on(first).is_ok());
    assert!(harness.runtime.block_on(second).is_ok());
    assert_eq!(harness.reply(b"fourth").unwrap(), b"fourth");
    assert_eq!(harness.handle.report().unwrap().depth, 0);

    assert!(running.run_until_drained(drain_now(harness.drain)));
}

/// Starts a worker whose inference spins on `hold` and then answers, taking no locks, so a
/// test can keep requests outstanding without ever holding a mutex the worker wants.
fn start_with_hold(
    fake: &Fake,
    queue_capacity: usize,
    drain: Duration,
    hold: &Arc<AtomicBool>,
) -> (worker::Running, Harness) {
    let gate = Arc::clone(hold);
    let running = worker::spawn(
        checkpoint(),
        Options {
            queue_capacity,
            drain,
        },
        |_path| Ok(()),
        |_engine| Ok(()),
        move |request: Request<'_, ()>| {
            let spin = Instant::now() + Duration::from_micros(20);
            while gate.load(Ordering::Acquire) && Instant::now() < spin {
                thread::yield_now();
            }
            Ok(Answer {
                body: request.body.to_vec(),
                ..request.answer()
            })
        },
    )
    .unwrap();
    let harness = Harness {
        handle: Arc::clone(&running.handle),
        runtime: Runtime::new().unwrap(),
        drain,
    };
    let _ = fake;
    (running, harness)
}

/// A worker held on one request, so a test can decide exactly what is queued behind it.
struct Blocked {
    open: mpsc::Sender<()>,
    entered: mpsc::Receiver<()>,
}

impl Blocked {
    /// Starts a worker whose next inference will block until [`Blocked::release`], and
    /// signals when that inference has actually started. The caller submits the request that
    /// takes the gate, so the depth it has to fit into is the caller's decision.
    fn wait(fake: &Fake, queue_capacity: usize) -> (worker::Running, Harness, Self) {
        let (open, held) = mpsc::channel();
        fake.gates.lock().unwrap().push((open.clone(), held));
        let (entered, started) = mpsc::channel();
        *fake.entered.lock().unwrap() = Some(entered);
        let (ready, harness) = start(fake, queue_capacity, Duration::from_secs(2)).unwrap();
        let blocked = Self {
            open,
            entered: started,
        };
        (ready, harness, blocked)
    }

    fn entered(&self) {
        self.entered
            .recv_timeout(Duration::from_secs(5))
            .expect("the held inference never started");
    }

    fn release(&self) {
        let _ = self.open.send(());
    }
}

/// The configured limit covers requests that are *outstanding*, not only requests that are
/// waiting in the channel. The worker frees its channel slot the moment it dequeues a job,
/// so admission has to count the running one too — otherwise capacity 1 serves two at once
/// while the README promises a `503` for the second.
#[test]
fn a_running_request_counts_against_capacity() {
    let fake = Fake::new();
    let (running, harness, blocked) = Blocked::wait(&fake, 1);
    // The first request is inside inference, so the channel is empty and only the
    // outstanding count can refuse the next one.
    let _first = harness.submit(b"first").expect("the first request fits");
    blocked.entered();

    match harness.submit(b"second") {
        Err(EngineError::Busy) => {}
        Err(other) => panic!("expected Busy while one request is running, got {other:?}"),
        Ok(_) => panic!("a second request was accepted while one was already running"),
    }
    let report = harness
        .handle
        .report()
        .expect("the worker reports its queue");
    assert_eq!(report.depth, 1, "a refused request must not be counted");
    assert_eq!(report.capacity, 1);
    assert_eq!(report.rejected, 1);

    // The refusal is not a lost slot: admission resumes once the running request has
    // finished. The slot is released as the worker sends its reply, so the next request can
    // be refused for a moment after the reply arrives — retry rather than assume ordering.
    blocked.release();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match harness.try_reply(b"third") {
            Ok(Ok(body)) => {
                assert_eq!(body, b"third");
                break;
            }
            // Refused, not lost: retry until the finished request has released its slot.
            Err(EngineError::Busy) if Instant::now() < deadline => thread::yield_now(),
            other => panic!("admission never resumed: {other:?}"),
        }
    }
    assert_eq!(harness.handle.report().unwrap().depth, 0);
    assert!(running.run_until_drained(drain_now(harness.drain)));
}

#[test]
fn cancelled_and_expired_requests_never_reach_the_engine() {
    let fake = Fake::new();
    let (running, harness, blocked) = Blocked::wait(&fake, 4);
    // The worker is parked inside inference, so anything submitted now is certainly queued
    // rather than already picked up. Without this handshake the test races the worker.
    let _blocker = harness.submit(b"blocker").unwrap();
    blocked.entered();

    // A client that gave up while its request waited.
    let abandoned = harness.submit(b"abandoned").unwrap();
    drop(abandoned);

    // A request whose budget expires while it waits. The wait below is what makes it
    // expire: the worker is held inside inference, which is exactly the queueing delay the
    // budget covers.
    let late = harness
        .handle
        .submit(b"late".to_vec(), Instant::now() + Duration::from_millis(20))
        .unwrap();
    thread::sleep(Duration::from_millis(80));

    assert!(
        fake.served.load(Ordering::Acquire) == 1,
        "only the blocker may have started"
    );
    blocked.release();
    assert_eq!(harness.reply(b"served").unwrap(), b"served");

    assert_eq!(
        fake.served.load(Ordering::Acquire),
        2,
        "only the blocker and the later request may run"
    );
    assert_eq!(
        harness.handle.report().unwrap().rejected,
        2,
        "both skipped requests are reported as work the worker refused to run"
    );
    assert_eq!(
        harness.handle.report().unwrap().depth,
        0,
        "a skipped request must release its queue slot"
    );
    assert!(
        harness.runtime.block_on(late).is_err(),
        "the expired request must have been dropped, not answered"
    );

    assert!(running.run_until_drained(drain_now(harness.drain)));
}

#[test]
fn a_failed_inference_retires_the_worker() {
    let mut fake = Fake::new();
    fake.fail = true;
    let (running, harness) = start(&fake, 1, Duration::from_secs(2)).unwrap();

    let answer = harness.reply(b"body");
    assert!(
        matches!(answer, Err(EngineError::InferenceFailed)),
        "{answer:?}"
    );

    // An engine that could not run one request is not offered the next one.
    assert_eq!(
        wait_for(&harness.handle, Readiness::Failed),
        Readiness::Failed
    );
    assert!(matches!(
        harness.submit(b"body"),
        Err(EngineError::Unavailable)
    ));
    assert!(running.run_until_drained(drain_now(harness.drain)));
}

#[test]
fn a_panicking_engine_answers_then_retires() {
    let mut fake = Fake::new();
    fake.panic = true;
    let (running, harness) = start(&fake, 1, Duration::from_secs(2)).unwrap();

    let answer = harness.reply(b"body");

    // The request is answered rather than dropped, so no client is left waiting, and the
    // engine is retired because a panic leaves its state unknown.
    assert!(
        matches!(answer, Err(EngineError::InferenceFailed)),
        "{answer:?}"
    );
    assert_eq!(
        wait_for(&harness.handle, Readiness::Failed),
        Readiness::Failed
    );
    let reason = harness
        .handle
        .failure()
        .expect("a failed engine explains itself");
    assert!(reason.contains("on purpose"), "{reason}");
    assert!(running.run_until_drained(drain_now(harness.drain)));
}

#[test]
fn drain_reports_when_it_cannot_finish() {
    let fake = Fake::new();
    let (running, harness) = start(&fake, 4, Duration::from_millis(150)).unwrap();
    // A gate nobody opens stands in for a wedged kernel: nothing here can interrupt it.
    // The second request queues behind it, so drain has both kinds of work to wait for.
    fake.expect_in_flight();
    let _stuck = harness.submit(b"stuck").unwrap();
    let _queued = harness.submit(b"queued").unwrap();
    assert_eq!(
        wait_for(&harness.handle, Readiness::Ready),
        Readiness::Ready
    );
    // The deadline is measured from here, as a caller's would be when shutdown begins.
    let start = Instant::now();
    assert!(
        !running.run_until_drained(drain_now(harness.drain)),
        "drain must not claim success while work is outstanding"
    );
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "drain must respect its budget, took {:?}",
        start.elapsed()
    );
}

#[test]
fn a_drained_worker_refuses_further_work() {
    let fake = Fake::new();
    let (running, harness) = start(&fake, 1, Duration::from_secs(2)).unwrap();
    assert_eq!(harness.handle.readiness(), Readiness::Ready);
    assert!(running.run_until_drained(drain_now(harness.drain)));

    // The worker thread has exited, so nothing can answer even though the handle is alive.
    // An engine that was never ready would also read as `Starting`, so check that this one
    // did become ready before it was drained.
    assert!(matches!(
        harness.submit(b"body"),
        Err(EngineError::Unavailable)
    ));
    assert_eq!(harness.handle.readiness(), Readiness::Starting);
    assert!(fake.warmed.load(Ordering::Acquire));
}

#[test]
fn capacity_zero_is_rejected_at_startup() {
    let error = worker::spawn(
        checkpoint(),
        Options {
            queue_capacity: 0,
            drain: Duration::from_secs(1),
        },
        |_path| Ok(()),
        |_engine| Ok(()),
        |request: Request<'_, ()>| {
            Ok(Answer {
                body: Vec::new(),
                ..request.answer()
            })
        },
    )
    .err()
    .expect("a zero-length queue cannot admit anything");
    assert!(error.to_string().contains("at least 1"), "{error}");
}

/// Queue depth is reserved before the job is published, so a consumer that wins the race to
/// dequeue cannot release a slot that was never taken. The bug this pins: incrementing after
/// the send lets the worker finish and decrement first, which reads as a wrong depth and
/// wraps the counter, leaving `/health` reporting a huge number.
#[test]
fn depth_is_reserved_before_the_job_is_published() {
    let fake = Fake::new();
    // Each inference spins briefly. That is the window in which the worker and the caller
    // race hardest, and it also keeps jobs outstanding long enough for the count below to
    // mean something.
    let hold = Arc::new(AtomicBool::new(true));
    let (running, harness) = start_with_hold(&fake, 64, Duration::from_secs(2), &hold);

    // Every reply is held rather than awaited, so nothing accepted can have completed and
    // the accepted count is exactly what the reported depth has to be.
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut replies = Vec::new();
    while replies.len() < 20_000 && Instant::now() < deadline {
        if let Ok(reply) = harness.submit(b"storm") {
            replies.push(reply);
            let depth = harness.handle.report().unwrap().depth;
            // A slot cannot be counted that was never taken, so depth can never exceed the
            // requests this caller has accepted. Counting after the send broke it in the
            // other direction: the released-before-reserved decrement wraps a `usize`, so
            // the failure shows up here as a huge number.
            assert!(
                depth <= replies.len(),
                "depth {depth} exceeded the {} requests accepted",
                replies.len()
            );
        }
    }
    assert!(replies.len() > 100, "the storm never got going");
    let accepted = replies.len();

    // Stop holding inference, then let everything finish: depth has to return to zero
    // rather than staying wrapped at a huge number.
    hold.store(false, Ordering::Release);
    drop(replies);
    let deadline = Instant::now() + Duration::from_secs(20);
    while harness.handle.report().unwrap().depth > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        harness.handle.report().unwrap().depth,
        0,
        "the queue did not drain after {accepted} requests"
    );
    assert!(running.run_until_drained(drain_now(harness.drain)));
}
