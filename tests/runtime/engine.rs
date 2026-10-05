//! The worker-side engine contract: what an implementation must do, and what a caller may
//! assume. Everything here goes through the public API, because that is all a worker or a
//! transport sees.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use omni_runtime::engine::{Answer, Engine, EngineError, Readiness, Reply, Report};
use tokio::sync::oneshot;

/// The smallest worker: ready immediately, answers with its own body, and reports nothing.
struct Echo {
    seen: Arc<AtomicUsize>,
}

impl Echo {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            seen: Arc::new(AtomicUsize::new(0)),
        })
    }
}

impl Engine for Echo {
    fn readiness(&self) -> Readiness {
        Readiness::Ready
    }

    fn submit(&self, body: Vec<u8>, _deadline: Instant) -> Reply {
        self.seen.fetch_add(1, Ordering::AcqRel);
        let (reply, receiver) = oneshot::channel();
        if body.is_empty() {
            let _ = reply.send(Err(EngineError::InvalidRequest(
                "empty request body".into(),
            )));
        } else {
            let _ = reply.send(Ok(Answer { body, depth: 0 }));
        }
        receiver
    }
}

/// A worker that owns a queue: it admits nothing beyond its capacity, and reports the depth it
/// is holding. This is the shape the trait exists for.
struct Queued {
    capacity: usize,
    depth: AtomicUsize,
    rejected: AtomicUsize,
    held: std::sync::Mutex<Vec<oneshot::Sender<Result<Answer, EngineError>>>>,
}

impl Queued {
    fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            depth: AtomicUsize::new(0),
            rejected: AtomicUsize::new(0),
            held: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Answers everything admitted so far, releasing the depth it was holding.
    fn release_all(&self) {
        for reply in self.held.lock().unwrap().drain(..) {
            self.depth.fetch_sub(1, Ordering::AcqRel);
            let _ = reply.send(Ok(Answer {
                body: b"answered".to_vec(),
                depth: 0,
            }));
        }
    }
}

impl Engine for Queued {
    fn readiness(&self) -> Readiness {
        Readiness::Ready
    }

    fn submit(&self, _body: Vec<u8>, _deadline: Instant) -> Reply {
        let (reply, receiver) = oneshot::channel();
        // Admission and the answer travel on one path, so a refusal is sent rather than
        // returned: the caller has a single thing to await either way.
        if self.depth.load(Ordering::Acquire) >= self.capacity {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            let _ = reply.send(Err(EngineError::Busy));
            return receiver;
        }
        self.depth.fetch_add(1, Ordering::AcqRel);
        self.held.lock().unwrap().push(reply);
        receiver
    }

    fn report(&self) -> Option<Report> {
        Some(Report {
            depth: self.depth.load(Ordering::Acquire),
            capacity: self.capacity,
            rejected: self.rejected.load(Ordering::Relaxed) as u64,
        })
    }
}

#[tokio::test]
async fn an_answer_arrives_on_the_reply() {
    let engine: Arc<dyn Engine> = Echo::new();
    let answer = engine
        .submit(b"hello".to_vec(), Instant::now())
        .await
        .expect("the sender is held until it answers")
        .expect("a non-empty body is answered");
    assert_eq!(answer.body, b"hello");
}

#[tokio::test]
async fn a_refusal_arrives_on_the_same_reply() {
    let engine: Arc<dyn Engine> = Echo::new();
    let refused = engine
        .submit(Vec::new(), Instant::now())
        .await
        .expect("the refusal is the message, not a dropped channel")
        .expect_err("an empty body is refused");
    assert!(
        matches!(refused, EngineError::InvalidRequest(_)),
        "{refused:?}"
    );
}

#[tokio::test]
async fn a_worker_without_observability_reports_nothing() {
    let echo = Echo::new();
    let seen = Arc::clone(&echo.seen);
    let engine: Arc<dyn Engine> = echo;
    assert_eq!(engine.readiness(), Readiness::Ready);
    assert!(engine.report().is_none());
    assert!(engine.failure().is_none());
    assert_eq!(
        seen.load(Ordering::Acquire),
        0,
        "asking a worker how it is must not run any work"
    );
}

/// The trait has to carry a queue, not just a pass-through: admission, the depth it reports,
/// and the fact that a refusal does not release a slot.
#[tokio::test]
async fn a_queued_worker_admits_to_capacity_and_says_so() {
    let engine = Queued::new(2);
    let first = engine.submit(b"one".to_vec(), Instant::now());
    let second = engine.submit(b"two".to_vec(), Instant::now());
    assert_eq!(
        engine
            .report()
            .map(|report| (report.depth, report.capacity)),
        Some((2, 2)),
        "both accepted requests are outstanding"
    );

    let third = engine.submit(b"three".to_vec(), Instant::now());
    assert!(
        matches!(third.await, Ok(Err(EngineError::Busy))),
        "the third has nowhere to go and is told so on its own reply"
    );
    let report = engine.report().expect("this worker reports its queue");
    assert_eq!(report.depth, 2, "a refused request must not be counted");
    assert_eq!(report.rejected, 1);

    // A refusal is not a lost slot: answering the accepted work frees the queue again, and the
    // next request is admitted rather than refused.
    engine.release_all();
    assert!(matches!(first.await, Ok(Ok(_))));
    assert!(matches!(second.await, Ok(Ok(_))));
    assert_eq!(engine.report().map(|report| report.depth), Some(0));

    let fourth = engine.submit(b"fourth".to_vec(), Instant::now());
    assert_eq!(
        engine.report().map(|report| report.depth),
        Some(1),
        "the freed slot is usable again"
    );
    engine.release_all();
    assert!(matches!(fourth.await, Ok(Ok(_))));
}

/// A dropped receiver is how cancellation is observed. Dropping it must not disturb the
/// worker's own accounting, because work it already admitted still has to finish.
#[tokio::test]
async fn dropping_a_reply_does_not_disturb_the_queue() {
    let engine = Queued::new(2);
    let abandoned = engine.submit(b"one".to_vec(), Instant::now());
    drop(abandoned);
    assert_eq!(engine.report().map(|report| report.depth), Some(1));
    engine.release_all();
    assert_eq!(engine.report().map(|report| report.depth), Some(0));
}

#[test]
fn an_error_says_what_it_is() {
    assert_eq!(
        EngineError::InvalidRequest("no model field".into()).to_string(),
        "invalid request: no model field"
    );
    assert_eq!(EngineError::Busy.to_string(), "no queue capacity");
    assert_eq!(EngineError::Unavailable.to_string(), "engine unavailable");
    assert_eq!(EngineError::InferenceFailed.to_string(), "inference failed");
}

/// The transport reads a deadline off the contract; it has to be the caller's own instant.
#[tokio::test]
async fn a_deadline_is_an_instant_the_caller_owns() {
    let engine = Queued::new(1);
    let deadline = Instant::now() + Duration::from_millis(50);
    let reply = engine.submit(b"one".to_vec(), deadline);
    assert!(deadline > Instant::now(), "the budget is in the future");
    engine.release_all();
    assert!(matches!(reply.await, Ok(Ok(_))));
}
