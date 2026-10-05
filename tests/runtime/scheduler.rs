use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::task::Poll;
use std::time::Duration;

use anyhow::Result;
use omni_runtime::SerialScheduler;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

async fn deadline<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .expect("scheduler made no progress")
}

async fn poll_waiting<T>(mut future: Pin<&mut impl Future<Output = T>>) {
    poll_fn(|cx| {
        assert!(future.as_mut().poll(cx).is_pending());
        Poll::Ready(())
    })
    .await;
}

async fn blocked(scheduler: SerialScheduler) -> (JoinHandle<Result<()>>, mpsc::Sender<()>) {
    let (started, ready) = oneshot::channel();
    let (release, wait) = mpsc::channel();
    let task = tokio::spawn(async move {
        scheduler
            .run(move || {
                started.send(()).unwrap();
                wait.recv().unwrap();
                Ok(())
            })
            .await
    });
    deadline(ready).await.unwrap();
    (task, release)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_units_run_serially_in_fifo_order() {
    let scheduler = SerialScheduler::default();
    let (first, release) = blocked(scheduler.clone()).await;
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut second = Box::pin(scheduler.run({
        let order = order.clone();
        move || {
            order.lock().unwrap().push(2);
            Ok(20)
        }
    }));
    let mut third = Box::pin(scheduler.run({
        let order = order.clone();
        move || {
            order.lock().unwrap().push(3);
            Ok(30)
        }
    }));
    poll_waiting(second.as_mut()).await;
    poll_waiting(third.as_mut()).await;
    // A spare blocking thread exists, but neither queued unit may execute yet.
    assert!(
        tokio::time::timeout(Duration::from_millis(100), second.as_mut())
            .await
            .is_err()
    );
    assert!(order.lock().unwrap().is_empty());
    release.send(()).unwrap();
    deadline(first).await.unwrap().unwrap();
    let (a, b) = deadline(async { tokio::join!(second, third) }).await;
    assert_eq!((a.unwrap(), b.unwrap()), (20, 30));
    assert_eq!(*order.lock().unwrap(), [2, 3]);
}

#[test]
fn cancelling_queued_work_never_dispatches_it() {
    // One blocking thread makes premature dispatch observable after the held job.
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(1)
        .enable_time()
        .build()
        .unwrap()
        .block_on(async {
            let scheduler = SerialScheduler::default();
            let (first, release) = blocked(scheduler.clone()).await;
            let calls = Arc::new(AtomicUsize::new(0));
            let mut cancelled = Box::pin(scheduler.run({
                let calls = calls.clone();
                move || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            }));
            poll_waiting(cancelled.as_mut()).await;
            drop(cancelled);
            release.send(()).unwrap();
            deadline(first).await.unwrap().unwrap();
            assert_eq!(deadline(scheduler.run(|| Ok(42))).await.unwrap(), 42);
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        });
}

struct DeviceState(Arc<AtomicBool>);

impl Drop for DeviceState {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelling_dispatched_work_retains_admission_and_resources_until_completion() {
    let scheduler = SerialScheduler::default();
    let dropped = Arc::new(AtomicBool::new(false));
    let state = Arc::new(DeviceState(dropped.clone()));
    let (started, ready) = oneshot::channel();
    let (release, wait) = mpsc::channel();
    let first = tokio::spawn({
        let scheduler = scheduler.clone();
        let state = state.clone();
        async move {
            scheduler
                .run(move || {
                    started.send(()).unwrap();
                    wait.recv().unwrap();
                    drop(state);
                    Ok(())
                })
                .await
        }
    });
    deadline(ready).await.unwrap();
    first.abort();
    assert!(deadline(first).await.unwrap_err().is_cancelled());
    drop(state);
    assert!(!dropped.load(Ordering::SeqCst));
    let mut next = Box::pin(scheduler.run({
        let dropped = dropped.clone();
        move || {
            assert!(dropped.load(Ordering::SeqCst));
            Ok(())
        }
    }));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), next.as_mut())
            .await
            .is_err()
    );
    assert!(!dropped.load(Ordering::SeqCst));
    release.send(()).unwrap();
    deadline(next).await.unwrap();
    assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn errors_and_panics_release_admission() {
    let scheduler = SerialScheduler::default();
    let error = scheduler
        .run(|| -> Result<()> { anyhow::bail!("forward failed") })
        .await;
    assert_eq!(error.unwrap_err().to_string(), "forward failed");
    let panic = scheduler
        .run(|| -> Result<()> { panic!("executor panicked") })
        .await;
    assert!(
        panic
            .unwrap_err()
            .to_string()
            .contains("execution task failed")
    );
    assert_eq!(deadline(scheduler.run(|| Ok(42))).await.unwrap(), 42);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loaded_executors_have_independent_admission() {
    let first_scheduler = SerialScheduler::default();
    let second_scheduler = SerialScheduler::default();
    let (first, release) = blocked(first_scheduler).await;
    assert_eq!(deadline(second_scheduler.run(|| Ok(42))).await.unwrap(), 42);
    release.send(()).unwrap();
    deadline(first).await.unwrap().unwrap();
}
