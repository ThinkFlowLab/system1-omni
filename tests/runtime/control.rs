use anyhow::Result;
use omni_runtime::{SerialScheduler, admission::BoundedAdmission, control::ExecutionContext};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn expired_queued_work_does_not_execute() {
    let scheduler = SerialScheduler::default();
    let (started, ready) = oneshot::channel();
    let (release, wait) = mpsc::channel();
    let first = tokio::spawn({
        let scheduler = scheduler.clone();
        async move {
            scheduler
                .run(move || {
                    started.send(()).unwrap();
                    wait.recv().unwrap();
                    Ok(())
                })
                .await
        }
    });
    ready.await.unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let context = ExecutionContext::new(Some(Instant::now() + Duration::from_millis(40)));
    let queued = tokio::spawn({
        let scheduler = scheduler.clone();
        let calls = calls.clone();
        async move {
            scheduler
                .run_with_context(context, move || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(80)).await;
    release.send(()).unwrap();
    first.await.unwrap().unwrap();
    assert!(
        queued
            .await
            .unwrap()
            .unwrap_err()
            .to_string()
            .contains("deadline")
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_after_dispatch_keeps_both_reservations_until_work_finishes() {
    let admission = BoundedAdmission::new(1, Some(10));
    let tokens = BoundedAdmission::new(1, Some(20));
    let context = ExecutionContext::new(None)
        .retain(admission.try_admit(3).unwrap())
        .retain(tokens.try_admit(20).unwrap());
    let (started, ready) = oneshot::channel();
    let (release, wait) = mpsc::channel();
    let scheduler = SerialScheduler::default();
    let task = tokio::spawn({
        let scheduler = scheduler.clone();
        async move {
            let _cancel = context.cancel_on_drop();
            scheduler
                .run_with_context(context, move || {
                    started.send(()).unwrap();
                    wait.recv().unwrap();
                    Ok(())
                })
                .await
        }
    });
    ready.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(admission.try_admit(1).is_none());
    assert!(tokens.try_admit(1).is_none());
    release.send(()).unwrap();
    scheduler_barrier(&scheduler).await;
    assert!(admission.try_admit(10).is_some());
    assert!(tokens.try_admit(20).is_some());
}

async fn scheduler_barrier(scheduler: &SerialScheduler) {
    tokio::time::timeout(Duration::from_secs(5), scheduler.run(|| Ok(())))
        .await
        .unwrap()
        .unwrap();
}

#[test]
fn blocking_pool_backlog_checks_cancellation_before_starting_work() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(1)
        .enable_time()
        .build()
        .unwrap()
        .block_on(async {
            let (started, ready) = oneshot::channel();
            let (release, wait) = mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                wait.recv().unwrap();
            });
            ready.await.unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let context = ExecutionContext::new(None);
            let cancel = context.cancel_on_drop();
            let scheduler = SerialScheduler::default();
            let mut work = Box::pin(scheduler.run_with_context(context, {
                let calls = calls.clone();
                move || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            }));
            std::future::poll_fn(|cx| {
                assert!(std::future::Future::poll(work.as_mut(), cx).is_pending());
                std::task::Poll::Ready(())
            })
            .await;
            drop(cancel);
            drop(work);
            release.send(()).unwrap();
            blocker.await.unwrap();
            scheduler_barrier(&scheduler).await;
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        });
}

#[tokio::test]
async fn errors_and_panics_release_owned_capacity() {
    let admission = BoundedAdmission::new(1, None);
    let scheduler = SerialScheduler::default();
    for panic in [false, true] {
        let context = ExecutionContext::new(None).retain(admission.try_admit(1).unwrap());
        let result: Result<()> = scheduler
            .run_with_context(context, move || {
                if panic {
                    panic!("expected panic")
                }
                anyhow::bail!("expected error")
            })
            .await;
        assert!(result.is_err());
        assert_eq!(admission.report().depth, 0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_while_waiting_releases_request_capacity_without_dispatch() {
    let scheduler = SerialScheduler::default();
    let (started, ready) = oneshot::channel();
    let (release, wait) = mpsc::channel();
    let first = tokio::spawn({
        let scheduler = scheduler.clone();
        async move {
            scheduler
                .run(move || {
                    started.send(()).unwrap();
                    wait.recv().unwrap();
                    Ok(())
                })
                .await
        }
    });
    ready.await.unwrap();
    let admission = BoundedAdmission::new(1, None);
    let context = ExecutionContext::new(None).retain(admission.try_admit(1).unwrap());
    let cancel = context.cancel_on_drop();
    let mut queued = Box::pin(scheduler.run_with_context(context, || -> Result<()> {
        panic!("cancelled work executed")
    }));
    std::future::poll_fn(|cx| {
        assert!(std::future::Future::poll(queued.as_mut(), cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    assert_eq!(admission.report().depth, 1);
    drop(cancel);
    drop(queued);
    assert_eq!(admission.report().depth, 0);
    release.send(()).unwrap();
    first.await.unwrap().unwrap();
    scheduler_barrier(&scheduler).await;
}

#[test]
fn blocking_pool_backlog_checks_deadline_before_starting_work() {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(1)
        .enable_time()
        .build()
        .unwrap()
        .block_on(async {
            let (started, ready) = oneshot::channel();
            let (release, wait) = mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started.send(()).unwrap();
                wait.recv().unwrap();
            });
            ready.await.unwrap();
            let scheduler = SerialScheduler::default();
            let calls = Arc::new(AtomicUsize::new(0));
            let context = ExecutionContext::new(Some(Instant::now() + Duration::from_millis(30)));
            let mut work = Box::pin(scheduler.run_with_context(context, {
                let calls = calls.clone();
                move || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            }));
            std::future::poll_fn(|cx| {
                assert!(std::future::Future::poll(work.as_mut(), cx).is_pending());
                std::task::Poll::Ready(())
            })
            .await;
            tokio::time::sleep(Duration::from_millis(60)).await;
            release.send(()).unwrap();
            blocker.await.unwrap();
            assert!(work.await.unwrap_err().to_string().contains("deadline"));
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        });
}
