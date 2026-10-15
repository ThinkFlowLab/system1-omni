use super::{AtomicBool, guarded};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
struct Sentinel(Arc<AtomicUsize>);
impl Drop for Sentinel {
    fn drop(&mut self) {
        self.0.fetch_add(
            if self.0.load(Ordering::SeqCst) == 1 {
                1
            } else {
                100
            },
            Ordering::SeqCst,
        );
    }
}
#[test]
fn execution_error_synchronizes_and_retires_loaded_state() {
    let event = Arc::new(AtomicUsize::new(0));
    let mut state = Some(Sentinel(event.clone()));
    let ready = AtomicBool::new(true);
    let result = guarded(
        &mut state,
        |_| Err::<(), _>(anyhow::anyhow!("forward failed")),
        |s| {
            assert!(
                !ready.load(Ordering::SeqCst),
                "unavailable before synchronization"
            );
            s.0.store(1, Ordering::SeqCst);
            Ok(())
        },
        || {
            ready.store(false, Ordering::SeqCst);
        },
    );
    assert!(result.is_err());
    assert!(state.is_none());
    assert_eq!(event.load(Ordering::SeqCst), 2);
}
#[test]
fn execution_panic_synchronizes_and_retires_loaded_state() {
    let event = Arc::new(AtomicUsize::new(0));
    let mut state = Some(Sentinel(event.clone()));
    let ready = AtomicBool::new(true);
    let result = guarded(
        &mut state,
        |_| -> anyhow::Result<()> { panic!("forward panic") },
        |s| {
            assert!(
                !ready.load(Ordering::SeqCst),
                "unavailable before synchronization"
            );
            s.0.store(1, Ordering::SeqCst);
            Ok(())
        },
        || {
            ready.store(false, Ordering::SeqCst);
        },
    );
    assert!(result.is_err());
    assert!(state.is_none());
    assert_eq!(event.load(Ordering::SeqCst), 2);
}
#[test]
fn health_availability_does_not_wait_for_execution_mutex() {
    use super::{AtomicBool, Executor, Mutex};
    use std::sync::mpsc;
    let executor = Executor {
        state: Arc::new(Mutex::new(None)),
        ready: Arc::new(AtomicBool::new(true)),
    };
    let state = executor.state.clone();
    let held = state.lock().unwrap();
    let (tx, rx) = mpsc::channel();
    let thread = std::thread::spawn(move || tx.send(executor.available()).unwrap());
    let response = rx.recv_timeout(std::time::Duration::from_secs(1));
    drop(held);
    thread.join().unwrap();
    assert!(response.unwrap());
}
