use omni_jev::engine::{self, Engine, EngineError, Reply};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{net::TcpListener, sync::oneshot};
type PendingReply = oneshot::Sender<Result<Vec<u8>, EngineError>>;
struct Mock {
    ready: bool,
    mode: u8,
    held: Mutex<Vec<PendingReply>>,
}
impl Engine for Mock {
    fn ready(&self) -> bool {
        self.ready
    }
    fn submit(&self, body: Vec<u8>, _: Instant) -> Result<Reply, EngineError> {
        if self.mode == 1 {
            return Err(EngineError::Busy);
        }
        let (tx, rx) = oneshot::channel();
        if self.mode == 2 {
            self.held.lock().unwrap().push(tx);
        } else {
            let _ = tx.send(Ok(body));
        }
        Ok(rx)
    }
}
async fn start(m: Arc<Mock>) -> (String, tokio::task::JoinHandle<()>) {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", l.local_addr().unwrap());
    let app = engine::app(m, Duration::from_millis(40), 64);
    let task = tokio::spawn(async move {
        axum::serve(l, app).await.unwrap();
    });
    (url, task)
}
#[tokio::test]
async fn native_status_body_limit_and_readiness() {
    for (ready, mode, status) in [(true, 0, 200), (true, 1, 503), (false, 0, 503)] {
        let m = Arc::new(Mock {
            ready,
            mode,
            held: Mutex::new(vec![]),
        });
        let (url, task) = start(m).await;
        let c = reqwest::Client::new();
        let r = c
            .post(format!("{url}/v1/systemone"))
            .body("{\"ok\":true}")
            .send()
            .await
            .unwrap();
        assert_eq!(r.status().as_u16(), status);
        if status == 200 {
            assert_eq!(r.text().await.unwrap(), "{\"ok\":true}");
        }
        assert_eq!(
            c.get(format!("{url}/health"))
                .send()
                .await
                .unwrap()
                .status()
                .as_u16(),
            if ready { 200 } else { 503 }
        );
        if ready && mode == 0 {
            assert_eq!(
                c.post(format!("{url}/v1/systemone"))
                    .body(vec![b'x'; 65])
                    .send()
                    .await
                    .unwrap()
                    .status(),
                413
            );
        }
        task.abort();
        let _ = task.await;
    }
}
#[tokio::test]
async fn timeout_closes_reply_without_cancelling_inflight_owner() {
    let m = Arc::new(Mock {
        ready: true,
        mode: 2,
        held: Mutex::new(vec![]),
    });
    let (url, task) = start(m.clone()).await;
    let r = reqwest::Client::new()
        .post(format!("{url}/v1/systemone"))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 504);
    let tx = m.held.lock().unwrap().pop().unwrap();
    assert!(tx.is_closed());
    assert!(tx.send(Ok(b"late".to_vec())).is_err());
    task.abort();
    let _ = task.await;
}
