//! One worker owns the model, stream and graph cache. HTTP requests never share GPU buffers.
use crate::{
    decision,
    model::Model,
    preprocess::{Preprocessor, Request},
};
use anyhow::{Result, anyhow};
use omni_jev::engine::{Engine, EngineError, Reply};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Instant,
};
use tokio::sync::{mpsc, oneshot};

struct Job {
    body: Vec<u8>,
    deadline: Instant,
    reply: oneshot::Sender<Result<Vec<u8>, EngineError>>,
}

pub struct Handle {
    sender: mpsc::Sender<Job>,
    ready: Arc<AtomicBool>,
    admission: Mutex<()>,
}

impl Engine for Handle {
    fn submit(&self, body: Vec<u8>, deadline: Instant) -> Result<Reply, EngineError> {
        let _guard = self
            .admission
            .lock()
            .map_err(|_| EngineError::Unavailable)?;
        if !self.ready() {
            return Err(EngineError::Unavailable);
        }
        let (tx, rx) = oneshot::channel();
        self.sender
            .try_send(Job {
                body,
                deadline,
                reply: tx,
            })
            .map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => EngineError::Busy,
                mpsc::error::TrySendError::Closed(_) => EngineError::Unavailable,
            })?;
        Ok(rx)
    }

    fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire) && !self.sender.is_closed()
    }
}

impl Handle {
    pub fn stop_accepting(&self) {
        let _guard = self.admission.lock().unwrap_or_else(|e| e.into_inner());
        self.ready.store(false, Ordering::Release);
    }
}

pub async fn start(
    checkpoint: PathBuf,
    bundle: PathBuf,
    queue_size: usize,
) -> Result<(Arc<Handle>, thread::JoinHandle<()>)> {
    anyhow::ensure!(
        queue_size > 0 && queue_size <= 256,
        "queue size must be 1..256"
    );
    let (tx, mut rx) = mpsc::channel::<Job>(queue_size);
    let ready = Arc::new(AtomicBool::new(false));
    let worker_ready = ready.clone();
    let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();
    let worker = thread::Builder::new()
        .name("laya-gpu".into())
        .spawn(move || {
            let (preprocessor, mut model) = match load_and_warmup(&checkpoint, &bundle) {
                Ok(loaded) => loaded,
                Err(error) => {
                    let _ = ready_tx.send(Err(format!("{error:#}")));
                    return;
                }
            };
            worker_ready.store(true, Ordering::Release);
            if ready_tx.send(Ok(())).is_err() {
                return;
            }

            while let Some(job) = rx.blocking_recv() {
                if job.reply.is_closed() || Instant::now() >= job.deadline {
                    continue;
                }
                let result = infer_request(&preprocessor, &mut model, &job);
                let failed = matches!(result, Err(EngineError::InferenceFailed));
                let _ = job.reply.send(result);
                if failed {
                    worker_ready.store(false, Ordering::Release);
                    break;
                }
            }
            worker_ready.store(false, Ordering::Release);
        })?;
    match ready_rx.await {
        Ok(Ok(())) => Ok((
            Arc::new(Handle {
                sender: tx,
                ready,
                admission: Mutex::new(()),
            }),
            worker,
        )),
        other => {
            drop(tx);
            let _ = worker.join();
            Err(anyhow!("native startup failed: {other:?}"))
        }
    }
}

fn load_and_warmup(checkpoint: &Path, bundle: &Path) -> Result<(Preprocessor, Model)> {
    let preprocessor = Preprocessor::load(checkpoint)?;
    let mut model = Model::load(checkpoint, bundle, true, false)?;
    let request: Request = serde_json::from_str(
        r#"{"model":"english","state":"I was charged twice for my order. Please refund the duplicate today.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for a refund?"}}}"#,
    )?;
    let batch = preprocessor.prepare(&request)?;
    model.infer(&batch)?;
    Ok((preprocessor, model))
}

fn infer_request(
    preprocessor: &Preprocessor,
    model: &mut Model,
    job: &Job,
) -> Result<Vec<u8>, EngineError> {
    let request: Request = serde_json::from_slice(&job.body)
        .map_err(|error| EngineError::InvalidRequest(error.to_string()))?;
    let batch = preprocessor
        .prepare(&request)
        .map_err(|error| EngineError::InvalidRequest(error.to_string()))?;
    // Cancellation before GPU submission is cheap. After submission, infer must finish copyback.
    if job.reply.is_closed() || Instant::now() >= job.deadline {
        return Err(EngineError::Unavailable);
    }
    let (logits, actions) = model.infer(&batch).map_err(|error| {
        eprintln!("native inference failed: {error:#}");
        EngineError::InferenceFailed
    })?;
    let response = decision::decode(&batch, &model.config.agent, &logits, &actions)
        .map_err(|_| EngineError::InferenceFailed)?;
    serde_json::to_vec(&response).map_err(|_| EngineError::InferenceFailed)
}
