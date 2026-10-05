//! Keep CUDA's Rc state on one thread; admission belongs to the shared runtime.
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
};

use anyhow::{Context, Result, anyhow};
use omni_runtime::SerialScheduler;

use crate::{
    model::{Model, ModelOutput},
    packing::Batch,
};

struct Job {
    inputs: Batch,
    reply: mpsc::Sender<Result<ModelOutput>>,
}

struct Worker {
    sender: Option<mpsc::SyncSender<Job>>,
    thread: Option<thread::JoinHandle<()>>,
    ready: Arc<AtomicBool>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        drop(self.sender.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

pub struct Executor {
    worker: Arc<Worker>,
}

impl Executor {
    pub async fn load(checkpoint: &Path, bundle: &Path) -> Result<Self> {
        let (checkpoint, bundle) = (checkpoint.to_path_buf(), bundle.to_path_buf());
        tokio::task::spawn_blocking(move || Self::load_blocking(checkpoint, bundle)).await?
    }

    fn load_blocking(checkpoint: PathBuf, bundle: PathBuf) -> Result<Self> {
        // Rendezvous transport has no pending queue; SerialScheduler admits one request.
        let (sender, receiver) = mpsc::sync_channel::<Job>(0);
        let (loaded, initialized) = mpsc::channel();
        let ready = Arc::new(AtomicBool::new(false));
        let worker_ready = ready.clone();
        let thread = thread::Builder::new()
            .name("laya-cuda".into())
            .spawn(move || {
                let mut model = match Model::load(&checkpoint, &bundle, true, false) {
                    Ok(model) => model,
                    Err(e) => {
                        let _ = loaded.send(Err(format!("{e:#}")));
                        return;
                    }
                };
                worker_ready.store(true, Ordering::Release);
                if loaded.send(Ok(())).is_err() {
                    return;
                }
                while let Ok(job) = receiver.recv() {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        model.infer(&job.inputs)
                    }));
                    match result {
                        Ok(Ok(output)) => {
                            let _ = job.reply.send(Ok(output));
                        }
                        failed => {
                            worker_ready.store(false, Ordering::Release);
                            // Drop synchronizes outstanding CUDA work before the caller releases admission.
                            drop(model);
                            let error = match failed {
                                Ok(Err(e)) => e,
                                _ => anyhow!("CUDA executor panicked"),
                            };
                            let _ = job.reply.send(Err(error));
                            return;
                        }
                    }
                }
                worker_ready.store(false, Ordering::Release);
            })?;
        let worker = Arc::new(Worker {
            sender: Some(sender),
            thread: Some(thread),
            ready,
        });
        initialized
            .recv()
            .context("CUDA worker stopped during startup")?
            .map_err(anyhow::Error::msg)?;
        Ok(Self { worker })
    }

    pub fn ready(&self) -> bool {
        self.worker.ready.load(Ordering::Acquire)
    }

    pub async fn execute(&self, scheduler: &SerialScheduler, inputs: Batch) -> Result<ModelOutput> {
        let worker = self.worker.clone();
        scheduler
            .run(move || {
                let (reply, result) = mpsc::channel();
                worker
                    .sender
                    .as_ref()
                    .context("CUDA worker unavailable")?
                    .send(Job { inputs, reply })
                    .map_err(|_| anyhow!("CUDA worker unavailable"))?;
                result
                    .recv()
                    .context("CUDA worker stopped during execution")?
            })
            .await
    }
}
