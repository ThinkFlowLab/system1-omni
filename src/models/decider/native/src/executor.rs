//! Eager complete-row prefills and a selected tied-embedding CUDA readout.
use crate::{
    Limits, RowInput,
    checkpoint::{Checkpoint, HIDDEN, LABELS, PADDED_LABELS, VOCAB},
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use omni_qwen3_5_native::{
    cuda::{self, DeviceBuffer, Stream},
    model::Model,
};
use omni_runtime::SerialScheduler;
use std::{
    ffi::c_void,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

struct ProjectionStream(Stream);
impl Drop for ProjectionStream {
    fn drop(&mut self) {
        // SAFETY: this owner destroys its created stream after Head synchronizes it.
        unsafe {
            (cuda::api().cs1_stream_destroy)(self.0);
        }
    }
}
struct Gemm(*mut c_void);
// SAFETY: only the serialized executor uses this handle; each call selects device 0.
unsafe impl Send for Gemm {}
impl Drop for Gemm {
    fn drop(&mut self) {
        // SAFETY: handle created by cs1_gemm_create, owned exclusively here.
        unsafe {
            (cuda::api().cs1_gemm_destroy)(self.0);
        }
    }
}
struct Head {
    stream: ProjectionStream,
    gemm: Gemm,
    weights: DeviceBuffer,
    input: DeviceBuffer,
    output: DeviceBuffer,
}
impl Drop for Head {
    fn drop(&mut self) {
        let _ = cuda::set_device(0);
        let _ = cuda::synchronize(self.stream.0);
    }
}
impl Head {
    fn new(weights: &[u8]) -> Result<Self> {
        ensure!(
            weights.len() == PADDED_LABELS * HIDDEN * 2,
            "invalid selected head buffer"
        );
        let stream = ProjectionStream(cuda::new_stream()?);
        // SAFETY: null-checked immediately, then retained in the owning guard.
        let gemm = Gemm(unsafe { (cuda::api().cs1_gemm_create)(32 << 20) });
        ensure!(!gemm.0.is_null(), "Decider cuBLASLt setup failed");
        let head = Self {
            stream,
            gemm,
            weights: DeviceBuffer::new(weights.len())?,
            input: DeviceBuffer::new(HIDDEN * 2)?,
            output: DeviceBuffer::new(PADDED_LABELS * 2)?,
        };
        // SAFETY: the device allocation has exactly weights.len() bytes.
        unsafe {
            cuda::upload(head.weights.at(0), weights, head.stream.0)?;
        }
        Ok(head)
    }
    fn project(&self, hidden: &[f32], count: usize) -> Result<Vec<f32>> {
        ensure!(
            hidden.len() == HIDDEN
                && hidden.iter().all(|x| x.is_finite())
                && (2..=LABELS).contains(&count),
            "invalid hidden state/readout"
        );
        let input: Vec<u8> = hidden
            .iter()
            .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
            .collect();
        // SAFETY: [1,HIDDEN] * [256,HIDDEN]^T -> [1,256]; allocated buffers match.
        // The last weight row is zero padding and is never returned or normalized.
        unsafe {
            cuda::upload(self.input.at(0), &input, self.stream.0)?;
            cuda::check(
                (cuda::api().cs1_gemm)(
                    self.gemm.0,
                    self.input.at(0),
                    self.weights.at(0),
                    self.output.at(0),
                    1,
                    PADDED_LABELS as i32,
                    HIDDEN as i32,
                    PADDED_LABELS as i32,
                    self.stream.0,
                ),
                "Decider label projection",
            )?;
        }
        let mut output = vec![0; PADDED_LABELS * 2];
        // SAFETY: download waits for GEMM and copies exactly the output allocation.
        unsafe {
            cuda::download(&mut output, self.output.at(0), self.stream.0)?;
        }
        let logits: Vec<f32> = output.as_chunks::<2>().0[..count]
            .iter()
            .map(|&b| bf16::from_le_bytes(b).to_f32())
            .collect();
        ensure!(
            logits.iter().all(|x| x.is_finite()),
            "nonfinite candidate logits"
        );
        Ok(logits)
    }
}
struct Loaded {
    model: Model,
    head: Head,
}
impl Loaded {
    fn execute(&mut self, rows: &[RowInput]) -> Result<Vec<Vec<f32>>> {
        cuda::set_device(0)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rows.iter()
                .map(|row| {
                    let hidden = self.model.forward(&row.ids)?;
                    self.head.project(&hidden, row.candidate_ids.len())
                })
                .collect::<Result<Vec<_>>>()
        }))
        .map_err(|_| anyhow::anyhow!("Decider execution panicked"))
        .and_then(|result| result);
        // Complete both streams on success and failure before admission is released.
        let model_sync = self.model.synchronize();
        let head_sync = cuda::synchronize(self.head.stream.0);
        model_sync?;
        head_sync?;
        result
    }
}

pub struct Executor {
    loaded: Arc<Mutex<Option<Loaded>>>,
    labels: Vec<u32>,
    ready: Arc<AtomicBool>,
}
impl Executor {
    pub(crate) async fn load(
        dir: &Path,
        library: &Path,
        checkpoint: Checkpoint,
        labels: Vec<u32>,
    ) -> Result<Self> {
        ensure!(
            std::env::var("CUA_S1_GRAPH").as_deref() != Ok("1"),
            "Decider eager execution requires CUA_S1_GRAPH unset or 0"
        );
        let (dir, library) = (dir.to_owned(), library.to_owned());
        let loaded = tokio::task::spawn_blocking(move || -> Result<Loaded> {
            let model = Model::load(&dir, &library)?;
            let head = Head::new(&checkpoint.head)?;
            Ok(Loaded { model, head })
        })
        .await??;
        Ok(Self {
            loaded: Arc::new(Mutex::new(Some(loaded))),
            labels,
            ready: Arc::new(AtomicBool::new(true)),
        })
    }
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// One admitted unit holds every row and its GPU head; no partial answers escape.
    /// Cancellation after dispatch retains resources and admission until synchronization.
    pub async fn execute(
        &self,
        scheduler: &SerialScheduler,
        rows: Vec<RowInput>,
    ) -> Result<Vec<Vec<f32>>> {
        validate_rows(&rows, &self.labels)?;
        ensure!(self.is_ready(), "executor unavailable");
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let (loaded, ready) = (self.loaded.clone(), self.ready.clone());
        scheduler
            .run(move || {
                let mut guard = match loaded.lock() {
                    Ok(guard) => guard,
                    Err(_) => {
                        ready.store(false, Ordering::Release);
                        return Err(anyhow::anyhow!("poisoned executor"));
                    }
                };
                let result = guard
                    .as_mut()
                    .context("executor unavailable")?
                    .execute(&rows);
                if result.is_err() {
                    ready.store(false, Ordering::Release);
                    // Loaded::execute synchronized both streams, including failure paths.
                    // Retire failed device state before releasing the scheduler permit.
                    guard.take();
                }
                result
            })
            .await
    }
}

fn validate_rows(rows: &[RowInput], labels: &[u32]) -> Result<()> {
    let limits = Limits::default();
    ensure!(rows.len() <= limits.max_rows, "too many executor rows");
    let mut total = 0usize;
    for row in rows {
        let n = row.ids.len();
        ensure!(
            n > 0 && n <= limits.max_row_tokens && row.readout_position == n - 1,
            "invalid complete row/readout position"
        );
        ensure!(
            row.ids.iter().all(|&id| (id as usize) < VOCAB),
            "token outside vocabulary"
        );
        let count = row.candidate_ids.len();
        ensure!(
            (2..=LABELS).contains(&count)
                && labels.get(..count) == Some(row.candidate_ids.as_slice()),
            "candidate IDs differ from tied head ordering"
        );
        total = total.checked_add(n).context("executor token overflow")?;
        ensure!(
            total <= limits.max_request_tokens,
            "executor processed-token budget exceeded"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../../tests/decider/executor.rs"]
mod tests;
