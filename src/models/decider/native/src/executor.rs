//! Eager complete-row prefills and a selected tied-embedding CUDA readout.
use crate::{
    Limits, RowInput,
    batching::BatchLimits,
    checkpoint::{Checkpoint, HIDDEN, LABELS, PADDED_LABELS, VOCAB},
    prefix::{PrefixMode, PrefixPlan, PrefixStats},
};
use anyhow::{Context, Result, ensure};
use half::bf16;
use omni_qwen3_5_native::{
    cuda::{self, DeviceBuffer, Stream},
    model::{GraphStats, Model},
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
    capacity: usize,
}
impl Drop for Head {
    fn drop(&mut self) {
        let _ = cuda::set_device(0);
        let _ = cuda::synchronize(self.stream.0);
    }
}
impl Head {
    fn new(weights: &[u8], capacity: usize) -> Result<Self> {
        ensure!((1..=4).contains(&capacity), "invalid head batch capacity");
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
            input: DeviceBuffer::new(capacity * HIDDEN * 2)?,
            output: DeviceBuffer::new(capacity * PADDED_LABELS * 2)?,
            capacity,
        };
        // SAFETY: the device allocation has exactly weights.len() bytes.
        unsafe {
            cuda::upload(head.weights.at(0), weights, head.stream.0)?;
        }
        Ok(head)
    }
    fn project(&self, hidden: &[f32], count: usize) -> Result<Vec<f32>> {
        Ok(self.project_batch(&[hidden], &[count])?.pop().unwrap())
    }
    fn project_batch(&self, hidden: &[&[f32]], counts: &[usize]) -> Result<Vec<Vec<f32>>> {
        let rows = hidden.len();
        ensure!(
            rows > 0
                && rows <= self.capacity
                && rows == counts.len()
                && hidden
                    .iter()
                    .all(|h| h.len() == HIDDEN && h.iter().all(|x| x.is_finite()))
                && counts.iter().all(|n| (2..=LABELS).contains(n)),
            "invalid hidden batch/readout"
        );
        let input: Vec<u8> = hidden
            .iter()
            .flat_map(|h| h.iter())
            .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
            .collect();
        // SAFETY: [rows,HIDDEN] * [256,HIDDEN]^T -> [rows,256]; rows <= capacity.
        // The final zero weight row is padding and never participates in normalization.
        unsafe {
            cuda::upload(self.input.at(0), &input, self.stream.0)?;
            cuda::check(
                (cuda::api().cs1_gemm)(
                    self.gemm.0,
                    self.input.at(0),
                    self.weights.at(0),
                    self.output.at(0),
                    rows as i32,
                    PADDED_LABELS as i32,
                    HIDDEN as i32,
                    PADDED_LABELS as i32,
                    self.stream.0,
                ),
                "Decider label projection",
            )?;
        }
        let mut output = vec![0; rows * PADDED_LABELS * 2];
        // SAFETY: every returned row is inside the allocation; download synchronizes GEMM.
        unsafe {
            cuda::download(&mut output, self.output.at(0), self.stream.0)?;
        }
        output
            .as_chunks::<{ PADDED_LABELS * 2 }>()
            .0
            .iter()
            .zip(counts)
            .map(|(row, &count)| {
                let logits: Vec<f32> = row.as_chunks::<2>().0[..count]
                    .iter()
                    .map(|&b| bf16::from_le_bytes(b).to_f32())
                    .collect();
                ensure!(
                    logits.iter().all(|x| x.is_finite()),
                    "nonfinite candidate logits"
                );
                Ok(logits)
            })
            .collect()
    }
}
struct Loaded {
    model: Model,
    head: Head,
    batching: BatchLimits,
    prefix_mode: PrefixMode,
    prefix_stats: PrefixStats,
}
impl Loaded {
    fn execute(&mut self, rows: &[RowInput]) -> Result<Vec<Vec<f32>>> {
        cuda::set_device(0)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let lengths: Vec<usize> = rows.iter().map(|row| row.ids.len()).collect();
            let plan = match self.prefix_mode {
                PrefixMode::Shared => PrefixPlan::new(rows),
                PrefixMode::Auto => PrefixPlan::new(rows).filter(|plan| plan.worth_auto(rows)),
                _ => None,
            };
            if self.prefix_mode == PrefixMode::Fixed
                || self.prefix_mode == PrefixMode::Shared
                || plan.is_some()
            {
                let hidden = if let Some(plan) = plan {
                    let hidden = self
                        .model
                        .forward_shared(&plan.prompts)?
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>();
                    self.prefix_stats.shared_requests += 1;
                    self.prefix_stats.saved_tokens += plan.saved_tokens as u64;
                    hidden
                } else {
                    rows.iter()
                        .map(|row| self.model.forward_fixed(&row.ids))
                        .collect::<Result<Vec<_>>>()?
                };
                ensure!(
                    hidden.len() == rows.len(),
                    "shared backbone output count mismatch"
                );
                let mut logits = Vec::with_capacity(rows.len());
                for range in self.batching.ranges(&lengths) {
                    let batch = &rows[range.clone()];
                    let hidden: Vec<&[f32]> = hidden[range].iter().map(Vec::as_slice).collect();
                    let counts: Vec<_> = batch.iter().map(|row| row.candidate_ids.len()).collect();
                    logits.extend(self.head.project_batch(&hidden, &counts)?);
                }
                return Ok(logits);
            }
            let mut logits = Vec::with_capacity(rows.len());
            for range in self.batching.ranges(&lengths) {
                let batch = &rows[range];
                if batch.len() == 1 {
                    let hidden = self.model.forward(&batch[0].ids)?;
                    logits.push(self.head.project(&hidden, batch[0].candidate_ids.len())?);
                } else {
                    let inputs: Vec<&[u32]> = batch.iter().map(|row| row.ids.as_slice()).collect();
                    let hidden = self.model.forward_batch(&inputs)?;
                    ensure!(
                        hidden.len() == batch.len(),
                        "backbone batch output count mismatch"
                    );
                    let hidden: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
                    let counts: Vec<usize> =
                        batch.iter().map(|row| row.candidate_ids.len()).collect();
                    logits.extend(self.head.project_batch(&hidden, &counts)?);
                }
            }
            if self.prefix_mode == PrefixMode::Auto {
                self.prefix_stats.auto_independent_requests += 1;
            }
            Ok(logits)
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
    graph_stats: Arc<Mutex<GraphStats>>,
    prefix_stats: Arc<Mutex<PrefixStats>>,
}
impl Executor {
    pub(crate) async fn load(
        dir: &Path,
        library: &Path,
        checkpoint: Checkpoint,
        labels: Vec<u32>,
        batching: BatchLimits,
        graph: bool,
        prefix_mode: PrefixMode,
    ) -> Result<Self> {
        ensure!(
            !graph || prefix_mode == PrefixMode::Off,
            "shared/fixed execution requires Graph off"
        );
        let (dir, library) = (dir.to_owned(), library.to_owned());
        let loaded = tokio::task::spawn_blocking(move || -> Result<Loaded> {
            let model = Model::load_with_graph(&dir, &library, graph)?;
            let head = Head::new(&checkpoint.head, batching.max_rows())?;
            Ok(Loaded {
                model,
                head,
                batching,
                prefix_mode,
                prefix_stats: PrefixStats::default(),
            })
        })
        .await??;
        let graph_stats = Arc::new(Mutex::new(loaded.model.graph_stats()));
        Ok(Self {
            graph_stats,
            prefix_stats: Arc::new(Mutex::new(PrefixStats::default())),
            loaded: Arc::new(Mutex::new(Some(loaded))),
            labels,
            ready: Arc::new(AtomicBool::new(true)),
        })
    }
    pub fn graph_stats(&self) -> GraphStats {
        *self
            .graph_stats
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
    pub fn prefix_stats(&self) -> PrefixStats {
        *self
            .prefix_stats
            .lock()
            .unwrap_or_else(|error| error.into_inner())
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
        let (loaded, ready, stats) = (
            self.loaded.clone(),
            self.ready.clone(),
            self.graph_stats.clone(),
        );
        let prefix_stats = self.prefix_stats.clone();
        scheduler
            .run(move || {
                let mut guard = match loaded.lock() {
                    Ok(guard) => guard,
                    Err(_) => {
                        let mut snapshot = stats.lock().unwrap_or_else(|error| error.into_inner());
                        snapshot.enabled = false;
                        snapshot.cached_shapes = 0;
                        ready.store(false, Ordering::Release);
                        return Err(anyhow::anyhow!("poisoned executor"));
                    }
                };
                let loaded = guard.as_mut().context("executor unavailable")?;
                let result = loaded.execute(&rows);
                *prefix_stats
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = loaded.prefix_stats;
                let mut snapshot = loaded.model.graph_stats();
                if result.is_err() {
                    snapshot.enabled = false;
                    snapshot.cached_shapes = 0;
                }
                *stats.lock().unwrap_or_else(|error| error.into_inner()) = snapshot;
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
