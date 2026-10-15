//! One prefill per question and BF16 projection of the untied LM-head label rows.
use crate::processing::Input;
use anyhow::{Result, ensure};
use half::bf16;
use omni_qwen3_5_native::{
    cuda::{self, DeviceBuffer, Stream},
    inputs::MultimodalInput,
    model::{Config, Model},
    vision::VisionModel,
};
use omni_runtime::SerialScheduler;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::{
    ffi::c_void,
    panic::{AssertUnwindSafe, catch_unwind},
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
/// Validate selected LM-head rows, then pad the output dimension for CUDA GEMM.
pub fn head_bytes(data: &[u8], hidden: usize) -> Result<Vec<u8>> {
    let tensors = SafeTensors::deserialize(data)?;
    ensure!(
        tensors.names().len() == 1,
        "expected only selected LM-head weight"
    );
    let weight = tensors.tensor("weight")?;
    ensure!(
        weight.dtype() == Dtype::BF16 && weight.shape() == [32, hidden],
        "expected selected BF16 lm_head rows [32,hidden]"
    );
    ensure!(
        weight
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| bf16::from_le_bytes([b[0], b[1]]).is_finite()),
        "non-finite LM head"
    );
    let mut bytes = vec![0u8; 256 * hidden * 2];
    bytes[..weight.data().len()].copy_from_slice(weight.data());
    Ok(bytes)
}
struct Gemm(*mut c_void);
// SAFETY: each handle is exclusively used under the loaded-state mutex and scheduler.
unsafe impl Send for Gemm {}
impl Drop for Gemm {
    fn drop(&mut self) {
        // SAFETY: this handle came from cs1_gemm_create and is owned exclusively.
        unsafe { (cuda::api().cs1_gemm_destroy)(self.0) }
    }
}
struct OwnedStream(Stream);
impl Drop for OwnedStream {
    fn drop(&mut self) {
        let _ = cuda::synchronize(self.0);
        // SAFETY: the owned stream has been synchronized before destruction.
        unsafe {
            (cuda::api().cs1_stream_destroy)(self.0);
        }
    }
}
struct LabelHead {
    stream: OwnedStream,
    gemm: Gemm,
    weight: DeviceBuffer,
    input: DeviceBuffer,
    output: DeviceBuffer,
    hidden: usize,
}
impl LabelHead {
    fn load(bytes: &[u8], hidden: usize) -> Result<Self> {
        let stream = OwnedStream(cuda::new_stream()?);
        let weight = DeviceBuffer::new(bytes.len())?;
        // SAFETY: the weight allocation has exactly bytes.len() bytes.
        unsafe {
            cuda::upload(weight.at(0), bytes, stream.0)?;
        }
        // SAFETY: backend initialization returns an owning handle or null.
        let gemm = unsafe { (cuda::api().reference()?.gemm_create)(32 << 20) };
        ensure!(!gemm.is_null(), "LM head cuBLASLt setup failed");
        Ok(Self {
            stream,
            gemm: Gemm(gemm),
            weight,
            input: DeviceBuffer::new(hidden * 2)?,
            output: DeviceBuffer::new(256 * 2)?,
            hidden,
        })
    }
    fn score(&mut self, last: &[f32], count: usize) -> Result<Vec<f32>> {
        ensure!(
            last.len() == self.hidden && (2..=32).contains(&count),
            "invalid hidden state or label count"
        );
        let bytes = last
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect::<Vec<_>>();
        // SAFETY: input/weight/output allocations match GEMM dimensions; readback
        // waits for queued operations on this exclusively owned stream.
        unsafe {
            cuda::upload(self.input.at(0), &bytes, self.stream.0)?;
            cuda::check(
                (cuda::api().cs1_gemm)(
                    self.gemm.0,
                    self.input.at(0),
                    self.weight.at(0),
                    self.output.at(0),
                    1,
                    256,
                    self.hidden as i32,
                    256,
                    self.stream.0,
                ),
                "BF16 label LM-head GEMM",
            )?;
            let mut out = vec![0u8; 256 * 2];
            cuda::download(&mut out, self.output.at(0), self.stream.0)?;
            let logits = out
                .as_chunks::<2>()
                .0
                .iter()
                .take(count)
                .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect::<Vec<_>>();
            ensure!(
                logits.iter().all(|v| v.is_finite()),
                "non-finite LM-head logits"
            );
            Ok(logits)
        }
    }
    fn synchronize(&self) -> Result<()> {
        cuda::synchronize(self.stream.0)
    }
}
impl Drop for LabelHead {
    fn drop(&mut self) {
        let _ = self.synchronize();
    }
}
struct Loaded {
    language: Model,
    vision: VisionModel,
    head: LabelHead,
}
impl Loaded {
    fn synchronize(&self) -> Result<()> {
        let a = self.language.synchronize();
        let b = self.vision.synchronize();
        let c = self.head.synchronize();
        a.and(b).and(c)
    }
    fn execute(&mut self, inputs: &[Input]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        if let Some(first) = inputs.first() {
            embeddings = self.vision.forward_images(&first.images)?;
        }
        let mut rows = Vec::with_capacity(inputs.len());
        for input in inputs {
            let last = if input.images.is_empty() {
                self.language.forward(&input.ids)?
            } else {
                self.language.forward_multimodal(&MultimodalInput {
                    token_ids: &input.ids,
                    image_token_indices: &input.image_indices,
                    image_embeddings: &embeddings,
                    position_ids: [
                        &input.positions[0],
                        &input.positions[1],
                        &input.positions[2],
                    ],
                })?
            };
            rows.push(self.head.score(&last, input.n_candidates)?);
        }
        self.synchronize()?;
        Ok(rows)
    }
}
pub struct Executor {
    state: Arc<Mutex<Option<Loaded>>>,
    ready: Arc<AtomicBool>,
}
impl Executor {
    pub async fn load(dir: &Path, library: &Path, manifest: &Value) -> Result<Self> {
        crate::artifacts::verify_export(dir, manifest)?;
        let (adapter_file, scale) = crate::artifacts::lora_contract(manifest)?;
        let adapter_path = dir.join(adapter_file);
        let cfg = Config::load(dir)?;
        ensure!(
            (
                cfg.hidden,
                cfg.intermediate,
                cfg.full_attention.len(),
                cfg.heads,
                cfg.kv_heads,
                cfg.lin_k_heads,
                cfg.lin_v_heads
            ) == (5120, 17408, 64, 24, 4, 16, 48),
            "expected Qwen3.8-27B backbone"
        );
        ensure!(
            manifest["label_head_file"] == "jemm_lm_head.safetensors",
            "expected selected LM-head file"
        );
        let bytes = head_bytes(
            &std::fs::read(dir.join("jemm_lm_head.safetensors"))?,
            cfg.hidden,
        )?;
        let (dir, library) = (dir.to_path_buf(), library.to_path_buf());
        let state = tokio::task::spawn_blocking(move || -> Result<Loaded> {
            let mut language = Model::load_with_lora(&dir, &library, &adapter_path, scale)?;
            language.enable_reference_numerics()?;
            let mut vision = VisionModel::load(&dir, &library)?;
            vision.enable_reference_numerics()?;
            let head = LabelHead::load(&bytes, cfg.hidden)?;
            Ok(Loaded {
                language,
                vision,
                head,
            })
        })
        .await??;
        Ok(Self {
            state: Arc::new(Mutex::new(Some(state))),
            ready: Arc::new(AtomicBool::new(true)),
        })
    }
    pub fn available(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
    /// Admission covers all image, language and head work for one entire request.
    pub async fn execute(
        &self,
        scheduler: &SerialScheduler,
        inputs: Vec<Input>,
    ) -> Result<Vec<Vec<f32>>> {
        let state = self.state.clone();
        let ready = self.ready.clone();
        scheduler
            .run(move || {
                let mut state = match state.lock() {
                    Ok(state) => state,
                    Err(poison) => {
                        ready.store(false, Ordering::Release);
                        let mut state = poison.into_inner();
                        if let Some(loaded) = state.as_ref() {
                            let _ = loaded.synchronize();
                        }
                        state.take();
                        return Err(anyhow::anyhow!("poisoned model state"));
                    }
                };
                guarded(
                    &mut state,
                    |loaded| loaded.execute(&inputs),
                    Loaded::synchronize,
                    || ready.store(false, Ordering::Release),
                )
            })
            .await
    }
}

fn guarded<T, R>(
    state: &mut Option<T>,
    work: impl FnOnce(&mut T) -> Result<R>,
    synchronize: impl FnOnce(&T) -> Result<()>,
    mark_unavailable: impl FnOnce(),
) -> Result<R> {
    let Some(loaded) = state.as_mut() else {
        mark_unavailable();
        return Err(anyhow::anyhow!("model is unavailable"));
    };
    let result = catch_unwind(AssertUnwindSafe(|| work(loaded)))
        .unwrap_or_else(|_| Err(anyhow::anyhow!("model execution panicked")));
    if result.is_err() {
        mark_unavailable();
        let _ = synchronize(loaded);
        state.take();
    }
    result
}

#[cfg(test)]
#[path = "../../../../../tests/jemm/cleanup.rs"]
mod cleanup_tests;

#[cfg(test)]
#[path = "../../../../../tests/jemm/cuda.rs"]
mod cuda_tests;
