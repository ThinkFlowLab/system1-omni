//! GPU vision forward for the structurally validated, unmerged checkpoint.
use super::{VisionCheckpoint, geometry::Geometry};
use crate::{
    cuda::{self, DeviceBuffer, Stream, api, check},
    image_preprocess::ProcessedImage,
};
use anyhow::{Result, ensure};
use half::bf16;
use std::{collections::BTreeMap, ffi::c_void, path::Path};

type Trace<'a> = Option<&'a mut dyn FnMut(&str, &[bf16]) -> Result<()>>;

struct OwnedStream(Stream);
impl Drop for OwnedStream {
    fn drop(&mut self) {
        unsafe { (api().cs1_stream_destroy)(self.0) };
    }
}
struct Gemm(*mut c_void);
// SAFETY: the model uses its GEMM handle and stream serially.
unsafe impl Send for Gemm {}
impl Drop for Gemm {
    fn drop(&mut self) {
        unsafe { (api().cs1_gemm_destroy)(self.0) };
    }
}

/// The base remains BF16; all 50 LoRA A/B pairs remain FP32, applied at scale 2.
/// A model is used by one request at a time and returns row-major [image_tokens,2560].
pub struct VisionModel {
    // Field order retires graph/scratch before weights, GEMM workspace and stream.
    scratch: Option<Scratch>,
    base: BTreeMap<String, DeviceBuffer>,
    lora: BTreeMap<String, DeviceBuffer>,
    gemm: Gemm,
    stream: OwnedStream,
    graph_enabled: bool,
}
impl VisionModel {
    pub fn load(base: impl AsRef<Path>, adapter: impl AsRef<Path>, library: &Path) -> Result<Self> {
        let checkpoint = VisionCheckpoint::load(base, adapter)?;
        cuda::load(library)?;
        cuda::set_device(0)?;
        let stream = OwnedStream(cuda::new_stream()?);
        let gemm = Gemm(unsafe { (api().cs1_gemm_create)(32 << 20) });
        ensure!(!gemm.0.is_null(), "cannot create vision cuBLAS handle");
        let mut model = Self {
            scratch: None,
            graph_enabled: std::env::var("CUA_S1_VISION_GRAPH").as_deref() == Ok("1"),
            base: BTreeMap::new(),
            lora: BTreeMap::new(),
            stream,
            gemm,
        };
        for name in checkpoint.base_names() {
            let tensor = checkpoint.base_tensor(name)?;
            model.base.insert(
                name.strip_prefix("model.visual.").unwrap().into(),
                model.upload(tensor.data())?,
            );
        }
        for name in checkpoint.adapter_names() {
            let tensor = checkpoint.adapter_tensor(name)?;
            model.lora.insert(
                name.strip_prefix("base_model.model.model.visual.")
                    .unwrap()
                    .into(),
                model.upload(tensor.data())?,
            );
        }
        model
            .base
            .insert("__zero_bias".into(), model.upload(&[0; 2048])?);
        Ok(model)
    }
    fn upload(&self, bytes: &[u8]) -> Result<DeviceBuffer> {
        let buffer = DeviceBuffer::new(bytes.len())?;
        unsafe {
            cuda::upload(buffer.at(0), bytes, self.stream.0)?;
        }
        Ok(buffer)
    }
    fn weight(&self, name: &str) -> *const c_void {
        self.base[name].at(0)
    }
    fn norm(&self, name: &str, x: &DeviceBuffer, y: &DeviceBuffer, rows: usize) -> Result<()> {
        unsafe {
            check(
                (api().cs1_vision_norm)(
                    x.at(0),
                    self.weight(&format!("{name}.weight")),
                    self.weight(&format!("{name}.bias")),
                    y.at(0),
                    rows as i32,
                    1024,
                    self.stream.0,
                ),
                name,
            )
        }
    }
    #[allow(clippy::too_many_arguments)] // Mirrors the fixed-shape GEMM operation.
    fn linear(
        &self,
        name: &str,
        x: &DeviceBuffer,
        y: &DeviceBuffer,
        rows: usize,
        output: usize,
        input: usize,
        work: &Work,
    ) -> Result<()> {
        // SAFETY: each caller provides rows*input/output sized allocations. All dimensions
        // derive from validated fixed architecture and bounded image geometry.
        let bias_name = if name == "patch_embed.proj" {
            "__zero_bias".into()
        } else {
            format!("{name}.bias")
        };
        unsafe {
            check(
                (api().cs1_vision_linear)(
                    self.gemm.0,
                    x.at(0),
                    self.weight(&format!("{name}.weight")),
                    self.weight(&bias_name),
                    y.at(0),
                    rows as i32,
                    output as i32,
                    input as i32,
                    self.stream.0,
                ),
                name,
            )?;
            if name == "patch_embed.proj" {
                check(
                    (api().cs1_vision_bias)(
                        y.at(0),
                        self.weight("patch_embed.proj.bias"),
                        rows * output,
                        output as i32,
                        self.stream.0,
                    ),
                    "vision patch bias",
                )?;
            }
            if let Some(a) = self.lora.get(&format!("{name}.lora_A.weight")) {
                let b = &self.lora[&format!("{name}.lora_B.weight")];
                check(
                    (api().cs1_vision_to_float)(
                        x.at(0),
                        work.float_input.at(0).cast(),
                        rows * input,
                        self.stream.0,
                    ),
                    "vision LoRA input",
                )?;
                check(
                    (api().cs1_gemm_f32)(
                        self.gemm.0,
                        work.float_input.at(0).cast(),
                        a.at(0).cast(),
                        work.rank.at(0).cast(),
                        rows as i32,
                        16,
                        input as i32,
                        self.stream.0,
                    ),
                    "vision LoRA A",
                )?;
                check(
                    (api().cs1_gemm_f32)(
                        self.gemm.0,
                        work.rank.at(0).cast(),
                        b.at(0).cast(),
                        work.delta.at(0).cast(),
                        rows as i32,
                        output as i32,
                        16,
                        self.stream.0,
                    ),
                    "vision LoRA B",
                )?;
                check(
                    (api().cs1_vision_lora_add)(
                        y.at(0),
                        work.delta.at(0).cast(),
                        rows * output,
                        2.,
                        self.stream.0,
                    ),
                    "vision LoRA add",
                )?;
            }
        }
        Ok(())
    }
    fn read(&self, x: &DeviceBuffer, n: usize) -> Result<Vec<bf16>> {
        let mut bytes = vec![0u8; n * 2];
        unsafe {
            cuda::download(&mut bytes, x.at(0), self.stream.0)?;
        }
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])))
            .collect())
    }
    fn trace(
        &self,
        callback: &mut Trace<'_>,
        name: &str,
        x: &DeviceBuffer,
        n: usize,
    ) -> Result<()> {
        if let Some(callback) = callback.as_mut() {
            callback(name, &self.read(x, n)?)?;
        }
        Ok(())
    }
    pub fn synchronize(&self) -> Result<()> {
        cuda::set_device(0)?;
        cuda::synchronize(self.stream.0)
    }

    pub fn forward(&mut self, image: &ProcessedImage) -> Result<Vec<bf16>> {
        self.run(image, None)
    }
    /// Optional synchronized stage downloads for parity diagnosis; ordinary forward skips them.
    pub fn forward_with_trace(
        &mut self,
        image: &ProcessedImage,
        mut callback: impl FnMut(&str, &[bf16]) -> Result<()>,
    ) -> Result<Vec<bf16>> {
        self.run(image, Some(&mut callback))
    }
    fn run(&mut self, image: &ProcessedImage, mut callback: Trace<'_>) -> Result<Vec<bf16>> {
        cuda::set_device(0)?;
        let geo = if self
            .scratch
            .as_ref()
            .is_some_and(|s| s.grid == image.image_grid_thw)
        {
            None
        } else {
            Some(Geometry::new(image.image_grid_thw)?)
        };
        let n = image.image_grid_thw[1] * image.image_grid_thw[2];
        ensure!(
            image.pixel_values.len() == n * 1536,
            "vision pixel_values length does not match grid"
        );
        ensure!(
            image.resized_height == image.image_grid_thw[1] * 16
                && image.resized_width == image.image_grid_thw[2] * 16,
            "vision resized geometry does not match grid"
        );
        ensure!(
            image.pixel_values.iter().all(|v| v.is_finite()),
            "vision pixels must be finite"
        );
        let pixels: Vec<u8> = image
            .pixel_values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect();
        if let Some(geo) = geo {
            self.synchronize()?;
            // Destroy the executable before releasing any address it references.
            self.scratch = None;
            self.scratch = Some(Scratch::new(self, image.image_grid_thw, geo)?);
        }
        let scratch = self.scratch.as_ref().unwrap();
        // SAFETY: pixels have the validated exact size of the resident allocation.
        unsafe {
            cuda::upload(scratch.pixels.at(0), &pixels, self.stream.0)?;
        }
        if callback.is_none()
            && self.graph_enabled
            && let Some(graph) = &scratch.graph
        {
            graph.launch(self.stream.0)?;
            graph_trace("replayed", scratch.grid);
            return self.read(&scratch.out, n / 4 * 2560);
        }
        self.enqueue(scratch, &mut callback)?;
        let result = self.read(&scratch.out, n / 4 * 2560)?;
        if let Some(callback) = callback.as_mut() {
            callback("merger.output", &result)?;
        } else if self.graph_enabled {
            // The eager read completed warmup. Capture does not execute the operators;
            // return the already completed result even if recording fails.
            let captured = cuda::Graph::capture(self.stream.0, || self.enqueue(scratch, &mut None));
            self.finish_capture(captured);
        }
        Ok(result)
    }
    fn finish_capture(&mut self, captured: Result<cuda::Graph>) {
        match captured {
            Ok(graph) => {
                let scratch = self.scratch.as_mut().unwrap();
                scratch.graph = Some(graph);
                graph_trace("captured", scratch.grid);
            }
            Err(error) => {
                eprintln!("Vision CUDA Graph capture failed; using eager execution: {error:#}");
                self.graph_enabled = false;
                if let Some(scratch) = self.scratch.as_mut() {
                    scratch.graph = None;
                }
            }
        }
    }
    fn enqueue(&self, scratch: &Scratch, callback: &mut Trace<'_>) -> Result<()> {
        let n = scratch.grid[1] * scratch.grid[2];
        let Scratch {
            pixels,
            indices,
            weights,
            co,
            si,
            w,
            x,
            norm,
            qkv,
            q,
            k,
            attn,
            delta,
            mlp,
            out,
            ..
        } = scratch;
        self.linear("patch_embed.proj", pixels, x, n, 1024, 1536, w)?;
        self.trace(callback, "patch_embed", x, n * 1024)?;
        unsafe {
            check(
                (api().cs1_vision_position)(
                    x.at(0),
                    self.weight("pos_embed.weight"),
                    indices.at(0).cast(),
                    weights.at(0).cast(),
                    n as i32,
                    self.stream.0,
                ),
                "vision learned positions",
            )?;
        }
        self.trace(callback, "position", x, n * 1024)?;
        for i in 0..24 {
            let p = format!("blocks.{i}");
            self.norm(&format!("{p}.norm1"), x, norm, n)?;
            self.linear(&format!("{p}.attn.qkv"), norm, qkv, n, 3072, 1024, w)?;
            unsafe {
                check(
                    (api().cs1_vision_rope)(
                        qkv.at(0),
                        co.at(0).cast(),
                        si.at(0).cast(),
                        q.at(0),
                        k.at(0),
                        n as i32,
                        self.stream.0,
                    ),
                    "vision rotary",
                )?;
                check(
                    (api().cs1_vision_attention)(
                        q.at(0),
                        k.at(0),
                        qkv.at(2048 * 2),
                        attn.at(0),
                        n as i32,
                        self.stream.0,
                    ),
                    "vision attention",
                )?;
            }
            self.linear(&format!("{p}.attn.proj"), attn, delta, n, 1024, 1024, w)?;
            unsafe {
                check(
                    (api().cs1_vision_add)(x.at(0), delta.at(0), n * 1024, self.stream.0),
                    "vision attention residual",
                )?;
            }
            self.norm(&format!("{p}.norm2"), x, norm, n)?;
            self.linear(&format!("{p}.mlp.linear_fc1"), norm, mlp, n, 4096, 1024, w)?;
            unsafe {
                check(
                    (api().cs1_vision_gelu)(mlp.at(0), n * 4096, 0, self.stream.0),
                    "vision tanh GELU",
                )?;
            }
            self.linear(&format!("{p}.mlp.linear_fc2"), mlp, delta, n, 1024, 4096, w)?;
            unsafe {
                check(
                    (api().cs1_vision_add)(x.at(0), delta.at(0), n * 1024, self.stream.0),
                    "vision MLP residual",
                )?;
            }
            self.trace(callback, &p, x, n * 1024)?;
        }
        self.norm("merger.norm", x, norm, n)?;
        self.trace(callback, "merger.norm", norm, n * 1024)?;
        // Consecutive groups of four patches already have the required 2x2 merge order.
        self.linear("merger.linear_fc1", norm, mlp, n / 4, 4096, 4096, w)?;
        self.trace(callback, "merger.linear_fc1", mlp, n * 1024)?;
        unsafe {
            check(
                (api().cs1_vision_gelu)(mlp.at(0), n * 1024, 1, self.stream.0),
                "vision exact GELU",
            )?;
        }
        self.linear("merger.linear_fc2", mlp, out, n / 4, 2560, 4096, w)?;
        Ok(())
    }
}
impl Drop for VisionModel {
    fn drop(&mut self) {
        // Error paths can leave work queued; all referenced resources are still alive here.
        let _ = self.synchronize();
    }
}
fn graph_trace(action: &str, grid: [usize; 3]) {
    if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
        eprintln!("Vision CUDA Graph {action} grid={grid:?}");
    }
}
struct Scratch {
    // Executable is dropped first. One exact grid bounds resident activation memory.
    graph: Option<cuda::Graph>,
    grid: [usize; 3],
    pixels: DeviceBuffer,
    indices: DeviceBuffer,
    weights: DeviceBuffer,
    co: DeviceBuffer,
    si: DeviceBuffer,
    w: Work,
    x: DeviceBuffer,
    norm: DeviceBuffer,
    qkv: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    attn: DeviceBuffer,
    delta: DeviceBuffer,
    mlp: DeviceBuffer,
    out: DeviceBuffer,
}
impl Scratch {
    fn new(model: &VisionModel, grid: [usize; 3], geo: Geometry) -> Result<Self> {
        let n = grid[1] * grid[2];
        Ok(Self {
            graph: None,
            grid,
            pixels: DeviceBuffer::new(n * 1536 * 2)?,
            indices: model.upload(
                &geo.indices
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            weights: model.upload(
                &geo.weights
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            co: model.upload(
                &geo.cos
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            si: model.upload(
                &geo.sin
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            w: Work::new(n)?,
            x: DeviceBuffer::new(n * 1024 * 2)?,
            norm: DeviceBuffer::new(n * 1024 * 2)?,
            qkv: DeviceBuffer::new(n * 3072 * 2)?,
            q: DeviceBuffer::new(n * 1024 * 2)?,
            k: DeviceBuffer::new(n * 1024 * 2)?,
            attn: DeviceBuffer::new(n * 1024 * 2)?,
            delta: DeviceBuffer::new(n * 1024 * 2)?,
            mlp: DeviceBuffer::new(n * 4096 * 2)?,
            out: DeviceBuffer::new(n / 4 * 2560 * 2)?,
        })
    }
}
struct Work {
    float_input: DeviceBuffer,
    rank: DeviceBuffer,
    delta: DeviceBuffer,
}
impl Work {
    fn new(n: usize) -> Result<Self> {
        Ok(Self {
            float_input: DeviceBuffer::new(n * 4096 * 4)?,
            rank: DeviceBuffer::new(n * 16 * 4)?,
            delta: DeviceBuffer::new(n * 4096 * 4)?,
        })
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/cua_s1/vision_graph.rs"]
mod graph_tests;
