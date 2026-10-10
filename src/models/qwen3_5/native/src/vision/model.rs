//! GPU vision forward for the structurally validated, unmerged checkpoint.
use super::{VisionCheckpoint, VisionConfig, geometry::VisionGeometry};
use crate::{
    cuda::{self, DeviceBuffer, Stream, api, check},
    image_preprocess::ProcessedImage,
};
use anyhow::{Result, ensure};
use half::bf16;
use safetensors::{Dtype, tensor::TensorView};
use std::{
    collections::{BTreeMap, VecDeque},
    ffi::c_void,
    path::Path,
};

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

/// BF16 base with optional separate FP32 Cua LoRA pairs at scale2.
/// One request at a time; returns row-major [image_tokens, config.out_hidden_size].
pub struct VisionModel {
    config: VisionConfig,
    // Captures and their buffers retire before weights, GEMM workspace and stream.
    scratch: VecDeque<Scratch>,
    graph_enabled: bool,
    base: BTreeMap<String, DeviceBuffer>,
    lora: BTreeMap<String, DeviceBuffer>,
    gemm: Gemm,
    stream: OwnedStream,
}
impl VisionModel {
    pub fn load(dir: impl AsRef<Path>, library: &Path) -> Result<Self> {
        let checkpoint = VisionCheckpoint::load(dir)?;
        let tensors = checkpoint
            .base_names()
            .map(|name| Ok((name.to_owned(), checkpoint.base_tensor(name)?)))
            .collect::<Result<Vec<_>>>()?;
        Self::from_tensors(checkpoint.config().clone(), tensors, Vec::new(), library)
    }
    /// Upload structurally validated BF16 base tensors and optional FP32 Cua LoRA.
    /// Prefixes match original checkpoint names; LoRA is supported only for the 4B layout.
    pub fn from_tensors(
        config: VisionConfig,
        base: Vec<(String, TensorView<'_>)>,
        lora: Vec<(String, TensorView<'_>)>,
        library: &Path,
    ) -> Result<Self> {
        let config = VisionConfig::from_value(serde_json::to_value(config)?)?;
        let expected = config.inventory();
        let mut names = std::collections::BTreeSet::new();
        for (name, tensor) in &base {
            ensure!(
                expected
                    .get(name)
                    .is_some_and(|shape| shape == tensor.shape())
                    && tensor.dtype() == Dtype::BF16
                    && names.insert(name),
                "invalid or duplicate base vision tensor {name}"
            );
        }
        ensure!(
            names.len() == expected.len(),
            "incomplete base vision tensors"
        );
        if !lora.is_empty() {
            ensure!(
                config.hidden_size == 1024,
                "vision LoRA only supported for Cua 4B"
            );
            let mut expected_lora = BTreeMap::new();
            for (name, shape) in &expected {
                if name.ends_with(".weight")
                    && (name.contains(".linear_fc1.") || name.contains(".linear_fc2."))
                {
                    let module = name
                        .strip_prefix("model.visual.")
                        .unwrap()
                        .strip_suffix(".weight")
                        .unwrap();
                    expected_lora.insert(
                        format!("base_model.model.model.visual.{module}.lora_A.weight"),
                        vec![16, shape[1]],
                    );
                    expected_lora.insert(
                        format!("base_model.model.model.visual.{module}.lora_B.weight"),
                        vec![shape[0], 16],
                    );
                }
            }
            let mut names = std::collections::BTreeSet::new();
            for (name, tensor) in &lora {
                ensure!(
                    expected_lora.get(name).is_some_and(|s| s == tensor.shape())
                        && tensor.dtype() == Dtype::F32
                        && names.insert(name),
                    "invalid or duplicate vision LoRA tensor {name}"
                );
            }
            ensure!(
                names.len() == expected_lora.len(),
                "incomplete vision LoRA tensors"
            );
        }
        cuda::load(library)?;
        // ABI6 is required; head-64 Cua layouts use the original vision entry points.
        if config.hidden_size != 1024 {
            api().vision_v2()?;
        }
        cuda::set_device(0)?;
        let stream = OwnedStream(cuda::new_stream()?);
        let gemm = Gemm(unsafe { (api().cs1_gemm_create)(32 << 20) });
        ensure!(!gemm.0.is_null(), "cannot create vision cuBLAS handle");
        let mut model = Self {
            config,
            scratch: VecDeque::new(),
            graph_enabled: std::env::var("CUA_S1_VISION_GRAPH").as_deref() == Ok("1"),
            base: BTreeMap::new(),
            lora: BTreeMap::new(),
            stream,
            gemm,
        };
        for (name, tensor) in base {
            model.base.insert(
                name.strip_prefix("model.visual.").unwrap().into(),
                model.upload(tensor.data())?,
            );
        }
        for (name, tensor) in lora {
            model.lora.insert(
                name.strip_prefix("base_model.model.model.visual.")
                    .unwrap()
                    .into(),
                model.upload(tensor.data())?,
            );
        }
        model.base.insert(
            "__zero_bias".into(),
            model.upload(&vec![0; model.config.hidden_size * 2])?,
        );
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
                    self.config.hidden_size as i32,
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
        let cached = self
            .scratch
            .iter()
            .position(|s| s.grid == image.image_grid_thw);
        let geometry = if cached.is_none() {
            Some(VisionGeometry::new(image.image_grid_thw, &self.config)?)
        } else {
            None
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
        let index = if let Some(index) = cached {
            index
        } else {
            self.synchronize()?;
            // FIFO is bounded to four exact grids; graph drops before its buffers.
            if self.scratch.len() == 4 {
                self.scratch.pop_front();
            }
            let scratch = Scratch::new(self, image.image_grid_thw, geometry.unwrap())?;
            self.scratch.push_back(scratch);
            self.scratch.len() - 1
        };
        let scratch = &self.scratch[index];
        // SAFETY: the current image has the validated exact resident buffer size.
        unsafe {
            cuda::upload(scratch.pixels.at(0), &pixels, self.stream.0)?;
        }
        if callback.is_none()
            && self.graph_enabled
            && let Some(graph) = &scratch.graph
        {
            graph.launch(self.stream.0)?;
            graph_trace("replayed", scratch.grid);
            return self.read(&scratch.out, n / 4 * self.config.out_hidden_size);
        }
        self.enqueue(scratch, &mut callback)?;
        let result = self.read(&scratch.out, n / 4 * self.config.out_hidden_size)?;
        if let Some(callback) = callback.as_mut() {
            callback("merger.output", &result)?;
        } else if self.graph_enabled {
            // Readback completed warmup. Capture records without executing; keep
            // the eager result even when recording/instantiation fails.
            let captured = cuda::Graph::capture(self.stream.0, || self.enqueue(scratch, &mut None));
            match captured {
                Ok(graph) => {
                    self.scratch[index].graph = Some(graph);
                    graph_trace("captured", image.image_grid_thw);
                }
                Err(error) => {
                    eprintln!("Vision CUDA Graph capture failed; using eager execution: {error:#}");
                    self.graph_enabled = false;
                    for scratch in &mut self.scratch {
                        scratch.graph = None;
                    }
                }
            }
        }
        Ok(result)
    }
    fn enqueue(&self, scratch: &Scratch, callback: &mut Trace<'_>) -> Result<()> {
        let n = scratch.grid[1] * scratch.grid[2];
        let h = self.config.hidden_size;
        let intermediate = self.config.intermediate_size;
        let merged = h * 4;
        let output = self.config.out_hidden_size;
        let Scratch {
            pixels,
            indices,
            weights,
            co,
            si,
            w,
            attention_workspace,
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
        self.linear("patch_embed.proj", pixels, x, n, h, 1536, w)?;
        self.trace(callback, "patch_embed", x, n * h)?;
        unsafe {
            if h == 1024 {
                check(
                    (api().cs1_vision_position)(
                        x.at(0),
                        self.weight("pos_embed.weight"),
                        indices.at(0).cast(),
                        weights.at(0).cast(),
                        n as i32,
                        self.stream.0,
                    ),
                    "vision positions",
                )?;
            } else {
                check(
                    (api().vision_v2()?.position)(
                        x.at(0),
                        self.weight("pos_embed.weight"),
                        indices.at(0).cast(),
                        weights.at(0).cast(),
                        n as i32,
                        h as i32,
                        self.stream.0,
                    ),
                    "vision positions",
                )?;
            }
        }
        self.trace(callback, "position", x, n * h)?;
        for i in 0..self.config.depth {
            let p = format!("blocks.{i}");
            self.norm(&format!("{p}.norm1"), x, norm, n)?;
            self.linear(&format!("{p}.attn.qkv"), norm, qkv, n, h * 3, h, w)?;
            unsafe {
                if h == 1024 {
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
                            qkv.at(h * 2 * 2),
                            attn.at(0),
                            n as i32,
                            self.stream.0,
                        ),
                        "vision attention",
                    )?;
                } else {
                    let v2 = api().vision_v2()?;
                    check(
                        (v2.rope)(
                            qkv.at(0),
                            co.at(0).cast(),
                            si.at(0).cast(),
                            q.at(0),
                            k.at(0),
                            n as i32,
                            h as i32,
                            self.config.head_dim() as i32,
                            self.stream.0,
                        ),
                        "vision rotary",
                    )?;
                    check(
                        (v2.attention)(
                            q.at(0),
                            k.at(0),
                            qkv.at(h * 2 * 2),
                            attn.at(0),
                            n as i32,
                            self.config.num_heads as i32,
                            self.config.head_dim() as i32,
                            attention_workspace.at(0),
                            self.stream.0,
                        ),
                        "vision attention",
                    )?;
                }
            }
            self.linear(&format!("{p}.attn.proj"), attn, delta, n, h, h, w)?;
            unsafe {
                check(
                    (api().cs1_vision_add)(x.at(0), delta.at(0), n * h, self.stream.0),
                    "vision attention residual",
                )?;
            }
            self.norm(&format!("{p}.norm2"), x, norm, n)?;
            self.linear(
                &format!("{p}.mlp.linear_fc1"),
                norm,
                mlp,
                n,
                intermediate,
                h,
                w,
            )?;
            unsafe {
                check(
                    (api().cs1_vision_gelu)(mlp.at(0), n * intermediate, 0, self.stream.0),
                    "vision tanh GELU",
                )?;
            }
            self.linear(
                &format!("{p}.mlp.linear_fc2"),
                mlp,
                delta,
                n,
                h,
                intermediate,
                w,
            )?;
            unsafe {
                check(
                    (api().cs1_vision_add)(x.at(0), delta.at(0), n * h, self.stream.0),
                    "vision MLP residual",
                )?;
            }
            self.trace(callback, &p, x, n * h)?;
        }
        self.norm("merger.norm", x, norm, n)?;
        self.trace(callback, "merger.norm", norm, n * h)?;
        // Consecutive groups of four patches already have the required 2x2 merge order.
        self.linear("merger.linear_fc1", norm, mlp, n / 4, merged, merged, w)?;
        self.trace(callback, "merger.linear_fc1", mlp, n * h)?;
        unsafe {
            check(
                (api().cs1_vision_gelu)(mlp.at(0), n * h, 1, self.stream.0),
                "vision exact GELU",
            )?;
        }
        self.linear("merger.linear_fc2", mlp, out, n / 4, output, merged, w)?;
        Ok(())
    }
}
impl Drop for VisionModel {
    fn drop(&mut self) {
        // Even failure paths retain every referenced address through synchronization.
        let _ = self.synchronize();
    }
}
fn graph_trace(action: &str, grid: [usize; 3]) {
    if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
        eprintln!("Vision CUDA Graph {action} grid={grid:?}");
    }
}
struct Scratch {
    graph: Option<cuda::Graph>,
    grid: [usize; 3],
    pixels: DeviceBuffer,
    indices: DeviceBuffer,
    weights: DeviceBuffer,
    co: DeviceBuffer,
    si: DeviceBuffer,
    w: Work,
    attention_workspace: DeviceBuffer,
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
    fn new(model: &VisionModel, grid: [usize; 3], geo: VisionGeometry) -> Result<Self> {
        let n = grid[1] * grid[2];
        let h = model.config.hidden_size;
        let width = model
            .config
            .intermediate_size
            .max(h * 4)
            .max(model.config.out_hidden_size);
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
            w: Work::new(if model.lora.is_empty() { 0 } else { n }, width)?,
            attention_workspace: DeviceBuffer::new(if h == 1024 {
                0
            } else {
                n * model.config.num_heads * 80 * 4 * 2
            })?,
            x: DeviceBuffer::new(n * h * 2)?,
            norm: DeviceBuffer::new(n * h * 2)?,
            qkv: DeviceBuffer::new(n * h * 3 * 2)?,
            q: DeviceBuffer::new(n * h * 2)?,
            k: DeviceBuffer::new(n * h * 2)?,
            attn: DeviceBuffer::new(n * h * 2)?,
            delta: DeviceBuffer::new(n * h * 2)?,
            mlp: DeviceBuffer::new(n * width * 2)?,
            out: DeviceBuffer::new(n / 4 * model.config.out_hidden_size * 2)?,
        })
    }
}

struct Work {
    float_input: DeviceBuffer,
    rank: DeviceBuffer,
    delta: DeviceBuffer,
}
impl Work {
    fn new(n: usize, width: usize) -> Result<Self> {
        Ok(Self {
            float_input: DeviceBuffer::new(n * width * 4)?,
            rank: DeviceBuffer::new(n * 16 * 4)?,
            delta: DeviceBuffer::new(n * width * 4)?,
        })
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/qwen3_5/vision_graph.rs"]
mod graph_tests;
