//! GPU vision forward for the structurally validated, unmerged checkpoint.
use super::{VisionCheckpoint, VisionConfig, geometry::VisionGeometry};
use crate::{
    cuda::{self, DeviceBf16Matrix, DeviceBuffer, Stream, api, check},
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
    // Uncached device output remains owned until its borrow is consumed.
    transient: Option<Scratch>,
    // None preserves the legacy four-grid FIFO policy for generic consumers.
    cache_limits: Option<CacheLimits>,
    graph_enabled: bool,
    base: BTreeMap<String, DeviceBuffer>,
    lora: BTreeMap<String, DeviceBuffer>,
    gemm: Gemm,
    stream: OwnedStream,
}
#[derive(Clone, Copy)]
enum ScratchLocation {
    Retained(usize),
    Transient,
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
        Self::build(config, base, lora, library, None)
    }
    /// Explicit retained-byte/count LRU policy; environment settings belong to callers.
    pub fn from_tensors_with_cache(
        config: VisionConfig,
        base: Vec<(String, TensorView<'_>)>,
        lora: Vec<(String, TensorView<'_>)>,
        library: &Path,
        cache_limits: CacheLimits,
    ) -> Result<Self> {
        Self::build(config, base, lora, library, Some(cache_limits))
    }
    fn build(
        config: VisionConfig,
        base: Vec<(String, TensorView<'_>)>,
        lora: Vec<(String, TensorView<'_>)>,
        library: &Path,
        cache_limits: Option<CacheLimits>,
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
        // ABI8 is required; head-64 Cua layouts use the original vision entry points.
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
            transient: None,
            cache_limits,
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
    /// Synchronized resident merger output; the borrow prevents cache mutation/retirement.
    pub fn forward_device(&mut self, image: &ProcessedImage) -> Result<DeviceBf16Matrix<'_>> {
        let result: Result<(usize, ScratchLocation)> = (|| {
            let (n, index, capture) = self.enqueue_image(image, &mut None)?;
            self.synchronize()?;
            if let (true, ScratchLocation::Retained(index)) = (capture, index) {
                self.capture_current(index);
            }
            Ok((n, index))
        })();
        if result.is_err() {
            // Drain while every referenced allocation and Graph is still owned.
            self.synchronize()?;
            self.release_transient()?;
        }
        let (n, index) = result?;
        self.scratch_at(index)
            .out
            .bf16_matrix(n / 4, self.config.out_hidden_size)
    }
    fn run(&mut self, image: &ProcessedImage, mut callback: Trace<'_>) -> Result<Vec<bf16>> {
        let result = (|| {
            let (n, index, capture) = self.enqueue_image(image, &mut callback)?;
            let output = self.read(
                &self.scratch_at(index).out,
                n / 4 * self.config.out_hidden_size,
            )?;
            if let Some(callback) = callback.as_mut() {
                callback("merger.output", &output)?;
            } else if let (true, ScratchLocation::Retained(index)) = (capture, index) {
                self.capture_current(index);
            }
            Ok(output)
        })();
        if result.is_err() {
            self.synchronize()?;
        }
        self.release_transient()?;
        result
    }
    /// Release uncached output after every consuming stream has completed.
    /// A live DeviceBf16Matrix borrow prevents calling this method.
    /// ```compile_fail
    /// use omni_qwen3_5_native::{vision::VisionModel, image_preprocess::ProcessedImage};
    /// fn retire_live_output(model: &mut VisionModel, image: &ProcessedImage) {
    ///     let output = model.forward_device(image).unwrap();
    ///     model.release_transient().unwrap();
    ///     let _ = output.download();
    /// }
    /// ```
    pub fn release_transient(&mut self) -> Result<()> {
        if self.transient.is_some() {
            self.synchronize()?;
            self.transient = None;
        }
        Ok(())
    }
    fn scratch_at(&self, location: ScratchLocation) -> &Scratch {
        match location {
            ScratchLocation::Retained(index) => &self.scratch[index],
            ScratchLocation::Transient => self.transient.as_ref().unwrap(),
        }
    }
    fn capture_current(&mut self, index: usize) {
        let scratch = &self.scratch[index];
        // The caller completes warmup before recording; capture never reruns math.
        let captured = cuda::Graph::capture(self.stream.0, || self.enqueue(scratch, &mut None));
        match captured {
            Ok(graph) => {
                let scratch = &mut self.scratch[index];
                scratch.graph = Some(graph);
                graph_trace("captured", scratch.grid);
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

    fn enqueue_image(
        &mut self,
        image: &ProcessedImage,
        callback: &mut Trace<'_>,
    ) -> Result<(usize, ScratchLocation, bool)> {
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
        self.release_transient()?;
        let bytes =
            Scratch::bytes_for_grid(&self.config, !self.lora.is_empty(), image.image_grid_thw);
        let index = if self
            .cache_limits
            .is_some_and(|limits| !limits.retains(bytes))
        {
            self.synchronize()?;
            let geometry = match geometry {
                Some(geometry) => geometry,
                None => VisionGeometry::new(image.image_grid_thw, &self.config)?,
            };
            self.transient = Some(Scratch::new(self, image.image_grid_thw, geometry)?);
            self.cache_trace("transient", image.image_grid_thw);
            ScratchLocation::Transient
        } else {
            let index = if let Some(index) = cached {
                if self.cache_limits.is_some() {
                    // Restore the exact grid as MRU, keeping inactive entries oldest first.
                    let scratch = self.scratch.remove(index).unwrap();
                    self.scratch.push_back(scratch);
                    self.cache_trace("hit", image.image_grid_thw);
                    self.scratch.len() - 1
                } else {
                    index
                }
            } else {
                self.synchronize()?;
                if let Some(limits) = self.cache_limits {
                    while limits.needs_eviction(self.scratch.len(), self.retained_bytes(), bytes) {
                        let retired = self.scratch.pop_front().unwrap();
                        let grid = retired.grid;
                        drop(retired); // Graph field retires before every captured buffer.
                        self.cache_trace("evicted", grid);
                    }
                } else if self.scratch.len() == 4 {
                    self.scratch.pop_front();
                }
                let scratch = Scratch::new(self, image.image_grid_thw, geometry.unwrap())?;
                self.scratch.push_back(scratch);
                self.cache_trace("miss", image.image_grid_thw);
                self.scratch.len() - 1
            };
            ScratchLocation::Retained(index)
        };
        let scratch = self.scratch_at(index);
        // SAFETY: the current image has the validated exact resident buffer size.
        unsafe {
            cuda::upload(scratch.pixels.at(0), &pixels, self.stream.0)?;
        }
        if callback.is_none()
            && matches!(index, ScratchLocation::Retained(_))
            && self.graph_enabled
            && let Some(graph) = &scratch.graph
        {
            graph.launch(self.stream.0)?;
            graph_trace("replayed", scratch.grid);
            return Ok((n, index, false));
        }
        self.enqueue(scratch, callback)?;
        Ok((
            n,
            index,
            callback.is_none()
                && self.graph_enabled
                && matches!(index, ScratchLocation::Retained(_)),
        ))
    }
    fn retained_bytes(&self) -> usize {
        self.scratch
            .iter()
            .map(|s| Scratch::bytes_for_grid(&self.config, !self.lora.is_empty(), s.grid))
            .sum()
    }
    fn cache_trace(&self, action: &str, grid: [usize; 3]) {
        if let Some(limits) = self.cache_limits
            && std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1")
        {
            eprintln!(
                "Vision cache {action} grid={grid:?} retained_entries={} retained_bytes={} entry_limit={} byte_limit={}",
                self.scratch.len(),
                self.retained_bytes(),
                limits.entries,
                limits.bytes
            );
        }
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
/// Budgets cover every retained Scratch allocation, excluding shared weights/GEMM workspace.
#[derive(Clone, Copy)]
pub struct CacheLimits {
    pub entries: usize,
    pub bytes: usize,
}
impl CacheLimits {
    fn retains(self, bytes: usize) -> bool {
        self.entries > 0 && bytes <= self.bytes
    }
    fn needs_eviction(self, entries: usize, retained_bytes: usize, incoming_bytes: usize) -> bool {
        entries >= self.entries || incoming_bytes > self.bytes.saturating_sub(retained_bytes)
    }
}

impl Scratch {
    fn layout(config: &VisionConfig, lora: bool, grid: [usize; 3]) -> [usize; 18] {
        let n = grid[1] * grid[2];
        let h = config.hidden_size;
        let width = config
            .intermediate_size
            .max(h * 4)
            .max(config.out_hidden_size);
        let work_rows = if lora { n } else { 0 };
        [
            n * 1536 * 2,
            n * 4 * 4,
            n * 4 * 4,
            n * (config.head_dim() / 2) * 4,
            n * (config.head_dim() / 2) * 4,
            work_rows * width * 4,
            work_rows * 16 * 4,
            work_rows * width * 4,
            if h == 1024 {
                0
            } else {
                n * config.num_heads * 80 * 4 * 2
            },
            n * h * 2,
            n * h * 2,
            n * h * 3 * 2,
            n * h * 2,
            n * h * 2,
            n * h * 2,
            n * h * 2,
            n * width * 2,
            n / 4 * config.out_hidden_size * 2,
        ]
    }
    fn bytes_for_grid(config: &VisionConfig, lora: bool, grid: [usize; 3]) -> usize {
        Self::layout(config, lora, grid).iter().sum()
    }
    fn new(model: &VisionModel, grid: [usize; 3], geo: VisionGeometry) -> Result<Self> {
        let sizes = Self::layout(&model.config, !model.lora.is_empty(), grid);
        ensure!(
            geo.indices.len() * 4 == sizes[1]
                && geo.weights.len() * 4 == sizes[2]
                && geo.cos.len() * 4 == sizes[3]
                && geo.sin.len() * 4 == sizes[4],
            "vision geometry storage does not match layout"
        );
        Ok(Self {
            graph: None,
            grid,
            pixels: DeviceBuffer::new(sizes[0])?,
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
            w: Work::new([sizes[5], sizes[6], sizes[7]])?,
            attention_workspace: DeviceBuffer::new(sizes[8])?,
            x: DeviceBuffer::new(sizes[9])?,
            norm: DeviceBuffer::new(sizes[10])?,
            qkv: DeviceBuffer::new(sizes[11])?,
            q: DeviceBuffer::new(sizes[12])?,
            k: DeviceBuffer::new(sizes[13])?,
            attn: DeviceBuffer::new(sizes[14])?,
            delta: DeviceBuffer::new(sizes[15])?,
            mlp: DeviceBuffer::new(sizes[16])?,
            out: DeviceBuffer::new(sizes[17])?,
        })
    }
}

struct Work {
    float_input: DeviceBuffer,
    rank: DeviceBuffer,
    delta: DeviceBuffer,
}
impl Work {
    fn new(sizes: [usize; 3]) -> Result<Self> {
        Ok(Self {
            float_input: DeviceBuffer::new(sizes[0])?,
            rank: DeviceBuffer::new(sizes[1])?,
            delta: DeviceBuffer::new(sizes[2])?,
        })
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/qwen3_5/vision_graph.rs"]
mod vision_tests;
