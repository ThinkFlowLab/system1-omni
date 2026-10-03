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
    base: BTreeMap<String, DeviceBuffer>,
    lora: BTreeMap<String, DeviceBuffer>,
    stream: OwnedStream,
    gemm: Gemm,
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
        let geo = Geometry::new(image.image_grid_thw)?;
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
        let pixels = self.upload(&pixels)?;
        let indices = self.upload(
            &geo.indices
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let weights = self.upload(
            &geo.weights
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let co = self.upload(
            &geo.cos
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let si = self.upload(
            &geo.sin
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let w = Work::new(n)?;
        let x = DeviceBuffer::new(n * 1024 * 2)?;
        let norm = DeviceBuffer::new(n * 1024 * 2)?;
        let qkv = DeviceBuffer::new(n * 3072 * 2)?;
        let q = DeviceBuffer::new(n * 1024 * 2)?;
        let k = DeviceBuffer::new(n * 1024 * 2)?;
        let attn = DeviceBuffer::new(n * 1024 * 2)?;
        let delta = DeviceBuffer::new(n * 1024 * 2)?;
        let mlp = DeviceBuffer::new(n * 4096 * 2)?;
        self.linear("patch_embed.proj", &pixels, &x, n, 1024, 1536, &w)?;
        self.trace(&mut callback, "patch_embed", &x, n * 1024)?;
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
        self.trace(&mut callback, "position", &x, n * 1024)?;
        for i in 0..24 {
            let p = format!("blocks.{i}");
            self.norm(&format!("{p}.norm1"), &x, &norm, n)?;
            self.linear(&format!("{p}.attn.qkv"), &norm, &qkv, n, 3072, 1024, &w)?;
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
            self.linear(&format!("{p}.attn.proj"), &attn, &delta, n, 1024, 1024, &w)?;
            unsafe {
                check(
                    (api().cs1_vision_add)(x.at(0), delta.at(0), n * 1024, self.stream.0),
                    "vision attention residual",
                )?;
            }
            self.norm(&format!("{p}.norm2"), &x, &norm, n)?;
            self.linear(
                &format!("{p}.mlp.linear_fc1"),
                &norm,
                &mlp,
                n,
                4096,
                1024,
                &w,
            )?;
            unsafe {
                check(
                    (api().cs1_vision_gelu)(mlp.at(0), n * 4096, 0, self.stream.0),
                    "vision tanh GELU",
                )?;
            }
            self.linear(
                &format!("{p}.mlp.linear_fc2"),
                &mlp,
                &delta,
                n,
                1024,
                4096,
                &w,
            )?;
            unsafe {
                check(
                    (api().cs1_vision_add)(x.at(0), delta.at(0), n * 1024, self.stream.0),
                    "vision MLP residual",
                )?;
            }
            self.trace(&mut callback, &p, &x, n * 1024)?;
        }
        self.norm("merger.norm", &x, &norm, n)?;
        self.trace(&mut callback, "merger.norm", &norm, n * 1024)?;
        // Consecutive groups of four patches already have the required 2x2 merge order.
        self.linear("merger.linear_fc1", &norm, &mlp, n / 4, 4096, 4096, &w)?;
        self.trace(&mut callback, "merger.linear_fc1", &mlp, n * 1024)?;
        unsafe {
            check(
                (api().cs1_vision_gelu)(mlp.at(0), n * 1024, 1, self.stream.0),
                "vision exact GELU",
            )?;
        }
        let out = DeviceBuffer::new(n / 4 * 2560 * 2)?;
        self.linear("merger.linear_fc2", &mlp, &out, n / 4, 2560, 4096, &w)?;
        let result = self.read(&out, n / 4 * 2560)?;
        if let Some(callback) = callback.as_mut() {
            callback("merger.output", &result)?;
        }
        Ok(result)
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
