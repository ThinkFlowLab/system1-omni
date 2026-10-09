//! CPU preprocessing for the fixed Qwen3.5-4B / Cua-S1 image processor.
//!
//! Input is already decoded, interleaved RGB8. No image codecs, GPU, model
//! weights, or HTTP handling are involved. The output is row-major
//! `[patches, 1536]`, ready for a separate vision encoder.
//!
//! The resize algorithm is a Rust adaptation of PyTorch's CPU uint8 bicubic
//! antialias implementation (which credits Pillow). Smart resize and packing
//! follow Transformers' Qwen2VLImageProcessor. See `../THIRD_PARTY_NOTICES.md`.

use std::borrow::Cow;

use anyhow::{Result, ensure};

const PATCH_SIZE: usize = 16;
const MERGE_SIZE: usize = 2;
const FACTOR: usize = PATCH_SIZE * MERGE_SIZE;
const PATCH_VALUES: usize = 3 * 2 * PATCH_SIZE * PATCH_SIZE;
const MIN_PIXELS: usize = 65_536;
const MAX_PIXELS: usize = 16_777_216;

/// Normalized image patches and the spatial metadata used by the vision model.
#[derive(Debug)]
pub struct ProcessedImage {
    /// Contiguous row-major `[patches, 1536]` float32 values.
    pub pixel_values: Vec<f32>,
    /// `[1, resized_height / 16, resized_width / 16]`.
    pub image_grid_thw: [usize; 3],
    pub resized_width: usize,
    pub resized_height: usize,
}

impl ProcessedImage {
    /// Number of image tokens after the model's 2-by-2 spatial merge.
    pub fn image_tokens(&self) -> usize {
        self.pixel_values.len() / PATCH_VALUES / (MERGE_SIZE * MERGE_SIZE)
    }
}

/// Preprocess a decoded RGB8 image using the fixed 4B processor settings.
///
/// Rejects empty dimensions, sides over 2048, area over 1,048,576 pixels,
/// aspect ratios over 200, and buffers whose length is not `width * height * 3`.
/// Geometry and buffer arithmetic are checked before allocating image buffers.
pub fn preprocess_rgb8(width: usize, height: usize, rgb: &[u8]) -> Result<ProcessedImage> {
    ensure!(width > 0 && height > 0, "image dimensions must be nonzero");
    ensure!(
        width <= 2048 && height <= 2048,
        "image sides must not exceed 2048"
    );
    let area = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("image area overflow"))?;
    ensure!(
        area <= 1_048_576,
        "image area must not exceed 1048576 pixels"
    );
    ensure!(
        width.max(height) <= width.min(height) * 200,
        "image aspect ratio must not exceed 200"
    );
    let input_len = area
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("RGB buffer length overflow"))?;
    ensure!(
        rgb.len() == input_len,
        "RGB buffer length must be {input_len}, got {}",
        rgb.len()
    );

    let (resized_width, resized_height) = smart_resize(width, height);
    let resized_area = resized_width
        .checked_mul(resized_height)
        .ok_or_else(|| anyhow::anyhow!("resized area overflow"))?;
    let resized_len = resized_area
        .checked_mul(3)
        .ok_or_else(|| anyhow::anyhow!("resized buffer length overflow"))?;
    let horizontal_len = resized_width
        .checked_mul(height)
        .and_then(|area| area.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("horizontal buffer length overflow"))?;
    let output_len = resized_area
        .checked_mul(6)
        .ok_or_else(|| anyhow::anyhow!("patch buffer length overflow"))?;
    output_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("patch buffer byte length overflow"))?;

    let mut resized = Cow::Borrowed(rgb);
    if resized_width != width {
        let axis = AxisWeights::new(width, resized_width);
        let mut horizontal = vec![0; horizontal_len];
        for y in 0..height {
            for (x, kernel) in axis.kernels.iter().enumerate() {
                for channel in 0..3 {
                    horizontal[(y * resized_width + x) * 3 + channel] =
                        axis.apply(kernel, |source_x| rgb[(y * width + source_x) * 3 + channel]);
                }
            }
        }
        resized = Cow::Owned(horizontal);
    }
    if resized_height != height {
        let axis = AxisWeights::new(height, resized_height);
        let mut vertical = vec![0; resized_len];
        for (y, kernel) in axis.kernels.iter().enumerate() {
            for x in 0..resized_width {
                for channel in 0..3 {
                    vertical[(y * resized_width + x) * 3 + channel] = axis
                        .apply(kernel, |source_y| {
                            resized[(source_y * resized_width + x) * 3 + channel]
                        });
                }
            }
        }
        resized = Cow::Owned(vertical);
    }

    let mut pixel_values = Vec::with_capacity(output_len);
    for block_y in 0..resized_height / FACTOR {
        for block_x in 0..resized_width / FACTOR {
            for merge_y in 0..MERGE_SIZE {
                for merge_x in 0..MERGE_SIZE {
                    for channel in 0..3 {
                        for _temporal in 0..2 {
                            for patch_y in 0..PATCH_SIZE {
                                for patch_x in 0..PATCH_SIZE {
                                    let y = block_y * FACTOR + merge_y * PATCH_SIZE + patch_y;
                                    let x = block_x * FACTOR + merge_x * PATCH_SIZE + patch_x;
                                    let pixel = resized[(y * resized_width + x) * 3 + channel];
                                    // Match the fused float32 torchvision normalization,
                                    // including its operation order (no reciprocal multiply).
                                    pixel_values.push((f32::from(pixel) - 127.5) / 127.5);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(ProcessedImage {
        pixel_values,
        image_grid_thw: [1, resized_height / PATCH_SIZE, resized_width / PATCH_SIZE],
        resized_width,
        resized_height,
    })
}

fn smart_resize(width: usize, height: usize) -> (usize, usize) {
    // Python round uses ties-to-even; Rust's ordinary round does not.
    let mut w = (width as f64 / FACTOR as f64).round_ties_even() as usize * FACTOR;
    let mut h = (height as f64 / FACTOR as f64).round_ties_even() as usize * FACTOR;
    if w * h > MAX_PIXELS {
        let beta = ((width * height) as f64 / MAX_PIXELS as f64).sqrt();
        w = ((width as f64 / beta / FACTOR as f64).floor() as usize * FACTOR).max(FACTOR);
        h = ((height as f64 / beta / FACTOR as f64).floor() as usize * FACTOR).max(FACTOR);
    } else if w * h < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (width * height) as f64).sqrt();
        w = (width as f64 * beta / FACTOR as f64).ceil() as usize * FACTOR;
        h = (height as f64 * beta / FACTOR as f64).ceil() as usize * FACTOR;
    }
    (w, h)
}

struct Kernel {
    start: usize,
    weights: Vec<i16>,
}

struct AxisWeights {
    kernels: Vec<Kernel>,
    precision: u32,
}

impl AxisWeights {
    fn new(input: usize, output: usize) -> Self {
        let scale = input as f64 / output as f64;
        let support = 2.0 * scale.max(1.0);
        let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let max_size = support.ceil() as usize * 2 + 1;
        let mut maximum = 0.0_f64;
        let mut floating = Vec::with_capacity(output);
        for index in 0..output {
            let center = scale * (index as f64 + 0.5);
            // C++ conversion truncates toward zero before clamping the bounds.
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let count = end.saturating_sub(start).min(max_size);
            let mut weights: Vec<f64> = (0..count)
                .map(|j| cubic((j as f64 + start as f64 - center + 0.5) * invscale))
                .collect();
            let total: f64 = weights.iter().sum();
            if total != 0.0 {
                for weight in &mut weights {
                    *weight /= total;
                    maximum = maximum.max(*weight);
                }
            }
            floating.push((start, weights));
        }
        // One precision for the whole axis, as in PyTorch's int16 path.
        let mut precision = 0;
        while precision < 22 {
            if (0.5 + maximum * f64::from(1 << (precision + 1))) as i32 >= (1 << 15) {
                break;
            }
            precision += 1;
        }
        let multiplier = f64::from(1 << precision);
        let kernels = floating
            .into_iter()
            .map(|(start, weights)| Kernel {
                start,
                weights: weights
                    .into_iter()
                    .map(|weight| {
                        let value = weight * multiplier;
                        (value + if value < 0.0 { -0.5 } else { 0.5 }) as i16
                    })
                    .collect(),
            })
            .collect();
        Self { kernels, precision }
    }

    fn apply(&self, kernel: &Kernel, pixel: impl Fn(usize) -> u8) -> u8 {
        let mut accumulator = 1_i32 << (self.precision - 1);
        for (offset, &weight) in kernel.weights.iter().enumerate() {
            accumulator += i32::from(pixel(kernel.start + offset)) * i32::from(weight);
        }
        (accumulator >> self.precision).clamp(0, 255) as u8
    }
}

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    const A: f64 = -0.5;
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}
