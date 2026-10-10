//! Geometry in the processor's 2×2 block-major patch order.
use anyhow::{Result, ensure};
pub(super) struct Geometry {
    pub indices: Vec<i32>,
    pub weights: Vec<f32>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}
impl Geometry {
    pub fn new([t, h, w]: [usize; 3]) -> Result<Self> {
        ensure!(
            t == 1 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0,
            "expected one image with an even, nonzero patch grid"
        );
        let n = h
            .checked_mul(w)
            .ok_or_else(|| anyhow::anyhow!("vision grid overflow"))?;
        // Processor bounds allow rounding up a 1,048,576-pixel source and narrow upscaled images.
        ensure!(
            n <= 4608 && h <= 512 && w <= 512,
            "vision grid exceeds processor bounds"
        );
        let mut g = Self {
            indices: Vec::with_capacity(n * 4),
            weights: Vec::with_capacity(n * 4),
            cos: Vec::with_capacity(n * 32),
            sin: Vec::with_capacity(n * 32),
        };
        for br in 0..h / 2 {
            for bc in 0..w / 2 {
                for ir in 0..2 {
                    for ic in 0..2 {
                        let row = br * 2 + ir;
                        let col = bc * 2 + ic;
                        let y = (row as f32 * 47.) / (h - 1) as f32;
                        let x = (col as f32 * 47.) / (w - 1) as f32;
                        let yl = y.floor() as usize;
                        let xl = x.floor() as usize;
                        let fy = y - yl as f32;
                        let fx = x - xl as f32;
                        for (yy, wy) in [(yl, 1. - fy), ((yl + 1).min(47), fy)] {
                            for (xx, wx) in [(xl, 1. - fx), ((xl + 1).min(47), fx)] {
                                g.indices.push((yy * 48 + xx) as i32);
                                g.weights.push(wy * wx);
                            }
                        }
                        for pos in [row, col] {
                            for i in 0..16 {
                                let angle = pos as f32 / 10000f32.powf(i as f32 / 16.);
                                g.cos.push(angle.cos());
                                g.sin.push(angle.sin());
                            }
                        }
                    }
                }
            }
        }
        Ok(g)
    }
}
