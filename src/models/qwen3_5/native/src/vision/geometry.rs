//! Geometry in the processor's 2×2 block-major patch order.
use super::VisionConfig;
use anyhow::{Result, ensure};
pub struct VisionGeometry {
    pub indices: Vec<i32>,
    pub weights: Vec<f32>,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}
impl VisionGeometry {
    fn patch_count([t, h, w]: [usize; 3], config: &VisionConfig) -> Result<usize> {
        config.validate()?;
        ensure!(
            t == 1 && h > 0 && w > 0 && h % 2 == 0 && w % 2 == 0,
            "expected one image with an even, nonzero patch grid"
        );
        let n = h
            .checked_mul(w)
            .ok_or_else(|| anyhow::anyhow!("vision grid overflow"))?;
        ensure!(
            if config.hidden_size == 1024 {
                n <= 4608 && h <= 512 && w <= 512
            } else {
                n <= 65536 && h <= 16384 && w <= 16384
            },
            "vision grid exceeds processor bounds"
        );
        Ok(n)
    }
    pub(crate) fn use_reference_angles(&mut self, grids: &[[usize; 3]], config: &VisionConfig) {
        self.cos.clear();
        let quarter = config.head_dim() / 4;
        for &[_, h, w] in grids {
            for br in 0..h / 2 {
                for bc in 0..w / 2 {
                    for ir in 0..2 {
                        for ic in 0..2 {
                            for pos in [br * 2 + ir, bc * 2 + ic] {
                                for i in 0..quarter {
                                    let inv = 1.0f32 / 10000f32.powf(i as f32 / quarter as f32);
                                    self.cos.push(pos as f32 * inv);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    pub fn new([t, h, w]: [usize; 3], config: &VisionConfig) -> Result<Self> {
        let n = Self::patch_count([t, h, w], config)?;
        let mut g = Self {
            indices: Vec::with_capacity(n * 4),
            weights: Vec::with_capacity(n * 4),
            cos: Vec::with_capacity(n * (config.head_dim() / 2)),
            sin: Vec::with_capacity(n * (config.head_dim() / 2)),
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
                            for i in 0..config.head_dim() / 4 {
                                let angle = pos as f32
                                    / 10000f32.powf(i as f32 / (config.head_dim() / 4) as f32);
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

/// Ordered image-local geometry concatenated for all-image projection row counts.
pub(crate) struct BatchGeometry {
    pub geometry: VisionGeometry,
}
impl BatchGeometry {
    pub fn lengths(grids: &[[usize; 3]], config: &VisionConfig) -> Result<Vec<usize>> {
        ensure!(
            config.hidden_size == 1152 && (2..=4).contains(&grids.len()),
            "batched vision requires 2..=4 unadapted 27B images"
        );
        let lengths = grids
            .iter()
            .map(|&grid| VisionGeometry::patch_count(grid, config))
            .collect::<Result<Vec<_>>>()?;
        let total = lengths.iter().try_fold(0usize, |n, &rows| {
            n.checked_add(rows)
                .ok_or_else(|| anyhow::anyhow!("vision batch row overflow"))
        })?;
        // Keep the aggregate allocation within the existing shared geometry bound.
        ensure!(total <= 65536, "vision batch exceeds 65536 patches");
        Ok(lengths)
    }
    pub fn new(grids: &[[usize; 3]], config: &VisionConfig) -> Result<Self> {
        let lengths = Self::lengths(grids, config)?;
        let total: usize = lengths.iter().sum();
        let mut geometry = VisionGeometry {
            indices: Vec::with_capacity(total * 4),
            weights: Vec::with_capacity(total * 4),
            cos: Vec::with_capacity(total * (config.head_dim() / 2)),
            sin: Vec::with_capacity(total * (config.head_dim() / 2)),
        };
        for &grid in grids {
            let image = VisionGeometry::new(grid, config)?;
            geometry.indices.extend(image.indices);
            geometry.weights.extend(image.weights);
            geometry.cos.extend(image.cos);
            geometry.sin.extend(image.sin);
        }
        Ok(Self { geometry })
    }
}
