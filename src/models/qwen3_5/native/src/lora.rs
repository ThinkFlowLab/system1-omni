//! Separate, immutable FP32 language LoRA weights and per-projection execution.
use crate::{
    cuda::{self, DeviceBuffer, Stream, check},
    model::Config,
};
use anyhow::{Context, Result, ensure};
use std::{collections::BTreeMap, ffi::c_void, ops::Range, path::Path};

const PREFIX: &str = "base_model.model.model.language_model.";
const MAX_RANK: usize = 128;
const MAX_HEADER: usize = 8 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Group {
    LinearInput,
    LinearOutput,
    AttentionInput,
    AttentionOutput,
    GateUp,
    Down,
}
impl Group {
    fn index(self) -> usize {
        self as usize
    }
}
#[derive(Clone, Debug)]
struct Target {
    name: String,
    layer: usize,
    group: Group,
    rows: usize,
    cols: usize,
    offset: usize,
    stride: usize,
}
struct Pair {
    target: Target,
    rank: usize,
    a: Range<usize>,
    b: Range<usize>,
}
type Components = [Option<(Range<usize>, usize)>; 2];

struct Parsed {
    pairs: Vec<Pair>,
    base_width: usize,
}

fn targets(cfg: &Config) -> Result<Vec<Target>> {
    let mul = |a: usize, b: usize| a.checked_mul(b).context("LoRA dimension overflow");
    let add = |a: usize, b: usize| a.checked_add(b).context("LoRA dimension overflow");
    ensure!(
        !cfg.full_attention.is_empty() && cfg.full_attention.len() <= 1024,
        "LoRA layer count"
    );
    let (h, i) = (cfg.hidden, cfg.intermediate);
    let (kd, vd) = (
        mul(cfg.lin_k_heads, cfg.lin_k_dim)?,
        mul(cfg.lin_v_heads, cfg.lin_v_dim)?,
    );
    let q = mul(mul(cfg.heads, cfg.head_dim)?, 2)?;
    let kv = mul(cfg.kv_heads, cfg.head_dim)?;
    let conv = add(mul(2, kd)?, vd)?;
    let lin_stride = add(add(conv, vd)?, mul(2, cfg.lin_v_heads)?)?;
    let attn_stride = add(q, mul(2, kv)?)?;
    let mut targets = Vec::new();
    for (layer, &full) in cfg.full_attention.iter().enumerate() {
        let mut push = |name: &str,
                        group: Group,
                        rows: usize,
                        cols: usize,
                        offset: usize,
                        stride: usize|
         -> Result<()> {
            ensure!(
                rows > 0 && cols > 0 && stride <= i32::MAX as usize && cols <= i32::MAX as usize,
                "LoRA dimensions"
            );
            ensure!(
                offset.checked_add(rows).is_some_and(|end| end <= stride),
                "LoRA output slice"
            );
            targets.push(Target {
                name: format!("layers.{layer}.{name}"),
                layer,
                group,
                rows,
                cols,
                offset,
                stride,
            });
            Ok(())
        };
        if full {
            push(
                "self_attn.q_proj",
                Group::AttentionInput,
                q,
                h,
                0,
                attn_stride,
            )?;
            push(
                "self_attn.k_proj",
                Group::AttentionInput,
                kv,
                h,
                q,
                attn_stride,
            )?;
            push(
                "self_attn.v_proj",
                Group::AttentionInput,
                kv,
                h,
                add(q, kv)?,
                attn_stride,
            )?;
            push("self_attn.o_proj", Group::AttentionOutput, h, q / 2, 0, h)?;
        } else {
            push(
                "linear_attn.in_proj_qkv",
                Group::LinearInput,
                conv,
                h,
                0,
                lin_stride,
            )?;
            push(
                "linear_attn.in_proj_z",
                Group::LinearInput,
                vd,
                h,
                conv,
                lin_stride,
            )?;
            push(
                "linear_attn.in_proj_b",
                Group::LinearInput,
                cfg.lin_v_heads,
                h,
                add(conv, vd)?,
                lin_stride,
            )?;
            push(
                "linear_attn.in_proj_a",
                Group::LinearInput,
                cfg.lin_v_heads,
                h,
                add(add(conv, vd)?, cfg.lin_v_heads)?,
                lin_stride,
            )?;
            push("linear_attn.out_proj", Group::LinearOutput, h, vd, 0, h)?;
        }
        push("mlp.gate_proj", Group::GateUp, i, h, 0, mul(2, i)?)?;
        push("mlp.up_proj", Group::GateUp, i, h, i, mul(2, i)?)?;
        push("mlp.down_proj", Group::Down, h, i, 0, h)?;
    }
    Ok(targets)
}

fn parse(cfg: &Config, bytes: &[u8], scale: f32) -> Result<Parsed> {
    ensure!(
        scale.is_finite() && scale > 0.,
        "LoRA scale must be finite and positive"
    );
    let allowed: BTreeMap<_, _> = targets(cfg)?
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect();
    let header_len = usize::try_from(u64::from_le_bytes(
        bytes.get(..8).context("LoRA header length")?.try_into()?,
    ))?;
    ensure!(header_len <= MAX_HEADER, "LoRA header too large");
    let start = header_len.checked_add(8).context("LoRA header overflow")?;
    let header = crate::json::parse(bytes.get(8..start).context("truncated LoRA header")?)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        header.len() <= allowed.len() * 2 + 1,
        "too many LoRA tensors"
    );
    let mut components: BTreeMap<String, Components> = BTreeMap::new();
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        let native = name
            .strip_prefix(PREFIX)
            .context("unsupported LoRA tensor prefix")?;
        let (target, letter) = if let Some(target) = native.strip_suffix(".lora_A.weight") {
            (target, 0)
        } else {
            (
                native
                    .strip_suffix(".lora_B.weight")
                    .context("unsupported LoRA tensor suffix")?,
                1,
            )
        };
        let layout = allowed
            .get(target)
            .with_context(|| format!("unsupported LoRA target {target}"))?;
        ensure!(info["dtype"] == "F32", "LoRA tensor {name} must be FP32");
        let shape: [usize; 2] = serde_json::from_value(info["shape"].clone())
            .context("LoRA tensor must be a matrix")?;
        let rank = if letter == 0 { shape[0] } else { shape[1] };
        ensure!(
            (1..=MAX_RANK).contains(&rank),
            "LoRA rank must be 1..={MAX_RANK}"
        );
        ensure!(
            shape
                == if letter == 0 {
                    [rank, layout.cols]
                } else {
                    [layout.rows, rank]
                },
            "LoRA shape mismatch for {name}"
        );
        let [begin, end]: [usize; 2] =
            serde_json::from_value(info["data_offsets"].clone()).context("LoRA tensor offsets")?;
        let size = shape[0]
            .checked_mul(shape[1])
            .and_then(|n| n.checked_mul(4))
            .context("LoRA tensor size overflow")?;
        ensure!(
            end.checked_sub(begin) == Some(size),
            "LoRA tensor byte count"
        );
        let range = start.checked_add(begin).context("LoRA offset overflow")?
            ..start.checked_add(end).context("LoRA offset overflow")?;
        let data = bytes
            .get(range.clone())
            .context("LoRA tensor exceeds source")?;
        ensure!(
            data.as_chunks::<4>()
                .0
                .iter()
                .all(|&v| f32::from_le_bytes(v).is_finite()),
            "nonfinite LoRA tensor {name}"
        );
        let halves = components.entry(target.into()).or_default();
        ensure!(
            halves[letter].replace((range, rank)).is_none(),
            "duplicate LoRA component"
        );
    }
    ensure!(!components.is_empty(), "empty LoRA adapter");
    // All sizes/offsets are bounded above before safetensors performs its payload checks.
    safetensors::SafeTensors::deserialize(bytes).context("invalid LoRA safetensors")?;
    let pairs = components
        .into_iter()
        .map(|(name, [a, b])| {
            let (a, rank) = a.with_context(|| format!("unpaired LoRA target {name}: missing A"))?;
            let (b, b_rank) =
                b.with_context(|| format!("unpaired LoRA target {name}: missing B"))?;
            ensure!(rank == b_rank, "LoRA A/B rank mismatch for {name}");
            Ok(Pair {
                target: allowed[&name].clone(),
                rank,
                a,
                b,
            })
        })
        .collect::<Result<_>>()?;
    Ok(Parsed {
        pairs,
        base_width: allowed.values().map(|t| t.rows).max().unwrap(),
    })
}

pub(crate) struct Checkpoint {
    map: memmap2::Mmap,
    parsed: Parsed,
    scale: f32,
}
impl Checkpoint {
    pub(crate) fn load(cfg: &Config, path: &Path, scale: f32) -> Result<Self> {
        ensure!(
            scale.is_finite() && scale > 0.,
            "LoRA scale must be finite and positive"
        );
        let file = std::fs::File::open(path)
            .with_context(|| format!("LoRA adapter {}", path.display()))?;
        let max_bytes = targets(cfg)?.iter().try_fold(MAX_HEADER + 8, |n, t| {
            t.rows
                .checked_add(t.cols)
                .and_then(|v| v.checked_mul(MAX_RANK * 4))
                .and_then(|v| n.checked_add(v))
                .context("LoRA maximum size overflow")
        })?;
        ensure!(
            file.metadata()?.len() <= u64::try_from(max_bytes)?,
            "LoRA source exceeds configured tensor bound"
        );
        // SAFETY: the source checkpoint must remain immutable while validation/upload runs.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        let parsed = parse(cfg, &map, scale)?;
        Ok(Self { map, parsed, scale })
    }
    pub(crate) fn upload(self, layers: usize, stream: Stream) -> Result<Lora> {
        let mut layout = WorkLayout::for_pairs(1, &self.parsed.pairs)?.unwrap();
        // Reuse the BF16 gather for contiguous base projection outputs. All configured
        // components must fit even when only a small subset has an adapter.
        layout.gather = self
            .parsed
            .base_width
            .checked_mul(2)
            .context("base projection work overflow")?;
        let mut result = Lora {
            layers: (0..layers)
                .map(|_| std::array::from_fn(|_| Vec::new()))
                .collect(),
            scale: self.scale,
            layout,
        };
        for pair in self.parsed.pairs {
            let a = DeviceBuffer::new(pair.a.len())?;
            let b = DeviceBuffer::new(pair.b.len())?;
            // SAFETY: source and device buffers have the validated exact FP32 byte counts.
            unsafe {
                cuda::upload(a.at(0), &self.map[pair.a], stream)?;
                cuda::upload(b.at(0), &self.map[pair.b], stream)?;
            }
            result.layers[pair.target.layer][pair.target.group.index()].push(GpuPair {
                target: pair.target,
                rank: pair.rank,
                a,
                b,
            });
        }
        Ok(result)
    }
}

#[derive(Clone, Copy)]
struct WorkLayout {
    input: usize,
    rank: usize,
    delta: usize,
    gather: usize,
}
impl WorkLayout {
    fn for_pairs(rows: usize, pairs: &[Pair]) -> Result<Option<Self>> {
        if pairs.is_empty() {
            return Ok(None);
        }
        let bytes = |width: usize, element: usize| {
            rows.checked_mul(width)
                .and_then(|n| n.checked_mul(element))
                .context("LoRA work size overflow")
        };
        let cols = pairs.iter().map(|p| p.target.cols).max().unwrap();
        let outputs = pairs.iter().map(|p| p.target.rows).max().unwrap();
        let rank = pairs.iter().map(|p| p.rank).max().unwrap();
        Ok(Some(Self {
            input: bytes(cols, 4)?,
            rank: bytes(rank, 4)?,
            delta: bytes(outputs, 4)?,
            gather: bytes(outputs, 2)?,
        }))
    }
    fn scaled(self, rows: usize) -> Result<Self> {
        let scale = |n: usize| n.checked_mul(rows).context("LoRA work size overflow");
        Ok(Self {
            input: scale(self.input)?,
            rank: scale(self.rank)?,
            delta: scale(self.delta)?,
            gather: scale(self.gather)?,
        })
    }
}
struct GpuPair {
    target: Target,
    rank: usize,
    a: DeviceBuffer,
    b: DeviceBuffer,
}
pub(crate) struct Lora {
    layers: Vec<[Vec<GpuPair>; 6]>,
    scale: f32,
    layout: WorkLayout,
}
pub(crate) struct Work {
    input: DeviceBuffer,
    rank: DeviceBuffer,
    delta: DeviceBuffer,
    gather: DeviceBuffer,
    rows: usize,
    gather_width: usize,
}
impl Lora {
    pub(crate) fn work(&self, rows: usize) -> Result<Work> {
        let layout = self.layout.scaled(rows)?;
        Ok(Work {
            input: DeviceBuffer::new(layout.input)?,
            rank: DeviceBuffer::new(layout.rank)?,
            delta: DeviceBuffer::new(layout.delta)?,
            gather: DeviceBuffer::new(layout.gather)?,
            rows,
            gather_width: self.layout.gather / 2,
        })
    }
    #[allow(clippy::too_many_arguments)] // Validated projection layout plus exclusively owned device work.
    pub(crate) unsafe fn apply(
        &self,
        layer: usize,
        group: Group,
        input: *const c_void,
        output: *mut c_void,
        lengths: &[usize],
        work: &Work,
        gemm: *mut c_void,
        stream: Stream,
    ) -> Result<()> {
        let pairs = &self.layers[layer][group.index()];
        if pairs.is_empty() {
            return Ok(());
        }
        let (cols, stride) = (pairs[0].target.cols, pairs[0].target.stride);
        let mut row = 0;
        for &rows in lengths {
            ensure!(rows > 0 && rows <= work.rows, "LoRA work row bound");
            // Each prompt keeps its own FP32 GEMM M, including packed base projections.
            unsafe {
                check(
                    (cuda::api().cs1_vision_to_float)(
                        input.wrapping_byte_add(row * cols * 2),
                        work.input.at(0).cast(),
                        rows * cols,
                        stream,
                    ),
                    "language LoRA input",
                )?;
                for pair in pairs {
                    let n = pair.target.rows;
                    check(
                        (cuda::api().cs1_gemm_f32)(
                            gemm,
                            work.input.at(0).cast(),
                            pair.a.at(0).cast(),
                            work.rank.at(0).cast(),
                            rows as i32,
                            pair.rank as i32,
                            cols as i32,
                            stream,
                        ),
                        "language LoRA A",
                    )?;
                    check(
                        (cuda::api().cs1_gemm_f32)(
                            gemm,
                            work.rank.at(0).cast(),
                            pair.b.at(0).cast(),
                            work.delta.at(0).cast(),
                            rows as i32,
                            n as i32,
                            pair.rank as i32,
                            stream,
                        ),
                        "language LoRA B",
                    )?;
                    let out = output.wrapping_byte_add((row * stride + pair.target.offset) * 2);
                    if n == stride {
                        check(
                            (cuda::api().cs1_vision_lora_add)(
                                out,
                                work.delta.at(0).cast(),
                                rows * n,
                                self.scale,
                                stream,
                            ),
                            "language LoRA add",
                        )?;
                    } else {
                        cuda::copy2d(
                            work.gather.at(0),
                            n * 2,
                            out,
                            stride * 2,
                            n * 2,
                            rows,
                            stream,
                        )?;
                        check(
                            (cuda::api().cs1_vision_lora_add)(
                                work.gather.at(0),
                                work.delta.at(0).cast(),
                                rows * n,
                                self.scale,
                                stream,
                            ),
                            "language LoRA add",
                        )?;
                        cuda::copy2d(
                            out,
                            stride * 2,
                            work.gather.at(0),
                            n * 2,
                            n * 2,
                            rows,
                            stream,
                        )?;
                    }
                }
            }
            row += rows;
        }
        Ok(())
    }
}

pub(crate) fn component_rows(cfg: &Config, group: Group) -> Result<[usize; 4]> {
    let mul = |a: usize, b: usize| {
        a.checked_mul(b)
            .context("base component dimension overflow")
    };
    Ok(match group {
        Group::LinearInput => {
            let kd = mul(cfg.lin_k_heads, cfg.lin_k_dim)?;
            let vd = mul(cfg.lin_v_heads, cfg.lin_v_dim)?;
            [
                mul(2, kd)?
                    .checked_add(vd)
                    .context("base component dimension overflow")?,
                vd,
                cfg.lin_v_heads,
                cfg.lin_v_heads,
            ]
        }
        Group::AttentionInput => [
            mul(mul(cfg.heads, cfg.head_dim)?, 2)?,
            mul(cfg.kv_heads, cfg.head_dim)?,
            mul(cfg.kv_heads, cfg.head_dim)?,
            0,
        ],
        Group::GateUp => [cfg.intermediate, cfg.intermediate, 0, 0],
        _ => anyhow::bail!("base component group is not fused"),
    })
}
/// Original component GEMMs write contiguous BF16 rows before copying into fused scratch.
///
/// # Safety
/// Input is packed row-major [sum(lengths),cols]; weights are contiguous row-stacked
/// component matrices; output holds sum(lengths) rows of sum(components). Work and
/// GEMM belong to the serialized Model on this stream.
#[allow(clippy::too_many_arguments)] // Exact device boundary and validated component layout.
pub(crate) unsafe fn project_components(
    input: *const c_void,
    weights: *const c_void,
    output: *mut c_void,
    lengths: &[usize],
    cols: usize,
    components: &[usize],
    work: &Work,
    gemm: *mut c_void,
    stream: Stream,
) -> Result<()> {
    let stride = components
        .iter()
        .try_fold(0usize, |sum, &n| sum.checked_add(n))
        .context("base component stride overflow")?;
    ensure!(
        stride > 0 && stride <= i32::MAX as usize && cols > 0 && cols <= i32::MAX as usize,
        "base component dimensions"
    );
    ensure!(
        components.iter().all(|&n| n <= work.gather_width),
        "base component exceeds work width"
    );
    let mut row = 0;
    for &rows in lengths {
        ensure!(
            rows > 0 && rows <= work.rows,
            "base component work row bound"
        );
        let mut first = 0;
        for &n in components {
            if n == 0 {
                continue;
            }
            // Match the original projection's M/N/K and contiguous input/weight/output
            // strides. A strided GEMM output itself can choose a different cuBLAS plan.
            unsafe {
                check(
                    (cuda::api().cs1_gemm)(
                        gemm,
                        input.wrapping_byte_add(row * cols * 2),
                        weights.wrapping_byte_add(first * cols * 2),
                        work.gather.at(0),
                        rows as i32,
                        n as i32,
                        cols as i32,
                        n as i32,
                        stream,
                    ),
                    "base component GEMM",
                )?;
                cuda::copy2d(
                    output.wrapping_byte_add((row * stride + first) * 2),
                    stride * 2,
                    work.gather.at(0),
                    n * 2,
                    n * 2,
                    rows,
                    stream,
                )?;
            }
            first += n;
        }
        row = row
            .checked_add(rows)
            .context("base component row overflow")?;
        ensure!(row <= i32::MAX as usize, "base component packed row bound");
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../../tests/qwen3_5/lora.rs"]
mod tests;
