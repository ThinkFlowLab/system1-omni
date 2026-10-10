//! The Qwen3.5/3.8 text backbone, prefill only: one
//! forward pass over a prompt, returning the final-norm hidden state of the last
//! position. The layer loop and buffers live here; the operations are the CUDA
//! kernels in `src/backends/cuda/qwen3_5`.
//!
//! The order of operations follows `modeling_qwen3_5.py`, and so do the points where
//! it rounds to bfloat16, except inside attention and the Gated DeltaNet prefill (see
//! their kernels). Text prompts use one position per
//! token. The explicit multimodal boundary inserts adapted image rows and supplies
//! the interleaved temporal/height/width rotary positions to the same layer loop.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value as Json;

use crate::cuda::{self, DeviceBuffer, Stream, check};
use crate::inputs::{MultimodalInput, rotary_tables};

const ALIGN: usize = 256;
const BF16: usize = 2;
const F32: usize = 4;
const GEMM_WORKSPACE: usize = 32 << 20;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub eps: f32,
    pub full_attention: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Half the number of rotary dims (rotate_half pairs dim i with dim i + half).
    pub rotary_half: usize,
    pub rope_theta: f64,
    pub mrope_section: [usize; 3],
    pub max_positions: usize,
    pub image_token_id: Option<u32>,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let root: Json = serde_json::from_str(
            &std::fs::read_to_string(&path).with_context(|| format!("{}", path.display()))?,
        )?;
        let c = root.get("text_config").unwrap_or(&root);
        let int = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .with_context(|| format!("config.json: `{k}` is missing"))
        };
        let rope = &c["rope_parameters"];
        let partial = rope["partial_rotary_factor"]
            .as_f64()
            .or(c["partial_rotary_factor"].as_f64())
            .unwrap_or(1.0);
        let head_dim = int("head_dim")?;
        let full_attention = c["layer_types"]
            .as_array()
            .context("config.json: `layer_types` is missing")?
            .iter()
            .map(|t| match t.as_str() {
                Some("full_attention") => Ok(true),
                Some("linear_attention") => Ok(false),
                other => bail!("unknown layer type {other:?}"),
            })
            .collect::<Result<Vec<_>>>()?;
        let sections = rope
            .get("mrope_section")
            .cloned()
            .unwrap_or(serde_json::json!([11, 11, 10]));
        let sections = sections
            .as_array()
            .context("mrope_section must be an array")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .context("mrope_section must contain integers")
            })
            .collect::<Result<Vec<_>>>()?;
        let mrope_section: [usize; 3] = sections
            .try_into()
            .map_err(|_| anyhow::anyhow!("mrope_section must have three entries"))?;
        let image_token_id = root
            .get("image_token_id")
            .map(|v| {
                v.as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .context("image_token_id must be a u32")
            })
            .transpose()?;
        let cfg = Config {
            hidden: int("hidden_size")?,
            intermediate: int("intermediate_size")?,
            eps: c["rms_norm_eps"].as_f64().context("rms_norm_eps")? as f32,
            heads: int("num_attention_heads")?,
            kv_heads: int("num_key_value_heads")?,
            head_dim,
            rotary_half: (head_dim as f64 * partial) as usize / 2,
            rope_theta: rope["rope_theta"]
                .as_f64()
                .or(c["rope_theta"].as_f64())
                .context("rope_theta")?,
            mrope_section,
            max_positions: int("max_position_embeddings")?,
            image_token_id,
            lin_k_heads: int("linear_num_key_heads")?,
            lin_v_heads: int("linear_num_value_heads")?,
            lin_k_dim: int("linear_key_head_dim")?,
            lin_v_dim: int("linear_value_head_dim")?,
            full_attention,
        };
        // What the kernels implement.
        ensure!(
            cfg.full_attention.len() == int("num_hidden_layers")?,
            "layer_types does not match num_hidden_layers"
        );
        ensure!(c["hidden_act"] == "silu", "hidden_act is not silu");
        ensure!(
            c["attn_output_gate"].as_bool().unwrap_or(true),
            "attention without the output gate"
        );
        ensure!(
            c["attention_bias"].as_bool() != Some(true),
            "attention with bias"
        );
        ensure!(
            rope["rope_type"].as_str().unwrap_or("default") == "default",
            "rope type {}",
            rope["rope_type"]
        );
        ensure!(int("linear_conv_kernel_dim")? == 4, "conv kernel is not 4");
        ensure!(
            cfg.head_dim == 256 && cfg.lin_k_dim == 128 && cfg.lin_v_dim == 128,
            "head dims {} / {} / {}",
            cfg.head_dim,
            cfg.lin_k_dim,
            cfg.lin_v_dim
        );
        ensure!(cfg.rotary_half == 32, "{} rotary dims", 2 * cfg.rotary_half);
        ensure!(
            cfg.rope_theta.is_finite() && cfg.rope_theta > 0.0,
            "invalid rope_theta"
        );
        ensure!(
            rope["mrope_interleaved"].as_bool() != Some(false),
            "non-interleaved mrope is unsupported"
        );
        ensure!(
            cfg.mrope_section
                .iter()
                .try_fold(0usize, |sum, &x| sum.checked_add(x))
                == Some(cfg.rotary_half),
            "mrope sections must sum to rotary_half"
        );
        ensure!(
            cfg.max_positions > 0 && cfg.max_positions <= (1 << 24),
            "unsupported max_position_embeddings"
        );
        ensure!(
            cfg.kv_heads > 0 && cfg.heads.is_multiple_of(cfg.kv_heads),
            "attention heads"
        );
        ensure!(
            cfg.lin_k_heads > 0 && cfg.lin_v_heads.is_multiple_of(cfg.lin_k_heads),
            "linear attention heads"
        );
        ensure!(cfg.hidden.is_multiple_of(8), "hidden size");
        Ok(cfg)
    }

    fn key_dim(&self) -> usize {
        self.lin_k_heads * self.lin_k_dim
    }

    fn value_dim(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }
}

/// A weight in the device arena.
#[derive(Clone)]
struct Tensor {
    ptr: *const c_void,
    shape: Vec<usize>,
}

impl Tensor {
    fn bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * BF16
    }
}

/// Projections that run as one GEMM, in the order their rows are stacked.
const GROUPS: &[&str] = &[
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
];

/// Upload order: by layer, and inside a layer the GROUPS members first and in order,
/// so each group's matrices sit back to back and form one [sum N, K] matrix.
fn upload_order(name: &str) -> (usize, usize, String) {
    if let Some(tail) = name.strip_prefix("layers.")
        && let Some((layer, rest)) = tail.split_once('.')
        && let Ok(layer) = layer.parse::<usize>()
    {
        let rank = GROUPS
            .iter()
            .position(|g| *g == rest)
            .unwrap_or(GROUPS.len());
        return (layer, rank, rest.to_string());
    }
    (usize::MAX, 0, name.to_string())
}

struct Weights {
    _arena: DeviceBuffer,
    tensors: HashMap<String, Tensor>,
    prefix: String,
}

impl Weights {
    /// Upload every bfloat16 tensor of the language model into one allocation.
    fn load(dir: &Path, stream: Stream) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let mut files: Vec<String> = if index.exists() {
            let index: Json = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
            index["weight_map"]
                .as_object()
                .context("weight_map")?
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        } else {
            vec!["model.safetensors".to_string()]
        };
        files.sort();
        files.dedup();
        let maps = files
            .iter()
            .map(|f| {
                let file = std::fs::File::open(dir.join(f)).with_context(|| f.clone())?;
                // SAFETY: the checkpoint is not modified while it is loaded.
                Ok(unsafe { memmap2::Mmap::map(&file)? })
            })
            .collect::<Result<Vec<_>>>()?;
        let sts = maps
            .iter()
            .map(|m| safetensors::SafeTensors::deserialize(m).map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        let names: Vec<(usize, String)> = sts
            .iter()
            .enumerate()
            .flat_map(|(i, st)| st.names().into_iter().map(move |n| (i, n.to_string())))
            .collect();
        let prefix = ["model.language_model.", "model.", ""]
            .into_iter()
            .find(|p| {
                names
                    .iter()
                    .any(|(_, n)| *n == format!("{p}embed_tokens.weight"))
            })
            .context("no embed_tokens.weight in the checkpoint")?
            .to_string();
        let mut ours: Vec<(usize, String)> = names
            .into_iter()
            .filter(|(_, n)| n.starts_with(&prefix))
            .collect();
        ours.sort_by_key(|(_, n)| upload_order(&n[prefix.len()..]));
        let mut plan = Vec::new();
        let mut total = 0usize;
        for (i, name) in ours {
            let view = sts[i].tensor(&name)?;
            ensure!(
                view.dtype() == safetensors::Dtype::BF16,
                "{name} is {:?}, not bfloat16",
                view.dtype()
            );
            plan.push((i, name, total));
            total = (total + view.data().len()).next_multiple_of(ALIGN);
        }
        let arena = DeviceBuffer::new(total)?;
        let mut tensors = HashMap::new();
        for (i, name, offset) in plan {
            let view = sts[i].tensor(&name)?;
            // SAFETY: the arena has room for every planned tensor at its offset.
            unsafe { cuda::upload(arena.at(offset), view.data(), stream)? };
            tensors.insert(
                name[prefix.len()..].to_string(),
                Tensor {
                    ptr: arena.at(offset),
                    shape: view.shape().to_vec(),
                },
            );
        }
        Ok(Self {
            _arena: arena,
            tensors,
            prefix,
        })
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let t = self
            .tensors
            .get(name)
            .with_context(|| format!("{}{name} is missing", self.prefix))?;
        ensure!(
            t.shape == shape,
            "{}{name}: shape {:?}, expected {:?}",
            self.prefix,
            t.shape,
            shape
        );
        Ok(t.clone())
    }

    /// The row-stacked matrix of tensors that were uploaded back to back.
    fn stacked(&self, parts: &[Tensor]) -> Result<Tensor> {
        let k = parts[0].shape[1];
        let mut rows = 0;
        for (i, p) in parts.iter().enumerate() {
            ensure!(
                p.shape.len() == 2 && p.shape[1] == k,
                "stacked shapes differ"
            );
            if i > 0 {
                let prev = &parts[i - 1];
                ensure!(
                    p.ptr == prev.ptr.wrapping_byte_add(prev.bytes()),
                    "stacked weights are not contiguous"
                );
            }
            rows += p.shape[0];
        }
        Ok(Tensor {
            ptr: parts[0].ptr,
            shape: vec![rows, k],
        })
    }
}

struct LinearAttention {
    /// in_proj_qkv | in_proj_z | in_proj_b | in_proj_a
    in_proj: Tensor,
    conv: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: Tensor,
    out: Tensor,
}

struct FullAttention {
    /// q_proj (query and gate per head) | k_proj | v_proj
    qkv: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}

enum Mixer {
    Linear(LinearAttention),
    Full(FullAttention),
}

struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    mixer: Mixer,
    /// gate_proj | up_proj
    gate_up: Tensor,
    down: Tensor,
}

/// Row widths of the stacked projection outputs.
struct Widths {
    conv: usize,
    gdn_in: usize,
    attn_q: usize,
    attn_in: usize,
}

impl Widths {
    fn of(cfg: &Config) -> Self {
        let conv = 2 * cfg.key_dim() + cfg.value_dim();
        let attn_q = cfg.heads * cfg.head_dim * 2;
        Self {
            conv,
            gdn_in: conv + cfg.value_dim() + 2 * cfg.lin_v_heads,
            attn_q,
            attn_in: attn_q + 2 * cfg.kv_heads * cfg.head_dim,
        }
    }
}

/// Per-request buffers for up to `cap` tokens, as byte offsets into one allocation,
/// plus the rotary tables for positions below `cap`.
struct Scratch {
    cap: usize,
    buf: DeviceBuffer,
    ids: usize,
    res: usize,
    x: usize,
    delta: usize,
    gdn_in: usize,
    beta: usize,
    g: usize,
    lq: usize,
    lk: usize,
    lv: usize,
    lo: usize,
    ln: usize,
    workspace: usize,
    attn_in: usize,
    aq: usize,
    agate: usize,
    ak: usize,
    ao: usize,
    gate_up: usize,
    act: usize,
    cos: usize,
    sin: usize,
    custom_cos: usize,
    custom_sin: usize,
}

impl Scratch {
    fn new(cfg: &Config, cap: usize, stream: Stream) -> Result<Self> {
        let (h, kd, vd, hv) = (cfg.hidden, cfg.key_dim(), cfg.value_dim(), cfg.lin_v_heads);
        let (hq, hk, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
        let w = Widths::of(cfg);
        let mut next = 0usize;
        let mut take = |bytes: usize| {
            let off = next;
            next = (off + bytes).next_multiple_of(ALIGN);
            off
        };
        // SAFETY: pure function of its arguments.
        let ws_floats = unsafe { (cuda::api().cs1_gdn_workspace_floats)(cap as i32, hv as i32) };
        let offsets = [
            take(cap * 4),
            take(cap * h * BF16),
            take(cap * h * BF16),
            take(cap * h * BF16),
            take(cap * w.gdn_in * BF16),
            take(cap * hv * BF16),
            take(cap * hv * F32),
            take(cap * kd * BF16),
            take(cap * kd * BF16),
            take(cap * vd * BF16),
            take(cap * vd * BF16),
            take(cap * vd * BF16),
            take(ws_floats * F32),
            take(cap * w.attn_in * BF16),
            take(cap * hq * hd * BF16),
            take(cap * hq * hd * BF16),
            take(cap * hk * hd * BF16),
            take(cap * hq * hd * BF16),
            take(cap * 2 * cfg.intermediate * BF16),
            take(cap * cfg.intermediate * BF16),
            take(cap * cfg.rotary_half * BF16),
            take(cap * cfg.rotary_half * BF16),
            take(cap * cfg.rotary_half * BF16),
            take(cap * cfg.rotary_half * BF16),
        ];
        let buf = DeviceBuffer::new(next)?;
        let [
            ids,
            res,
            x,
            delta,
            gdn_in,
            beta,
            g,
            lq,
            lk,
            lv,
            lo,
            ln,
            workspace,
            attn_in,
            aq,
            agate,
            ak,
            ao,
            gate_up,
            act,
            cos,
            sin,
            custom_cos,
            custom_sin,
        ] = offsets;
        let positions: Vec<i64> = (0..cap as i64).collect();
        let (cos_t, sin_t) = rotary_tables(
            [&positions; 3],
            cfg.rotary_half,
            cfg.rope_theta,
            cfg.mrope_section,
        );
        // SAFETY: both tables were laid out for cap * rotary_half bfloat16 values.
        unsafe {
            cuda::upload(buf.at(cos), &cos_t, stream)?;
            cuda::upload(buf.at(sin), &sin_t, stream)?;
        }
        Ok(Self {
            cap,
            buf,
            ids,
            res,
            x,
            delta,
            gdn_in,
            beta,
            g,
            lq,
            lk,
            lv,
            lo,
            ln,
            workspace,
            attn_in,
            aq,
            agate,
            ak,
            ao,
            gate_up,
            act,
            cos,
            sin,
            custom_cos,
            custom_sin,
        })
    }

    fn at(&self, offset: usize) -> *mut c_void {
        self.buf.at(offset)
    }
}

/// The cached device-side state of one token prefix (rows `[0, len)`) for the R2d
/// hybrid path: per full-attention layer the post-prep K rows plus the pre-GEMM V
/// rows, per Gated DeltaNet layer the float32 recurrent state and the three
/// pre-conv projection columns feeding the conv window. States belong to the
/// prefix itself; a continuation seeds its own scratch from them, read-only.
pub struct PrefixState {
    owner: Arc<()>,
    initialized: bool,
    /// Tokens this prefix covers; always a multiple of 64 (the GDN chunk length),
    /// which also aligns the flash-attention key tiles of the one-shot pass.
    len: usize,
    /// Post-prep K [len, Hk*Dh] and raw V [len, Hk*Dh] rows per full-attention layer.
    attn_kv: Vec<(DeviceBuffer, DeviceBuffer)>,
    /// Float32 [H, K, V] recurrent states, one per Gated DeltaNet layer.
    gdn_state: Vec<DeviceBuffer>,
    /// Three pre-conv projection columns [3, gdn_in width], one per Gated DeltaNet layer.
    conv_tail: Vec<DeviceBuffer>,
    /// Total device bytes, for cache-budget accounting by the caller.
    bytes: usize,
}

impl PrefixState {
    /// Number of tokens covered by this nonempty prefix.
    pub fn token_count(&self) -> usize {
        self.len
    }

    /// Device allocation size used by cache-budget accounting.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

pub struct Model {
    pub cfg: Config,
    prefix_owner: Arc<()>,
    _weights: Weights,
    embed: Tensor,
    final_norm: Tensor,
    layers: Vec<Layer>,
    stream: Stream,
    gemm: *mut c_void,
    /// Buffers for the largest packed token count so far; grows as needed.
    scratch: Option<Scratch>,
    /// Opt-in replay with at most 64 captures per mode, keyed by sequence lengths.
    graph_enabled: bool,
    graphs: VecDeque<(Vec<usize>, cuda::Graph)>,
    mm_graphs: VecDeque<(Vec<usize>, cuda::Graph)>,
}

// SAFETY: the raw pointers are device addresses and a cuBLASLt handle owned by the
// model; the engine runs one forward pass at a time behind a mutex.
unsafe impl Send for Model {}

impl Drop for Model {
    fn drop(&mut self) {
        self.graphs.clear();
        self.mm_graphs.clear();
        // SAFETY: created by cs1_gemm_create and not destroyed before.
        unsafe { (cuda::api().cs1_gemm_destroy)(self.gemm) };
    }
}

impl Model {
    /// Load the CUDA library and the weights.
    pub fn load(dir: &Path, library: &Path) -> Result<Self> {
        let cfg = Config::load(dir)?;
        cuda::load(library)?;
        cuda::set_device(0)?;
        let stream = cuda::new_stream()?;
        let weights = Weights::load(dir, stream)?;
        let (h, kd, vd) = (cfg.hidden, cfg.key_dim(), cfg.value_dim());
        let embed = weights
            .tensors
            .get("embed_tokens.weight")
            .context("embed_tokens.weight is missing")?
            .clone();
        ensure!(
            embed.shape.len() == 2 && embed.shape[1] == h,
            "embed_tokens.weight shape {:?}",
            embed.shape
        );
        let final_norm = weights.get("norm.weight", &[h])?;
        let mut layers = Vec::with_capacity(cfg.full_attention.len());
        for (i, &full) in cfg.full_attention.iter().enumerate() {
            let w = |n: &str, s: &[usize]| weights.get(&format!("layers.{i}.{n}"), s);
            let mixer = if full {
                let (hq, hk, hd) = (cfg.heads, cfg.kv_heads, cfg.head_dim);
                Mixer::Full(FullAttention {
                    qkv: weights.stacked(&[
                        w("self_attn.q_proj.weight", &[hq * hd * 2, h])?,
                        w("self_attn.k_proj.weight", &[hk * hd, h])?,
                        w("self_attn.v_proj.weight", &[hk * hd, h])?,
                    ])?,
                    o: w("self_attn.o_proj.weight", &[h, hq * hd])?,
                    q_norm: w("self_attn.q_norm.weight", &[hd])?,
                    k_norm: w("self_attn.k_norm.weight", &[hd])?,
                })
            } else {
                let hv = cfg.lin_v_heads;
                Mixer::Linear(LinearAttention {
                    in_proj: weights.stacked(&[
                        w("linear_attn.in_proj_qkv.weight", &[2 * kd + vd, h])?,
                        w("linear_attn.in_proj_z.weight", &[vd, h])?,
                        w("linear_attn.in_proj_b.weight", &[hv, h])?,
                        w("linear_attn.in_proj_a.weight", &[hv, h])?,
                    ])?,
                    conv: w("linear_attn.conv1d.weight", &[2 * kd + vd, 1, 4])?,
                    a_log: w("linear_attn.A_log", &[hv])?,
                    dt_bias: w("linear_attn.dt_bias", &[hv])?,
                    norm: w("linear_attn.norm.weight", &[cfg.lin_v_dim])?,
                    out: w("linear_attn.out_proj.weight", &[h, vd])?,
                })
            };
            layers.push(Layer {
                input_norm: w("input_layernorm.weight", &[h])?,
                post_norm: w("post_attention_layernorm.weight", &[h])?,
                mixer,
                gate_up: weights.stacked(&[
                    w("mlp.gate_proj.weight", &[cfg.intermediate, h])?,
                    w("mlp.up_proj.weight", &[cfg.intermediate, h])?,
                ])?,
                down: w("mlp.down_proj.weight", &[h, cfg.intermediate])?,
            });
        }
        // SAFETY: plain allocation; checked for null below.
        let gemm = unsafe { (cuda::api().cs1_gemm_create)(GEMM_WORKSPACE) };
        ensure!(!gemm.is_null(), "cuBLASLt setup failed");
        let model = Self {
            cfg,
            prefix_owner: Arc::new(()),
            _weights: weights,
            embed,
            final_norm,
            layers,
            stream,
            gemm,
            scratch: None,
            graph_enabled: std::env::var("CUA_S1_GRAPH").as_deref() == Ok("1"),
            graphs: VecDeque::new(),
            mm_graphs: VecDeque::new(),
        };
        Ok(model)
    }

    fn gemm(&self, s: &Scratch, x: usize, w: &Tensor, y: usize, m: usize) -> Result<()> {
        let (n, k) = (w.shape[0] as i32, w.shape[1] as i32);
        // SAFETY: x and y are scratch buffers sized for m rows of w's shape.
        check(
            unsafe {
                (cuda::api().cs1_gemm)(
                    self.gemm,
                    s.at(x),
                    w.ptr,
                    s.at(y),
                    m as i32,
                    n,
                    k,
                    n,
                    self.stream,
                )
            },
            "gemm",
        )
    }

    /// Preserve each prompt's output/down GEMM shape and split-K reduction order.
    fn gemm_sequences(
        &self,
        s: &Scratch,
        x: usize,
        w: &Tensor,
        y: usize,
        lengths: &[usize],
    ) -> Result<()> {
        let (n, k) = (w.shape[0], w.shape[1]);
        let mut offset = 0;
        for &length in lengths {
            self.gemm(s, x + offset * k * BF16, w, y + offset * n * BF16, length)?;
            offset += length;
        }
        Ok(())
    }

    /// Finish queued work before releasing external execution admission.
    pub fn synchronize(&self) -> Result<()> {
        cuda::set_device(0)?;
        cuda::synchronize(self.stream)
    }

    fn prepare_scratch(&mut self, t: usize) -> Result<()> {
        cuda::set_device(0)?;
        if self.scratch.as_ref().is_none_or(|s| t > s.cap) {
            self.graphs.clear();
            self.mm_graphs.clear();
            self.scratch = None;
            self.scratch = Some(Scratch::new(
                &self.cfg,
                t.next_multiple_of(1024),
                self.stream,
            )?);
        }
        Ok(())
    }

    /// The final-norm hidden state at the last position, as float32.
    pub fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        Ok(self.forward_batch(&[ids])?.pop().unwrap())
    }

    /// Pack independent text prompts for input and gate/up GEMMs.
    /// Mixers reset at each boundary; final-position hidden states retain input order.
    pub fn forward_batch(&mut self, inputs: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        ensure!(!inputs.is_empty(), "empty batch");
        ensure!(
            inputs
                .iter()
                .all(|ids| !ids.is_empty() && ids.len() <= self.cfg.max_positions),
            "empty or oversized prompt"
        );
        ensure!(
            inputs
                .iter()
                .flat_map(|ids| ids.iter())
                .all(|&id| (id as usize) < self.embed.shape[0]),
            "token id outside the vocabulary"
        );
        let lengths: Vec<usize> = inputs.iter().map(|ids| ids.len()).collect();
        let t = lengths
            .iter()
            .try_fold(0usize, |total, &length| total.checked_add(length))
            .context("packed token count overflow")?;
        ensure!(
            t <= i32::MAX as usize,
            "packed token count exceeds the CUDA layout"
        );
        self.prepare_scratch(t)?;
        let s = self.scratch.as_ref().unwrap();
        let ids: Vec<u32> = inputs.iter().flat_map(|ids| ids.iter().copied()).collect();
        self.embed_tokens(s, &ids)?;
        if self.graph_enabled {
            if let Some((_, graph)) = self.graphs.iter().find(|(shape, _)| *shape == lengths) {
                graph.launch(self.stream)?;
                if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
                    eprintln!("CUA_S1_GRAPH mode=text event=replay tokens={t}");
                }
            } else {
                // Warm plans and keep the eager result: run() advances the residual
                // in place, so replaying on this cache miss would advance it twice.
                self.run(s, &lengths, false)?;
                cuda::synchronize(self.stream)?;
                let captured = cuda::Graph::capture(self.stream, || self.run(s, &lengths, false));
                self.cache_graph(&lengths, false, captured);
            }
        } else {
            self.run(s, &lengths, false)?;
        }
        let s = self.scratch.as_ref().unwrap();
        let mut end = 0;
        lengths
            .iter()
            .map(|&length| {
                end += length;
                self.last_hidden(s, end)
            })
            .collect()
    }

    /// Prefill one unpadded prompt with already-adapted BF16 image embeddings and
    /// explicit `[3, 1, sequence]` T/H/W positions. No vision tower runs here.
    /// Load a checkpoint with the matching multimodal language adapter merged.
    pub fn forward_multimodal(&mut self, input: &MultimodalInput<'_>) -> Result<Vec<f32>> {
        let image_token = self
            .cfg
            .image_token_id
            .context("checkpoint has no image_token_id")?;
        input.validate(
            self.cfg.hidden,
            self.embed.shape[0],
            image_token,
            self.cfg.max_positions,
        )?;
        let t = input.token_ids.len();
        self.prepare_scratch(t)?;
        let s = self.scratch.as_ref().unwrap();
        self.upload_positions(s, input.position_ids)?;
        self.embed_tokens(s, input.token_ids)?;
        self.overwrite_image_rows(s, input, 0)?;
        if self.graph_enabled {
            if let Some((_, graph)) = self
                .mm_graphs
                .iter()
                .find(|(shape, _)| shape.as_slice() == [t])
            {
                graph.launch(self.stream)?;
                if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
                    eprintln!("CUA_S1_GRAPH mode=multimodal event=replay tokens={t}");
                }
            } else {
                // Uploads and embedding are outside capture. Keep the warmed eager
                // result; launching now would advance the residual a second time.
                self.run(s, &[t], true)?;
                cuda::synchronize(self.stream)?;
                let captured = cuda::Graph::capture(self.stream, || self.run(s, &[t], true));
                self.cache_graph(&[t], true, captured);
            }
        } else {
            self.run(s, &[t], true)?;
        }
        self.last_hidden(self.scratch.as_ref().unwrap(), t)
    }

    /// Both modes retain the eager miss result; capture records without executing.
    fn cache_graph(&mut self, lengths: &[usize], multimodal: bool, captured: Result<cuda::Graph>) {
        let mode = if multimodal { "multimodal" } else { "text" };
        match captured {
            Ok(graph) => {
                let cache = if multimodal {
                    &mut self.mm_graphs
                } else {
                    &mut self.graphs
                };
                if cache.len() == 64 {
                    cache.pop_front();
                }
                cache.push_back((lengths.to_vec(), graph));
                if std::env::var("CUA_S1_GRAPH_TRACE").as_deref() == Ok("1") {
                    let t: usize = lengths.iter().sum();
                    eprintln!("CUA_S1_GRAPH mode={mode} event=capture tokens={t}");
                }
            }
            Err(error) => {
                eprintln!(
                    "CUDA Graph capture failed in {mode} mode; using eager execution: {error:#}"
                );
                self.graph_enabled = false;
                self.graphs.clear();
                self.mm_graphs.clear();
            }
        }
    }

    fn upload_positions(&self, s: &Scratch, positions: [&[i64]; 3]) -> Result<()> {
        let (cos, sin) = rotary_tables(
            positions,
            self.cfg.rotary_half,
            self.cfg.rope_theta,
            self.cfg.mrope_section,
        );
        // SAFETY: tables contain at most s.cap rows of rotary_half BF16 values.
        unsafe {
            cuda::upload(s.at(s.custom_cos), &cos, self.stream)?;
            cuda::upload(s.at(s.custom_sin), &sin, self.stream)?;
        }
        Ok(())
    }

    /// Overwrite the residual rows of validated image placeholders with the adapted
    /// image embeddings. `qb` is the absolute row base of the passed slice inside
    /// the residual buffer (0 for a full prompt, the cached prefix length for a
    /// continuation — indices are local to the slice in both cases).
    fn overwrite_image_rows(
        &self,
        s: &Scratch,
        input: &MultimodalInput<'_>,
        qb: usize,
    ) -> Result<()> {
        let bytes: Vec<u8> = input
            .image_embeddings
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        // Coalesce adjacent placeholders. Text rows remain those of embed_tokens.
        let indices = input.image_token_indices;
        let mut begin = 0;
        while begin < indices.len() {
            let mut end = begin + 1;
            while end < indices.len() && indices[end] == indices[end - 1] + 1 {
                end += 1;
            }
            let row_bytes = self.cfg.hidden * BF16;
            // SAFETY: validated indices lie in the residual buffer, and features
            // contain exactly one hidden-size BF16 row per placeholder.
            unsafe {
                cuda::upload(
                    s.at(s.res + (qb + indices[begin]) * row_bytes),
                    &bytes[begin * row_bytes..end * row_bytes],
                    self.stream,
                )?;
            }
            begin = end;
        }
        Ok(())
    }

    /// Allocate an uninitialized prefix of `len` tokens (a multiple of 64).
    /// Capture must finish successfully on this model before continuation.
    pub fn alloc_prefix(&self, len: usize) -> Result<PrefixState> {
        ensure!(
            len >= 64 && len.is_multiple_of(64),
            "cached prefix length must be a positive multiple of 64"
        );
        ensure!(
            len <= self.cfg.max_positions,
            "cached prefix exceeds the configured maximum length"
        );
        let kvrow = self.cfg.kv_heads * self.cfg.head_dim * BF16;
        let w = Widths::of(&self.cfg);
        let nfull = self.cfg.full_attention.iter().filter(|&&f| f).count();
        let ngdn = self.cfg.full_attention.len() - nfull;
        let mut bytes = 0usize;
        let mut attn_kv = Vec::with_capacity(nfull);
        for _ in 0..nfull {
            let k = DeviceBuffer::new(len * kvrow)?;
            let v = DeviceBuffer::new(len * kvrow)?;
            bytes += 2 * len * kvrow;
            attn_kv.push((k, v));
        }
        let state_floats = self.cfg.lin_v_heads * self.cfg.lin_k_dim * self.cfg.lin_v_dim;
        let tail_bytes = 3 * w.gdn_in * BF16;
        let mut gdn_state = Vec::with_capacity(ngdn);
        let mut conv_tail = Vec::with_capacity(ngdn);
        for _ in 0..ngdn {
            gdn_state.push(DeviceBuffer::new(state_floats * F32)?);
            conv_tail.push(DeviceBuffer::new(tail_bytes)?);
            bytes += state_floats * F32 + tail_bytes;
        }
        Ok(PrefixState {
            owner: Arc::clone(&self.prefix_owner),
            initialized: false,
            len,
            attn_kv,
            gdn_state,
            conv_tail,
            bytes,
        })
    }

    /// Run the rows `[0, state.len)` of one prompt, collecting the per-layer
    /// prefix state (post-prep K/V columns, float32 GDN states, conv tails) into
    /// `state`. Synchronize before marking the state ready for continuation.
    /// The hidden states of these rows are computed but not returned.
    pub fn forward_multimodal_capture(
        &mut self,
        input: &MultimodalInput<'_>,
        state: &mut PrefixState,
    ) -> Result<()> {
        ensure!(
            Arc::ptr_eq(&self.prefix_owner, &state.owner),
            "prefix state belongs to a different model instance"
        );
        let t = input.token_ids.len();
        ensure!(
            t == state.len,
            "capture input must cover the cached prefix exactly"
        );
        state.initialized = false;
        let image_token = self
            .cfg
            .image_token_id
            .context("checkpoint has no image_token_id")?;
        input.validate(
            self.cfg.hidden,
            self.embed.shape[0],
            image_token,
            self.cfg.max_positions,
        )?;
        self.prepare_scratch(t)?;
        let s = self.scratch.as_ref().unwrap();
        self.upload_positions(s, input.position_ids)?;
        self.embed_tokens(s, input.token_ids)?;
        self.overwrite_image_rows(s, input, 0)?;
        self.run_window(s, 0, t, true, Some(state), None)?;
        self.synchronize()?;
        state.initialized = true;
        Ok(())
    }

    /// Run rows `[prefix.len, prefix.len + input.len)` of one prompt seeded from a
    /// captured prefix; returns the final-norm hidden state at the last position.
    pub fn forward_multimodal_continue(
        &mut self,
        input: &MultimodalInput<'_>,
        prefix: &PrefixState,
    ) -> Result<Vec<f32>> {
        ensure!(
            Arc::ptr_eq(&self.prefix_owner, &prefix.owner),
            "prefix state belongs to a different model instance"
        );
        ensure!(prefix.initialized, "prefix state has not completed capture");
        let image_token = self
            .cfg
            .image_token_id
            .context("checkpoint has no image_token_id")?;
        input.validate(
            self.cfg.hidden,
            self.embed.shape[0],
            image_token,
            self.cfg.max_positions,
        )?;
        let rows = input.token_ids.len();
        let tend = prefix
            .len
            .checked_add(rows)
            .context("cached prompt length overflow")?;
        ensure!(
            tend <= self.cfg.max_positions,
            "cached prompt exceeds the configured maximum length"
        );
        self.prepare_scratch(tend)?;
        let s = self.scratch.as_ref().unwrap();
        self.upload_positions(s, input.position_ids)?;
        self.embed_tokens_at(s, input.token_ids, prefix.len)?;
        self.overwrite_image_rows(s, input, prefix.len)?;
        self.run_window(s, prefix.len, tend, true, None, Some(prefix))?;
        self.last_hidden(s, tend)
    }

    fn embed_tokens(&self, s: &Scratch, ids: &[u32]) -> Result<()> {
        self.embed_tokens_at(s, ids, 0)
    }

    /// Embed `ids` into the residual buffer rows starting at absolute row `qb`.
    fn embed_tokens_at(&self, s: &Scratch, ids: &[u32], qb: usize) -> Result<()> {
        let ids32: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
        // SAFETY: IDs were checked against the vocabulary; scratch holds qb + t rows.
        unsafe {
            cuda::upload(s.at(s.ids), &ids32, self.stream)?;
            check(
                (cuda::api().cs1_embed)(
                    s.at(s.ids).cast(),
                    self.embed.ptr,
                    s.at(s.res + qb * self.cfg.hidden * BF16),
                    ids.len() as i32,
                    self.cfg.hidden as i32,
                    self.stream,
                ),
                "embed",
            )?;
        }
        Ok(())
    }

    fn last_hidden(&self, s: &Scratch, t: usize) -> Result<Vec<f32>> {
        let mut last = vec![0u8; self.cfg.hidden * BF16];
        // SAFETY: x holds at least t rows of the hidden size.
        unsafe {
            cuda::download(
                &mut last,
                s.at(s.x + (t - 1) * self.cfg.hidden * BF16),
                self.stream,
            )?;
        }
        let (pairs, _) = last.as_chunks::<2>();
        Ok(pairs
            .iter()
            .map(|&b| half::bf16::from_le_bytes(b).to_f32())
            .collect())
    }

    /// Queue language layers over packed embeddings, resetting sequence positions.
    /// Final-norm hidden states end up in `s.x`.
    fn run(&self, s: &Scratch, lengths: &[usize], custom_positions: bool) -> Result<()> {
        let t: usize = lengths.iter().sum();
        let mut start = 0;
        let sequences: Vec<(usize, i32)> = lengths
            .iter()
            .map(|&length| {
                let offset = start;
                start += length;
                (offset, length as i32)
            })
            .collect();
        let cfg = &self.cfg;
        let st = self.stream;
        let (ti, hi, eps) = (t as i32, cfg.hidden as i32, cfg.eps);
        let (kd, vd, hv) = (cfg.key_dim(), cfg.value_dim(), cfg.lin_v_heads);
        let (hq, hk, hd) = (cfg.heads as i32, cfg.kv_heads as i32, cfg.head_dim as i32);
        let w = Widths::of(cfg);
        let p = |off: usize| s.at(off);
        let (cos, sin) = if custom_positions {
            (s.custom_cos, s.custom_sin)
        } else {
            (s.cos, s.sin)
        };
        // SAFETY (every kernel call below): the pointers are weights in the arena or
        // scratch buffers laid out for at least t tokens with the widths used here.
        unsafe {
            check(
                (cuda::api().cs1_rms_norm)(
                    p(s.res),
                    self.layers[0].input_norm.ptr,
                    p(s.x),
                    ti,
                    hi,
                    eps,
                    st,
                ),
                "input norm",
            )?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Linear(la) => {
                    self.gemm(s, s.x, &la.in_proj, s.gdn_in, t)?;
                    let ld = w.gdn_in as i32;
                    let z = s.gdn_in + w.conv * BF16;
                    let b = z + vd * BF16;
                    let a = b + hv * BF16;
                    unsafe {
                        for &(offset, length) in &sequences {
                            check(
                                (cuda::api().cs1_gdn_conv)(
                                    p(s.gdn_in + offset * w.gdn_in * BF16),
                                    ld,
                                    la.conv.ptr,
                                    p(s.lq + offset * kd * BF16),
                                    p(s.lk + offset * kd * BF16),
                                    p(s.lv + offset * vd * BF16),
                                    length,
                                    kd as i32,
                                    vd as i32,
                                    st,
                                ),
                                "gdn conv",
                            )?;
                        }
                        check(
                            (cuda::api().cs1_gdn_gates)(
                                p(b),
                                p(a),
                                ld,
                                la.a_log.ptr,
                                la.dt_bias.ptr,
                                p(s.beta),
                                p(s.g).cast(),
                                ti,
                                hv as i32,
                                st,
                            ),
                            "gdn gates",
                        )?;
                        for &(offset, length) in &sequences {
                            check(
                                (cuda::api().cs1_gdn_prefill)(
                                    p(s.lq + offset * kd * BF16),
                                    p(s.lk + offset * kd * BF16),
                                    p(s.lv + offset * vd * BF16),
                                    p(s.g + offset * hv * F32).cast(),
                                    p(s.beta + offset * hv * BF16),
                                    p(s.lo + offset * vd * BF16),
                                    p(s.workspace).cast(),
                                    length,
                                    hv as i32,
                                    cfg.lin_k_heads as i32,
                                    (cfg.lin_k_dim as f32).powf(-0.5),
                                    st,
                                ),
                                "gdn prefill",
                            )?;
                        }
                        check(
                            (cuda::api().cs1_gated_rms_norm)(
                                p(s.lo),
                                p(z),
                                ld,
                                la.norm.ptr,
                                p(s.ln),
                                ti,
                                hv as i32,
                                cfg.lin_v_dim as i32,
                                eps,
                                st,
                            ),
                            "gated norm",
                        )?;
                    }
                    self.gemm_sequences(s, s.ln, &la.out, s.delta, lengths)?;
                }
                Mixer::Full(fa) => {
                    self.gemm(s, s.x, &fa.qkv, s.attn_in, t)?;
                    let ld = w.attn_in as i32;
                    let k = s.attn_in + w.attn_q * BF16;
                    let v = k + cfg.kv_heads * cfg.head_dim * BF16;
                    unsafe {
                        for &(offset, length) in &sequences {
                            check(
                                (cuda::api().cs1_attn_prep)(
                                    p(s.attn_in + offset * w.attn_in * BF16),
                                    p(k + offset * w.attn_in * BF16),
                                    ld,
                                    fa.q_norm.ptr,
                                    fa.k_norm.ptr,
                                    p(cos),
                                    p(sin),
                                    p(s.aq + offset * cfg.heads * cfg.head_dim * BF16),
                                    p(s.agate + offset * cfg.heads * cfg.head_dim * BF16),
                                    p(s.ak + offset * cfg.kv_heads * cfg.head_dim * BF16),
                                    length,
                                    hq,
                                    hk,
                                    hd,
                                    cfg.rotary_half as i32,
                                    eps,
                                    st,
                                ),
                                "attention prep",
                            )?;
                            check(
                                (cuda::api().cs1_attention_gated)(
                                    p(s.aq + offset * cfg.heads * cfg.head_dim * BF16),
                                    p(s.ak + offset * cfg.kv_heads * cfg.head_dim * BF16),
                                    p(v + offset * w.attn_in * BF16),
                                    ld,
                                    p(s.agate + offset * cfg.heads * cfg.head_dim * BF16),
                                    p(s.ao + offset * cfg.heads * cfg.head_dim * BF16),
                                    length,
                                    hq,
                                    hk,
                                    hd,
                                    (cfg.head_dim as f32).powf(-0.5),
                                    st,
                                ),
                                "gated attention",
                            )?;
                        }
                    }
                    self.gemm_sequences(s, s.ao, &fa.o, s.delta, lengths)?;
                }
            }
            unsafe {
                check(
                    (cuda::api().cs1_add_rms_norm)(
                        p(s.res),
                        p(s.delta),
                        layer.post_norm.ptr,
                        p(s.x),
                        ti,
                        hi,
                        eps,
                        st,
                    ),
                    "post-attention norm",
                )?;
            }
            self.gemm(s, s.x, &layer.gate_up, s.gate_up, t)?;
            unsafe {
                check(
                    (cuda::api().cs1_silu_mul)(
                        p(s.gate_up),
                        (2 * cfg.intermediate) as i32,
                        p(s.act),
                        ti,
                        cfg.intermediate as i32,
                        st,
                    ),
                    "silu mul",
                )?;
            }
            self.gemm_sequences(s, s.act, &layer.down, s.delta, lengths)?;
            let next = self
                .layers
                .get(i + 1)
                .map_or(&self.final_norm, |l| &l.input_norm);
            unsafe {
                check(
                    (cuda::api().cs1_add_rms_norm)(
                        p(s.res),
                        p(s.delta),
                        next.ptr,
                        p(s.x),
                        ti,
                        hi,
                        eps,
                        st,
                    ),
                    "input norm",
                )?;
            }
        }
        Ok(())
    }

    /// The R2d prefix-cache pass: run rows `[qb, tend)` of one prompt with
    /// explicit positions, touching per-layer buffers only for those rows.
    ///
    /// `capture` (qb must be 0): collect post-prep K and pre-GEMM V columns of each
    /// full-attention layer plus the float32 GDN state and the three pre-conv
    /// projection columns of each Gated DeltaNet layer. `prefix` (qb = prefix.len):
    /// seed the window from a captured entry — the conv tail columns are copied
    /// back into place before the convolution, and the prefix K/V columns before
    /// the attention. Everything else is `run()` with shifted row bases, so a
    /// captured+continued pass repeats the one-shot arithmetic token for token
    /// (qb is a multiple of the 64-row GDN chunk; the only remaining divergence
    /// is cuBLASLt's M-shaped plan choice, an f32-accumulation-order ulp).
    fn run_window(
        &self,
        s: &Scratch,
        qb: usize,
        tend: usize,
        custom_positions: bool,
        capture: Option<&mut PrefixState>,
        prefix: Option<&PrefixState>,
    ) -> Result<()> {
        ensure!(
            qb <= tend && (qb == 0) == prefix.is_none() && (qb == 0 || capture.is_none()),
            "prefix window shape"
        );
        let cfg = &self.cfg;
        let st = self.stream;
        let rows = tend - qb;
        let (ti, hi, eps) = (rows as i32, cfg.hidden as i32, cfg.eps);
        let (kd, vd, hv) = (cfg.key_dim(), cfg.value_dim(), cfg.lin_v_heads);
        let (hq, hk, hd) = (cfg.heads as i32, cfg.kv_heads as i32, cfg.head_dim as i32);
        let w = Widths::of(cfg);
        let p = |off: usize| s.at(off);
        let (cos, sin) = if custom_positions {
            (s.custom_cos, s.custom_sin)
        } else {
            (s.cos, s.sin)
        };
        let hb = cfg.hidden * BF16;
        let ob = cfg.heads * cfg.head_dim * BF16; // post-prep q/gate and ao row bytes
        let kvb = cfg.kv_heads * cfg.head_dim * BF16; // k and v row bytes
        let ldb = w.gdn_in; // gdn_in row, elements
        let ldbb = ldb * BF16;
        let ab = w.attn_in * BF16; // attn_in row bytes
        let kb = kd * BF16; // linear q/k row bytes
        let vb = vd * BF16; // linear v row bytes
        let scale = (cfg.head_dim as f32).powf(-0.5);
        let mut fa_i = 0usize;
        let mut la_i = 0usize;
        // SAFETY (every kernel call below): pointers are weights in the arena or
        // scratch buffers laid out for tend tokens at the widths used here; prefix
        // buffers hold exactly the shapes allocated by alloc_prefix(qb).
        unsafe {
            check(
                (cuda::api().cs1_rms_norm)(
                    p(s.res + qb * hb),
                    self.layers[0].input_norm.ptr,
                    p(s.x + qb * hb),
                    ti,
                    hi,
                    eps,
                    st,
                ),
                "input norm",
            )?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Linear(la) => {
                    self.gemm(s, s.x + qb * hb, &la.in_proj, s.gdn_in + qb * ldbb, rows)?;
                    let ld = ldb as i32;
                    let z = s.gdn_in + w.conv * BF16;
                    let b = z + vd * BF16;
                    let a = b + hv * BF16;
                    let (s_in, s_out): (*const c_void, *mut c_void) =
                        match (&prefix, capture.as_deref()) {
                            (Some(pre), _) => {
                                (pre.gdn_state[la_i].at(0).cast_const(), std::ptr::null_mut())
                            }
                            (None, Some(cap)) => (std::ptr::null(), cap.gdn_state[la_i].at(0)),
                            (None, None) => (std::ptr::null(), std::ptr::null_mut()),
                        };
                    unsafe {
                        if let Some(cap) = capture.as_deref() {
                            // conv window tail: the last three pre-conv columns.
                            cuda::copy_dd(
                                cap.conv_tail[la_i].at(0),
                                p(s.gdn_in + (rows - 3) * ldbb),
                                3 * ldbb,
                                st,
                            )?;
                        }
                        if qb > 0 {
                            cuda::copy_dd(
                                p(s.gdn_in + (qb - 3) * ldbb),
                                prefix.unwrap().conv_tail[la_i].at(0),
                                3 * ldbb,
                                st,
                            )?;
                        }
                        // The conv kernel zero-pads its local first three rows.
                        // Include the restored history in its input and discard
                        // those three outputs so continuation rows see all taps.
                        let conv_start = qb.saturating_sub(3);
                        check(
                            (cuda::api().cs1_gdn_conv)(
                                p(s.gdn_in + conv_start * ldbb),
                                ld,
                                la.conv.ptr,
                                p(s.lq + conv_start * kb),
                                p(s.lk + conv_start * kb),
                                p(s.lv + conv_start * vb),
                                (tend - conv_start) as i32,
                                kd as i32,
                                vd as i32,
                                st,
                            ),
                            "gdn conv",
                        )?;
                        check(
                            (cuda::api().cs1_gdn_gates)(
                                p(b + qb * ldbb),
                                p(a + qb * ldbb),
                                ld,
                                la.a_log.ptr,
                                la.dt_bias.ptr,
                                p(s.beta + qb * hv * BF16),
                                p(s.g + qb * hv * F32).cast(),
                                ti,
                                hv as i32,
                                st,
                            ),
                            "gdn gates",
                        )?;
                        check(
                            (cuda::api().cs1_gdn_prefill_x)(
                                p(s.lq + qb * kb),
                                p(s.lk + qb * kb),
                                p(s.lv + qb * vb),
                                p(s.g + qb * hv * F32).cast(),
                                p(s.beta + qb * hv * BF16),
                                p(s.lo + qb * vb),
                                p(s.workspace).cast(),
                                ti,
                                hv as i32,
                                cfg.lin_k_heads as i32,
                                (cfg.lin_k_dim as f32).powf(-0.5),
                                s_in,
                                s_out,
                                st,
                            ),
                            "gdn prefill",
                        )?;
                        check(
                            (cuda::api().cs1_gated_rms_norm)(
                                p(s.lo + qb * vb),
                                p(z + qb * ldbb),
                                ld,
                                la.norm.ptr,
                                p(s.ln + qb * vb),
                                ti,
                                hv as i32,
                                cfg.lin_v_dim as i32,
                                eps,
                                st,
                            ),
                            "gated norm",
                        )?;
                    }
                    self.gemm(s, s.ln + qb * vb, &la.out, s.delta + qb * hb, rows)?;
                    la_i += 1;
                }
                Mixer::Full(fa) => {
                    self.gemm(s, s.x + qb * hb, &fa.qkv, s.attn_in + qb * ab, rows)?;
                    let k = s.attn_in + w.attn_q * BF16;
                    let v = k + kvb;
                    let (q_base, t_flash) = (qb as i32, tend as i32);
                    unsafe {
                        if qb > 0 {
                            let pre = prefix.unwrap();
                            // Restore the prefix V columns and post-prep K rows in place;
                            // the windowed flash then sees the identical k/v layout.
                            cuda::copy2d(p(v), ab, pre.attn_kv[fa_i].1.at(0), kvb, kvb, qb, st)?;
                            cuda::copy_dd(p(s.ak), pre.attn_kv[fa_i].0.at(0), qb * kvb, st)?;
                        }
                        check(
                            (cuda::api().cs1_attn_prep)(
                                p(s.attn_in + qb * ab),
                                p(k + qb * ab),
                                w.attn_in as i32,
                                fa.q_norm.ptr,
                                fa.k_norm.ptr,
                                p(cos),
                                p(sin),
                                p(s.aq + qb * ob),
                                p(s.agate + qb * ob),
                                p(s.ak + qb * kvb),
                                ti,
                                hq,
                                hk,
                                hd,
                                cfg.rotary_half as i32,
                                eps,
                                st,
                            ),
                            "attention prep",
                        )?;
                        if let Some(cap) = capture.as_deref() {
                            // K must be post-prep (normed + rotary); V is the raw column.
                            cuda::copy_dd(cap.attn_kv[fa_i].0.at(0), p(s.ak), tend * kvb, st)?;
                            cuda::copy2d(cap.attn_kv[fa_i].1.at(0), kvb, p(v), ab, kvb, tend, st)?;
                        }
                        check(
                            (cuda::api().cs1_attention_gated_prefix)(
                                p(s.aq),
                                p(s.ak),
                                p(v),
                                w.attn_in as i32,
                                p(s.agate),
                                p(s.ao),
                                t_flash,
                                hq,
                                hk,
                                hd,
                                scale,
                                q_base,
                                st,
                            ),
                            "gated attention",
                        )?;
                    }
                    self.gemm(s, s.ao + qb * ob, &fa.o, s.delta + qb * hb, rows)?;
                    fa_i += 1;
                }
            }
            unsafe {
                check(
                    (cuda::api().cs1_add_rms_norm)(
                        p(s.res + qb * hb),
                        p(s.delta + qb * hb),
                        layer.post_norm.ptr,
                        p(s.x + qb * hb),
                        ti,
                        hi,
                        eps,
                        st,
                    ),
                    "post-attention norm",
                )?;
            }
            self.gemm(
                s,
                s.x + qb * hb,
                &layer.gate_up,
                s.gate_up + qb * 2 * cfg.intermediate * BF16,
                rows,
            )?;
            unsafe {
                check(
                    (cuda::api().cs1_silu_mul)(
                        p(s.gate_up + qb * 2 * cfg.intermediate * BF16),
                        (2 * cfg.intermediate) as i32,
                        p(s.act + qb * cfg.intermediate * BF16),
                        ti,
                        cfg.intermediate as i32,
                        st,
                    ),
                    "silu mul",
                )?;
            }
            self.gemm(
                s,
                s.act + qb * cfg.intermediate * BF16,
                &layer.down,
                s.delta + qb * hb,
                rows,
            )?;
            let next = self
                .layers
                .get(i + 1)
                .map_or(&self.final_norm, |l| &l.input_norm);
            unsafe {
                check(
                    (cuda::api().cs1_add_rms_norm)(
                        p(s.res + qb * hb),
                        p(s.delta + qb * hb),
                        next.ptr,
                        p(s.x + qb * hb),
                        ti,
                        hi,
                        eps,
                        st,
                    ),
                    "input norm",
                )?;
            }
        }
        Ok(())
    }
}
#[cfg(test)]
#[path = "../../../../../tests/qwen3_5/multimodal_graph.rs"]
mod graph_tests;
