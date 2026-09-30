//! The Qwen3.5 text model (the language model of Qwen/Qwen3.5-4B), prefill only: one
//! forward pass over a prompt, returning the final-norm hidden state of the last
//! position. The layer loop, buffers, CUDA graphs and kernel choice live here; the
//! operations are the CUDA kernels in `src/backends/cuda/qwen3_5`.
//!
//! Prompts up to GRAPH_MAX_TOKENS run as a CUDA graph captured for their exact length
//! on first use (the most recent GRAPH_CACHE lengths are kept); longer ones run
//! eagerly. A graph queues the same kernels with the same GEMM algorithms as the eager
//! pass of that length, so both give bitwise identical results. GEMM algorithms are
//! timed at startup for the lengths in TUNE_ROWS; other lengths borrow a nearby one
//! (see gemm.cu). Candidates within a few percent of each other can swap between
//! restarts, which changes the last bits of the results.
//!
//! The order of operations follows `modeling_qwen3_5.py`, and so do the points where
//! it rounds to bfloat16, except inside attention and the Gated DeltaNet prefill (see
//! their kernels). Text prompts use one position per
//! token, so the multimodal rotary sections all get the same position and the
//! rotary embedding is the plain one.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value as Json;

use crate::cuda::{self, DeviceBuffer, Graph, Stream, check};

const ALIGN: usize = 256;
const BF16: usize = 2;
const F32: usize = 4;
const GEMM_WORKSPACE: usize = 32 << 20;

const GRAPH_MAX_TOKENS: usize = 2048;
const GRAPH_CACHE: usize = 128;

/// Prompt lengths the GEMM algorithms are timed for, at most twice apart, so that a
/// length up to the last one borrows the choice for at most twice its length and longer
/// ones borrow the last one's. Past these, cuBLASLt's first choice for long prompts is
/// an older, half-rate tensor-core kernel on sm_89.
const TUNE_ROWS: &[usize] = &[
    64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 1024, 1536, 2048, 4096, 8192,
    16384,
];

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
        let prefix = ["model.language_model.", "model."]
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
        ] = offsets;
        // Rotary tables close to how Qwen3_5TextRotaryEmbedding builds them: inv_freq and
        // freqs = inv_freq * position in float32, cos and sin rounded to bfloat16. Here
        // cos and sin are taken in float64 on the host rather than in float32 on the
        // GPU, so a few of the rounded values can differ by one bfloat16 step.
        let half = cfg.rotary_half;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / (cfg.rope_theta as f32).powf((2 * i) as f32 / (2 * half) as f32))
            .collect();
        let mut cos_t = Vec::with_capacity(cap * half * BF16);
        let mut sin_t = Vec::with_capacity(cap * half * BF16);
        for pos in 0..cap {
            for &f in &inv {
                let freq = (f * pos as f32) as f64;
                cos_t.extend(half::bf16::from_f32(freq.cos() as f32).to_le_bytes());
                sin_t.extend(half::bf16::from_f32(freq.sin() as f32).to_le_bytes());
            }
        }
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
        })
    }

    fn at(&self, offset: usize) -> *mut c_void {
        self.buf.at(offset)
    }
}

pub struct Model {
    pub cfg: Config,
    _weights: Weights,
    embed: Tensor,
    final_norm: Tensor,
    layers: Vec<Layer>,
    stream: Stream,
    gemm: *mut c_void,
    /// Graphs by prompt length, least recently used first in `graph_lru`; they point
    /// into `graph_scratch`, which is never reallocated.
    graphs: HashMap<usize, Graph>,
    graph_lru: VecDeque<usize>,
    graph_scratch: Option<Scratch>,
    /// For prompts longer than `graph_scratch` holds; grows as needed.
    eager_scratch: Option<Scratch>,
}

// SAFETY: the raw pointers are device addresses and a cuBLASLt handle owned by the
// model; the engine runs one forward pass at a time behind a mutex.
unsafe impl Send for Model {}

impl Drop for Model {
    fn drop(&mut self) {
        self.graphs.clear();
        // SAFETY: created by cs1_gemm_create and not destroyed before.
        unsafe { (cuda::api().cs1_gemm_destroy)(self.gemm) };
    }
}

impl Model {
    /// Load the weights, allocate the graph buffers and time the GEMM algorithms.
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
        let mut model = Self {
            cfg,
            _weights: weights,
            embed,
            final_norm,
            layers,
            stream,
            gemm,
            graphs: HashMap::new(),
            graph_lru: VecDeque::new(),
            graph_scratch: None,
            eager_scratch: None,
        };
        let s = Scratch::new(&model.cfg, GRAPH_MAX_TOKENS, stream)?;
        // one eager pass over the longest prompt fills every buffer and sets up the kernels
        let zeros = vec![0u8; GRAPH_MAX_TOKENS * 4];
        // SAFETY: the ids buffer holds GRAPH_MAX_TOKENS int32 values.
        unsafe { cuda::upload(s.at(s.ids), &zeros, stream)? };
        model.run(&s, GRAPH_MAX_TOKENS)?;
        cuda::synchronize(stream)?;
        // the longer lengths are timed in a temporary buffer
        let big = Scratch::new(&model.cfg, *TUNE_ROWS.last().unwrap(), stream)?;
        for &m in TUNE_ROWS {
            model.tune(if m <= GRAPH_MAX_TOKENS { &s } else { &big }, m)?;
        }
        drop(big);
        // SAFETY: frees only the tuning buffers.
        unsafe { (cuda::api().cs1_gemm_tune_done)(gemm) };
        model.graph_scratch = Some(s);
        Ok(model)
    }

    /// Time the GEMM algorithms for `m` rows on the first layer of each kind.
    fn tune(&self, s: &Scratch, m: usize) -> Result<()> {
        let layer = &self.layers[0];
        let mut jobs = vec![
            (s.x, &layer.gate_up, s.gate_up),
            (s.act, &layer.down, s.delta),
        ];
        if let Some(Mixer::Linear(la)) = self
            .layers
            .iter()
            .map(|l| &l.mixer)
            .find(|m| matches!(m, Mixer::Linear(_)))
        {
            jobs.extend([(s.x, &la.in_proj, s.gdn_in), (s.ln, &la.out, s.delta)]);
        }
        if let Some(Mixer::Full(fa)) = self
            .layers
            .iter()
            .map(|l| &l.mixer)
            .find(|m| matches!(m, Mixer::Full(_)))
        {
            jobs.extend([(s.x, &fa.qkv, s.attn_in), (s.ao, &fa.o, s.delta)]);
        }
        for (x, w, y) in jobs {
            let (n, k) = (w.shape[0] as i32, w.shape[1] as i32);
            // SAFETY: x and y are scratch buffers sized for at least m rows of this shape.
            let code = unsafe {
                (cuda::api().cs1_gemm_tune)(
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
            };
            check(code, "gemm tuning")?;
        }
        Ok(())
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

    /// The final-norm hidden state at the last position, as float32.
    /// Runs from the graph for this length when graphs are on and it fits, else eagerly.
    pub fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        let t = ids.len();
        ensure!(t > 0, "empty prompt");
        let (vocab, h) = (self.embed.shape[0], self.cfg.hidden);
        ensure!(
            ids.iter().all(|&i| (i as usize) < vocab),
            "token id outside the vocabulary"
        );
        cuda::set_device(0)?;
        let in_graph_scratch = self.graph_scratch.as_ref().is_some_and(|s| t <= s.cap);
        if !in_graph_scratch && self.eager_scratch.as_ref().is_none_or(|s| t > s.cap) {
            self.eager_scratch = None;
            self.eager_scratch = Some(Scratch::new(
                &self.cfg,
                t.next_multiple_of(1024),
                self.stream,
            )?);
        }
        let s = if in_graph_scratch {
            self.graph_scratch.as_ref()
        } else {
            self.eager_scratch.as_ref()
        }
        .unwrap();
        let ids32: Vec<u8> = ids.iter().flat_map(|&i| (i as i32).to_le_bytes()).collect();
        // SAFETY: the ids buffer holds at least t int32 values.
        unsafe { cuda::upload(s.at(s.ids), &ids32, self.stream)? };
        if in_graph_scratch {
            self.graph_for(t)?;
            self.graphs[&t].launch(self.stream)?;
        } else {
            self.run(s, t)?;
        }
        let s = if in_graph_scratch {
            self.graph_scratch.as_ref()
        } else {
            self.eager_scratch.as_ref()
        }
        .unwrap();
        let mut last = vec![0u8; h * BF16];
        // SAFETY: x holds at least t rows of the hidden size.
        unsafe { cuda::download(&mut last, s.at(s.x + (t - 1) * h * BF16), self.stream)? };
        let (pairs, _) = last.as_chunks::<2>();
        Ok(pairs
            .iter()
            .map(|&b| half::bf16::from_le_bytes(b).to_f32())
            .collect())
    }

    /// Make sure a graph for `t` tokens exists (capturing it if needed) and mark it
    /// as the most recently used, dropping the least recently used beyond the cache.
    fn graph_for(&mut self, t: usize) -> Result<()> {
        if self.graphs.contains_key(&t) {
            self.graph_lru.retain(|&x| x != t);
        } else {
            while self.graphs.len() >= GRAPH_CACHE {
                let Some(old) = self.graph_lru.pop_front() else {
                    break;
                };
                self.graphs.remove(&old);
            }
            let s = self.graph_scratch.as_ref().context("no graph buffers")?;
            let graph = Graph::capture(self.stream, || self.run(s, t))?;
            self.graphs.insert(t, graph);
        }
        self.graph_lru.push_back(t);
        Ok(())
    }

    /// Queue one forward pass over the first `t` ids in `s` (nothing else is queued,
    /// so it can be captured). The final-norm hidden states end up in `s.x`.
    fn run(&self, s: &Scratch, t: usize) -> Result<()> {
        let cfg = &self.cfg;
        let st = self.stream;
        let (ti, hi, eps) = (t as i32, cfg.hidden as i32, cfg.eps);
        let (kd, vd, hv) = (cfg.key_dim(), cfg.value_dim(), cfg.lin_v_heads);
        let (hq, hk, hd) = (cfg.heads as i32, cfg.kv_heads as i32, cfg.head_dim as i32);
        let w = Widths::of(cfg);
        let p = |off: usize| s.at(off);
        // SAFETY (every kernel call below): the pointers are weights in the arena or
        // scratch buffers laid out for at least t tokens with the widths used here.
        unsafe {
            check(
                (cuda::api().cs1_embed)(p(s.ids).cast(), self.embed.ptr, p(s.res), ti, hi, st),
                "embed",
            )?;
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
                        check(
                            (cuda::api().cs1_gdn_conv)(
                                p(s.gdn_in),
                                ld,
                                la.conv.ptr,
                                p(s.lq),
                                p(s.lk),
                                p(s.lv),
                                ti,
                                kd as i32,
                                vd as i32,
                                st,
                            ),
                            "gdn conv",
                        )?;
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
                        check(
                            (cuda::api().cs1_gdn_prefill)(
                                p(s.lq),
                                p(s.lk),
                                p(s.lv),
                                p(s.g).cast(),
                                p(s.beta),
                                p(s.lo),
                                p(s.workspace).cast(),
                                ti,
                                hv as i32,
                                cfg.lin_k_heads as i32,
                                (cfg.lin_k_dim as f32).powf(-0.5),
                                st,
                            ),
                            "gdn prefill",
                        )?;
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
                    self.gemm(s, s.ln, &la.out, s.delta, t)?;
                }
                Mixer::Full(fa) => {
                    self.gemm(s, s.x, &fa.qkv, s.attn_in, t)?;
                    let ld = w.attn_in as i32;
                    let k = s.attn_in + w.attn_q * BF16;
                    let v = k + cfg.kv_heads * cfg.head_dim * BF16;
                    unsafe {
                        check(
                            (cuda::api().cs1_attn_prep)(
                                p(s.attn_in),
                                p(k),
                                ld,
                                fa.q_norm.ptr,
                                fa.k_norm.ptr,
                                p(s.cos),
                                p(s.sin),
                                p(s.aq),
                                p(s.agate),
                                p(s.ak),
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
                        check(
                            (cuda::api().cs1_attention)(
                                p(s.aq),
                                p(s.ak),
                                p(v),
                                ld,
                                p(s.ao),
                                ti,
                                hq,
                                hk,
                                hd,
                                (cfg.head_dim as f32).powf(-0.5),
                                st,
                            ),
                            "attention",
                        )?;
                        check(
                            (cuda::api().cs1_sigmoid_gate)(
                                p(s.ao),
                                p(s.agate),
                                t * cfg.heads * cfg.head_dim,
                                st,
                            ),
                            "attention gate",
                        )?;
                    }
                    self.gemm(s, s.ao, &fa.o, s.delta, t)?;
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
            self.gemm(s, s.act, &layer.down, s.delta, t)?;
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
}
