//! Single-GPU Laya executor. All CUDA buffers and graphs remain on the owning worker thread.
use crate::{config::Config, preprocess::Batch, weights::Weights};
use anyhow::{Context, Result, ensure};
use half::bf16;
use omni_cuda::{Buffer, Cuda, Graph, Ptr};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::Path,
};
const D: usize = 1024;
pub type ModelOutput = (Vec<Vec<f32>>, Vec<[f32; 2]>);
fn bytes16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn bytes32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i64bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn decode_bf16(v: &[u8]) -> Vec<f32> {
    v.as_chunks::<2>()
        .0
        .iter()
        .map(|x| bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
        .collect()
}

type Linear = unsafe extern "C" fn(Ptr, Ptr, Ptr, Ptr, Ptr, i32, i32, i32, i32, Ptr) -> i32;
struct Blas {
    cuda: Cuda,
    p: Ptr,
}
impl Blas {
    fn new(cuda: &Cuda) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        unsafe {
            let f =
                cuda.symbol::<unsafe extern "C" fn(*mut Ptr, Ptr) -> i32>(b"laya_blas_create\0")?;
            cuda.check(f(&mut p, cuda.stream()))?;
        }
        Ok(Self {
            cuda: cuda.clone(),
            p,
        })
    }
    #[allow(clippy::too_many_arguments)] // Mirrors the checked GEMM boundary.
    fn linear(
        &self,
        a: &Buffer,
        w: &Buffer,
        bias: &Buffer,
        out: &Buffer,
        rows: usize,
        n: usize,
        k: usize,
        gelu: bool,
    ) -> Result<()> {
        ensure!(
            a.bytes() >= rows * k * 2
                && w.bytes() == n * k * 2
                && bias.bytes() == n * 2
                && out.bytes() >= rows * n * 2,
            "linear buffer shape mismatch"
        );
        unsafe {
            let f = self.cuda.symbol::<Linear>(b"laya_linear\0")?;
            self.cuda.check(f(
                self.p,
                a.ptr(),
                w.ptr(),
                bias.ptr(),
                out.ptr(),
                rows as i32,
                n as i32,
                k as i32,
                gelu as i32,
                self.cuda.stream(),
            ))
        }
    }
}
impl Drop for Blas {
    fn drop(&mut self) {
        let _ = self.cuda.sync();
        unsafe {
            if let Ok(f) = self
                .cuda
                .symbol::<unsafe extern "C" fn(Ptr) -> i32>(b"laya_blas_free\0")
            {
                f(self.p);
            }
        }
    }
}

struct Workspace {
    // Destruction order matters: destroy the graph before any referenced allocation.
    graph: Option<Graph>,
    b: usize,
    l: usize,
    bytes: usize,
    ids: Buffer,
    lens: Buffer,
    types: Buffer,
    x: Buffer,
    y: Buffer,
    qkv: Buffer,
    o: Buffer,
    g: Buffer,
    ff: Buffer,
    indices: Buffer,
    offsets: Buffer,
    markers: Buffer,
    scored: Buffer,
    logits: Buffer,
    features: Buffer,
    action_hidden: Buffer,
    actions: Buffer,
}
impl Workspace {
    fn new(c: &Cuda, b: usize, l: usize) -> Result<Self> {
        let m = b * l;
        let mut bytes = 0;
        let mut alloc = |n| {
            bytes += n;
            c.alloc(n)
        };
        Ok(Self {
            graph: None,
            b,
            l,
            ids: alloc(m * 8)?,
            lens: alloc(b * 4)?,
            types: alloc(b * 8)?,
            x: alloc(m * D * 4)?,
            y: alloc(m * D * 2)?,
            qkv: alloc(m * D * 6)?,
            o: alloc(m * D * 2)?,
            g: alloc(m * 2624 * 2)?,
            ff: alloc(m * 4096 * 2)?,
            indices: alloc(2048 * 4)?,
            offsets: alloc(17 * 4)?,
            markers: alloc(2048 * D * 2)?,
            scored: alloc(2048 * D * 2)?,
            logits: alloc(2048 * 2)?,
            features: alloc(16 * 1028 * 2)?,
            action_hidden: alloc(16 * 256 * 2)?,
            actions: alloc(16 * 2 * 2)?,
            bytes,
        })
    }
}

pub struct Model {
    pub config: Config,
    cuda: Cuda,
    blas: Blas,
    weights: HashMap<String, Buffer>,
    cache: VecDeque<Workspace>,
    graphs: bool,
    original_rope: bool,
}
impl Drop for Model {
    fn drop(&mut self) {
        let _ = self.cuda.sync();
        self.cache.clear();
    }
}
impl Model {
    pub fn load(
        checkpoint: &Path,
        bundle: &Path,
        graphs: bool,
        original_rope: bool,
    ) -> Result<Self> {
        let config = Config::load(checkpoint)?;
        validate_bundle(checkpoint, bundle)?;
        // SAFETY: bundle is an explicit trusted build artifact supplied by the operator.
        let cuda = unsafe { Cuda::load(&bundle.join("liblaya_cuda.so")) }?;
        let blas = Blas::new(&cuda)?;
        let source = Weights::open(&checkpoint.join("model.safetensors"))?;
        let mut weights = HashMap::new();
        let mut add = |name: &str, shape: &[usize], dtype: &str| -> Result<()> {
            let data = match dtype {
                "f32" => bytes32(&source.f32(name, shape)?),
                "f16" => bytes16(&source.f16(name, shape)?),
                _ => bytes16(&source.bf16(name, shape)?),
            };
            weights.insert(
                name.to_owned(),
                cuda.upload(&data)
                    .with_context(|| format!("upload {name}"))?,
            );
            Ok(())
        };
        add(
            "encoder.embeddings.tok_embeddings.weight",
            &[50368, D],
            "f16",
        )?;
        add("encoder.embeddings.norm.weight", &[D], "f32")?;
        add("encoder.final_norm.weight", &[D], "f32")?;
        for i in 0..28 {
            let p = format!("encoder.layers.{i}");
            if i > 0 {
                add(&format!("{p}.attn_norm.weight"), &[D], "f32")?;
            }
            add(&format!("{p}.mlp_norm.weight"), &[D], "f32")?;
            for (name, shape) in [
                ("attn.Wqkv.weight", vec![3 * D, D]),
                ("attn.Wo.weight", vec![D, D]),
                ("mlp.Wi.weight", vec![5248, D]),
                ("mlp.Wo.weight", vec![D, 2624]),
            ] {
                add(&format!("{p}.{name}"), &shape, "bf16")?;
            }
        }
        add("type_emb.weight", &[3, D], "bf16")?;
        for i in 0..2 {
            let p = format!("head.layers.{i}");
            for n in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"] {
                add(&format!("{p}.{n}"), &[D], "f32")?;
            }
            for (n, rows, cols) in [
                ("self_attn.in_proj_weight", 3 * D, D),
                ("self_attn.out_proj.weight", D, D),
                ("linear1.weight", 4 * D, D),
                ("linear2.weight", D, 4 * D),
            ] {
                add(&format!("{p}.{n}"), &[rows, cols], "bf16")?;
            }
            for (n, len) in [
                ("self_attn.in_proj_bias", 3 * D),
                ("self_attn.out_proj.bias", D),
                ("linear1.bias", 4 * D),
                ("linear2.bias", D),
            ] {
                add(&format!("{p}.{n}"), &[len], "f32")?;
            }
        }
        for n in ["scorer.0.weight", "scorer.0.bias"] {
            add(n, &[D], "f32")?;
        }
        for (p, n, k) in [
            ("scorer.1", D, D),
            ("scorer.3", 1, D),
            ("act_head.0", 256, 1028),
            ("act_head.2", 2, 256),
        ] {
            add(&format!("{p}.weight"), &[n, k], "bf16")?;
            add(&format!("{p}.bias"), &[n], "bf16")?;
        }
        for n in [D, 3 * D, 4 * D] {
            weights.insert(format!("zeros.{n}"), cuda.upload(&vec![0; n * 4])?);
        }
        for kind in ["full", "local"] {
            for part in ["cos", "sin"] {
                let key = format!("rope_{kind}_{part}");
                let data = fs::read(bundle.join(format!("{key}.f32")))?;
                ensure!(data.len() == 512 * 32 * 4, "invalid rotary table size");
                weights.insert(key, cuda.upload(&data)?);
            }
        }
        Ok(Self {
            config,
            cuda,
            blas,
            weights,
            cache: VecDeque::new(),
            graphs,
            original_rope,
        })
    }
    fn w(&self, n: &str) -> &Buffer {
        &self.weights[n]
    }
    fn encode(&self, s: &Workspace) -> Result<()> {
        let (b, l) = (s.b, s.l);
        let z = self.w("zeros.1024").ptr();
        let attention = |label: &str| {
            if l == 512 && (b == 1 || b == 4) {
                format!("attn_{label}_b{b}_l512")
            } else {
                format!("attn_{label}")
            }
        };
        // All pointers refer to checked fixed-shape, resident allocations in this worker.
        let call = |name: &str, args: &[Ptr]| unsafe { self.cuda.launch(name, args, b, l) };
        call(
            "embed",
            &[
                s.ids.ptr(),
                self.w("encoder.embeddings.tok_embeddings.weight").ptr(),
                self.w("encoder.embeddings.norm.weight").ptr(),
                s.x.ptr(),
                s.y.ptr(),
            ],
        )?;
        self.dump("embedding", &s.x, false)?;
        for i in 0..28 {
            let p = format!("encoder.layers.{i}");
            let w = |n: &str| self.w(&format!("{p}.{n}")).ptr();
            call(
                "qkv",
                &[
                    s.y.ptr(),
                    w("attn.Wqkv.weight"),
                    self.w("zeros.3072").ptr(),
                    s.qkv.ptr(),
                ],
            )?;
            let kind = if i % 3 == 0 { "full" } else { "local" };
            call(
                if self.original_rope {
                    "rope_original"
                } else {
                    "rope"
                },
                &[
                    s.qkv.ptr(),
                    self.w(&format!("rope_{kind}_cos")).ptr(),
                    self.w(&format!("rope_{kind}_sin")).ptr(),
                ],
            )?;
            call(
                &attention(if i % 3 == 0 { "full" } else { "local" }),
                &[s.qkv.ptr(), s.lens.ptr(), s.o.ptr()],
            )?;
            call("out", &[s.o.ptr(), w("attn.Wo.weight"), z, s.y.ptr()])?;
            call(
                "addln",
                &[s.x.ptr(), s.y.ptr(), w("mlp_norm.weight"), z, s.y.ptr()],
            )?;
            call("geglu", &[s.y.ptr(), w("mlp.Wi.weight"), s.g.ptr()])?;
            call("down", &[s.g.ptr(), w("mlp.Wo.weight"), z, s.y.ptr()])?;
            let next = if i < 27 {
                self.w(&format!("encoder.layers.{}.attn_norm.weight", i + 1))
            } else {
                self.w("encoder.final_norm.weight")
            };
            call("addln", &[s.x.ptr(), s.y.ptr(), next.ptr(), z, s.y.ptr()])?;
            if [0, 1, 2, 27].contains(&i) {
                self.dump(&format!("encoder{i}_residual"), &s.x, false)?;
                self.dump(&format!("encoder{i}_normalized"), &s.y, true)?;
            }
        }
        call(
            "type",
            &[
                s.y.ptr(),
                self.w("type_emb.weight").ptr(),
                s.types.ptr(),
                s.x.ptr(),
            ],
        )?;
        for i in 0..2 {
            let p = format!("head.layers.{i}");
            let w = |n: &str| self.w(&format!("{p}.{n}")).ptr();
            call(
                "ln_bias",
                &[
                    s.x.ptr(),
                    s.y.ptr(),
                    w("norm1.weight"),
                    w("norm1.bias"),
                    s.y.ptr(),
                ],
            )?;
            call(
                "head_in",
                &[
                    s.y.ptr(),
                    w("self_attn.in_proj_weight"),
                    w("self_attn.in_proj_bias"),
                    s.qkv.ptr(),
                ],
            )?;
            call(&attention("full"), &[s.qkv.ptr(), s.lens.ptr(), s.o.ptr()])?;
            call(
                "head_out",
                &[
                    s.o.ptr(),
                    w("self_attn.out_proj.weight"),
                    w("self_attn.out_proj.bias"),
                    s.y.ptr(),
                ],
            )?;
            call(
                "addln_bias",
                &[
                    s.x.ptr(),
                    s.y.ptr(),
                    w("norm2.weight"),
                    w("norm2.bias"),
                    s.y.ptr(),
                ],
            )?;
            call(
                "ffn1",
                &[
                    s.y.ptr(),
                    w("linear1.weight"),
                    w("linear1.bias"),
                    s.ff.ptr(),
                ],
            )?;
            call(
                "ffn2",
                &[
                    s.ff.ptr(),
                    w("linear2.weight"),
                    w("linear2.bias"),
                    s.y.ptr(),
                ],
            )?;
            call("residual", &[s.x.ptr(), s.y.ptr()])?;
        }
        Ok(())
    }
    fn dump(&self, name: &str, buffer: &Buffer, bf: bool) -> Result<()> {
        if !self.graphs
            && let Ok(dir) = std::env::var("LAYA_DUMP_DIR")
        {
            fs::create_dir_all(&dir)?;
            let data = buffer.read(buffer.bytes())?;
            fs::write(
                Path::new(&dir).join(format!("{name}.f32")),
                if bf {
                    bytes32(&decode_bf16(&data))
                } else {
                    data
                },
            )?;
        }
        Ok(())
    }
    pub fn infer(&mut self, batch: &Batch) -> Result<ModelOutput> {
        if batch.questions.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        ensure!(
            batch.b <= 16
                && batch.l <= 512
                && batch.input_ids.iter().all(|i| *i >= 0 && *i < 50368),
            "invalid packed input"
        );
        let n = batch.questions.len();
        ensure!(
            batch.b == n.next_power_of_two()
                && batch.l >= 16
                && batch.l.is_multiple_of(16)
                && batch.input_ids.len() == batch.b * batch.l
                && batch.lens.len() == batch.b
                && batch.qtypes.len() == batch.b,
            "inconsistent packed dimensions"
        );
        for i in 0..batch.b {
            ensure!((0..=2).contains(&batch.qtypes[i]), "invalid question type");
            if i < n {
                ensure!(
                    batch.lens[i] > 0
                        && batch.lens[i] as usize <= batch.l
                        && !batch.questions[i].markers.is_empty()
                        && batch.questions[i]
                            .markers
                            .iter()
                            .all(|m| *m < batch.lens[i] as usize),
                    "invalid sequence/marker bounds"
                );
            } else {
                ensure!(batch.lens[i] == 0, "dummy row must have zero length");
            }
        }
        ensure!(
            batch
                .questions
                .iter()
                .map(|q| q.markers.len())
                .sum::<usize>()
                <= 2048,
            "too many markers"
        );
        let found = self
            .cache
            .iter()
            .position(|s| s.b == batch.b && s.l == batch.l);
        let mut s = if let Some(i) = found {
            self.cache.remove(i).unwrap()
        } else {
            Workspace::new(&self.cuda, batch.b, batch.l)?
        };
        s.ids.write(&i64bytes(&batch.input_ids))?;
        s.lens.write(&i32bytes(&batch.lens))?;
        s.types.write(&i64bytes(&batch.qtypes))?;
        if self.graphs {
            if s.graph.is_none() {
                self.encode(&s)?;
                self.encode(&s)?;
                self.cuda.sync()?;
                // Graph uses weights owned by self and allocations owned by s; self clears cache first on Drop.
                s.graph = Some(unsafe { self.cuda.capture(|| self.encode(&s)) }?);
            }
            s.graph.as_ref().unwrap().replay()?;
        } else {
            self.encode(&s)?;
        }
        if let Ok(path) = std::env::var("LAYA_DUMP_HIDDEN") {
            fs::write(path, s.x.read(batch.b * batch.l * D * 4)?)?;
        }
        let mut indices = Vec::new();
        let mut offsets = vec![0i32];
        for (i, q) in batch.questions.iter().enumerate() {
            for &m in &q.markers {
                ensure!(m < batch.l, "marker outside sequence");
                indices.push((i * batch.l + m) as i32);
            }
            offsets.push(indices.len() as i32);
        }
        let rows = indices.len();
        ensure!(rows <= 2048, "too many markers");
        s.indices.write(&i32bytes(&indices))?;
        s.offsets.write(&i32bytes(&offsets))?;
        unsafe {
            self.cuda.launch_rows(
                "gather",
                &[
                    s.x.ptr(),
                    s.indices.ptr(),
                    self.w("scorer.0.weight").ptr(),
                    self.w("scorer.0.bias").ptr(),
                    s.markers.ptr(),
                ],
                1,
                1,
                rows,
            )?;
        }
        self.blas.linear(
            &s.markers,
            self.w("scorer.1.weight"),
            self.w("scorer.1.bias"),
            &s.scored,
            rows,
            D,
            D,
            true,
        )?;
        self.blas.linear(
            &s.scored,
            self.w("scorer.3.weight"),
            self.w("scorer.3.bias"),
            &s.logits,
            rows,
            1,
            D,
            false,
        )?;
        let n = batch.questions.len();
        unsafe {
            self.cuda.launch_rows(
                "features",
                &[s.x.ptr(), s.logits.ptr(), s.offsets.ptr(), s.features.ptr()],
                n,
                batch.l,
                rows,
            )?;
        }
        self.blas.linear(
            &s.features,
            self.w("act_head.0.weight"),
            self.w("act_head.0.bias"),
            &s.action_hidden,
            n,
            256,
            1028,
            true,
        )?;
        self.blas.linear(
            &s.action_hidden,
            self.w("act_head.2.weight"),
            self.w("act_head.2.bias"),
            &s.actions,
            n,
            2,
            256,
            false,
        )?;
        let raw = decode_bf16(&s.logits.read(rows * 2)?);
        let acts = decode_bf16(&s.actions.read(n * 4)?);
        let logits = offsets
            .windows(2)
            .map(|w| raw[w[0] as usize..w[1] as usize].to_vec())
            .collect();
        let actions = acts
            .as_chunks::<2>()
            .0
            .iter()
            .map(|x| [x[0], x[1]])
            .collect();
        while self.cache.len() >= 4
            || self.cache.iter().map(|s| s.bytes).sum::<usize>() + s.bytes > 512 * 1024 * 1024
        {
            if self.cache.pop_front().is_none() {
                break;
            }
        }
        self.cache.push_back(s);
        Ok((logits, actions))
    }
}

fn validate_bundle(checkpoint: &Path, bundle: &Path) -> Result<()> {
    let tables: serde_json::Value = serde_json::from_slice(&fs::read(bundle.join("tables.json"))?)?;
    let build: serde_json::Value =
        serde_json::from_slice(&fs::read(bundle.join("build-manifest.json"))?)?;
    ensure!(
        tables["abi"] == 1
            && tables["laya"] == "0.3.20"
            && tables["hidden_size"] == 1024
            && tables["head_dim"] == 64
            && tables["max_len"] == 512
            && build["abi"] == 1
            && build["arch"] == "sm_90a",
        "unsupported CUDA bundle"
    );
    let check = |path: std::path::PathBuf, expected: &serde_json::Value| -> Result<()> {
        let hash = format!("{:x}", Sha256::digest(fs::read(&path)?));
        ensure!(
            expected.as_str() == Some(hash.as_str()),
            "bundle hash mismatch: {}",
            path.display()
        );
        Ok(())
    };
    for name in ["rl_agent_config.json", "encoder/config.json"] {
        check(checkpoint.join(name), &tables["config_sha256"][name])?;
    }
    for name in [
        "rope_full_cos.f32",
        "rope_full_sin.f32",
        "rope_local_cos.f32",
        "rope_local_sin.f32",
    ] {
        check(bundle.join(name), &tables["tables"][name])?;
    }
    check(bundle.join("liblaya_cuda.so"), &build["library_sha256"])?;
    Ok(())
}
