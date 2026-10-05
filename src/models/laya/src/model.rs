//! Single-GPU Laya executor. All CUDA buffers and graphs remain on the owning worker thread.
use crate::{config::Config, packing::Batch, weights::Weights};
use anyhow::{Context, Result, ensure};
use half::bf16;
use omni_cuda::{Buffer, Cuda, Graph, Kernel, Ptr};
use std::{
    collections::{HashMap, VecDeque},
    fs,
    path::Path,
};
const D: usize = 1024;
pub type ModelOutput = (Vec<Vec<f32>>, Vec<[f32; 2]>);
fn storage_dtype(name: &str) -> &'static str {
    if name == "encoder.embeddings.tok_embeddings.weight" {
        "f16"
    } else if name.starts_with("scorer.0.")
        || name == "temperature"
        || name.contains("norm")
        || (name.starts_with("head.layers.") && name.ends_with("bias"))
    {
        "f32"
    } else {
        "bf16"
    }
}
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

#[derive(Clone, Copy)]
enum Scratch {
    Ids,
    Lens,
    Types,
    X,
    Y,
    Qkv,
    O,
    G,
    Ff,
}
impl Scratch {
    fn buffer(self, s: &Workspace) -> &Buffer {
        match self {
            Self::Ids => &s.ids,
            Self::Lens => &s.lens,
            Self::Types => &s.types,
            Self::X => &s.x,
            Self::Y => &s.y,
            Self::Qkv => &s.qkv,
            Self::O => &s.o,
            Self::G => &s.g,
            Self::Ff => &s.ff,
        }
    }
}
#[derive(Clone, Copy)]
enum Argument {
    Weight(Ptr),
    Scratch(Scratch),
}
impl Argument {
    fn pointer(self, s: &Workspace) -> Ptr {
        match self {
            Self::Weight(p) => p,
            Self::Scratch(slot) => slot.buffer(s).ptr(),
        }
    }
}
enum SelectedKernel {
    Fixed(Kernel),
    Attention([Option<Kernel>; 3]),
}
impl SelectedKernel {
    fn resolve(cuda: &Cuda, name: &str) -> Result<Self> {
        if name == "attn_full" || name == "attn_local" {
            // Sparse trusted bundles may omit wrappers for shapes never requested.
            // Preserve their eager usability and fail when a missing shape is used.
            Ok(Self::Attention([
                Some(cuda.resolve(name)?),
                cuda.resolve(&format!("{name}_b1_l512")).ok(),
                cuda.resolve(&format!("{name}_b4_l512")).ok(),
            ]))
        } else {
            Ok(Self::Fixed(cuda.resolve(name)?))
        }
    }
    fn select(&self, b: usize, l: usize) -> Result<&Kernel> {
        match self {
            Self::Fixed(k) => Ok(k),
            Self::Attention(k) => k[if l == 512 && b == 1 {
                1
            } else if l == 512 && b == 4 {
                2
            } else {
                0
            }]
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("missing specialized attention kernel for B={b}, L={l}")
            }),
        }
    }
}
enum Step {
    Launch {
        kernel: SelectedKernel,
        args: [Argument; 5],
        count: usize,
    },
    Dump {
        name: String,
        slot: Scratch,
        bf: bool,
    },
}

fn prepare_encoder(
    cuda: &Cuda,
    weights: &HashMap<String, Buffer>,
    original_rope: bool,
) -> Result<Vec<Step>> {
    let steps = std::cell::RefCell::new(Vec::new());
    let weight = |name: &str| Argument::Weight(weights[name].ptr());
    let call = |name: &str, args: &[Argument]| -> Result<()> {
        ensure!(args.len() <= 5, "encoder argument count");
        let mut resolved = [Argument::Weight(std::ptr::null_mut()); 5];
        resolved[..args.len()].copy_from_slice(args);
        steps.borrow_mut().push(Step::Launch {
            kernel: SelectedKernel::resolve(cuda, name)?,
            args: resolved,
            count: args.len(),
        });
        Ok(())
    };
    let dump = |name: &str, slot: Scratch, bf: bool| {
        steps.borrow_mut().push(Step::Dump {
            name: name.to_owned(),
            slot,
            bf,
        });
        Ok::<_, anyhow::Error>(())
    };
    let z = weight("zeros.1024");
    let attention = |label: &str| format!("attn_{label}");
    call(
        "embed",
        &[
            Argument::Scratch(Scratch::Ids),
            weight("encoder.embeddings.tok_embeddings.weight"),
            weight("encoder.embeddings.norm.weight"),
            Argument::Scratch(Scratch::X),
            Argument::Scratch(Scratch::Y),
        ],
    )?;
    dump("embedding", Scratch::X, false)?;
    for i in 0..28 {
        let p = format!("encoder.layers.{i}");
        let w = |n: &str| weight(&format!("{p}.{n}"));
        call(
            "qkv",
            &[
                Argument::Scratch(Scratch::Y),
                w("attn.Wqkv.weight"),
                weight("zeros.3072"),
                Argument::Scratch(Scratch::Qkv),
            ],
        )?;
        let kind = if i % 3 == 0 { "full" } else { "local" };
        call(
            if original_rope {
                "rope_original"
            } else {
                "rope"
            },
            &[
                Argument::Scratch(Scratch::Qkv),
                weight(&format!("rope_{kind}_cos")),
                weight(&format!("rope_{kind}_sin")),
            ],
        )?;
        call(
            &attention(if i % 3 == 0 { "full" } else { "local" }),
            &[
                Argument::Scratch(Scratch::Qkv),
                Argument::Scratch(Scratch::Lens),
                Argument::Scratch(Scratch::O),
            ],
        )?;
        call(
            "out",
            &[
                Argument::Scratch(Scratch::O),
                w("attn.Wo.weight"),
                z,
                Argument::Scratch(Scratch::Y),
            ],
        )?;
        call(
            "addln",
            &[
                Argument::Scratch(Scratch::X),
                Argument::Scratch(Scratch::Y),
                w("mlp_norm.weight"),
                z,
                Argument::Scratch(Scratch::Y),
            ],
        )?;
        call(
            "geglu",
            &[
                Argument::Scratch(Scratch::Y),
                w("mlp.Wi.weight"),
                Argument::Scratch(Scratch::G),
            ],
        )?;
        call(
            "down",
            &[
                Argument::Scratch(Scratch::G),
                w("mlp.Wo.weight"),
                z,
                Argument::Scratch(Scratch::Y),
            ],
        )?;
        let next = if i < 27 {
            weight(&format!("encoder.layers.{}.attn_norm.weight", i + 1))
        } else {
            weight("encoder.final_norm.weight")
        };
        call(
            "addln",
            &[
                Argument::Scratch(Scratch::X),
                Argument::Scratch(Scratch::Y),
                next,
                z,
                Argument::Scratch(Scratch::Y),
            ],
        )?;
        if [0, 1, 2, 27].contains(&i) {
            dump(&format!("encoder{i}_residual"), Scratch::X, false)?;
            dump(&format!("encoder{i}_normalized"), Scratch::Y, true)?;
        }
    }
    call(
        "type",
        &[
            Argument::Scratch(Scratch::Y),
            weight("type_emb.weight"),
            Argument::Scratch(Scratch::Types),
            Argument::Scratch(Scratch::X),
        ],
    )?;
    for i in 0..2 {
        let p = format!("head.layers.{i}");
        let w = |n: &str| weight(&format!("{p}.{n}"));
        call(
            "ln_bias",
            &[
                Argument::Scratch(Scratch::X),
                Argument::Scratch(Scratch::Y),
                w("norm1.weight"),
                w("norm1.bias"),
                Argument::Scratch(Scratch::Y),
            ],
        )?;
        call(
            "head_in",
            &[
                Argument::Scratch(Scratch::Y),
                w("self_attn.in_proj_weight"),
                w("self_attn.in_proj_bias"),
                Argument::Scratch(Scratch::Qkv),
            ],
        )?;
        call(
            &attention("full"),
            &[
                Argument::Scratch(Scratch::Qkv),
                Argument::Scratch(Scratch::Lens),
                Argument::Scratch(Scratch::O),
            ],
        )?;
        call(
            "head_out",
            &[
                Argument::Scratch(Scratch::O),
                w("self_attn.out_proj.weight"),
                w("self_attn.out_proj.bias"),
                Argument::Scratch(Scratch::Y),
            ],
        )?;
        call(
            "addln_bias",
            &[
                Argument::Scratch(Scratch::X),
                Argument::Scratch(Scratch::Y),
                w("norm2.weight"),
                w("norm2.bias"),
                Argument::Scratch(Scratch::Y),
            ],
        )?;
        call(
            "ffn1",
            &[
                Argument::Scratch(Scratch::Y),
                w("linear1.weight"),
                w("linear1.bias"),
                Argument::Scratch(Scratch::Ff),
            ],
        )?;
        call(
            "ffn2",
            &[
                Argument::Scratch(Scratch::Ff),
                w("linear2.weight"),
                w("linear2.bias"),
                Argument::Scratch(Scratch::Y),
            ],
        )?;
        call(
            "residual",
            &[Argument::Scratch(Scratch::X), Argument::Scratch(Scratch::Y)],
        )?;
    }
    Ok(steps.into_inner())
}

pub struct Model {
    pub config: Config,
    cuda: Cuda,
    blas: Blas,
    weights: HashMap<String, Buffer>,
    plan: Vec<Step>,
    cache: VecDeque<Workspace>,
    graphs: bool,
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
        crate::artifacts::validate_bundle(checkpoint, bundle)?;
        // SAFETY: bundle is an explicit trusted build artifact supplied by the operator.
        let cuda = unsafe { Cuda::load(&bundle.join("liblaya_cuda.so")) }?;
        let blas = Blas::new(&cuda)?;
        let source = Weights::open(&checkpoint.join("model.safetensors"))?;
        let mut weights = HashMap::new();
        let verify_weights = std::env::var_os("LAYA_VERIFY_WEIGHTS").is_some();
        let upload = |name: &str, data: &[u8]| -> Result<Buffer> {
            let buffer = cuda
                .upload(data)
                .with_context(|| format!("upload {name}"))?;
            if verify_weights {
                ensure!(
                    buffer
                        .read(data.len())
                        .with_context(|| format!("read back {name}"))?
                        == data,
                    "resident weight bytes mismatch: {name}"
                );
            }
            Ok(buffer)
        };
        let mut add = |name: &str, shape: &[usize], dtype: &str| -> Result<()> {
            let data = match dtype {
                "f32" => bytes32(&source.f32(name, shape)?),
                "f16" => bytes16(&source.f16(name, shape)?),
                _ => bytes16(&source.bf16(name, shape)?),
            };
            weights.insert(name.to_owned(), upload(name, &data)?);
            Ok(())
        };
        for spec in crate::weights::checkpoint_tensors() {
            add(&spec.name, &spec.shape, storage_dtype(&spec.name))?;
        }
        // Check only source tensors here, before adding synthetic buffers/tables.
        source.validate_names(weights.keys().map(String::as_str))?;
        for n in [D, 3 * D, 4 * D] {
            let key = format!("zeros.{n}");
            weights.insert(key.clone(), upload(&key, &vec![0; n * 4])?);
        }
        for kind in ["full", "local"] {
            for part in ["cos", "sin"] {
                let key = format!("rope_{kind}_{part}");
                let data = fs::read(bundle.join(format!("{key}.f32")))?;
                ensure!(data.len() == 512 * 32 * 4, "invalid rotary table size");
                weights.insert(key.clone(), upload(&key, &data)?);
            }
        }
        if verify_weights {
            eprintln!(
                "LAYA_VERIFY_WEIGHTS verified {} resident buffers",
                weights.len()
            );
        }
        let plan = prepare_encoder(&cuda, &weights, original_rope)?;
        Ok(Self {
            config,
            cuda,
            blas,
            weights,
            plan,
            cache: VecDeque::new(),
            graphs,
        })
    }
    fn w(&self, n: &str) -> &Buffer {
        &self.weights[n]
    }
    fn encode(&self, s: &Workspace) -> Result<()> {
        for step in &self.plan {
            match step {
                Step::Launch {
                    kernel,
                    args,
                    count,
                } => {
                    let pointers = args.map(|arg| arg.pointer(s));
                    unsafe {
                        kernel
                            .select(s.b, s.l)
                            .launch(&pointers[..*count], s.b, s.l)
                    }?;
                }
                Step::Dump { name, slot, bf } => self.dump(name, slot.buffer(s), *bf)?,
            }
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
        if batch.markers.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        ensure!(
            batch.b <= 16
                && batch.l <= 512
                && batch.input_ids.iter().all(|i| *i >= 0 && *i < 50368),
            "invalid packed input"
        );
        let n = batch.markers.len();
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
                        && !batch.markers[i].is_empty()
                        && batch.markers[i].iter().all(|m| *m < batch.lens[i] as usize),
                    "invalid sequence/marker bounds"
                );
            } else {
                ensure!(batch.lens[i] == 0, "dummy row must have zero length");
            }
        }
        ensure!(
            batch.markers.iter().map(Vec::len).sum::<usize>() <= 2048,
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
        for (i, markers) in batch.markers.iter().enumerate() {
            for &m in markers {
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
        let n = batch.markers.len();
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

#[cfg(test)]
#[path = "../../../../tests/laya/model.rs"]
mod tests;
