//! The CUDA side, loaded at run time from `libqwen3_5_cuda.so` (built by
//! `src/backends/cuda/qwen3_5/build.sh`): the Qwen3.5 operations and the few CUDA
//! runtime calls the model needs. Building this crate needs no CUDA toolkit.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, bail, ensure};

/// `CS1_ABI_VERSION` in ops.h.
const ABI_VERSION: u32 = 4;
pub const LIBRARY: &str = "libqwen3_5_cuda.so";

/// A `cudaStream_t`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Stream(*mut c_void);

// SAFETY: a stream handle may be used from any thread; the model queues work on it
// from one thread at a time.
unsafe impl Send for Stream {}

macro_rules! api {
    ($($name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?;)*) => {
        /// The functions of the library, as declared in ops.h.
        pub struct Api {
            _lib: libloading::Library,
            $(pub $name: unsafe extern "C" fn($($ty),*) $(-> $ret)?,)*
        }

        impl Api {
            fn resolve(lib: libloading::Library) -> Result<Self> {
                $(
                    // SAFETY: the signature is the one ops.h declares for this symbol.
                    let $name = unsafe {
                        lib.get::<unsafe extern "C" fn($($ty),*) $(-> $ret)?>(
                            concat!(stringify!($name), "\0").as_bytes(),
                        )
                        .map(|f| *f)
                    }
                    .with_context(|| format!("{} has no {}", LIBRARY, stringify!($name)))?;
                )*
                Ok(Self { _lib: lib, $($name,)* })
            }
        }
    };
}

api! {
    cs1_vision_linear(gemm: *mut c_void, x: *const c_void, w: *const c_void, bias: *const c_void, y: *mut c_void, m: c_int, n: c_int, k: c_int, stream: Stream) -> c_int;
    cs1_gemm_f32(gemm: *mut c_void, x: *const f32, w: *const f32, y: *mut f32, m: c_int, n: c_int, k: c_int, stream: Stream) -> c_int;
    cs1_vision_norm(x: *const c_void, w: *const c_void, b: *const c_void, y: *mut c_void, rows: c_int, d: c_int, stream: Stream) -> c_int;
    cs1_vision_position(x: *mut c_void, table: *const c_void, indices: *const i32, weights: *const f32, n: c_int, stream: Stream) -> c_int;
    cs1_vision_rope(qkv: *const c_void, co: *const f32, si: *const f32, q: *mut c_void, k: *mut c_void, n: c_int, stream: Stream) -> c_int;
    cs1_vision_attention(q: *const c_void, k: *const c_void, v: *const c_void, out: *mut c_void, n: c_int, stream: Stream) -> c_int;
    cs1_vision_bias(x: *mut c_void, bias: *const c_void, n: usize, d: c_int, stream: Stream) -> c_int;
    cs1_vision_gelu(x: *mut c_void, n: usize, exact: c_int, stream: Stream) -> c_int;
    cs1_vision_add(x: *mut c_void, delta: *const c_void, n: usize, stream: Stream) -> c_int;
    cs1_vision_to_float(x: *const c_void, out: *mut f32, n: usize, stream: Stream) -> c_int;
    cs1_vision_lora_add(x: *mut c_void, delta: *const f32, n: usize, scale: f32, stream: Stream) -> c_int;
    cs1_abi_version() -> u32;
    cs1_error_string(code: c_int) -> *const c_char;
    cs1_set_device(device: c_int) -> c_int;
    cs1_malloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    cs1_free(ptr: *mut c_void) -> c_int;
    cs1_stream_create(stream: *mut Stream) -> c_int;
    cs1_stream_sync(stream: Stream) -> c_int;
    cs1_stream_destroy(stream: Stream) -> c_int;
    cs1_graph_begin(stream: Stream) -> c_int;
    cs1_graph_end(stream: Stream, exec: *mut *mut c_void) -> c_int;
    cs1_graph_launch(exec: *mut c_void, stream: Stream) -> c_int;
    cs1_graph_destroy(exec: *mut c_void) -> c_int;
    cs1_upload(dst: *mut c_void, src: *const c_void, bytes: usize, stream: Stream) -> c_int;
    cs1_download(dst: *mut c_void, src: *const c_void, bytes: usize, stream: Stream) -> c_int;
    cs1_embed(ids: *const i32, table: *const c_void, out: *mut c_void, t: c_int, d: c_int, stream: Stream) -> c_int;
    cs1_rms_norm(
        x: *const c_void, w: *const c_void, out: *mut c_void, rows: c_int, d: c_int, eps: f32, stream: Stream,
    ) -> c_int;
    cs1_add_rms_norm(
        residual: *mut c_void, delta: *const c_void, w: *const c_void, out: *mut c_void, rows: c_int, d: c_int,
        eps: f32, stream: Stream,
    ) -> c_int;
    cs1_gated_rms_norm(
        x: *const c_void, z: *const c_void, ldz: c_int, w: *const c_void, out: *mut c_void, t: c_int, h: c_int,
        d: c_int, eps: f32, stream: Stream,
    ) -> c_int;
    cs1_gdn_conv(
        qkv: *const c_void, ld: c_int, w: *const c_void, q: *mut c_void, k: *mut c_void, v: *mut c_void, t: c_int,
        key_dim: c_int, value_dim: c_int, stream: Stream,
    ) -> c_int;
    cs1_gdn_gates(
        b: *const c_void, a: *const c_void, ld: c_int, a_log: *const c_void, dt_bias: *const c_void,
        beta: *mut c_void, g: *mut f32, t: c_int, h: c_int, stream: Stream,
    ) -> c_int;
    cs1_gdn_workspace_floats(t: c_int, h: c_int) -> usize;
    cs1_gdn_prefill(
        q: *const c_void, k: *const c_void, v: *const c_void, g: *const f32, beta: *const c_void, o: *mut c_void,
        workspace: *mut f32, t: c_int, h: c_int, hk: c_int, scale: f32, stream: Stream,
    ) -> c_int;
    cs1_attn_prep(
        qg: *const c_void, kr: *const c_void, ld: c_int, qw: *const c_void, kw: *const c_void, cos: *const c_void,
        sin: *const c_void, q: *mut c_void, gate: *mut c_void, k: *mut c_void, t: c_int, hq: c_int, hk: c_int,
        dh: c_int, half: c_int, eps: f32, stream: Stream,
    ) -> c_int;
    cs1_attention(
        q: *const c_void, k: *const c_void, v: *const c_void, ldv: c_int, out: *mut c_void, t: c_int, hq: c_int,
        hk: c_int, dh: c_int, scale: f32, stream: Stream,
    ) -> c_int;
    cs1_sigmoid_gate(x: *mut c_void, gate: *const c_void, n: usize, stream: Stream) -> c_int;
    cs1_silu_mul(gate_up: *const c_void, ld: c_int, out: *mut c_void, t: c_int, i: c_int, stream: Stream) -> c_int;
    cs1_gemm_create(workspace_bytes: usize) -> *mut c_void;
    cs1_gemm_destroy(gemm: *mut c_void);
    cs1_gemm(
        gemm: *mut c_void, x: *const c_void, w: *const c_void, y: *mut c_void, m: c_int, n: c_int, k: c_int,
        ldy: c_int, stream: Stream,
    ) -> c_int;
}

static API: OnceLock<Api> = OnceLock::new();

/// The library next to the running executable.
pub fn default_library() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    Ok(exe
        .parent()
        .context("the executable has no directory")?
        .join(LIBRARY))
}

/// Load the library (once per process) and check its ABI version.
pub fn load(path: &Path) -> Result<&'static Api> {
    if let Some(api) = API.get() {
        return Ok(api);
    }
    // SAFETY: loading runs the library's initializers; it is the library these
    // sources build.
    let lib = unsafe { libloading::Library::new(path) }.with_context(|| {
        format!(
            "loading {} (build it with src/backends/cuda/qwen3_5/build.sh)",
            path.display()
        )
    })?;
    let api = Api::resolve(lib)?;
    // SAFETY: takes no arguments.
    let abi = unsafe { (api.cs1_abi_version)() };
    ensure!(
        abi == ABI_VERSION,
        "{} has ABI version {abi}, this build needs {ABI_VERSION}; rebuild it",
        path.display()
    );
    Ok(API.get_or_init(|| api))
}

/// The loaded library; `load` must have succeeded before.
pub fn api() -> &'static Api {
    API.get().expect("the CUDA library is not loaded")
}

/// Turn a return code of the library into an error; codes from 1000 up are cuBLAS
/// statuses (see gemm.cu).
pub fn check(code: c_int, what: &str) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    if code >= 1000 {
        bail!("{what}: cuBLAS status {}", code - 1000);
    }
    // SAFETY: cudaGetErrorString returns a static string for any code.
    let msg = unsafe { CStr::from_ptr((api().cs1_error_string)(code)) };
    bail!("{what}: {} ({code})", msg.to_string_lossy());
}

pub fn set_device(device: i32) -> Result<()> {
    // SAFETY: plain runtime call.
    check(unsafe { (api().cs1_set_device)(device) }, "cudaSetDevice")
}

/// A device allocation, freed on drop.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    bytes: usize,
}

// SAFETY: the pointer is a device address; access is serialized by the owner.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn new(bytes: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        if bytes > 0 {
            // SAFETY: `ptr` is a valid out-pointer.
            check(unsafe { (api().cs1_malloc)(&mut ptr, bytes) }, "cudaMalloc")?;
        }
        Ok(Self { ptr, bytes })
    }

    /// The device address `offset` bytes into the buffer.
    pub fn at(&self, offset: usize) -> *mut c_void {
        debug_assert!(offset <= self.bytes);
        self.ptr.wrapping_byte_add(offset)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated by cs1_malloc and not freed before.
            unsafe { (api().cs1_free)(self.ptr) };
        }
    }
}

pub fn new_stream() -> Result<Stream> {
    let mut stream = Stream(std::ptr::null_mut());
    // SAFETY: `stream` is a valid out-pointer.
    check(
        unsafe { (api().cs1_stream_create)(&mut stream) },
        "cudaStreamCreateWithFlags",
    )?;
    Ok(stream)
}

pub fn synchronize(stream: Stream) -> Result<()> {
    // SAFETY: a Stream only comes from new_stream.
    check(
        unsafe { (api().cs1_stream_sync)(stream) },
        "cudaStreamSynchronize",
    )
}

/// Copy host bytes to `dst` and wait for the copy.
///
/// # Safety
/// `dst` must be a device allocation with room for `src.len()` bytes.
pub unsafe fn upload(dst: *mut c_void, src: &[u8], stream: Stream) -> Result<()> {
    // SAFETY: see above; the library waits for the copy before returning.
    check(
        unsafe { (api().cs1_upload)(dst, src.as_ptr().cast(), src.len(), stream) },
        "copy to device",
    )
}

/// Copy `dst.len()` bytes from `src` to the host, after the work queued before it.
///
/// # Safety
/// `src` must be a device allocation holding at least `dst.len()` bytes.
pub unsafe fn download(dst: &mut [u8], src: *const c_void, stream: Stream) -> Result<()> {
    // SAFETY: see above; the library waits for the copy before returning.
    check(
        unsafe { (api().cs1_download)(dst.as_mut_ptr().cast(), src, dst.len(), stream) },
        "copy to host",
    )
}

pub struct Graph {
    exec: *mut c_void,
}

// SAFETY: the executable graph is only launched by its owner, one launch at a time.
unsafe impl Send for Graph {}

impl Graph {
    /// Capture the work `record` queues on `stream` (nothing runs) and instantiate it.
    pub fn capture(stream: Stream, record: impl FnOnce() -> Result<()>) -> Result<Self> {
        // SAFETY: plain runtime calls on a stream from new_stream; the capture is
        // always ended, also when `record` fails.
        unsafe {
            check((api().cs1_graph_begin)(stream), "cudaStreamBeginCapture")?;
            let recorded = record();
            let mut exec = std::ptr::null_mut();
            let ended = check(
                (api().cs1_graph_end)(stream, &mut exec),
                "capturing a CUDA graph",
            );
            match recorded.and(ended) {
                Ok(()) => Ok(Graph { exec }),
                Err(e) => {
                    if !exec.is_null() {
                        (api().cs1_graph_destroy)(exec);
                    }
                    Err(e)
                }
            }
        }
    }

    pub fn launch(&self, stream: Stream) -> Result<()> {
        // SAFETY: an instantiated graph whose buffers outlive it (see Model).
        check(
            unsafe { (api().cs1_graph_launch)(self.exec, stream) },
            "cudaGraphLaunch",
        )
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        // SAFETY: instantiated by capture and not destroyed before.
        unsafe { (api().cs1_graph_destroy)(self.exec) };
    }
}
