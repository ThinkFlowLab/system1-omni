//! Single-threaded CUDA ownership. Loading a compiled bundle is explicit; CPU builds need no CUDA.
use anyhow::{Result, anyhow, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, c_void},
    path::Path,
    rc::Rc,
};
pub type Ptr = *mut c_void;
type Launch = unsafe extern "C" fn(*mut Ptr, i32, i32, i32, Ptr) -> i32;
struct Context {
    lib: Library,
    stream: Ptr,
}
impl Context {
    fn check(&self, code: i32) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        unsafe {
            let f = self
                .lib
                .get::<unsafe extern "C" fn(i32) -> *const i8>(b"laya_error\0")?;
            Err(anyhow!(
                "CUDA {code}: {}",
                CStr::from_ptr(f(code)).to_string_lossy()
            ))
        }
    }
    fn symbol<T: Copy>(&self, name: &[u8]) -> Result<T> {
        unsafe { Ok(*self.lib.get::<T>(name)?) }
    }
    fn sync(&self) -> Result<()> {
        let f = self.symbol::<unsafe extern "C" fn(Ptr) -> i32>(b"laya_sync\0")?;
        self.check(unsafe { f(self.stream) })
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        let _ = self.sync();
        unsafe {
            if let Ok(f) = self
                .lib
                .get::<unsafe extern "C" fn(Ptr) -> i32>(b"laya_stream_free\0")
            {
                f(self.stream);
            }
        }
    }
}
#[derive(Clone)]
pub struct Cuda {
    ctx: Rc<Context>,
}
impl Cuda {
    /// Load a trusted library produced by this backend's build tools.
    /// # Safety
    /// The path must point to the matching native ABI, not arbitrary/untrusted code.
    pub unsafe fn load(path: &Path) -> Result<Self> {
        let lib = unsafe { Library::new(path) }?;
        let mut stream = std::ptr::null_mut();
        let init = unsafe { lib.get::<unsafe extern "C" fn(*mut Ptr) -> i32>(b"laya_init\0") }?;
        let code = unsafe { init(&mut stream) };
        let ctx = Rc::new(Context { lib, stream });
        ctx.check(code)?;
        Ok(Self { ctx })
    }
    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        ensure!(bytes > 0, "zero CUDA allocation");
        let f = self
            .ctx
            .symbol::<unsafe extern "C" fn(*mut Ptr, usize) -> i32>(b"laya_alloc\0")?;
        let mut p = std::ptr::null_mut();
        self.ctx.check(unsafe { f(&mut p, bytes) })?;
        Ok(Buffer {
            ctx: self.ctx.clone(),
            p,
            bytes,
        })
    }
    pub fn upload(&self, bytes: &[u8]) -> Result<Buffer> {
        let b = self.alloc(bytes.len())?;
        b.write(bytes)?;
        Ok(b)
    }
    /// Complete a prevalidated group with one synchronization before host borrows end.
    /// A failed submission is also drained; this does not overlap copies and compute.
    pub fn write_many(&self, writes: &[(&Buffer, &[u8])]) -> Result<()> {
        for (buffer, bytes) in writes {
            ensure!(
                Rc::ptr_eq(&self.ctx, &buffer.ctx),
                "upload buffer belongs to another CUDA context"
            );
            ensure!(bytes.len() <= buffer.bytes, "upload exceeds allocation");
        }
        if writes.iter().all(|(_, bytes)| bytes.is_empty()) {
            return Ok(());
        }
        let upload = self
            .ctx
            .symbol::<unsafe extern "C" fn(Ptr, *const u8, usize, Ptr) -> i32>(b"laya_upload\0")?;
        let sync = self
            .ctx
            .symbol::<unsafe extern "C" fn(Ptr) -> i32>(b"laya_sync\0")?;
        let mut copied = 0;
        for (buffer, bytes) in writes {
            if bytes.is_empty() {
                continue;
            }
            copied = unsafe { upload(buffer.p, bytes.as_ptr(), bytes.len(), self.ctx.stream) };
            if copied != 0 {
                break;
            }
        }
        // Resolve both entry points before submission and keep every source borrowed
        // through the drain even when a copy reports an error after queuing work.
        let synced = unsafe { sync(self.ctx.stream) };
        match (self.ctx.check(copied), self.ctx.check(synced)) {
            (Err(copy), Err(sync)) => Err(anyhow!(
                "{copy}; stream synchronization also failed: {sync}"
            )),
            (Err(e), _) | (_, Err(e)) => Err(e),
            (Ok(()), Ok(())) => Ok(()),
        }
    }
    pub fn sync(&self) -> Result<()> {
        self.ctx.sync()
    }
    /// Resolve a kernel once while retaining the owning runtime and native code.
    pub fn resolve(&self, name: &str) -> Result<Kernel> {
        Ok(Kernel {
            ctx: self.ctx.clone(),
            launch: self
                .ctx
                .symbol::<Launch>(format!("laya_{name}\0").as_bytes())?,
        })
    }
    /// # Safety
    /// Tensor shape, dtype, layout, aliasing and allocation sizes must match the generated kernel.
    /// Buffers must belong to this context and stay alive until synchronization or graph destruction.
    pub unsafe fn launch(&self, name: &str, args: &[Ptr], b: usize, l: usize) -> Result<()> {
        ensure!(
            b > 0 && b <= 16 && l > 0 && l <= 512 && l.is_multiple_of(16),
            "invalid CUDA shape"
        );
        let k = self
            .ctx
            .symbol::<Launch>(format!("laya_{name}\0").as_bytes())?;
        self.ctx.check(unsafe {
            k(
                args.as_ptr() as *mut Ptr,
                b as i32,
                l as i32,
                (b * l) as i32,
                self.ctx.stream,
            )
        })
    }
    /// # Safety
    /// Every allocation referenced by `work` must outlive the returned graph. No allocation or copy
    /// that may synchronize is permitted inside work. This context is confined to one OS thread.
    pub unsafe fn capture(&self, work: impl FnOnce() -> Result<()>) -> Result<Graph> {
        self.sync()?;
        let begin = self
            .ctx
            .symbol::<unsafe extern "C" fn(Ptr) -> i32>(b"laya_capture_begin\0")?;
        let end = self
            .ctx
            .symbol::<unsafe extern "C" fn(Ptr, *mut Ptr) -> i32>(b"laya_capture_end\0")?;
        self.ctx.check(unsafe { begin(self.ctx.stream) })?;
        let result = work();
        let mut p = std::ptr::null_mut();
        let code = unsafe { end(self.ctx.stream, &mut p) };
        if result.is_err() || code != 0 {
            if !p.is_null() {
                let f = self
                    .ctx
                    .symbol::<unsafe extern "C" fn(Ptr) -> i32>(b"laya_graph_free\0")?;
                unsafe {
                    f(p);
                }
            }
            result?;
            self.ctx.check(code)?;
        }
        Ok(Graph {
            ctx: self.ctx.clone(),
            p,
        })
    }
    /// # Safety
    /// Caller supplies the precise dimensions and allocation sizes required by this glue kernel.
    pub unsafe fn launch_rows(
        &self,
        name: &str,
        args: &[Ptr],
        b: usize,
        l: usize,
        rows: usize,
    ) -> Result<()> {
        let k = self
            .ctx
            .symbol::<Launch>(format!("laya_{name}\0").as_bytes())?;
        self.ctx.check(unsafe {
            k(
                args.as_ptr() as *mut Ptr,
                b as i32,
                l as i32,
                rows as i32,
                self.ctx.stream,
            )
        })
    }
    pub fn stream(&self) -> Ptr {
        self.ctx.stream
    }
    /// # Safety
    /// `T` must exactly match the ABI and signature of the named bundle symbol.
    pub unsafe fn symbol<T: Copy>(&self, name: &[u8]) -> Result<T> {
        self.ctx.symbol(name)
    }
    pub fn check(&self, code: i32) -> Result<()> {
        self.ctx.check(code)
    }
}
pub struct Buffer {
    ctx: Rc<Context>,
    p: Ptr,
    bytes: usize,
}
impl Buffer {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn ptr(&self) -> Ptr {
        self.p
    }
    pub fn write(&self, bytes: &[u8]) -> Result<()> {
        Cuda {
            ctx: self.ctx.clone(),
        }
        .write_many(&[(self, bytes)])
    }
    pub fn read(&self, bytes: usize) -> Result<Vec<u8>> {
        ensure!(bytes <= self.bytes, "download exceeds allocation");
        let mut data = vec![0; bytes];
        let f = self
            .ctx
            .symbol::<unsafe extern "C" fn(*mut u8, Ptr, usize, Ptr) -> i32>(b"laya_download\0")?;
        self.ctx
            .check(unsafe { f(data.as_mut_ptr(), self.p, bytes, self.ctx.stream) })?;
        self.ctx.sync()?;
        Ok(data)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = self.ctx.sync();
        if let Ok(f) = self
            .ctx
            .symbol::<unsafe extern "C" fn(Ptr) -> i32>(b"laya_free\0")
        {
            unsafe {
                f(self.p);
            }
        }
    }
}
pub struct Graph {
    ctx: Rc<Context>,
    p: Ptr,
}
impl Graph {
    pub fn replay(&self) -> Result<()> {
        let f = self
            .ctx
            .symbol::<unsafe extern "C" fn(Ptr, Ptr) -> i32>(b"laya_graph_run\0")?;
        self.ctx.check(unsafe { f(self.p, self.ctx.stream) })
    }
}
impl Drop for Graph {
    fn drop(&mut self) {
        let _ = self.ctx.sync();
        if let Ok(f) = self
            .ctx
            .symbol::<unsafe extern "C" fn(Ptr) -> i32>(b"laya_graph_free\0")
        {
            unsafe {
                f(self.p);
            }
        }
    }
}

/// A resolved entry point retaining its stream and native library.
#[derive(Clone)]
pub struct Kernel {
    ctx: Rc<Context>,
    launch: Launch,
}
impl Kernel {
    /// # Safety
    /// Shapes, dtype, layout, aliasing and pointer lifetimes must match this kernel.
    /// Every pointer belongs to this context and stays alive through synchronization
    /// or destruction of any graph that captures the launch.
    pub unsafe fn launch(&self, args: &[Ptr], b: usize, l: usize) -> Result<()> {
        ensure!(
            b > 0 && b <= 16 && l > 0 && l <= 512 && l.is_multiple_of(16),
            "invalid CUDA shape"
        );
        self.ctx.check(unsafe {
            (self.launch)(
                args.as_ptr() as *mut Ptr,
                b as i32,
                l as i32,
                (b * l) as i32,
                self.ctx.stream,
            )
        })
    }
}
