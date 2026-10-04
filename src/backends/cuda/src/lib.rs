//! CUDA resources confined to one thread. CPU builds do not link CUDA.
use anyhow::{Result, anyhow, ensure};
use libloading::Library;
use std::{
    cell::Cell,
    ffi::{CStr, c_char, c_void},
    path::Path,
    rc::Rc,
};

type Ptr = *mut c_void;

struct Functions {
    error: unsafe extern "C" fn(i32) -> *const c_char,
    set_device: unsafe extern "C" fn(i32) -> i32,
    stream_create: unsafe extern "C" fn(*mut Ptr) -> i32,
    alloc: unsafe extern "C" fn(*mut Ptr, usize) -> i32,
    free: unsafe extern "C" fn(Ptr) -> i32,
    upload: unsafe extern "C" fn(Ptr, *const u8, usize, Ptr) -> i32,
    download: unsafe extern "C" fn(*mut u8, Ptr, usize, Ptr) -> i32,
    sync: unsafe extern "C" fn(Ptr) -> i32,
    stream_free: unsafe extern "C" fn(Ptr) -> i32,
}

impl Functions {
    fn check(&self, code: i32) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let message = unsafe { (self.error)(code) };
        if message.is_null() {
            return Err(anyhow!("CUDA error {code}"));
        }
        Err(anyhow!("CUDA {code}: {}", unsafe {
            CStr::from_ptr(message).to_string_lossy()
        }))
    }
}

struct Context {
    _library: Library,
    functions: Functions,
    device: i32,
    stream: Ptr,
    capturing: Cell<bool>,
}

impl Context {
    fn ensure_not_capturing(&self) -> Result<()> {
        ensure!(
            !self.capturing.get(),
            "operation prohibited during CUDA capture"
        );
        Ok(())
    }

    fn activate(&self) -> Result<()> {
        self.functions
            .check(unsafe { (self.functions.set_device)(self.device) })
    }

    fn sync(&self) -> Result<()> {
        self.ensure_not_capturing()?;
        self.activate()?;
        self.functions
            .check(unsafe { (self.functions.sync)(self.stream) })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if self.activate().is_ok() {
            unsafe {
                (self.functions.sync)(self.stream);
                (self.functions.stream_free)(self.stream);
            }
        }
    }
}

#[derive(Clone)]
pub struct Cuda {
    ctx: Rc<Context>,
}

impl Cuda {
    /// # Safety
    /// `path` must name a trusted library implementing the complete runtime ABI.
    pub unsafe fn load(path: &Path, device: i32) -> Result<Self> {
        let library = unsafe { Library::new(path) }?;
        let version =
            unsafe { library.get::<unsafe extern "C" fn() -> u32>(b"laya_abi_version\0")?() };
        ensure!(version == 1, "unsupported CUDA runtime ABI {version}");
        let functions = unsafe {
            Functions {
                error: *library.get(b"laya_error_string\0")?,
                set_device: *library.get(b"laya_set_device\0")?,
                stream_create: *library.get(b"laya_stream_create\0")?,
                alloc: *library.get(b"laya_alloc\0")?,
                free: *library.get(b"laya_free\0")?,
                upload: *library.get(b"laya_upload\0")?,
                download: *library.get(b"laya_download\0")?,
                sync: *library.get(b"laya_sync\0")?,
                stream_free: *library.get(b"laya_stream_free\0")?,
            }
        };
        functions.check(unsafe { (functions.set_device)(device) })?;
        let mut stream = std::ptr::null_mut();
        functions.check(unsafe { (functions.stream_create)(&mut stream) })?;
        ensure!(!stream.is_null(), "CUDA runtime returned a null stream");
        Ok(Self {
            ctx: Rc::new(Context {
                _library: library,
                functions,
                device,
                stream,
                capturing: Cell::new(false),
            }),
        })
    }

    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        self.ctx.ensure_not_capturing()?;
        ensure!(bytes > 0, "zero CUDA allocation");
        self.ctx.activate()?;
        let mut ptr = std::ptr::null_mut();
        self.ctx
            .functions
            .check(unsafe { (self.ctx.functions.alloc)(&mut ptr, bytes) })?;
        ensure!(!ptr.is_null(), "CUDA runtime returned a null allocation");
        Ok(Buffer {
            inner: Rc::new(Allocation {
                ctx: self.ctx.clone(),
                ptr,
                bytes,
            }),
        })
    }

    pub fn upload(&self, bytes: &[u8]) -> Result<Buffer> {
        let buffer = self.alloc(bytes.len())?;
        buffer.write(bytes)?;
        Ok(buffer)
    }

    /// Uploads a validated group, synchronizing once before releasing host borrows.
    /// Empty groups (or groups of empty slices) do not enter the runtime. This
    /// groups transfers; it does not provide transfer/compute overlap.
    pub fn write_many(&self, writes: &[(&Buffer, &[u8])]) -> Result<()> {
        self.ctx.ensure_not_capturing()?;
        for (buffer, bytes) in writes {
            ensure!(
                Rc::ptr_eq(&self.ctx, &buffer.inner.ctx),
                "upload buffer belongs to a different CUDA context"
            );
            ensure!(
                bytes.len() <= buffer.inner.bytes,
                "upload exceeds allocation"
            );
        }
        if writes.iter().all(|(_, bytes)| bytes.is_empty()) {
            return Ok(());
        }
        self.ctx.activate()?;
        let functions = &self.ctx.functions;
        let mut copied = 0;
        for (buffer, bytes) in writes {
            if bytes.is_empty() {
                continue;
            }
            copied = unsafe {
                (functions.upload)(
                    buffer.inner.ptr,
                    bytes.as_ptr(),
                    bytes.len(),
                    self.ctx.stream,
                )
            };
            if copied != 0 {
                break;
            }
        }
        // A failing submission may still queue a copy. Drain even on failure,
        // while every source slice and destination allocation is still borrowed.
        let synced = unsafe { (functions.sync)(self.ctx.stream) };
        match (functions.check(copied), functions.check(synced)) {
            (Err(copy), Err(sync)) => Err(anyhow!(
                "{copy}; stream synchronization also failed: {sync}"
            )),
            (Err(error), _) | (_, Err(error)) => Err(error),
            (Ok(()), Ok(())) => Ok(()),
        }
    }

    pub fn sync(&self) -> Result<()> {
        self.ctx.sync()
    }
}

/// Clones share one allocation and keep its CUDA context alive.
#[derive(Clone)]
pub struct Buffer {
    inner: Rc<Allocation>,
}

struct Allocation {
    ctx: Rc<Context>,
    ptr: Ptr,
    bytes: usize,
}

impl Buffer {
    pub fn bytes(&self) -> usize {
        self.inner.bytes
    }

    pub fn write(&self, bytes: &[u8]) -> Result<()> {
        self.inner.ctx.ensure_not_capturing()?;
        ensure!(bytes.len() <= self.inner.bytes, "upload exceeds allocation");
        if bytes.is_empty() {
            return Ok(());
        }
        self.inner.ctx.activate()?;
        let functions = &self.inner.ctx.functions;
        let copied = unsafe {
            (functions.upload)(
                self.inner.ptr,
                bytes.as_ptr(),
                bytes.len(),
                self.inner.ctx.stream,
            )
        };
        // Even a failed copy may have queued work using the borrowed host memory.
        let synced = unsafe { (functions.sync)(self.inner.ctx.stream) };
        functions.check(copied)?;
        functions.check(synced)
    }

    pub fn read(&self, bytes: usize) -> Result<Vec<u8>> {
        self.inner.ctx.ensure_not_capturing()?;
        ensure!(bytes <= self.inner.bytes, "download exceeds allocation");
        let mut data = vec![0; bytes];
        if bytes == 0 {
            return Ok(data);
        }
        self.inner.ctx.activate()?;
        let functions = &self.inner.ctx.functions;
        let copied = unsafe {
            (functions.download)(
                data.as_mut_ptr(),
                self.inner.ptr,
                bytes,
                self.inner.ctx.stream,
            )
        };
        let synced = unsafe { (functions.sync)(self.inner.ctx.stream) };
        functions.check(copied)?;
        functions.check(synced)?;
        Ok(data)
    }
}

impl Drop for Allocation {
    fn drop(&mut self) {
        if self.ctx.activate().is_ok() {
            unsafe {
                (self.ctx.functions.sync)(self.ctx.stream);
                (self.ctx.functions.free)(self.ptr);
            }
        }
    }
}

pub mod kernels;

pub mod graph;
