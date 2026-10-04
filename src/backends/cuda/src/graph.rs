//! Optional CUDA Graph execution on the context's stream.
use crate::{Cuda, Ptr};
use anyhow::{Result, ensure};

#[derive(Clone, Copy)]
struct GraphFunctions {
    begin: unsafe extern "C" fn(Ptr) -> i32,
    end: unsafe extern "C" fn(Ptr, *mut Ptr) -> i32,
    run: unsafe extern "C" fn(Ptr, Ptr) -> i32,
    free: unsafe extern "C" fn(Ptr) -> i32,
}

/// Captured device work. The context and runtime remain loaded until drop.
/// Buffer and kernel-library lifetimes are the responsibility of the capturer.
pub struct Graph {
    cuda: Cuda,
    functions: GraphFunctions,
    executable: Ptr,
}

impl Graph {
    /// Captures work without executing it. Graph symbols are loaded on demand.
    ///
    /// # Safety
    /// All device buffers and kernel code referenced by `enqueue` must remain
    /// alive until this graph is dropped. `enqueue` must only launch work on this
    /// context's stream; it must not release resources, use other streams, or
    /// invoke CUDA operations incompatible with stream capture (including via FFI).
    pub unsafe fn capture(cuda: &Cuda, enqueue: impl FnOnce() -> Result<()>) -> Result<Self> {
        cuda.ctx.ensure_not_capturing()?;
        let library = &cuda.ctx._library;
        let functions = unsafe {
            GraphFunctions {
                begin: *library.get(b"laya_capture_begin\0")?,
                end: *library.get(b"laya_capture_end\0")?,
                run: *library.get(b"laya_graph_run\0")?,
                free: *library.get(b"laya_graph_free\0")?,
            }
        };
        cuda.ctx.activate()?;
        cuda.ctx
            .functions
            .check(unsafe { (functions.begin)(cuda.ctx.stream) })?;
        cuda.ctx.capturing.set(true);
        let guard = CaptureGuard {
            cuda,
            functions,
            active: true,
        };
        enqueue()?;
        guard.finish()
    }

    /// Enqueues a replay. Synchronize before observing its results.
    pub fn run(&self) -> Result<()> {
        self.cuda.ctx.ensure_not_capturing()?;
        self.cuda.ctx.activate()?;
        self.cuda
            .ctx
            .functions
            .check(unsafe { (self.functions.run)(self.executable, self.cuda.ctx.stream) })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if self.cuda.ctx.activate().is_ok() {
            unsafe {
                (self.cuda.ctx.functions.sync)(self.cuda.ctx.stream);
                (self.functions.free)(self.executable);
            }
        }
    }
}

struct CaptureGuard<'a> {
    cuda: &'a Cuda,
    functions: GraphFunctions,
    active: bool,
}

impl CaptureGuard<'_> {
    fn finish(mut self) -> Result<Graph> {
        // Keep the guard armed if activation fails: Drop still attempts to end capture.
        self.cuda.ctx.activate()?;
        let mut executable = std::ptr::null_mut();
        let status = unsafe { (self.functions.end)(self.cuda.ctx.stream, &mut executable) };
        self.active = false;
        self.cuda.ctx.capturing.set(false);
        // Take ownership before checking status, so even a partial handle is freed.
        let graph = (!executable.is_null()).then(|| Graph {
            cuda: self.cuda.clone(),
            functions: self.functions,
            executable,
        });
        self.cuda.ctx.functions.check(status)?;
        ensure!(graph.is_some(), "CUDA runtime returned a null graph");
        Ok(graph.unwrap())
    }
}

impl Drop for CaptureGuard<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.cuda.ctx.activate();
            let mut executable = std::ptr::null_mut();
            unsafe {
                (self.functions.end)(self.cuda.ctx.stream, &mut executable);
                if !executable.is_null() {
                    (self.functions.free)(executable);
                }
            }
            self.cuda.ctx.capturing.set(false);
        }
    }
}
