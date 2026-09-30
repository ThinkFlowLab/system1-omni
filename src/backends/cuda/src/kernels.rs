//! Calls into a separately built, trusted CUDA kernel library.
use super::*;
use std::collections::HashMap;

type Launch = unsafe extern "C" fn(*mut Ptr, i32, i32, i32, Ptr) -> i32;

pub struct Kernels {
    cuda: Cuda,
    _library: Library,
    functions: HashMap<String, Launch>,
}

impl Kernels {
    /// # Safety
    /// The library must implement the named pointer-array launch ABI and
    /// `laya_kernels_init`, and be compiled for this device's architecture.
    pub unsafe fn load(cuda: &Cuda, path: &Path, names: &[&str]) -> Result<Self> {
        cuda.ctx.activate()?;
        let library = unsafe { Library::new(path) }?;
        let mut functions = HashMap::new();
        for name in names {
            let symbol = format!("laya_{name}\0");
            let launch = unsafe { *library.get::<Launch>(symbol.as_bytes())? };
            functions.insert((*name).to_owned(), launch);
        }
        let init = unsafe { library.get::<unsafe extern "C" fn() -> i32>(b"laya_kernels_init\0")? };
        cuda.ctx.functions.check(unsafe { init() })?;
        Ok(Self {
            cuda: cuda.clone(),
            _library: library,
            functions,
        })
    }

    /// # Safety
    /// Argument count, sizes, contents, dtypes and aliasing must match the kernel.
    /// This checks context identity and shape bounds, not tensor semantics.
    pub unsafe fn launch(
        &self,
        name: &str,
        args: &[&Buffer],
        batch: usize,
        sequence: usize,
    ) -> Result<()> {
        ensure!(
            batch.is_power_of_two()
                && batch <= 16
                && (16..=512).contains(&sequence)
                && sequence.is_multiple_of(16),
            "invalid kernel shape"
        );
        ensure!(
            args.iter().all(|b| Rc::ptr_eq(&b.ctx, &self.cuda.ctx)),
            "kernel buffer belongs to another CUDA context"
        );
        let launch = self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("kernel not loaded: {name}"))?;
        self.cuda.ctx.activate()?;
        let mut pointers: Vec<_> = args.iter().map(|b| b.ptr).collect();
        self.cuda.ctx.functions.check(unsafe {
            launch(
                pointers.as_mut_ptr(),
                batch as i32,
                sequence as i32,
                (batch * sequence) as i32,
                self.cuda.ctx.stream,
            )
        })
    }
}

impl Drop for Kernels {
    fn drop(&mut self) {
        // Pending launches must finish before unloading their device code.
        let _ = self.cuda.sync();
    }
}
