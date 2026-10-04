//! Calls into a separately built, trusted CUDA kernel library.
use super::*;
use std::collections::HashMap;

type Launch = unsafe extern "C" fn(*mut Ptr, i32, i32, i32, Ptr) -> i32;
const MAX_ARGS: usize = 16;

struct KernelLibrary {
    // Unload device code before releasing its runtime context.
    _library: Library,
    cuda: Cuda,
}

pub struct Kernels {
    library: Rc<KernelLibrary>,
    functions: HashMap<String, Launch>,
}

/// A resolved entry point retaining both its library and CUDA context.
#[derive(Clone)]
pub struct Kernel {
    library: Rc<KernelLibrary>,
    launch: Launch,
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
            library: Rc::new(KernelLibrary {
                cuda: cuda.clone(),
                _library: library,
            }),
            functions,
        })
    }

    pub fn resolve(&self, name: &str) -> Result<Kernel> {
        let launch = *self
            .functions
            .get(name)
            .ok_or_else(|| anyhow!("kernel not loaded: {name}"))?;
        Ok(Kernel {
            library: self.library.clone(),
            launch,
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
        let kernel = self.resolve(name)?;
        kernel.validate(args, batch, sequence)?;
        // Keep the slice API compatible, including unusually large argument lists.
        if args.len() > MAX_ARGS {
            let mut pointers: Vec<_> = args.iter().map(|b| b.inner.ptr).collect();
            unsafe { kernel.dispatch(pointers.as_mut_ptr(), batch, sequence) }
        } else {
            let mut pointers = [std::ptr::null_mut(); MAX_ARGS];
            for (slot, buffer) in pointers.iter_mut().zip(args) {
                *slot = buffer.inner.ptr;
            }
            unsafe { kernel.dispatch(pointers.as_mut_ptr(), batch, sequence) }
        }
    }
}

impl Kernel {
    /// Launch without name lookup or heap allocation; at most 16 arguments.
    /// # Safety
    /// Argument count, sizes, contents, dtypes and aliasing must match the kernel.
    pub unsafe fn launch<const N: usize>(
        &self,
        args: [&Buffer; N],
        batch: usize,
        sequence: usize,
    ) -> Result<()> {
        const {
            assert!(N <= MAX_ARGS, "too many kernel arguments");
        }
        self.validate(&args, batch, sequence)?;
        let mut pointers = args.map(|b| b.inner.ptr);
        unsafe { self.dispatch(pointers.as_mut_ptr(), batch, sequence) }
    }

    fn validate(&self, args: &[&Buffer], batch: usize, sequence: usize) -> Result<()> {
        ensure!(
            batch.is_power_of_two()
                && batch <= 16
                && (16..=512).contains(&sequence)
                && sequence.is_multiple_of(16),
            "invalid kernel shape"
        );
        ensure!(
            args.iter()
                .all(|b| Rc::ptr_eq(&b.inner.ctx, &self.library.cuda.ctx)),
            "kernel buffer belongs to another CUDA context"
        );
        Ok(())
    }

    unsafe fn dispatch(&self, pointers: *mut Ptr, batch: usize, sequence: usize) -> Result<()> {
        let ctx = &self.library.cuda.ctx;
        ctx.activate()?;
        ctx.functions.check(unsafe {
            (self.launch)(
                pointers,
                batch as i32,
                sequence as i32,
                (batch * sequence) as i32,
                ctx.stream,
            )
        })
    }
}

impl Drop for KernelLibrary {
    fn drop(&mut self) {
        // The last resolved handle must finish pending work before unloading code.
        let _ = self.cuda.sync();
    }
}
