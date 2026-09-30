# CUDA backend

`omni-cuda` loads Laya's CUDA resource library at runtime. It owns one device and
stream per context, plus the buffers allocated through that context. Rust builds
and CPU tests need no CUDA toolkit.

The resource library covers allocation, copies, synchronization and cleanup.
`Kernels` loads operator code separately; model execution order belongs to Laya.
Graphs and hardware-specific optimizations remain separate.

## Build and check

On a machine with the CUDA toolkit, build the resource library:

```sh
nvcc -shared -Xcompiler=-fPIC -O2 src/backends/cuda/kernels/runtime.cu -o /tmp/liblaya_cuda.so
LAYA_CUDA_LIBRARY=/tmp/liblaya_cuda.so LAYA_CUDA_DEVICE=0 \
  cargo test --locked -p omni-cuda --test runtime -- --ignored
```

The device is an ordinal after `CUDA_VISIBLE_DEVICES` filtering. The library
contains no generated kernels and needs neither TileLang nor cuBLAS. This command
builds only the resource slice; the complete model bundle has a separate build.

The normal CPU tests compile a small C fixture with `cc`. They check the dynamic
loader, errors, copy bounds and resource lifetime. They do not validate CUDA or
hardware support. The ignored test exercises real allocation and copy roundtrips.

## Ownership and ABI

Load only a trusted library with the matching ABI. `Cuda::load(path, device)`
checks `laya_abi_version() == 1` and all required symbols before creating a stream.
The old prototype's `laya_init` library has no version symbol and is rejected.

`Cuda` and `Buffer` stay on their creating thread. A buffer keeps its stream and
library alive even after the caller drops `Cuda`. Operations select the owning
device before using its resources. Destruction attempts synchronization and
cleanup; call `sync()` explicitly when errors need to reach the caller.

`write` and `read` check byte limits and synchronize before returning, so borrowed
host memory cannot outlive a queued copy. They are not Graph-capture operations.
Allocation of zero bytes is rejected; empty reads and writes are no-ops.

The native resource entry points return zero on success and CUDA error codes on
failure; code 1000 means an invalid runtime argument. `laya_error_string` explains
the code. Upload and download take the caller's stream as their last argument and
do not synchronize internally. No Hopper requirement or model initialization is
hidden in stream creation.

These are Laya's resource entry points, not a new shared tensor interface. A common
runtime can be extracted when another model needs the same implementation.

## Kernel library

`Kernels::load(&cuda, path, names)` resolves the requested Laya pointer-array
entry points and initializes their launch attributes on the owning device.
`launch` checks the supported batch/sequence bounds and rejects buffers from
another context, including another stream on the same device. Tensor sizes,
dtypes, argument counts, contents and aliasing remain the unsafe caller's contract.
The library stays loaded until pending work has synchronized. Loading requires
trusted native code compiled for the selected GPU; the resource library itself
does not impose the operator bundle's architecture restrictions.
