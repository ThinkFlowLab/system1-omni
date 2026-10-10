# Laya CPU host regressions

`fixture.c` implements the Laya native ABI using host allocations, deterministic
copy/failure counters, a pointer launch trace and a minimal captured fill
operation. It performs no Laya tensor math and needs no CUDA toolkit, GPU,
checkpoint, Python or network access. The trace oracle in
`tests/laya/original_encode.rs` freezes the former encoder launch sequence from
`7f39ac40902c374803992407bb26eeba29c8a588`; it checks dispatch pointers/order,
not numerical outputs.

Run from the repository root with stable Rust and a C compiler available as `cc`:

```sh
cargo test --locked -p omni-cuda --test laya_runtime
cargo test --locked -p omni-laya --features serve --lib model::tests
```

The Rust helper embeds the maintained C source and compiles it into a unique
`tempfile` directory for every test. It uses `cc -shared -fPIC` on Linux and
`cc -dynamiclib` on macOS. Unique library paths isolate process-global counters
while Cargo runs tests concurrently; test-owned temporary directories retain the
library until their CUDA handles are dropped. These native fixture tests are
Unix-only. CI enables `omni-laya/serve`, compiling the native model/worker and
running its root-tree private tests on Linux.

The registered tests at this stage cover resolved-handle lifetimes, exact dispatch traces and sparse wrappers.\nThey validate host behavior without Laya tensor math or a CUDA driver.\n