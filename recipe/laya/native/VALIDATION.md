# Native Laya validation

GPU measurements: 2026-09-27, v8. [Results, timing samples, and measured source hashes](validation-v8.json).

## Environment and scope

One NVIDIA H800, BF16, Hopper `sm_90a`; Laya 0.3.20, PyTorch 2.11.0+cu128, TileLang 0.1.14, nvcc 13.0.88. Checkpoint: `convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.

The reference uses official fast with CUDA Graphs and the same selected RoPE as the native engine. Tests use the repository's [requests](fixtures.json) and [held-out requests](heldout-fixtures.json).

## GPU checks

| Check | Result |
| --- | --- |
| Regular requests | 12/12 exact responses in eager mode and 12/12 with Graph |
| Held-out requests | 8/8 exact responses |
| Option-count / temperature cases | 7/7 exact responses |
| Five boundary requests | Same decisions; maximum numeric error 0.0014, within the existing 0.002 limit |
| Attention boundaries | 12 shape/window cases; rows with keys match the reference, empty key ranges produce zero |
| HTTP | C1/C8 responses consistent; oversized request rejected; server remains healthy; SIGTERM exits normally |

Twenty named intermediate checks match bitwise on valid tokens. The Attention fix sets empty-key padding rows to zero. Validation covers numerical and serving parity on these inputs.

## Performance

Each case used 10 warmup requests and 50 timed requests per round, at concurrency 1 without a profiler. Execution order was fast, native, native, fast. Values below are the mean of the two per-round medians.

| Request | Official fast + same RoPE | Native + RoPE |
| --- | ---: | ---: |
| Short, one question | 2.842 ms | 1.809 ms |
| Short, three questions | 3.667 ms | 2.203 ms |
| Long, one question | 4.357 ms | 3.098 ms |
| Long, three questions | 9.054 ms | 7.355 ms |

Native timing includes JSON parsing, packing, CUDA execution, decoding, and JSON/pipe output. Reference timing covers `Router.predict` and completion synchronization. Loading and first-shape allocation, warmup, and capture are excluded.

Native HTTP C1 median was 2.004 ms, measured separately from the engine comparison above.

Build using the [README](README.md), then run the paired benchmark on the same otherwise idle GPU:

```sh
python recipe/laya/native/benchmark.py fast "$CHECKPOINT" "$BUNDLE" fast-r1.json
python recipe/laya/native/benchmark.py native "$CHECKPOINT" "$BUNDLE" native-r1.json
python recipe/laya/native/benchmark.py native "$CHECKPOINT" "$BUNDLE" native-r2.json
python recipe/laya/native/benchmark.py fast "$CHECKPOINT" "$BUNDLE" fast-r2.json
python recipe/laya/native/http_acceptance.py "$CHECKPOINT" "$BUNDLE" http-results.json
```

The native process uses the supplied checkpoint; the reference helper pins and loads the revision above. Use that same snapshot for `$CHECKPOINT`.

## Code cleanup and CPU checks

After the GPU runs, `0b4876e` added decoder comments and a CPU regression test without changing production behavior. The later `460f263` cleanup changed formatting, build-script structure, and private HTTP helpers. CUDA token comparisons, Python AST comparisons, build command/output comparisons, and code review passed. Model execution, RoPE, Attention, preprocessing, weights, and dependencies remained unchanged.

Recorded local checks for `460f263`: formatting, strict Clippy, 22 CPU tests, and the release build passed. Four checkpoint/oracle tests were ignored in that local run.

The GitHub CPU workflow checks all Rust features. It does not compile CUDA kernels or validate model execution, numerical parity, or GPU latency; those results must be reported separately.
