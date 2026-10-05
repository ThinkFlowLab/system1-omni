# Laya CUDA validation

The combination contains the selected RoPE tile, short QKV/output dispatch,
down N64, short GEGLU selection and long GEGLU BN32/static registers. It preserves
the original arithmetic and checkpoint precision. Failed/HOLD candidates are
excluded; this does not claim zero GPU bubbles or maximum occupancy.

## Frozen native engine comparison

These measurements precede the current-main integration. They used one frozen
Rust CLI with the original and optimized CUDA configurations, on one shared
H800, concurrency 1, FP16 token embeddings, BF16 projection matrices/activations and FP32 residuals/norms. Clocks
were not locked. JSON parsing, tokenization, padding/upload, forward/heads,
readback, decoding and response JSON write are included. HTTP, process/model
startup and warmup/cold Graph construction are excluded.

| Graph input | Original p50 | Combined p50 | Latency reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| choice, L48 | 2.5426 ms | 1.5957 ms | 37.24% | 1.59× |
| score, L64 | 2.5819 ms | 1.6101 ms | 37.64% | 1.60× |
| short, L48 | 2.5419 ms | 1.5959 ms | 37.22% | 1.59× |
| medium, L176 | 2.8666 ms | 1.9207 ms | 33.00% | 1.49× |
| long, L512 | 3.9926 ms | 3.0001 ms | 24.86% | 1.33× |

Each entry pools two 100-sample passes. The 12 processes measured 6000 requests
across original/retained/combined variants, Graph/eager, in forward/reverse order.
All raw engine/client timings and original file hashes are in
[measurements.json](measurements.json); private host paths are omitted.
The original uses `--original-rope` and the reverse-patched exporter; both sides
share the same runtime and Graph setting. There is no single workload-weighted
percentage. The full pipeline was measured directly, not by adding kernel gains.

The frozen CLI binary SHA256 is
`4d31e8593ff67127c5c908c3bdba26db393165036027e58a5554b70567a71c85`;
the combined library is
`eeb94478354b5b6f0b26c756263261dd61908a0aafd82f10b04b907c3d46cc36`.
Source kernels/runtime and exported library identity are retained in the evidence.
The earlier three-way numeric check covered 210 responses, 36 hidden-state
comparisons and six 17-request reuse sequences, bitwise equal to the original
native configuration. This tests implementation equivalence for fixed inputs,
not general model quality or all-input agreement with official PyTorch.

## Integration checks

The current-main integration retains checkpoint inventory checks and imports the
reviewed CPU parsing/decoding fixes. Local checks passed: workspace fmt/clippy,
83 Rust tests (10 fixture/checkpoint/GPU tests ignored), release build, eight
CUDA build-entry tests, eight benchmark tests and strict docs build.

Linux `laya-run`, `omni-laya` and `omni-jev` were built from commit
`10c14a34c870d9d4a6562e1dcf7ca787a716730f`. Subsequent changes only update docs
and normalize blank context lines in the baseline reproduction patch; execution
source and dependencies are byte-identical. The build used Rust 1.98.1 and the
same pinned CUDA library, checkpoint and fixtures as the engine comparison.

The new binaries passed actual H800 acceptance with those 12 fixtures:

- Graph and eager CLI: 24 complete JSON responses exactly match the fixed
  official reference; all 24 raw-head results match the original native engine
  with exact FP32 bits and request-derived row widths.
- Native worker and frontend proxy: 24 successful HTTP responses exactly match
  the same official JSON. Four malformed-JSON/content-type requests return
  422/415, and four health checks report ready before and after rejection.
- All four child processes exit with code 0; the worker and frontend shut down
  through SIGINT. Native process maps contain the registered CUDA library and
  no Python/Torch runtime. The run's GPU processes and personal lock are released.

JSON and raw-head comparisons use different references. The official reference
comes from Laya 0.3.20's fast Graph path with selected RoPE and BF16 autocast.
Four action values in `long_3`/`truncated_3` already differed from that reference
in the original native engine; they remain unchanged here. Therefore this check
establishes native implementation equivalence and exact response JSON for the
fixed inputs, not bitwise raw-head agreement with PyTorch or general model quality.
The first runner incorrectly mixed these two references; its failure was retained
and corrected by binding the original native raw outputs, without adding tolerance.

Complete responses, per-call output digests, test counts, binary/fixture/library
identities and the original native raw reference are in `integration_validation` in
[measurements.json](measurements.json). These are functional checks; current
HTTP latency and inference-failure injection were not measured. The timing table
above remains historical engine evidence.

Build and reproduction commands are in the [recipe](README.md). Full model
weight/tokenizer oracle checks remain explicit opt-in tests, with their pinned
artifact hashes; see the [model contract](../../../src/models/laya/README.md).
