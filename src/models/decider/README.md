# Decider-2B v11 native worker

The native Rust/CUDA worker serves text `choice`, `noul` and isolated-level `score`
through `POST /v1/systemone`. Follow the [native recipe](../../../recipe/decider/README.md).
The [validation protocol](../../../recipe/decider/validation.md) distinguishes
CPU contract checks, real checkpoint inference and HTTP checks.

## Pinned artifacts and contract

- Model/tokenizer/config: Mapika/decider-2b
  `533964dae8be954c5b5e19fa4948e48408094c1e` (released 2B v11).
- Reference: Mapika/decider `50d0be0d7cb43d2066965ce5fa7f3fe4e489a60f`, decider-ai 1.8.1.
- BF16, 24 Qwen3.5 layers, hidden width 2048, 18 Gated DeltaNet and six full-attention
  layers. Output embeddings are tied; there is no adapter merge or separate trained head.
- Plain state-first, independent questions; Score levels become separate no/yes rows.
  Choice supports 2–255 single-token labels A through JT; Score supports 2–10 levels.
- Per-type temperatures are Choice 1.164, Noul 1.624 and Score 1.124, read from the
  verified released configuration. The worker accepts only the original calibration
  artifact. The CPU library's configuration API also supports explicit calibration
  overrides, but those are outside the pinned worker's supported scope.
- Preserve ordered structured state/question rendering, long-array annotations,
  per-question output ordering, confidence/certainty/legend/level-fit fields and
  Python-style response rounding. `usage.input_tokens` counts the common prefix once;
  `usage.output_tokens` is zero. Empty questions return empty answers without execution.

State truncation caps the tokenized `Context:` prefix at 32,768 tokens; the question
suffix is additional. Native admission bounds complete rows to 36,864 tokens,
expanded rows to 1,024, total processed row tokens to 1,048,576 and raw bodies to
8 MiB. Unsupported modes, duplicate JSON keys, depth >=128, nonfinite literals
and integers outside i64/u64 are rejected. No image/video, chat/schema-first,
packed-question execution, neutralization, quantization, CPU/Metal inference,
cross-request prefix cache is included. CUDA Graph replay and request-local prefix
reuse are optional and default off. Prefix integration is a pending integration until
Qwen PRs #98/#99 merge.

## Ownership and execution

`processing.rs` compiles the entire request before device work. Prepared rows own
unpadded token IDs, final-position readout, candidate IDs and original identities;
`ResponseContext` owns whole-question calibration and answer reconstruction.

`checkpoint.rs` verifies fixed sizes and SHA-256 hashes for the checkpoint,
configuration, calibration and tokenizer before CUDA initialization. It rejects
sharded indices that could redirect the shared loader away from the verified file.
Keep artifacts immutable for the entire worker lifetime. Selected BF16 output rows
are loaded from the pinned tied input embedding; all 255 rows are retained, with a
256th zero row for aligned CUDA GEMM storage.

`executor.rs` owns the shared Qwen model and a separate CUDA projection stream,
GEMM handle and persistent head buffers. Each complete request gets one shared
`SerialScheduler` permit. Default execution runs rows eagerly in order. Optional request-local packing
uses `DECIDER_BATCH_MAX_ROWS` (1–4, default 1) and
`DECIDER_BATCH_MAX_TOKENS` (1–4096, default 4096). Contiguous complete rows are
packed with the shared backbone `forward_batch`; each sequence resets its own
GDN/attention state. A row above the packing budget runs alone under the unchanged
complete-row limit. A persistent head projects all batch hidden rows in one GEMM.
Score levels can cross batch boundaries; response normalization still uses the
complete question. This does not combine separate requests or change admission. The BF16 selected projection uses existing backend GEMM, then converts
its BF16 outputs to FP32 before CPU temperature scaling and normalization. Padding
never participates in softmax or token accounting. Current GEMM/reduction algorithms
can differ from Transformers; numerical agreement is measured rather than assumed.

Decider uses an explicit shared-model loader option controlled only by
`DECIDER_GRAPH`; existing workers retain their `CUA_S1_GRAPH` constructor behavior.
The backbone caches up to 64 ordered sequence-length shapes, clears them before
scratch growth, and falls back permanently to eager if capture fails. Effective
mode and cumulative capture/replay counters appear in health/diagnostic metadata.
The selected head and calibrated response remain outside the graph.

`prefix.rs` independently plans exact request/question token prefixes from complete
prepared rows. `DECIDER_PREFIX=1` uses shared Qwen continuation and fixed GEMM
selection; `DECIDER_FIXED=1` provides independent fixed full rows for a matched
control. Short/no-sharing cases run independent fixed rows. These paths require
Graph off. The selected head keeps identical batch grouping; calibrated responses
and unique-prefix usage remain unchanged. Prefix snapshots/KV are overwritten per
request and never reused across calls. The backend/consumers require ABI7 rebuilds.

Both streams synchronize on completion, errors and caught execution panics before
admission is released. An execution failure retires the loaded model/head and makes
health unavailable. Validation errors preserve readiness. Queued cancellation
removes the caller; after dispatch, resources and admission remain held until work
completes. FIFO waiting is serial and inherits the current runtime's unbounded
pending queue; no new queue/token scheduling policy is claimed.

`engine.rs` assembles the processor/executor and performs a real warmup before
binding a socket. `serve.rs` handles transport, status codes and worker health.
HTTP CPU preparation uses a blocking task outside GPU admission and the model lock.
The existing Rust frontend forwards bytes to this separately running worker.

## Tests

All test bodies and fixtures live under root `tests/decider`. Normal Cargo tests
require neither CUDA nor downloads. Explicit checkpoint/accelerator tests are
registered and ignored until their prerequisites are supplied; see the recipe.
