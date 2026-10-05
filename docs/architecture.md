# Architecture and integration contracts

System1-Omni's target design separates processing and scheduling from model
execution. These contracts define ownership and integration boundaries for that
design. Concrete input/output types follow each executor's supported layout.

## Implementation status

The [Rust frontend](../src/frontend/README.md) currently forwards HTTP requests
to separately running workers. Cua-S1 and Open-Jev have native Rust/CUDA workers
that share the [Qwen3.5/3.8 executor](../src/models/qwen3_5/native/). Their
model-specific workers coordinate independent processor and executor modules
through `prepare` → `execute` → `finish`. The shared Qwen executor accepts one
prompt per forward call. Both workers use the
[native runtime](../src/runtime/README.md) for FIFO admission and blocking dispatch
per loaded executor. Shared processing orchestration, batch budgets,
compatibility grouping and dynamic batching are planned.

The native workers currently compute their decision heads on the CPU after
downloading the final hidden state. GPU head execution belongs to the target
model/backend integration. LAYA's native executor and the Metal backend are
also planned; Python workers retain their documented reference/serving roles.

## Native worker boundaries

Both native workers separate `processing.rs` from `executor.rs`; `engine.rs`
assembles them with a `SerialScheduler` per loaded executor, and the HTTP handler
coordinates the three stages. Preparation validates the entire request before
any forward call and returns executor inputs plus a response context. The context retains question
and candidate identity, usage, and response metadata outside the executor.

| Worker | Prepared executor inputs | Executor outputs | Response finishing |
| --- | --- | --- | --- |
| Cua-S1 | One unpadded token-ID vector and option count per question, in request order. | One FP32 answer-letter logit vector per question. | Per-question softmax, choice/confidence, ordered answers, and token usage. |
| Open-Jev | Token-ID vectors grouped by question, then independent candidate, in request order. | One FP32 learned scalar per candidate in the same grouping. | Add the `noul` false logit of zero, calibrate across each complete question, and restore typed answers, usage, and metadata. |

These input collections are serial work, not GPU batches. Shared runtime
admission precedes blocking dispatch: Cua-S1 admits one question forward at a
time; Open-Jev admits one complete request. Cua-S1's CPU letter projection stays
outside admission; Open-Jev's scalar heads remain inside its request unit. The
model mutexes guard mutable state, retaining per-question/request granularity.
Executors own the loaded Qwen model and CPU head weights, preserving FP64 accumulation and
the existing FP32 rounding and bias order. Finishing checks output cardinality
before reconstruction. HTTP validation, error status/body conventions, and real
warmup before readiness remain model-specific and unchanged.

## Layer ownership and implementation language

| Component | Owns | Native target implementation |
| --- | --- | --- |
| Frontend | HTTP transport, forwarding, and response delivery. | Rust. |
| Pre/postprocessors | Model-specific request validation, prompt/token preparation, modality transforms, calibration, and response interpretation. | Rust CPU processing; optional GPU transforms use hardware backends. |
| Scheduler / batcher | Admission, queues, batch budgets, compatibility grouping, request bookkeeping, and result routing. | Rust host policy; device packing operations use hardware backends. |
| Model executor | Weights, forward orchestration, learned heads, device state, supported layouts, and kernel selection. | Rust orchestration calling CPU or GPU operations. |
| Hardware backends | Allocation/stream interfaces, dispatch, and hardware operations used by executors or processing modules. | Rust bindings/dispatch with CUDA C++ kernels or Metal shaders; native host glue follows the backend ABI. |

The runtime and executor can share one worker process. Request bookkeeping
belongs to the runtime; device buffers, scratch space, and graph caches belong
to the executor/backend integration. CUDA and Metal can use different internal
layouts and kernels while satisfying the same model-specific input/output
semantics.

## Processor and executor contracts

- A preprocessor converts a request into prepared work and the context needed
  to reconstruct its response. It preserves model identifiers, question and
  candidate order, prompt templates, special tokens, input limits, and token
  accounting specified by the model's contract.
- Prepared work retains its originating request, question, and candidate
  identity. The scheduler uses its validated lengths and executor constraints
  to select work; model-specific batch adapters assemble the supported input
  layout and unpack outputs back to those identities.
- The executor consumes the declared layout and produces corresponding model
  outputs. Learned projection/head computation belongs to model execution;
  calibration and API interpretation remain model-specific postprocessing.
  Changing where an operation runs preserves its numerical contract.
- Postprocessing receives every output needed for the original question and
  applies that model's normalization, decision, confidence, and response rules.
  Preserve whole-request failure behavior, error status/body, and usage fields
  when splitting or regrouping work.

Cua-S1 prepares one prompt per question and reads option-letter logits.
Open-Jev prepares independent candidate prompts and normalizes across the
complete question's candidates. A request, question, and GPU batch therefore
have different boundaries. Scheduler grouping must preserve those distinctions;
probabilities must not be normalized across unrelated questions or requests.

## Scheduling and batching contracts

The implemented serial scheduler limits admitted execution to one unit per loaded
executor, with FIFO waiting. Engines own scheduler instances; model-specific
execution adapters declare the unit and submit owned closures. Waiting is async,
so queued requests do not occupy blocking threads waiting for a model lock.
Schedulers are local to each worker/executor; this does not coordinate separate
processes. Queue-length limits and token budgets are not implemented yet.

- Batch only work accepted by the same loaded executor, with compatible
  checkpoint/adapter identity, device, dtype, and input layout. Models declare
  additional constraints and limits; the scheduler owns queue and admission
  policy rather than duplicating it inside each model implementation.
- Preserve attention masks, positions, sequence boundaries, recurrent-state
  isolation, and the final-position readout when padding or packing prompts.
  Padding and batch storage do not add tokens to the API's input-token usage.
- Return outputs to their originating requests, questions, and candidates in
  the model's defined order. Regrouping independent candidates must still wait
  for the complete question before applying its calibrated normalization.
- Dynamic batching requires an executor and kernels that support the selected
  batch layout. Moving a serial prompt loop into a shared module alone does not
  provide batched GPU execution. Begin with bounded prefill batching within the
  implementation's declared limits.

## Backend and lifetime contracts

Rust host code selects and launches backend operations. CUDA/Metal perform
device tensor computation, including model math and optional processing
transforms or batch packing. Those operations retain their processor/batcher
ownership even when their kernels live under `src/backends/`.

Keep Rust declarations and the native ABI in sync, and rebuild all consumers
when the ABI changes. Buffer handles and captured graphs must outlive queued
device operations; synchronize before reading host results or reclaiming
storage. Cancellation while waiting for admission removes the queued caller
without dispatch. Once dispatched, the runtime retains admission and captured executor
resources until the blocking closure completes, even if its caller is cancelled.
The executor must synchronize device work before returning; cancellation or
timeout does not authorize freeing buffers still in use by submitted work. Keep
readiness tied to successful initialization and the worker's real warmup;
extraction into shared layers preserves that behavior.

## Integration and validation

Keep model-specific processors and batch adapters separate from forward
implementations, even when colocated under `src/models/<model>/`. Reuse the
current worker interface and serial runtime while further runtime components
remain unimplemented.
Extract common orchestration when implementing that layer, with explicit scope
and current consumers; adding a model alone does not require a new framework.

Use the model's pinned reference and fixtures to check processing/execution
boundaries. For batching changes, verify ordering, state isolation, masks,
readout positions, whole-question normalization, errors, and token accounting,
plus numerical parity for affected executors. Follow the
[contribution checks](../CONTRIBUTING.md) and
[benchmark protocol](../benchmarks/README.md) for the changed behavior; report
implemented support separately from target design and observed validation.
