# Native execution runtime

`omni-runtime` provides `SerialScheduler` for the Cua-S1, Open-Jev and Laya native
workers. Their engines assemble a processor, one scheduler for the loaded
executor, and the executor. HTTP handlers still coordinate
`prepare` → `execute` → `finish`.

The scheduler owns FIFO admission and blocking dispatch. A caller waits
asynchronously for one execution permit before entering Tokio's blocking pool.
Cloning a scheduler shares that admission queue; separate loaded executors have
separate queues. The model mutex remains a guard for Rust's mutable executor
state. Waiting requests no longer occupy blocking threads waiting for that mutex.

| Worker | Admitted unit | Work outside admission |
| --- | --- | --- |
| Cua-S1 | One question's unpadded Qwen forward. Release before admitting its next question. | Request preparation, CPU letter projection and response finishing. |
| Open-Jev | One complete request's independent candidate forwards and CPU scalar heads. | Request preparation and calibrated response finishing. |
| Laya | One complete padded request, including GPU scorer/action head and copyback. | Request preparation/padding and calibrated response finishing. |

Models retain weights, heads, device state, scratch buffers and graph caches.
The scheduler takes an owned closure and returns its result; it does not know
model identifiers, question/candidate mappings, tensor layouts or backend APIs.
The existing numerical, ordering, whole-request failure and readiness contracts
remain in the processors, executors and worker handlers.

Cancellation while waiting removes the caller from admission without dispatch.
After dispatch, the blocking closure retains the permit and its captured
resources until completion, even if the caller disconnects or stops waiting.
Executors must finish queued device work before returning; the current Qwen
forward synchronizes the final hidden-state download. Errors and panics release
admission when the blocking task finishes. Model mutex poisoning retains its
existing failure behavior.

This implements serial admission per executor within a worker process. Admitted
concurrency is one. Queue-length limits, token budgets, compatibility grouping,
shared processing orchestration and cross-request GPU batching remain planned
for this scheduler. The shared Qwen executor supports single prompts and bounded
packed prefill: Open-Jev-27B-v1.1's model adapter groups up to 16 candidates and
4096 tokens, with longer prompts run alone, while Open-Jev-9B uses single-prompt
calls. `SerialScheduler` does not pack those inputs or coordinate separately
running workers. See the [architecture contracts](../../docs/architecture.md)
for grouping and reconstruction ownership.

## In-process engine interface

The separate [`engine` module](src/engine.rs) defines a byte-oriented `Engine`
contract: readiness, request submission with a deadline, a reply channel and
optional queue/failure reporting. Implementations own admission and their queue;
the trait does not supply a queue or batching policy. The
[frontend library](../frontend/README.md#in-process-library-transport) maps this
interface onto HTTP. The default `omni-jev` binary still forwards to an external
HTTP worker; the interface alone does not migrate existing native workers.

CPU tests cover FIFO serialization, queued cancellation, retention of admission
and resources after dispatched cancellation, errors/panics, and independent
loaded executors:

```sh
cargo test --locked -p omni-runtime
```

See the [architecture contracts](../../docs/architecture.md) and
[benchmark protocol](../../benchmarks/README.md) when changing admission policy.
