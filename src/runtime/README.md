# Native execution runtime

`omni-runtime` provides `SerialScheduler` for the Cua-S1 and Open-Jev native
workers. Their engines assemble a processor, one scheduler for the loaded
executor, and the executor. HTTP handlers still coordinate
`prepare` → `execute` → `finish`.

The scheduler owns FIFO admission and blocking dispatch. A caller waits
asynchronously for one execution permit before entering Tokio's blocking pool.
Cloning a scheduler shares that admission queue; separate loaded executors have
separate queues. The model mutex remains a guard for Rust's mutable executor
state. Waiting requests no longer occupy blocking threads waiting for that mutex.

Pending work is bounded to 64 execution units per loaded executor by default.
`OMNI_NATIVE_MAX_PENDING` sets a positive integer limit for either native worker;
invalid values fail startup before checkpoint or CUDA loading. The limit counts
waiting and dispatched units together, including work whose caller stopped
waiting. It is separate from the execution concurrency, which remains one.
At capacity, the scheduler immediately returns `Overloaded` without dispatching
the rejected unit. Workers return HTTP `503` with
`native execution capacity exhausted` in their existing error shape (`detail`
for Cua-S1, `error` for Open-Jev). The frontend forwards this response without
retrying. `SerialScheduler::new(limit)` configures the same bound for Rust callers;
`Default` uses 64 independently of environment variables.

| Worker | Admitted unit | Work outside admission |
| --- | --- | --- |
| Cua-S1 | One question's unpadded Qwen forward. Release before admitting its next question. | Request preparation, CPU letter projection and response finishing. |
| Open-Jev | One complete request's independent candidate forwards and CPU scalar heads. | Request preparation and calibrated response finishing. |

The limit bounds execution units, not ingress requests or prepared-token memory.
Preparation still precedes admission. Cua-S1 can be rejected on a later question
after earlier forwards; the entire request fails with no partial answer body.
Open-Jev is accepted or rejected as one complete request unit. Individual prompt
and HTTP body limits remain model-specific; aggregate token budgets are planned.

Models retain weights, heads, device state, scratch buffers and graph caches.
The scheduler takes an owned closure and returns its result; it does not know
model identifiers, question/candidate mappings, tensor layouts or backend APIs.
The existing numerical, ordering, whole-request failure and readiness contracts
remain in the processors, executors and worker handlers.

Cancellation while waiting removes the caller and releases pending capacity
without dispatch. After dispatch, the blocking closure retains both permits and
its captured resources until completion, even if the caller stops waiting.
Executors must finish queued device work before returning; the current Qwen
forward synchronizes the final hidden-state download. Errors and panics release
admission and pending capacity when the blocking task finishes. Model mutex
poisoning retains its existing failure behavior.

This implements serial admission per executor within a worker process. Admitted
concurrency is one. Token budgets, compatibility grouping,
shared processing orchestration and GPU batching remain planned. The Qwen
executor still accepts one prompt per forward; this scheduler does not pack
inputs or coordinate separately running workers.

CPU tests cover FIFO serialization, queued cancellation, retention of admission
and resources after dispatched cancellation, errors/panics, and independent
loaded executors, full-capacity rejection and recovery. Native worker tests also
cover startup configuration and overload/error HTTP responses:

```sh
cargo test --locked -p omni-runtime
cargo test --locked -p omni-cua-s1-native -p omni-open-jev-native
```

The [native admission comparison](../../docs/benchmarks/native-admission/README.md)
records normal-load parity and performance ranges, plus limit-one overload and
recovery on both real workers.

See the [architecture contracts](../../docs/architecture.md) and
[benchmark protocol](../../benchmarks/README.md) when changing admission policy.
