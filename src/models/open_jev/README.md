# Open-Jev-27B-v1.1

The [native Rust/CUDA worker](../../../recipe/open_jev/native.md) uses the
pinned Qwen3.8-27B backbone, merged LoRA adapter, trained scalar decision head,
and saved calibration temperature. It supports choice, ordinal score, and yes/no
text decisions through the existing Rust frontend.

The request compiler and response formulas follow
[Open-Jev @ 3308a15](https://github.com/Zefan-Cai/Open-Jev/tree/3308a15ccd7eea1df7a37d6ddc39b023b801ba16).
The CUDA prefill implementation is shared with Cua-S1 under
[`../qwen3_5/native/`](../qwen3_5/native/). See the recipe for preparation,
numerical limitations, validation and optimization scope.

The Rust contract is adapted from Open-Jev's MIT-licensed code; its copyright
and license are retained in [`native/LICENSE.open-jev`](native/LICENSE.open-jev).
No model weights are distributed here.

## Integration boundary

The native worker follows the [architecture contracts](../../../docs/architecture.md)
with independent [processing](native/src/processing.rs) and
[executor](native/src/executor.rs) modules. Preparation returns token IDs grouped
by question and candidate, plus the response context for identity/order,
calibration, usage, and metadata. The executor owns the trained head and returns
one FP32 scalar per candidate. Response finishing adds the `noul` false baseline
and applies calibrated normalization across each complete question. The worker
retains request-wide model locking and independent single-prompt execution, with
the scalar head on the CPU after CUDA prefill. The engine owns a
[shared serial scheduler](../../runtime/README.md) that admits the complete request
before blocking dispatch. GPU batching and batch budgets remain planned.

Pending complete request units are bounded to 64 by default, configured by
`OMNI_NATIVE_MAX_PENDING`. Full capacity rejects the request before its first
forward with HTTP `503` and `{"error":"native execution capacity exhausted"}`.
Preparation precedes admission; aggregate token budgets remain planned.
