# Audited repository map

Implementation baseline: `ThinkFlowLab/system1-omni@873655b4484dc2f537658644ebefd03630e9d507` (source and configuration checked 2026-10-04). The [architecture contracts](../../../../docs/architecture.md) describe the target design alongside implemented status. Revalidate both at the actual PR revision. Permalinks use `https://github.com/ThinkFlowLab/system1-omni/blob/<sha>/<path>`.

## Serving and model ownership

- `README.md`, `src/frontend/README.md`, `src/frontend/src/lib.rs`, `src/frontend/tests/frontend.rs`: Axum/Tokio/Reqwest frontend streams uploads and buffers responses so response-body timeouts can return 504; connection failures return 502. One pooled client uses a 60-second total deadline, no retries, no redirects, no environment proxies. Authorization/end-to-end headers pass through, hop-by-hop headers and connection-nominated headers do not. Backend configuration accepts a path prefix but rejects credentials/query/fragment; request query is forwarded. Health preserves the worker's status/body. Model-specific parsing belongs to workers.
- `docs/architecture.md`: independent processors and batch adapters preserve model semantics; the planned shared Rust runtime owns queues, admission, compatibility grouping, request bookkeeping, and result routing. Model executors own weights, forward orchestration, learned heads, and device state. Rust bindings dispatch CUDA/Metal device operations, including optional processing transforms and packing. Review supported executor layouts, state/lifetime isolation, origin/order reconstruction, whole-question normalization, and token accounting when these boundaries change.
- `Cargo.toml` and `.github/workflows/ci.yml`: baseline workspace members are frontend, Cua-S1 native, Open-Jev native, shared Qwen3.5/3.8 native execution, and the LAYA checkpoint reader. The default Rust suite can pass without CUDA because native code dynamically loads the kernel library. Do not infer GPU validation from workspace success or require GPU for transport-only unit tests.
- `src/models/qwen3_5/native/src/model.rs`, `src/models/cua_s1/native/src/engine.rs`, `src/models/open_jev/native/src/engine.rs`: the shared forward takes one prompt; current workers serialize access and compute model-specific heads on the CPU after CUDA prefill. Shared runtime orchestration, dynamic batching, and GPU heads in the target design are not existing capabilities.
- `src/models/laya/README.md`, `recipe/laya/README.md`, `src/backends/metal/README.md`: support is model/backend-specific. At this baseline Laya external Python worker support is documented while its native engine and Metal are planned. Inspect an incoming implementation on its own merits; do not repeat baseline status after it changes.

## Cua-S1 text reference contract

Read `src/models/cua_s1/README.md`, `src/models/cua_s1/text/contract.py`, `src/models/cua_s1/text/model.py`, `src/models/cua_s1/native/src/contract.rs`, `src/models/cua_s1/native/src/engine.rs`, shared `src/models/qwen3_5/native/src/json.rs`, `tests/cua_s1/test_text_contract.py`, and `tests/cua_s1/test_text_server.py` when affected.

- Pinned upstream reference: trycua/cua `0e75660ce4c2edda519e0c795fa3ad98abf4e76f`; base Qwen3.5-4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; adapter `16818868b0cc7813808aae4e87b417657046ab79`. The independent text and multimodal LoRA adapters are not interchangeable.
- One forward pass per question, no decoding; 1–26 options in request order map to A–Z. The chat template retains the upstream `<think>` suffix; disabling thinking changes tokenization. Preserve Python-compatible JSON spacing/Unicode/escaping and object insertion order, including null criteria fallback and text spelling special tokens.
- Text choice only: malformed JSON/UTF-8, duplicate keys, non-finite/out-of-range numbers and lone surrogates are 400; unsupported model/question or invalid semantic input is 422. Score/noul are not implicitly supported by this adapter.
- Probabilities read the final-position option logits; choice ties select the earliest option. Confidence is normalized entropy with single-option confidence 1; it is not upstream's p_max. Response identity includes adapter revision/modality; input usage sums prompts and output tokens are 0. Confirm native/reference numerical differences rather than assuming merged LoRA is exact.
- The native loader expects `cua_s1_export.json` before accepting merged weights. Check safetensors shape/dtype/index/tokenizer consistency and readiness failure paths when exports change.

## Open-Jev integration contract

Read `src/models/open_jev/README.md`, `src/models/open_jev/native/src/contract.rs`,
`src/models/open_jev/native/src/engine.rs`, `src/models/open_jev/native/src/main.rs`,
`tests/open_jev/contract.rs`, `tests/open_jev/tokenization.rs`, and
`recipe/open_jev/native.md` when affected.

- The reference compiler/readout is pinned to Open-Jev `3308a15ccd7eea1df7a37d6ddc39b023b801ba16`; backbone and checkpoint identities are checked in the exported manifest. Cua-S1's prompt, option limit, JSON ordering, and confidence formula do not apply to this model.
- Each candidate has an independent prompt and scalar head score. Preserve candidate order and sorted structured-input rendering. Normalize across each complete question using the saved temperature; `noul` uses logits `[0, score]`. Candidate regrouping must preserve question identity and token usage.
- The worker validates and tokenizes every candidate before inference, warms up before binding, and uses the shared Qwen executor/CUDA ABI. Current scoring is independent single-prompt execution with a CPU head. Use `recipe/open_jev/validation.md` for the scope and limitations of full-checkpoint comparisons.

## CUDA, graph and compatibility evidence

`src/backends/cuda/qwen3_5/{ops.h,runtime.cu,attention.cu,gdn_prefill.cu,build.sh}` and `src/models/qwen3_5/native/src/{cuda.rs,model.rs}` form the shared ABI/lifetime boundary used by Cua-S1 and Open-Jev. The audited ABI is 4; changes may legitimately bump it but Rust declarations and library must agree. Tensor-core code needs sm_80+; documented device coverage is specific to the workers and tests in `docs/supported-models.md` and the model recipes. Norm/elementwise/qk operations preserve reference BF16 rounding; attention/Gated DeltaNet have their own intermediate precision.

The shared graph path is opt-in through `CUA_S1_GRAPH=1` and keyed by exact prompt length, with at most 64 captures. Read actual code for scratch growth invalidation, updated input upload, buffer lifetime and capture error recovery; use `tests/qwen3_5/kernels.rs` as a starting point, not proof of execution. GPU tests are ignored by default. CUDA compilation, CPU fixture/tokenizer tests and checkpoint export do not establish full native parity or frontend-proxied inference. An ABI bump requires rebuilt consumers and library, not just a Rust test pass.

For a native/reference comparison use the target model's predeclared gates. Cua-S1's documentation specifies exact reference token IDs, direct-vs-proxy byte parity, and a native-vs-fp32 tolerance derived from BF16-reference drift plus a top-two-margin criterion; do not replace it with another model's ad hoc threshold.

## Test routing and measurements

- Rust CI: fmt, strict Clippy with all targets and locked dependencies, workspace tests, release build. Record test pass and ignored counts; inspect any newly relocated explicit Cargo test targets.
- `.github/workflows/ci.yml`, `benchmarks/README.md`, `benchmarks/requirements.txt`, `tests/benchmarks/test_bench.py`: `python benchmarks/bench.py validate benchmarks/smoke.jsonl` and `python -m unittest discover -s tests/benchmarks -p 'test_*.py' -v`. This uses Python 3.11+/httpx; four synthetic smoke requests are not a meaningful accuracy dataset or throughput workload.
- Benchmark runner preserves manifest/config hashes, raw responses and failures; hardware metadata is operator supplied and needs corroboration. Read the documented rounded-probability validator limitation and any later regression fix before relying on results. Incomplete A/B runs are not validated baselines. Keep successful-only latency denominators and all errors visible.
- `CONTRIBUTING.md`, `.github/workflows/docs.yml`, `mkdocs.yml`, `docs/hooks.py`: docs changes need verified repository/site links, claims and `mkdocs build --strict` in the documented environment; no GPU campaign for documentation-only edits.

Baseline inspection does not prove any future review loaded the skill or ran these checks. Every report must supply its own provenance and execution evidence.
