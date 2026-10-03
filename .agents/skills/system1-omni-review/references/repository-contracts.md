# Audited repository map

Baseline: `ThinkFlowLab/system1-omni@d2665e1fa867360cd96e45a87b6bc0b1506eac47` (main read 2026-10-01). Revalidate at the actual PR revision. Permalinks use `https://github.com/ThinkFlowLab/system1-omni/blob/<sha>/<path>`.

## Serving and model ownership

- `README.md`, `src/frontend/README.md`, `src/frontend/src/lib.rs`, `src/frontend/tests/frontend.rs`: Axum/Tokio/Reqwest frontend streams uploads and buffers responses so response-body timeouts can return 504; connection failures return 502. One pooled client uses a 60-second total deadline, no retries, no redirects, no environment proxies. Authorization/end-to-end headers pass through, hop-by-hop headers and connection-nominated headers do not. Backend configuration accepts a path prefix but rejects credentials/query/fragment; request query is forwarded. Health preserves the worker's status/body. Model-specific parsing belongs to workers.
- `Cargo.toml` and `.github/workflows/ci.yml`: baseline workspace members are frontend and Cua-S1 native. The default Rust suite can pass without CUDA because native code dynamically loads the kernel library. Do not infer GPU validation from workspace success or require GPU for transport-only unit tests.
- `src/models/laya/README.md`, `recipe/laya/README.md`, `src/backends/metal/README.md`: support is model/backend-specific. At this baseline Laya external Python worker support is documented while its native engine and Metal are planned. Inspect an incoming implementation on its own merits; do not repeat baseline status after it changes.

## Cua-S1 text reference contract

Read `src/models/cua_s1/README.md`, `text/contract.py`, `text/model.py`, `native/src/contract.rs`, `native/src/json.rs`, `native/src/engine.rs`, `tests/cua_s1/test_text_contract.py`, and `test_text_server.py` when affected.

- Pinned upstream reference: trycua/cua `0e75660ce4c2edda519e0c795fa3ad98abf4e76f`; base Qwen3.5-4B `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; adapter `16818868b0cc7813808aae4e87b417657046ab79`. The independent text and multimodal LoRA adapters are not interchangeable.
- One forward pass per question, no decoding; 1–26 options in request order map to A–Z. The chat template retains the upstream `<think>` suffix; disabling thinking changes tokenization. Preserve Python-compatible JSON spacing/Unicode/escaping and object insertion order, including null criteria fallback and text spelling special tokens.
- Text choice only: malformed JSON/UTF-8, duplicate keys, non-finite/out-of-range numbers and lone surrogates are 400; unsupported model/question or invalid semantic input is 422. Score/noul are not implicitly supported by this adapter.
- Probabilities read the final-position option logits; choice ties select the earliest option. Confidence is normalized entropy with single-option confidence 1; it is not upstream's p_max. Response identity includes adapter revision/modality; input usage sums prompts and output tokens are 0. Confirm native/reference numerical differences rather than assuming merged LoRA is exact.
- The native loader expects `cua_s1_export.json` before accepting merged weights. Check safetensors shape/dtype/index/tokenizer consistency and readiness failure paths when exports change.

## CUDA, graph and compatibility evidence

`src/backends/cuda/qwen3_5/{ops.h,runtime.cu,attention.cu,gdn_prefill.cu,build.sh}` and `src/models/cua_s1/native/src/{cuda.rs,model.rs}` form one ABI/lifetime boundary. The audited main ABI is 3; changes may legitimately bump it but Rust declarations and library must agree. Tensor-core code needs sm_80+, while documented execution covers sm_89, not all GPUs. Norm/elementwise/qk operations preserve reference BF16 rounding; attention/Gated DeltaNet have their own intermediate precision.

The graph path is opt-in and keyed by exact prompt length, with bounded captures. Read actual code for scratch growth invalidation, updated input upload, buffer lifetime and capture error recovery; use `native/tests/kernels.rs` as a starting point, not proof of execution. GPU tests are ignored by default. CUDA compilation, CPU fixture/tokenizer tests and checkpoint export do not establish full native parity or frontend-proxied inference. An ABI bump requires rebuilt consumers and library, not just a Rust test pass.

For a native/reference comparison use the target model's predeclared gates. Cua-S1's documentation specifies exact reference token IDs, direct-vs-proxy byte parity, and a native-vs-fp32 tolerance derived from BF16-reference drift plus a top-two-margin criterion; do not replace it with another model's ad hoc threshold.

## Test routing and measurements

- Rust CI: fmt, strict Clippy with all targets and locked dependencies, workspace tests, release build. Record test pass and ignored counts; inspect any newly relocated explicit Cargo test targets.
- `.github/workflows/ci.yml`, `benchmarks/README.md`, `benchmarks/requirements.txt`, `tests/benchmarks/test_bench.py`: `python benchmarks/bench.py validate benchmarks/smoke.jsonl` and `python -m unittest discover -s tests/benchmarks -p 'test_*.py' -v`. This uses Python 3.11+/httpx; four synthetic smoke requests are not a meaningful accuracy dataset or throughput workload.
- Benchmark runner preserves manifest/config hashes, raw responses and failures; hardware metadata is operator supplied and needs corroboration. Read the documented rounded-probability validator limitation and any later regression fix before relying on results. Incomplete A/B runs are not validated baselines. Keep successful-only latency denominators and all errors visible.
- `CONTRIBUTING.md`, `.github/workflows/docs.yml`, `mkdocs.yml`, `docs/hooks.py`: docs changes need verified repository/site links, claims and `mkdocs build --strict` in the documented environment; no GPU campaign for documentation-only edits.

Baseline inspection does not prove any future review loaded the skill or ran these checks. Every report must supply its own provenance and execution evidence.
