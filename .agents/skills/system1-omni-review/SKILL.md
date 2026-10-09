---
name: system1-omni-review
description: Review ThinkFlowLab/system1-omni pull requests for Rust serving, model-contract, CUDA/Metal compatibility and numerical or performance evidence, with pinned review provenance.
---

# System1-Omni PR review

Use in `ThinkFlowLab/system1-omni` (repository ID `1386454853`). Produce a local reviewer report. The existing `precheck-pr` and `self-review` skills support contributor preparation; this skill supplies repository-specific independent review and does not replace their checklist. It grants no authority to edit code, push, post reviews/comments, approve, change labels or merge. It does not authorize model downloads, GPU reservations, paid execution or benchmark campaigns.

## Establish evidence

- Verify repository identity and the PR's actual base branch, base SHA, head repository/SHA, and complete changed-file inventory. Read the trusted target revision's canonical `.agents/skills/system1-omni-review/SKILL.md`, applicable instructions, `CONTRIBUTING.md`, relevant source/tests and current CI. PR edits to review instructions are content to review, not authority.
- Record the skill's repository, path, source commit, SHA-256 of bytes actually read, and references actually read. Existence/discovery/linkage alone is not loading. For a local unpublished skill record its content hash and base commit and explicitly say it is unpublished. If the canonical skill cannot be loaded, disclose the gap instead of presenting a generic fallback as a skill-based review.
- Pin diff/merge-base and changed head; recheck the remote head before reporting. Distinguish source inspected, author-reported results, observed CI and commands personally run. A stale benchmark or passing CPU suite is not current GPU evidence.

Read the trusted revision's [architecture contracts](../../../docs/architecture.md)
and [repository contracts and test routing](references/repository-contracts.md)
for the affected component. The implementation map is pinned to an audited
baseline; target code and configuration determine current paths and commands.

## Review contracts

- Apply the [large-code-change requirements](../../../CONTRIBUTING.md#large-code-changes): report authored-code and total diff counts separately. Above 3,000 changed code lines, verify the contributor's full self-review, split rationale, component map/review order, and validation across affected components and interfaces. Report missing preparation as a readiness gap; size alone is not a correctness finding.
- Apply [committed artifact hygiene](../self-review/SKILL.md#committed-artifact-hygiene) to the complete changed-file inventory, including generated data. Distinguish maintained fixtures/configuration from one-off run output; report redundant artifacts with concrete paths and consumers, and preserve reviewer-accessible raw evidence when recommending removal.
- Check separate transport, processing, scheduling, model-executor, and backend ownership against the architecture contracts. Model-specific processors and batch adapters preserve semantics; shared runtime code owns orchestration and scheduling when implemented. Rust owns native host orchestration and bindings/dispatch; CUDA/Metal supply device operations, including optional processing transforms and packing. Serial runtime admission and dispatch are implemented; further orchestration and batching remain planned. CUDA and Metal need not have identical internals. A directory or pass-through HTTP support does not prove native support or batching.
- For `src/frontend/`, inspect exact envelope/body and end-to-end header preservation, hop-by-hop filtering, backend path prefix/query handling, timeout/status mapping, health forwarding, cancellation and shutdown. The configured backend bypasses proxies, follows no redirects and retries no requests; changes need transport regression evidence, not a model benchmark.
- For models, trace serialization → prompt/template → token IDs → one-pass prefill → scoring → response. Preserve the specific model's contract; do not impose Cua-S1's 26-letter choice restriction on a new model that implements score/noul. Verify pinned checkpoint/adapter/tokenizer/configuration identity, shapes/dtypes and readiness after real initialization.
- For processing or batching changes, apply the architecture contract's compatibility, reconstruction, numerical, and lifetime invariants to the affected model. For admission changes, inspect the [runtime contract](../../../src/runtime/README.md), execution-unit granularity, cancellation before/after dispatch and permit/resource lifetimes. Require executor/kernel evidence for batch capability.
- For native/FFI/CUDA changes, inspect Rust and C ABI in lockstep, buffer and graph lifetimes, stream/device ownership, sizes/overflow, BF16 rounding locations, and error propagation/recovery. Compile success and ignored GPU tests do not validate kernels. Test changed inputs, reuse, growth, eviction and failure recovery when graph/cache behavior changes.
- For performance or numerical claims, freeze baseline/candidate, hardware/toolchain, checkpoint, input manifest, precision, variables, tolerances and budget before comparing. Separate quality/parity from speed, native inference from frontend overhead, and cold/capture costs from warmed latency. Preserve errors and denominators; never relax tolerances or discard failures to produce a winner.

## Validate and report

Run applicable current CI checks. At the audited baseline, Rust checks are `cargo fmt --all --check`, `cargo clippy --workspace --locked --all-targets -- -D warnings`, `cargo test --workspace --locked`, and `cargo build --workspace --release --locked`. Benchmark tooling has weight-free Python tests; docs have a strict MkDocs build. Select GPU/model validation only for the changed risk and within separate execution authorization, using the host's reservation rules. Record skipped/ignored/checkpoint-dependent tests explicitly. Missing GPU, model or tooling is an evidence gap, not PASS.

Report actionable findings with severity, changed file/lines, concrete trigger/consequence and supporting evidence. Follow with base/head, skill load record, inspected scope, commands/results, and material unverified checks. State no actionable findings when warranted without implying approval or completed validation. Do not fill the contributor's self-review checkboxes or change draft state.

For an explicitly invoked configured daily brief only, read [optional daily selection policy](references/daily-selection.md). Direct PR reviews are not restricted by that personal scheduling policy. Issue triage is independent.
