---
name: add-new-model
description: Add a model to System1-Omni with a pinned inference contract, separate model-specific processors and executors, repository-level tests, recipes, documentation updates, and validation evidence. Use when implementing or extending model support in this repository.
---

# Add a System1-Omni model

Read [README.md](../../../README.md),
[CONTRIBUTING.md](../../../CONTRIBUTING.md), and applicable `AGENTS.md`
instructions from the target checkout. Paths below are relative to the repository
root. Inspect the branch, remotes, working-tree status, and affected components
before editing; preserve unrelated changes and follow the user's branch rules.
Read the [architecture contracts](../../../docs/architecture.md) for layer
ownership, Rust/device boundaries, and current versus planned integration.

## Establish the contract

- Pin the upstream reference code, checkpoint, tokenizer, adapters, and required
  dependency versions. Check attribution and license requirements before adapting
  upstream code.
- Identify the supported modalities and question types, model identifiers, input
  limits, validation errors, output fields, and token accounting. Record deliberate
  differences from the reference and unsupported behavior.
- Trace preprocessing through the final decision: structured JSON rendering and
  ordering, prompt/chat template, special tokens, tokenization, checkpoint layout,
  tensor shapes and dtypes, trained head, calibration, normalization, ties, and
  confidence. Sharing a backbone does not imply sharing prompts or readout.
- Capture small reference fixtures before implementing native execution. Preserve
  their provenance and generation commands; declare numerical tolerances before
  comparing outputs.
- Identify prepared-work and executor input/output layouts, batch compatibility
  constraints, request/question/candidate mapping, and state/buffer lifetimes.
  Keep prompt and response semantics model-specific; learned heads belong to
  execution, while calibrated response interpretation belongs to postprocessing.

Use the existing worker/frontend interface where it fits. Explain any required
interface change before implementing it; adding a model is not a reason to
redesign serving infrastructure or introduce a general model framework.

## Place and integrate the code

| Location | Changes that belong here |
| --- | --- |
| `src/models/<model>/` | Separate model-specific processors and batch adapters, plus executor checkpoint loading, weights, forward passes, learned heads, device state, and kernel selection. A native Rust worker may live in `native/`, following existing models. |
| `src/models/<family>/` | Model-family execution shared by implementations that actually need it. Reuse compatible code; extract a common implementation when there is a second user. |
| `src/backends/cuda/` or `src/backends/metal/` | Hardware operations and kernels, native host glue, build scripts, and backend integration. GPU processing transforms or tensor packing retain their processor/batcher ownership. |
| `src/runtime/` | Shared native admission and dispatch policy. Reuse the serial scheduler for single-executor work; keep model layouts and device state in model adapters/executors. |
| `src/frontend/` | Transport and existing worker adapters when integration requires them. Keep model-specific input/output semantics in processors rather than the transport layer. |
| Root `Cargo.toml` and affected crate manifests | Workspace membership, dependencies, binaries, and explicit test registration. Update `Cargo.lock` when dependencies change. |
| `recipe/<model>/` | Checkpoint preparation/export scripts, setup and launch instructions, and example requests. Reusable runtime implementation belongs in `src/`. |
| `tests/<model>/` and `tests/<shared-component>/` | All test bodies, test helpers, and fixtures, including contract, tokenizer, checkpoint, HTTP, and kernel tests. |

The [native runtime](../../../src/runtime/README.md) provides serial FIFO admission
and blocking dispatch per loaded executor. Integrate with available worker
interfaces while separating processors and batch adapters from forward code.
Models declare their execution unit, layouts and compatibility constraints;
shared runtime code owns admission policy. Processing orchestration, queue/token
budgets and dynamic batching remain planned. The native
target uses Rust host orchestration and bindings with CUDA/Metal device kernels.

**Do not add test code or fixtures under `src/`.** This includes inline Rust test
bodies and crate-local `tests/` directories nested under `src/models/` or another
`src/` directory. Existing violations are not a precedent for new code; avoid
unrelated test migrations.

Register Rust integration tests using `[[test]]` entries whose `path` points to
the repository-level `tests/` tree, relative to the crate's manifest. Tests of
private Rust items can be loaded from that tree using `#[cfg(test)]` and an
external `#[path = "..."] mod tests;`; only this registration wiring may remain
in `src/`. Verify discovery with `cargo test --locked -p <crate> -- --list` so
placing tests outside the crate does not silently omit them.

Keep normal workspace builds and CPU tests independent of GPU execution and
checkpoint downloads. Make checks requiring a checkpoint or accelerator explicit
opt-in checks and document their prerequisites. Follow the existing backend
loading/build convention; when a shared ABI changes, update its version and
consumers together and document required rebuilds.

## Validate behavior

- Test the supported request/response contract and rejection of unsupported or
  malformed inputs. Include ordering, structured values, special-token text,
  candidate limits, calibration, and token counts where they affect this model.
- Compare prompt text and token IDs exactly with the pinned reference. Compare
  numerical outputs under the declared tolerance; check decision agreement as
  well as probabilities. Report intentional API differences separately.
- Check new or changed kernels against a reference and cover relevant shapes,
  dtypes, and resource lifetimes. Revalidate existing consumers of shared execution
  or backend changes; results for the new model do not establish compatibility
  with an existing model.
- When processing or batching changes, check origin/order reconstruction,
  masks and readout positions, state isolation, whole-question normalization,
  error behavior, and token accounting. Real dynamic batching needs a
  batch-capable executor; a serial prompt loop does not establish it.
- For a native worker, complete loading and a real inference before reporting
  readiness. Verify its first request after readiness and run the same request
  directly and through the Rust frontend, checking status and response behavior.
- Run the Rust workspace checks listed in `CONTRIBUTING.md` and relevant
  component checks. Disclose checks not run and why; compilation, skipped tests,
  and CPU fixtures do not establish GPU or full-checkpoint parity.

Follow the execution host's reservation rules for all GPU device work. Run
performance experiments only within the requested scope. For measured claims,
follow [the benchmark protocol](../../../benchmarks/README.md): freeze revisions,
controls, tolerances and the run budget, preserve raw results, and separate
preparation, readiness, first inference, warm latency, and output fidelity.

## Update the documentation with the implementation

| File | Required content or update |
| --- | --- |
| `src/models/<model>/README.md` | Pinned artifacts, inference/API semantics, processor/executor boundary, layouts and batch constraints, state/lifetime ownership, supported scope, validation criteria, current status, and links to recipes/tests. |
| `recipe/<model>/README.md` or a focused recipe such as `native.md` | Prerequisites and dependencies, pinned downloads, export/setup, backend and worker build, environment variables, worker and frontend launch, health and a real example request, validation commands, and limitations. State the command working directory and any substantial RAM, storage, or device requirements. |
| Root `README.md` | Supported-model status and links. Update implementation-status prose or workspace descriptions when the new model makes them stale. Distinguish implemented support from planned work and tested hardware from build targets. |
| `recipe/README.md` | A discoverable link to the new recipe with its actual scope. |
| `mkdocs.yml` | Navigation for new public recipe or documentation pages; adjust `exclude_docs` only if a page or asset would otherwise be excluded. |
| Affected shared model/backend READMEs and recipes | New shared ownership, build/ABI requirements, and instructions for existing consumers when their setup changes. |
| `docs/architecture.md` | Changed integration contracts or implementation status when the new work affects shared processing, scheduling, execution, or backend boundaries. |
| `docs/` | Reproducible validation or benchmark evidence when needed; put documentation images in `docs/assets/`. |

Keep configuration names, commands, identifiers, limits, and examples consistent
with the code. Label unverified modalities, backends, hardware, and optimizations
explicitly. Do not describe reference fixtures or a narrow performance run as
general accuracy parity or an aggregate speedup.

Follow the documentation-site conventions in `CONTRIBUTING.md`: use repository
relative links, add pages to navigation when appropriate, and keep linked pages
and images within the published set. Hidden skill files are not site pages; the
existing MkDocs hook routes links to them and other unpublished source files to
GitHub. Run `mkdocs build --strict` with `docs/requirements.txt` installed in an
appropriate environment, and check links and commands in the model README and
skill files that the site build excludes.

## Prepare the review

Summarize the implemented scope, contract differences, changed shared components,
checks and results, and remaining coverage gaps. Follow `CONTRIBUTING.md` and the
PR template. Prepare the PR demo/evidence section per
[self-review](../self-review/SKILL.md): measured comparisons need pinned
provenance; an optional System1-Agents shared demo and illustrative video
supplement but never substitute for benchmark and parity evidence. Use
[precheck-pr](../precheck-pr/SKILL.md) for agent-assisted
self-review before requesting maintainer review. Follow the user's authorization
for commits, pushes, and PR creation.

## Worked examples

- [PR #19](https://github.com/ThinkFlowLab/system1-omni/pull/19): native Cua-S1
  execution checked against a reference worker, checkpoint export, CUDA build,
  and numerical validation.
- [PR #55](https://github.com/ThinkFlowLab/system1-omni/pull/55): Open-Jev's
  model-specific candidate scoring and calibration, shared Qwen execution, and
  registration of tests from the repository-level tree.

Read the relevant implementation and final diff when using an example. Historical
PR paths and behavior may differ from the target checkout; the placement rules
above apply even when an older example stores tests under `src/`.
