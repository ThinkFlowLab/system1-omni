# Contributing to System1-Omni

Review the [README](README.md) and
[architecture and integration contracts](docs/architecture.md) for the project's
ownership boundaries and current implementation status before making changes.

## Adding a model

Keep model-specific preprocessing, execution and response formatting under
`src/models/<model>/`; reusable hardware operations belong under `src/backends/`.
Share serving infrastructure and extract shared model execution when multiple
implementations need it.

Keep processors and batch adapters separate from the model's forward
implementation, even when they live in the same model directory. The README's
target architecture shares processing orchestration and scheduling across
models. Native workers already reuse
[serial admission and dispatch](src/runtime/README.md); processing orchestration
and GPU batching remain planned. Model executors own weights, forward passes,
learned heads, and device state.

For native implementations, Rust owns host processing, scheduling, model
orchestration, and backend bindings/dispatch. CUDA/Metal provide device
operations used by models or processing modules. Document processor inputs,
executor layouts and batch constraints, output reconstruction, and state/buffer
lifetimes alongside the existing API and numerical contract. Keep current worker
integration usable while further shared runtime components remain unimplemented.

All test bodies, test helpers and fixtures belong in the repository-level
`tests/` tree. Do not add inline test bodies or crate-local `tests/` directories
under `src/`. Register Rust integration tests with explicit `[[test]]` paths in
the crate manifest. For tests of private items, only the `#[cfg(test)]` and
external module-path wiring may remain in `src/`; the test implementation stays
under root `tests/`. Existing tests under `src/` do not change this rule for new
contributions.

Document the model contract in `src/models/<model>/README.md` and setup, launch,
example requests and validation in `recipe/<model>/`. Update the root README's
supported-model status, `recipe/README.md` and the documentation navigation in
`mkdocs.yml`. Update affected shared-component documentation when build or ABI
requirements change, and state support and validation limits explicitly.

For agent-assisted implementation, use the
[add-new-model skill](.agents/skills/add-new-model/SKILL.md). For example:

```text
Read .agents/skills/add-new-model/SKILL.md and use it to add support for this
model. Keep all tests under root tests/ and update the model docs and recipes.
```

## Self-review before requesting review

Self-review your PR before marking it ready for review or requesting maintainer
review. Open a draft PR if you want to share work in progress, and keep it in
draft until the self-review is complete.

1. Read the full diff against the target branch, including tests and documentation.
   Fix issues you find and remove unrelated changes, unused code introduced by the
   PR, and unnecessary abstractions.
2. Check ownership against the architecture contracts: shared serving,
   independent processing and scheduling, model-specific adapters and execution,
   and hardware operations. Distinguish implemented layers from the target
   design and keep the scope focused on the PR's stated problem.
3. Run checks appropriate to the change. For Rust changes, use the commands below
   from the repository root; add or update tests when behavior changes. For
   documentation changes, check links, examples, and claims against the actual
   implementation.
4. Fill in the PR's purpose, test plan, and test results. Report the commands you
   ran, their results, and any checks you could not run with the reason. Support
   accuracy or performance claims with reproducible evidence, including the
   configuration and hardware used.
5. Re-read the PR description and complete the template's self-review checklist.
   Repeat the relevant checks after subsequent changes.

The Rust checks used by [CI](.github/workflows/ci.yml) are:

```sh
cargo fmt --all --check
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo test --workspace --locked
cargo build --workspace --release --locked
```

For an agent-assisted self-review, use the repository's
[precheck-pr skill](.agents/skills/precheck-pr/SKILL.md). For example, ask your
coding agent:

```text
Read .agents/skills/precheck-pr/SKILL.md and use it to self-review my changes
before I mark this PR ready for review. Report findings and unverified checks.
```

The skill is optional; the self-review checklist applies to every contributor.
A coding agent can help review the diff and identify issues, but contributors
remain responsible for understanding the changes and verifying the results.
Self-review helps maintainers focus on design and correctness; it does not
replace maintainer review.

### Large code changes

PRs with **more than 3,000 changed lines of authored code** need extra contributor
attention before requesting review. Count additions plus deletions against the
PR's merge base in source files, tests, and build or validation scripts. Report
this count separately from the total diff size; exclude documentation, generated
output, lockfiles, and static fixtures from the code count, while still reviewing
those files for relevance and correctness.

- Complete a full self-review of every affected component and its integration
  boundaries. A quick precheck alone is insufficient; keep the PR in draft until
  the contributor self-review is complete.
- Consider splitting independent features, refactors, and cleanup into focused
  PRs. If the change needs to stay together, explain why in the PR description
  and provide a component map and suggested review order.
- Include the code-line count and a validation summary for each affected area in
  the PR description: commands, results, and unverified behavior with reasons.
  Cover changed interfaces between components as well as individual components.

Size signals the need for closer review; it is not itself a correctness finding.
Choose checks based on the changed behavior and risk. Crossing this threshold
alone does not require GPU benchmarks or other expensive experiments.

## Documentation site

The site at <https://thinkflowlab.github.io/system1-omni/> is built with MkDocs
from the README, this guide, the frontend README and the Markdown files under
`recipe/` and `docs/`. `mkdocs.yml` sets the navigation and, in `exclude_docs`,
the published files. Pages keep their repository paths, so write links as
relative paths that work on GitHub; links to files that are not published go to
GitHub. Put images in `docs/assets/`.

To run the check from the Docs workflow and preview the site, with Python 3.10 or
later:

```sh
python3 -m venv .venv-docs
.venv-docs/bin/pip install -r docs/requirements.txt
.venv-docs/bin/mkdocs build --strict
.venv-docs/bin/mkdocs serve    # http://127.0.0.1:8001/system1-omni/
```
