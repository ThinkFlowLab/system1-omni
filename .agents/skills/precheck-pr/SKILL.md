---
name: precheck-pr
description: Self-review a System1-Omni branch before opening a PR or marking it ready for review. Check scope, architecture, validation, and evidence, then report findings and unverified checks to the contributor.
---

# PR self-review

Review the contributor's changes against System1-Omni's conventions. Read
`CONTRIBUTING.md`, `README.md`, and applicable `AGENTS.md` instructions from the
checkout before starting. Paths below are relative to the repository root.

This skill produces a local report. It does not itself authorize edits, commits,
pushes, GitHub comments, or changing a PR's review status. Follow any separate
user instructions for those actions.

## Establish the review scope

- Inspect the branch, remotes, working-tree status, and PR target when available.
  Use the PR's actual base branch; otherwise verify the upstream default branch.
  Do not silently substitute another base when it cannot be resolved.
- Record the head and base commit IDs. Review the complete diff from their merge
  base, plus staged, unstaged, and relevant untracked changes. State which changes
  are not yet part of the PR. Disclose if the remote base could not be refreshed.
- Read changed files and enough surrounding code and callers to understand the
  behavior; do not rely only on diff fragments.

## Check the change

- Match the implementation to the stated problem. Flag unrelated changes, new
  unused code, duplicated logic, or abstractions without a current need.
- Preserve the architecture described in the README: the Rust frontend owns
  serving infrastructure, models own execution, and shared utilities stay minimal.
  Consult the relevant `src/frontend/`, `src/models/`, `src/backends/`, or
  `recipe/` documentation for the changed component's contract and current status.
- For frontend changes, inspect affected request validation, error handling,
  cancellation, and response behavior, including the model-worker boundary.
  Check for regressions against callers and existing tests.
- For model or backend changes, inspect affected input/output contracts, tensor
  shapes and dtypes, device assumptions, and resource lifetimes. Do not require
  CUDA and Metal to have identical implementations or claim an untested backend
  works. Distinguish planned functionality from implemented behavior.
- Check that tests exercise the changed behavior. For bug fixes, look for a
  regression case that fails before the fix. Verify documentation and examples
  against the implementation.

## Validate and assess evidence

Choose validation based on the changed files. For Rust changes, run the checks
in `.github/workflows/ci.yml`; `CONTRIBUTING.md` lists the current commands. For
documentation-only changes, check links, examples, and technical claims without
building unrelated code. Use component-specific checks when applicable.

Report exact commands and outcomes. If a dependency, tool, hardware resource, or
permission is unavailable, record what remains unverified and why; do not treat
skipped checks as passing. Follow applicable GPU reservation instructions before
any device execution.

For accuracy or performance claims, inspect reproduction commands, model and
revision, hardware, configurations, baseline, raw results, and variability. Flag
unsupported claims. Reviewing evidence does not authorize starting new benchmark
campaigns; stay within the user's execution scope and budget.

## Report to the contributor

Lead with actionable findings, ordered by severity, with file/line references,
the affected behavior, and a suggested correction. Then summarize:

- The reviewed base/head and any working-tree changes.
- Checks that passed, failed, or were not run, with commands and reasons.
- Remaining evidence gaps and whether they prevent recommending review.

If there are no findings, say so and still disclose unverified areas. Recommend
keeping the PR in draft while blocking findings remain. The contributor should
review the report, address findings, rerun affected checks after changes, and
complete the PR template checklist. Do not check boxes on the contributor's
behalf or claim that agent review replaces contributor or maintainer review.
