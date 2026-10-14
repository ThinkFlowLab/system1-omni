# Working in System1-Omni

Before implementation or contributor self-review, read [CONTRIBUTING.md](CONTRIBUTING.md),
including its human contributor-intent gate. Before changing or reviewing an area,
consult [lessons.md](lessons.md) and the relevant contracts below. Follow any more
specific instructions in the directory you are working in; report conflicts.
Independent read-only review stays outside the contributor-intent gate.

## Find the source of truth

- Setup and entrypoints: [README](README.md) and [first CPU decision](docs/getting-started.md).
- Ownership, implemented versus planned layers, numerical and lifetime invariants:
  [architecture contracts](docs/architecture.md).
- Commands, platforms and validation handoffs:
  [change-to-validation matrix](CONTRIBUTING.md#choose-validation-by-changed-area).
- Contributor self-review and independent maintainer review:
  [contribution process](CONTRIBUTING.md#self-review-before-requesting-review).
- Resuming work, evidence freshness, safe retries and measurements:
  [development loop](docs/development-loop.md).
- Preventing repeated mistakes and recording verified incidents: [lessons](lessons.md).

## Navigate the implementation

Transport lives in `src/frontend/`; admission and shared execution interfaces in
`src/runtime/`; model-specific processing/execution in `src/models/`; hardware
operations in `src/backends/`. Read the affected component README and model
recipe before editing. New tests belong under root `tests/`; preserve explicit
Cargo target wiring and the test-module wiring rules in CONTRIBUTING.

Use the relevant repository skill deliberately: [add-new-model](.agents/skills/add-new-model/SKILL.md),
[precheck-pr](.agents/skills/precheck-pr/SKILL.md),
[self-review](.agents/skills/self-review/SKILL.md), or
[independent review](.agents/skills/system1-omni-review/SKILL.md).
Their presence does not guarantee automatic invocation by any tool.

Keep code, contracts and user-facing documentation aligned in the same change.
Record exact revisions, personally run commands, observed CI and unverified
hardware separately. These instructions grant no permission to publish, merge,
change settings, install monitoring, or start paid/GPU runs.
