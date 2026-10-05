---
name: self-review
description: Self-review System1-Omni changes before requesting maintainer review, including deciding whether output-parity checks or controlled A/B measurements are needed. Use for self-review and pre-submit checks in this repository.
---

# System1-Omni self-review

Follow the target checkout's `CONTRIBUTING.md` when present and its PR template.
Review the full diff against the target branch for correctness, focused scope,
architecture alignment and tests. Verify that documentation and PR claims match
implemented behavior; report commands, outcomes and checks not run.

Read the [architecture contracts](../../../docs/architecture.md). Check that
processing and batch adapters remain separate from forward execution, shared
scheduling owns runtime policy when implemented, and Rust host orchestration
uses CUDA/Metal for device operations. Preserve model-specific API/numerical
semantics and distinguish target layers from current worker capabilities.

For processing or batching changes, trace prepared inputs through executor
layouts and output reconstruction. Apply the architecture contract's batching,
numerical, and lifetime invariants to the affected model. For shared runtime
admission changes, check FIFO execution units, cancellation before dispatch,
pending-unit limits, overload/error response mapping, startup configuration,
permit/resource retention after dispatch, and release on errors or panics against
the [runtime contract](../../../src/runtime/README.md).

## Conditional A/B checks

Read [the benchmark self-review guidance](../../../benchmarks/README.md#self-review-and-ab-evidence)
when the PR makes a performance claim or changes inference/serving behavior with
a concrete performance or numerical risk. Explain whether a comparison is
required, unnecessary or unverified. Use the narrowest comparison that resolves
the risk; ordinary documentation and smoke checks do not require GPU benchmarks.

For measured comparisons, follow the benchmark protocol: freeze the baseline and
candidate revisions, declare the variable, controls, tolerances and run budget,
and preserve raw results and failures. Use the execution host's GPU reservation
rules. Keep quality/parity separate from speed, and frontend overhead separate
from native inference. Do not present an aborted run as a completed comparison.

Inspect the runner revision and documented limitations before executing. Missing
GPU access or a harness failure is unverified evidence, not a pass. Remove or
qualify unsupported performance claims. Do not silently change tolerances or
extend the run budget; fixes require regression coverage and a recorded revised
protocol. This skill does not itself authorize external posts or paid execution.
