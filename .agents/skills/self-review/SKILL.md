---
name: self-review
description: Self-review System1-Omni changes before requesting maintainer review, including output-parity checks, controlled A/B measurements, and appropriate PR demo/evidence. Use for self-review and pre-submit checks in this repository.
---

# System1-Omni self-review

Follow the target checkout's `CONTRIBUTING.md` when present and its PR template.
Review the full diff against the target branch for correctness, focused scope,
architecture alignment and tests. Verify that documentation and PR claims match
implemented behavior; report commands, outcomes and checks not run.

This skill prepares a local contributor report. It does not itself authorize edits,
commits, pushes, external posts, or review-status changes; follow separate user
instructions for those actions. Distinguish personally run checks, author-reported
results, and observed CI when reporting evidence.

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

## PR demo/evidence

Prepare a **Demo / evidence** section that shows what changed in inference or
serving, with enough provenance for accurate review and public updates. Select
evidence for the changed behavior; there is no asset quota or requirement to run
production-scale workloads. Nonvisual, docs-only, and test-only changes do not
require video or an expensive GPU campaign. Use `N/A` with a reason for inapplicable
items, and label relevant but unavailable measurements **unverified**, not passing.
Stay within separately authorized resources and the declared run budget.

For a measured comparison, include:

- Exact baseline/head commits (and any local patches), model/checkpoint/tokenizer
  revisions, hardware and device count, driver/runtime/toolchain, precision,
  workload/input manifest, input shapes/lengths, concurrency/batch settings, and
  cache policy. Keep controls matched and explain necessary differences.
- Reproduction commands, dependency/setup requirements, and supported, unsupported,
  and untested configurations. Link raw repeated-run results and the protocol;
  report sample counts, variation, failures, and exclusions alongside summaries.
- Relevant latency, throughput, memory, and quality/output-parity evidence. State
  metric units, denominators, timing boundaries, and memory measurement method.
  Mark metrics not measured explicitly; do not imply all metrics improved.
- Separate cold startup/loading, first request/capture, and warmed runs. Keep
  kernel/microbenchmark, native inference, and end-to-end HTTP measurements
  distinct; a kernel speedup does not establish a user-visible serving speedup.
  Distinguish historical/reference validation from tests of the integrated head.

Use latency/throughput charts or memory comparisons when they clarify the raw
measurements, and simple architecture diagrams when they explain the change.
Label planned versus implemented components. A real video can help demonstrate
streaming responsiveness, startup, or an API interaction; show timing honestly,
including cuts, speed changes, stalls, failures, or human intervention. Video is
illustrative evidence of that run, not a substitute for benchmark or parity data.

For each asset, provide a durable, reviewer-accessible link and caption with its
workload, source run/commit, and status as an actual run, recorded replay, or
explanatory illustration. Label replay speed and historical/upstream provenance;
do not present another project's trajectory as a demonstration of this integration.
Never fabricate screens or results, and remove or qualify unsupported claims.

Before attaching publishable assets, check ownership, license/attribution, and
permission to share. Use safe sample data; redact credentials, private URLs,
personal/customer information, and sensitive screen/log content. Check the final
exports, captions, and metadata, retaining useful measurement context. Label
diagrams, mockups, and generated artwork as illustrations. If rights or safe
disclosure are unresolved, omit the asset and explain why. Preparing reusable
evidence does not authorize external posting.

An optional shared demo running System1-Agents on Omni may connect the user task
to the serving change. Pin both repositories and keep task completion evidence
separate from inference performance and parity claims.
