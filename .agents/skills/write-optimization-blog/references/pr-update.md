# A per-improvement progress update

Use this record for the evidence behind an optimization update. Omit
inapplicable metrics. One PR with independently tested changes
gets separate records; a bundled backend comparison remains a bundled result.

A concise post shows the headline, the A/B figure or compact table, a short
mechanism and the result/limit.
Link the full record for controls, repetitions, hashes and rejected variants;
avoid reproducing every field in the main narrative.

## Date, PR and implementation change

Link the PR and measured implementation. Explain where the previous path spent
time, the operation or data movement that changed, and the hypothesis. Describe
the concrete before/after execution steps, rounding points and fallbacks that
matter to the result. Keep profiler observations distinct from inferred causes.

## Frozen A/B pair and protocol

Record baseline/candidate commits, patches or artifact hashes. Identify the
isolated variable and necessary consequences. State the fixed exact device,
model/checkpoint, dtype, workload/token lengths, concurrency, affinity, graph and
cache settings. Give the timer's start/end boundaries and whether profiling was
active. Link the frozen protocol, preparation/reproduction commands and raw runs.

Record feasibility, first inference/capture, warmups and measured repetitions
separately, including arm order, success criteria and the declared stop condition.
Use the recorded protocol's run count; an ordinary two-run comparison shows both
runs per arm. Preserve failures and stop at the agreed budget.

## Results and acceptance

Use a table that exposes repetitions, rather than only an aggregate speedup:

| Metric and unit | Baseline measured runs | Candidate measured runs | Aggregate change / variability | Declared gate / result |
| --- | --- | --- | --- | --- |
| Relevant kernel/call metric | Each run's value | Each run's value | Change and observed spread | Pass, miss or unverified |
| Warm HTTP mean / p95 / throughput | Separate row for each relevant metric | Each run's value | Change and observed spread | Evaluate the original gate |

Keep standalone kernel timings, trace-family totals and complete HTTP latency in
separate rows or tables. Explain whether the gate applies to each run or to an
aggregate. Report numerical drift/tolerances, decision flips, failures and tested
sample counts alongside speed. Matching decisions alone is not probability parity.
Identify the acceptance result even when a kernel improves but HTTP misses its
gate; small refactor differences can support a regression check without a speedup.

## Evidence, limits and next comparison

Link raw samples, source hashes, measured code and figure inputs/regeneration.
State which full responses, traces or protocols remain outside the public record.
Figures should show the relevant repetitions and label their timing boundary.

Finish with what this pair establishes and the next unresolved comparison. Mark
missing isolated A/B evidence as unverified and name the variable to isolate.
Do not carry an older measurement forward as current-head performance, add gains
from independent campaigns, or fabricate a stage timeline from aggregate times.
