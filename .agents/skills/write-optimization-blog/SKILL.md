---
name: write-optimization-blog
description: Write or update Markdown engineering blog posts with PR-by-PR inference optimization progress, isolated A/B records, implementation explanations, and sourced reproducible comparison figures.
---

# Write an optimization blog

Turn implementation and experiment records into a readable engineering post.
Explain the bottleneck, the change that addresses it, and the measured result.
Keep measured behavior distinguishable from implemented support and future work.

## Choose the destination and evidence

Inspect the target repository's blog/docs conventions and contribution rules.
Use its existing Markdown, metadata, image paths and navigation. When the user
names another blog as a reference, study its presentation without silently
changing the publication destination. Read
[the vLLM examples](references/vllm-blog.md) when using that style or contributing
to `vllm-project/vllm-project.github.io`.

Record the inspected repository SHA, the post's evidence cutoff and the actual
measurement revisions. Current main can contain subsequent changes that were
never timed. Prefer immutable links to the measured code and raw records; link
current recipes separately for setup. Disclose missing raw artifacts rather
than implying every historical result can be reproduced from current main.

Build a compact evidence ledger before drafting. For each result, retain:

- Experiment identity/date and baseline/candidate revisions or artifact hashes.
- Hardware, model/checkpoint, dtype, workload/token lengths, concurrency, cache
  and graph settings, and the timed boundary.
- Excluded preparation, readiness, first inference and warmups; measured run
  counts and per-run values; output-fidelity checks and acceptance criteria.
- The source of the values and whether they are raw measurements, a documented
  summary, an inference, or a correctness-only check.

Use existing authorized records. A writing request does not authorize a new
benchmark campaign. If raw data is unavailable, a published summary can support
a labeled historical figure; preserve its source and displayed precision.

## Write the engineering story

For a concise post, lead with headline numbers and a comparison figure. Let
figures and compact tables carry the results; give each change a short mechanism
explanation. Avoid repeating the same results in a figure, table and paragraph.
Put full protocols, source hashes, per-run tables and rejected-variant detail in
linked evidence records. Keep workload/timer labels, regressions, missed gates
and material numerical limits visible in the post. Follow the user's requested
depth; keep the full evidence template in the linked record.

Give enough model/request-path context to understand what is accelerated. Connect
each observed cost to the implementation and its measured effect. Include rejected
approaches when they explain a precision constraint, cache policy or unresolved
limit. For compact presentation examples, see
[OpenJev-Fast alongside the vLLM references](references/vllm-blog.md#compact-reference-openjev-fast).

Separate complete backend comparisons, isolated optimization A/B runs, kernel
microbenchmarks and correctness checks. Do not add independently measured gains,
draw a cumulative speedup ladder from separate campaigns, or treat a faster
kernel as a measured end-to-end improvement. Keep different hardware/workloads
in distinct comparisons. Mark unmeasured combinations explicitly.

Preserve numerical limits: matching threshold decisions does not establish
identical probabilities; a small subset's score is not general accuracy; two
run means are variability observations rather than confidence intervals.
Explain a missed acceptance criterion alongside a favorable timing.

Use implementation and PR links near the relevant mechanism. Adapt the section
structure to the material; a performance-first post, a debugging narrative and
a stage-by-stage walkthrough need different emphasis. Verify quickstart commands
against current recipes and separate implemented features from the target design.

## Maintain a PR-by-PR progress record

When adding or revising a progress update, use
[the per-improvement A/B record](references/pr-update.md) for the detailed evidence;
summarize it in the article with figures and links.

Include an experiment map linking each relevant PR to its implementation changes
and A/B evidence. Split independently tested improvements within one PR into
separate updates; a PR containing normalization, activation and graph changes
needs more than one aggregate before/after row.

For each update's evidence record, keep the date, bottleneck, implementation
mechanism, isolated variable, measured baseline/candidate revisions or source
hashes, fixed controls, timer, run budget and exclusions. Show each measured run for both arms, relevant
kernel and serving results separately, numerical/parity checks, acceptance gates
and their outcome. Link raw samples, the frozen protocol and reproduction method.
Preserve regressions, missed gates and rejected variants alongside improvements.

Label regression checks as such; small nominal differences do not establish a
speedup. When an improvement lacks an isolated A/B, mark that gap and define the
missing comparison instead of assigning it part of a bundled backend gain.
Obtaining new measurements still requires the user's experiment scope and budget.

Append dated updates as PRs land without replacing earlier measurements with
current-head claims. Retain distinct hardware, workloads and timer boundaries.
Snapshot compact measured samples and source hashes when the underlying report is
local or a PR description can change; state what remains outside the public record.

## Make figures from the evidence

Use standard plotting tools for quantitative figures. Save the input values and
their provenance beside a CPU-only regeneration script. Derive aggregates from
raw records when available; otherwise label values taken from a summary/table.
Keep exact inputs in the data file and round only for display.

Use one coherent experiment per chart/panel. Label units, request counts,
hardware and warm/cold state. Bar axes start at zero; label any zoom or log axis.
Show individual run values when available and explain what markers represent.
Use readable colors, alt text and captions; export SVG for sharp embedding and
PNG when a raster preview or sharing format is useful. Inspect the rendered
figures and the Markdown/site, including labels, clipping and source links.

Mechanism diagrams should identify real components and data movement. Do not
invent GPU timeline spans, occupancy, stalls or bandwidth diagnoses from wall
times. Give profiler limitations where they affect the interpretation.

## Check and hand off

Check every headline against the ledger and recompute displayed ratios. Ensure
links/images resolve, scripts run without accelerator use, metadata matches the
target, and the site's normal documentation build passes where available.
Keep plot dependencies separate from inference dependencies. Run checks relevant
to the change instead of unrelated model/GPU tests.

Deliver the skill/post paths, figures, validation results and evidence gaps.
Commit, open a PR, publish or contact others only within the user's authorization
and repository workflow; this skill grants no additional permission. Preserve
contributor-owned review checklists when preparing an agent-assisted PR.
