# Development loop and handoff

This is a contributor-operated workflow using existing GitHub issues, PR bodies,
comments and check links. It defines no installed watcher, background runner,
webhook, hook or new permission. Automation deployment and its owner, storage,
concurrency controls and delivery verification remain a separate decision.

Start from [CONTRIBUTING](../CONTRIBUTING.md); agents also read
[AGENTS.md](../AGENTS.md). Consult [lessons](../lessons.md) before changing or
reviewing an area. Architecture and validation stay in their
[canonical contracts](architecture.md) and
[change-to-validation matrix](../CONTRIBUTING.md#choose-validation-by-changed-area).

## One change, one evidence trail

1. Complete the existing human intent gate, define the bounded change and select
   validation from the matrix. Record affected components and prerequisites.
2. Implement the bounded change. Add or update tests when behavior changes and
   update affected contracts/documentation. Capture revision,
   platform/tool versions, exact command, start/end time, result and evidence URL.
3. Self-review the complete diff and fill the existing PR template. A green CPU
   suite does not validate GPU behavior; author evidence does not replace
   maintainer review. Keep blocked validation and its handoff owner visible.
4. For multi-session work or an actual handoff, optionally keep a **Handoff
   checkpoint** in the authorized PR body or issue. Reuse existing PR evidence
   and record only fields needed to resume the work. If used, read it back to
   verify persistence. Without a remote item, keep relevant state in the
   contributor's private task workspace until publication is authorized; do not
   commit run state.
5. On resumption, read any checkpoint and current remote state before choosing the
   next action. Preserve past results as history, clearly separated from evidence
   applicable to the current head. A closed issue or merged PR does not complete
   every dependency automatically.

## Handoff checkpoint fields

When a checkpoint is useful, reuse the existing item rather than opening a
duplicate tracker. The following are optional fields to select for the actual
handoff, not a checklist required for every change:

- Repository identity and item number/URL; observation time and sources.
- Open/closed/merged state; actual PR base/head SHA. Issues have no inherent head;
  link their relevant PRs and revisions instead.
- Each relevant check's name, run/attempt, checked SHA, status/conclusion,
  platform, observation time and evidence link. Keep unknown/stale separate.
- Review/requirement status and validation owners, independently of author test
  results. Unknown branch-protection or merge authority remains unknown.
- Dependencies, blockers, next action and responsible person, or explicitly
  unassigned handoff. Include what evidence will unblock it.
- For an authorized external action: a stable transition/target key, purpose,
  planned/in-flight/confirmed/uncertain state and returned external identifier.
  Preserve enough history to avoid replaying it after a restart.

An action key includes repository/item, action type and meaningful evidence
version (for example head + check run/attempt + conclusion), not only the head.
Read before editing a shared record; coordinate concurrent writers. If a write's
result is uncertain, inspect the destination before retrying. A deduplication
record is not a guarantee of exactly-once delivery or authorization to act.

## Reconcile changes and demonstrate the protocol

This synthetic walkthrough exercises the rules, not a deployed tracker:

| Observation | Checkpoint / next action |
| --- | --- |
| Head A, run 10 pending | Save A/run 10 and the pending validation handoff; read it back after a pause. |
| Same A/run 10 failed | Refresh CI evidence despite unchanged head; record a new failure transition and diagnosis handoff. |
| Identical failure observed again | Retain the existing transition key; do not duplicate an action or notification. |
| Same A, rerun 11 succeeds | Record the new attempt and supersede current CI status; do not erase history or infer human approval. |
| Head B replaces A | A's check evidence is historical; select and obtain B's checks. Reassess review applicability under actual repository policy. |
| Base changes while head stays B | Reassess merge-result validation; refresh reviews, requirements and mergeability independently. |

Before a separately authorized merge or other consequential action, refresh the
actual head and relevant gates. Check runs and legacy commit statuses are separate
sources: an empty status list is not a failing check. If a source cannot be read,
record the last known result and its age without calling it current.

## Bounded execution and useful measurements

For benchmarks or resource-intensive work, follow time/attempt limits when the
applicable recipe or agreed execution plan specifies them. Do not impose a new
budget on ordinary contribution work. Retry only safe transient failures within
any applicable limit; a retry cannot expand authorization. Diagnose deterministic
failures instead of repeatedly running them. Stop on access denial, exhausted
agreed resources, scope change or missing human decision, and report the blocker.
Do not run parallel jobs against conflicting mutable resources.

CI timeouts apply to their configured jobs, not as universal local-test budgets.
Use the [benchmark protocol](../benchmarks/README.md) only where applicable;
GPU reservations, downloads and paid runs require separately agreed scope and
available resources. Record failures and skipped/ignored tests honestly.

For an iteration improvement claim, define the observation window and metric
before collecting it. Examples are elapsed time from first submitted revision to
first actionable validation result, or PR open-to-merge time. Record timestamps,
data sources, sample count, exclusions and unfinished items. Count independent
incidents using [lessons](../lessons.md), not alerts. No baseline or speedup is
claimed by this initialization; one successful PR cannot establish causality.
