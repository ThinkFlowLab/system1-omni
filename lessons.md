# Lessons

Read relevant active lessons before changing or reviewing an area. After a verified
failure or review correction, propose a concise prevention rule in the same PR or
its explicit follow-up. Keep normal architecture and review policy in their
canonical documents; this file records evidence-backed learning.

## Active lessons

No verified lessons recorded yet. This initialization does not infer historical
incidents from old PRs or count its acceptance examples as real failures.

## Entry format

For each actual lesson, record:

- Stable lesson ID, affected scope, and concise prevention rule.
- Incident key and reviewer-accessible source link; observed failure separately
  from suspected cause, with confidence when attribution is uncertain.
- Concrete preventive check and its canonical test/tool/document location.
- Named owner or explicitly unassigned next action, plus active/resolved status.
- Independent incident count and the distinct incident keys supporting it.

Do not add empty incident entries to satisfy this format. Keep secrets, private
conversation links, personal data and one-off execution dumps out of the file.

## Bounds, recurrence and archive

The active file limit is 200 lines, including this guidance.

- Keep this active file within that limit. Consolidate
  duplicate lessons before archiving resolved or superseded detail.
- Count independent failures, not retries, repeated observations or duplicate
  reports of one incident. Uncertain identity stays unresolved, not incremented.
- Use the observed failure mechanism and impact to propose an appropriate
  preventive change for review. There is no default incident-count threshold or
  automatic escalation; any new test or policy depends on repository needs and
  available resources.
- When there is material to archive, create `docs/lessons-archive/YYYY-MM.md`,
  retain source/incident history, and leave a short active-file reference.
  No archive is created until there is real content. Never hide an unresolved
  risk to meet the line limit.

Contributors propose updates through ordinary review; maintainers decide policy
changes. See [the development loop](docs/development-loop.md) for durable task
state and [CONTRIBUTING](CONTRIBUTING.md) for validation and review.
