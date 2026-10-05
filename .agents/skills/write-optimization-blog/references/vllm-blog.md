# vLLM blog references

Reviewed on 2026-10-05 at blog repository revision
`f9792a2e4f2b65c5ef3291396f941a1d831cc6b1`. These are presentation examples,
not benchmark evidence for another project. Refresh target conventions when
preparing a later contribution.

## Different useful structures

- [Qwen3-Omni optimization](https://github.com/vllm-project/vllm-project.github.io/blob/f9792a2e4f2b65c5ef3291396f941a1d831cc6b1/_posts/2026-07-01-qwen3-omni-optimization.md):
  introduce the execution path before individual optimizations; tie each change
  to a concrete bottleneck and implementation; place configuration sweeps and
  mechanism illustrations near the discussion. Its incremental sweep is valid
  because successive configurations were measured with earlier changes enabled.
  Do not imitate that cumulative table when your experiments are independent.
- [Kimi K3 performance](https://github.com/vllm-project/vllm-project.github.io/blob/f9792a2e4f2b65c5ef3291396f941a1d831cc6b1/_posts/2026-09-13-kimi-k3-performance-optimization.md):
  start with serving results, then select representative scheduling, state/cache,
  data-movement and kernel changes. A before/after illustration can explain the
  removed operation more clearly than a large diff. Link implementation PRs and
  the wider tracking issue without turning the article into a PR inventory.
- [Kimi K3 DSpark](https://github.com/vllm-project/vllm-project.github.io/blob/f9792a2e4f2b65c5ef3291396f941a1d831cc6b1/_posts/2026-09-15-kimi-k3-dspark.md):
  relate the headline to a defined workload and show performance alongside the
  mechanism, hardware setup and practical workflow. Keep throughput, per-user
  responsiveness and latency distinct rather than substituting one for another.
- [Kimi K2 tool-calling accuracy](https://github.com/vllm-project/vllm-project.github.io/blob/f9792a2e4f2b65c5ef3291396f941a1d831cc6b1/_posts/2025-10-28-Kimi-K2-Accuracy.md):
  a debugging narrative can organize the post around concrete compatibility
  failures and fixes. Preserve the validation context when explaining why a
  result changed; performance work also benefits from recording rejected changes
  that violate the output contract.

## When the destination is the vLLM blog repository

Read its current [README](https://github.com/vllm-project/vllm-project.github.io/blob/f9792a2e4f2b65c5ef3291396f941a1d831cc6b1/README.md)
and a nearby post before adding files. The reviewed conventions are:

- Markdown under `_posts/YYYY-MM-DD-slug.md`.
- YAML frontmatter with `layout: post`, title, author, tags, and a specific
  `summary` of at most 240 characters. Add image/social-image fields only for
  actual included assets; use verified authorship and the intended publication date.
- Figures under `assets/figures/<post-slug>/`, with captions and useful alt text.
- Build with the repository's Jekyll/Bundler setup. The main-branch workflow
  deploys the blog, so drafting files or a PR does not itself mean publishing.

For a different repository, adopt its renderer and relative-link conventions.
Do not add Jekyll or migrate a documentation site merely to mimic these examples.

## Compact reference: OpenJev-Fast

Reviewed [OpenJev-Fast](https://yiqilyu.me/open-jev-fast/) on 2026-10-05 for
presentation: headline metrics, an optimization chart, short mechanism blocks,
a benchmark figure/table, and brief correctness/setup notes. Detailed experiments
and failed attempts are linked in a separate report. Use that separation when
the user wants figures and numbers to carry the story. Its B300 forward/head
chart and HTTP benchmark use distinct timers; its values are not evidence for
System1-Omni's H200 experiments.
