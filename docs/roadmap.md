# System1-Omni roadmap

The next phase focuses on decision-inference performance, broader model support,
and community adoption through System1-Agents. This TODO list covers model
support, engine features, and agent integration over a roughly three-month
planning horizon. Milestones define order of work, not release dates or promised
speedups.

Use the [roadmap tracking issue, Omni #125](https://github.com/ThinkFlowLab/system1-omni/issues/125)
to claim tasks and track completion. This page records the planning snapshot;
keep it aligned when deliverables or priorities change.

Planning snapshot: **2026-10-08**, based on
[Omni main at 4a79980](https://github.com/ThinkFlowLab/system1-omni/tree/4a79980d8a75fd063cb3f8247e06215e288b18ac)
and
[Agents main at 5d89f85](https://github.com/ThinkFlowLab/system1-agents/tree/5d89f85e702fc8dbdb7b719721e3cb0dda0bef4f).
Check the [supported-model matrix](supported-models.md) for available paths.
[Omni #83][o83] tracks implementation separately from hardware validation.

## Working on a TODO

Each checkbox describes a remaining deliverable, even when prerequisites are
merged. Related PRs are proposals or foundations, not instructions to merge them
unchanged. Claim work in the linked issue, record an owner and dependencies, and
split large items into focused PRs. Where there is no dedicated issue, open one
using the linked design context.

Complete an item only when its scoped implementation is merged, its acceptance
checks pass, and its recipe or evidence is available. Preserve model-specific
prompts, heads, calibration, and probabilities. Follow the
[architecture contracts](architecture.md),
[contribution checks](../CONTRIBUTING.md), and
[benchmark protocol](../benchmarks/README.md).

## Model support

- [ ] **M1. Open-Jev-9B / 27B:** Maintain correctness and performance baselines
  across mixed `choice`/`score`/`noul`, prompt lengths, question/candidate counts,
  and concurrency; publish parity gates, memory, and repeated-run results.
  Merged foundations: [Omni #55][o55], [#92][o92], and [#102][o102].
  Broader benchmark tooling: [#116][o116].
- [ ] **M2. Cua-S1 4B:** Publish a reproducible native screenshot serving recipe
  and integrated reference-parity coverage; qualify language and vision graph
  replay separately. Native vision is merged in [Omni #64][o64].
  Remaining scope: [#10][o10], [#103][o103], and [#108][o108].
- [ ] **M3. Decider-2B:** Complete request/response compilation and eager native
  serving; qualify batching, graph replay, and prefix reuse as separate changes.
  Related: [Omni #57][o57], [#94][o94], [#110][o110], [#111][o111],
  [#112][o112], and [#113][o113].
- [ ] **M4. JEMM:** Complete native text/image serving, verified export and
  calibration, typed errors, and parity for supported multi-image requests.
  Related: [Omni #121][o121], [#118][o118], and [#122][o122].
- [ ] **M5. OmniJev-4B:** Complete native multimodal execution after the contract,
  preparation, head, and export work; validate readout, abstention, and probability
  semantics against the pinned reference. Related: [Omni #114][o114] and [#120][o120].
- [ ] **M6. openjev/openjev:** Complete native execution and serving after CPU
  request compilation; qualify this checkpoint's scoring/calibration independently
  of Open-Jev-9B/27B. Related: [Omni #95][o95] and [#124][o124].
- [ ] **M7. JEV-27B-VL:** Complete and validate native multimodal serving and
  prefix caching, with bounded memory and independent branch state.
  Related: [Omni #96][o96].
- [ ] **M8. CLM:** Complete real-encoder serving and verify the trained heads;
  define and validate bounded candidate caching with cold/warm measurements.
  Checkpoint/scoring support is merged in [Omni #28][o28]; remaining work:
  [#29][o29] and the design in [#9][o9].
- [ ] **M9. LAYA:** Complete resource/handle reuse, grouped uploads, failed graph
  cleanup, bounded workspace caching, and explicit build targets; publish matched
  comparisons with the original runtime. Native inference is merged in [Omni #90][o90].
  Related: [#66][o66], [#39][o39], [#71][o71], [#72][o72], [#73][o73],
  [#74][o74], and [#100][o100].
- [ ] **M10. Kev-4B:** Implement verified LoRA merging, tokenizer boundaries,
  request compilation, and the trained pointer head using shared Qwen execution.
  Related: [Omni #27][o27]. [#69][o69] is proposed head-kernel work,
  not a complete integration.
- [ ] **M11. LFM2.5-350M:** Complete reference-worker integration and candidate
  branching/cache-isolation checks. Evaluate native execution on a frozen workload;
  larger variants need separate scope and validation.
  Related: [Omni #31][o31] and [#32][o32].
- [ ] **M12. Valen:** Complete the reference worker and serving recipe, then use
  it to study shared-observation execution.
  Related: [Omni #84][o84], [#115][o115], and [#85][o85].
- [ ] **M13. Additional candidates:** Select owned first integrations for Julia-1,
  AFM-DE, Intern-Decision, or larger variants. Freeze artifacts and establish
  reference behavior before committing to native support. Selection context:
  [Omni #9][o9], [Agents #25][a25], and [Agents #52][a52].
  Intern-Decision has benchmark context, not an Omni integration issue.

For every promoted model, update the supported-model matrix and #83 with its
checkpoint/revision, question types, modalities, worker path, tested hardware,
reference parity, and reproducible measurements. Reference serving, native
execution, and optimized execution are separate milestones.

## Feature support

- [ ] **F1. Shared serving contract:** Agree on JSON errors, readiness, overload,
  model discovery, and capabilities; document model-specific confidence and token
  accounting. Related: [Omni #61][o61] and the client foundation in [Agents #35][a35].
- [ ] **F2. Bounded admission:** Converge pending queue-limit proposals on one
  policy per executor, including capacity, overload responses, cancellation, and
  resource retention after dispatch.
  Related: [Omni #35][o35], [#82][o82], and [#109][o109].
- [ ] **F3. Request-local observation reuse:** Execute compatible prefixes once,
  then branch attention, convolution, and recurrent state correctly. Validate
  changed observations, ordering, compatibility, memory budgets, and parity.
  Related: [Omni #85][o85], [#97][o97], and [#99][o99].
- [ ] **F4. Processing orchestration and batching:** Complete the worker-side
  processing/executor contract; extend within-request packing before bounded
  cross-request batching. Define compatibility, token budgets, maximum wait, and
  output reconstruction; require a batch-capable executor.
  Related foundations: [Omni #34][o34], [#102][o102], and [#111][o111].
  Cross-request batching needs a dedicated follow-up issue.
- [ ] **F5. Shared vision and CUDA Graphs:** Qualify shared vision and
  language/vision graph replay, mixed shapes, changed inputs, capture failures,
  cache budgets, and eager fallback.
  Related: [Omni #117][o117], [#123][o123], [#103][o103], [#108][o108], and [#106][o106].
- [ ] **F6. Performance and quality scorecard:** Publish parity/accuracy,
  calibration where applicable, p50/p95, requests/s, decisions/s, peak memory,
  startup, and first-inference results. Compare reference and optimized backends
  on matched workloads; retain failures and observed variability.
  Related: [Omni #116][o116], [#93][o93], [#39][o39], and [Agents #52][a52].
- [ ] **F7. Build and contribution path:** Compile every declared CUDA backend
  in CI; provide reproducible exports, worker builds, launch recipes, and claimable
  tasks for contracts, kernels, and independent hardware validation.
  Related: [Omni #77][o77], [#101][o101], and [#83][o83].
- [ ] **F8. Serving provenance and timing:** Define checkpoint/backend identity,
  request IDs, and queue/inference/proxy timing across workers and the frontend;
  test identity across restarts and routing. Design context:
  [Omni #61][o61] and [Agents #35][a35].
  Open a focused issue before extending the common response.
- [ ] **F9. Additional hardware and precision:** Scope native Metal and
  quantization as separate pilots with one model, hardware target, reference,
  numerical gate, and memory/performance criterion each. Native Metal remains
  planned; Python MPS is a foundation, not native Metal support.
  Related context: [Omni #9][o9] and [#30][o30]. Each pilot needs a design issue.

## Integration with System1-Agents

- [ ] **A1. Named served models:** Extend served-client plumbing to named
  Omni-backed models with local endpoints, optional authentication, correct model
  labels, and model-specific serializers. Exercise Open-Jev first and CLM when its
  worker is ready. Related: [Agents #35][a35], [#24][a24], [#46][a46], and [Omni #87][o87].
  Open a focused issue for the common Omni served adapter.
- [ ] **A2. Served screenshot decisions:** Send browser/desktop images through
  the decision-model interface; validate IDs/probabilities, preserve capture
  identity and window binding, reject stale targets, and keep image bytes out of
  observation logs. Related: [Agents #44][a44], [#14][a14], [#30][a30], and [#33][a33].
  Local Cua-S1 in merged [#32][a32] does not complete the HTTP screenshot bridge.
- [ ] **A3. Typed score questions:** Add typed score questions/answers,
  validation, serialization, and one rubric-based workflow. Agents currently
  exposes `choice` and `noul`. Design context:
  [Agents roadmap](https://github.com/ThinkFlowLab/system1-agents/blob/5d89f85e702fc8dbdb7b719721e3cb0dda0bef4f/docs/roadmap.md)
  and [Omni #61][o61]. Open a dedicated Agents issue before implementation.
- [ ] **A4. Front registration and capabilities:** Register served backends in
  applicable CLI, MCP, tool, browser, desktop, and rail paths; reject unsupported
  modalities/question types before inference.
  Related: [Agents #18][a18], [#35][a35], [#46][a46], and [#44][a44].
- [ ] **A5. Operational compatibility:** Align readiness, retries, overload,
  deadlines, and confidence thresholds with each worker's semantics. Reuse the
  configurable deadline already merged; verify unavailable, overloaded, invalid,
  and slow-response behavior.
  Related: [Omni #61][o61], [Agents #36][a36], and [#53][a53].
- [ ] **A6. Episode provenance:** Record model/checkpoint, revision, worker
  backend, request ID, and available server timings across supported fronts.
  Verify identity changes and separate capture, decision, and action time.
  Depends on F8; client foundation: [Agents #35][a35].
- [ ] **A7. Joint evaluation and demos:** Run pinned Agents/Omni revisions on
  repeatable text, browser, and desktop tasks. Publish task success, every failure,
  inference/episode time, commands, traces, and honestly timed replays. Related:
  [Omni #87][o87], [Agents #13][a13], [#44][a44], and [#52][a52].
  Keep public decision-set scores separate from live task completion.

## Milestones and dependencies

| Milestone | Focus | Exit criteria |
| --- | --- | --- |
| 1. Serving and measurement foundation | F1, F2, F6, F7; M1/M2 coverage; A1/A2/A5 | Agreed contracts, reproducible baselines, and one text plus one screenshot workflow through Omni. |
| 2. Reusable execution and model expansion | F3/F5; within-request F4; owned M3-M12 integrations; A3/A4 | Qualified shared fast paths and additional models with parity, recipes, and hardware evidence. |
| 3. Coherent serving release | Cross-request F4; F8; A6/A7 | Bounded compatible batching, serving/episode provenance, independent reproduction, and joint task results. |

Models can progress independently once prerequisites are ready. Select the next
integration by contributor ownership and validated readiness rather than promising
every candidate in one release. F9 and unowned M13 candidates stay exploratory
until their first deliverable is scoped.

Track performance against strong baselines, validated model/modality coverage,
the effort to add a model, and independent reproductions. Report variability and
negative results; faster kernels alone do not establish faster or more reliable
agents.

[o9]: https://github.com/ThinkFlowLab/system1-omni/issues/9
[o10]: https://github.com/ThinkFlowLab/system1-omni/issues/10
[o27]: https://github.com/ThinkFlowLab/system1-omni/issues/27
[o28]: https://github.com/ThinkFlowLab/system1-omni/pull/28
[o29]: https://github.com/ThinkFlowLab/system1-omni/pull/29
[o30]: https://github.com/ThinkFlowLab/system1-omni/pull/30
[o31]: https://github.com/ThinkFlowLab/system1-omni/issues/31
[o32]: https://github.com/ThinkFlowLab/system1-omni/pull/32
[o34]: https://github.com/ThinkFlowLab/system1-omni/pull/34
[o35]: https://github.com/ThinkFlowLab/system1-omni/pull/35
[o39]: https://github.com/ThinkFlowLab/system1-omni/issues/39
[o55]: https://github.com/ThinkFlowLab/system1-omni/pull/55
[o57]: https://github.com/ThinkFlowLab/system1-omni/issues/57
[o61]: https://github.com/ThinkFlowLab/system1-omni/issues/61
[o64]: https://github.com/ThinkFlowLab/system1-omni/pull/64
[o66]: https://github.com/ThinkFlowLab/system1-omni/issues/66
[o69]: https://github.com/ThinkFlowLab/system1-omni/pull/69
[o71]: https://github.com/ThinkFlowLab/system1-omni/pull/71
[o72]: https://github.com/ThinkFlowLab/system1-omni/pull/72
[o73]: https://github.com/ThinkFlowLab/system1-omni/pull/73
[o74]: https://github.com/ThinkFlowLab/system1-omni/pull/74
[o77]: https://github.com/ThinkFlowLab/system1-omni/pull/77
[o82]: https://github.com/ThinkFlowLab/system1-omni/pull/82
[o83]: https://github.com/ThinkFlowLab/system1-omni/issues/83
[o84]: https://github.com/ThinkFlowLab/system1-omni/issues/84
[o85]: https://github.com/ThinkFlowLab/system1-omni/issues/85
[o87]: https://github.com/ThinkFlowLab/system1-omni/issues/87
[o90]: https://github.com/ThinkFlowLab/system1-omni/pull/90
[o92]: https://github.com/ThinkFlowLab/system1-omni/pull/92
[o93]: https://github.com/ThinkFlowLab/system1-omni/pull/93
[o94]: https://github.com/ThinkFlowLab/system1-omni/pull/94
[o95]: https://github.com/ThinkFlowLab/system1-omni/issues/95
[o96]: https://github.com/ThinkFlowLab/system1-omni/pull/96
[o97]: https://github.com/ThinkFlowLab/system1-omni/pull/97
[o99]: https://github.com/ThinkFlowLab/system1-omni/pull/99
[o100]: https://github.com/ThinkFlowLab/system1-omni/pull/100
[o101]: https://github.com/ThinkFlowLab/system1-omni/pull/101
[o102]: https://github.com/ThinkFlowLab/system1-omni/pull/102
[o103]: https://github.com/ThinkFlowLab/system1-omni/pull/103
[o106]: https://github.com/ThinkFlowLab/system1-omni/pull/106
[o108]: https://github.com/ThinkFlowLab/system1-omni/pull/108
[o109]: https://github.com/ThinkFlowLab/system1-omni/pull/109
[o110]: https://github.com/ThinkFlowLab/system1-omni/pull/110
[o111]: https://github.com/ThinkFlowLab/system1-omni/pull/111
[o112]: https://github.com/ThinkFlowLab/system1-omni/pull/112
[o113]: https://github.com/ThinkFlowLab/system1-omni/pull/113
[o114]: https://github.com/ThinkFlowLab/system1-omni/issues/114
[o115]: https://github.com/ThinkFlowLab/system1-omni/pull/115
[o116]: https://github.com/ThinkFlowLab/system1-omni/pull/116
[o117]: https://github.com/ThinkFlowLab/system1-omni/pull/117
[o118]: https://github.com/ThinkFlowLab/system1-omni/pull/118
[o120]: https://github.com/ThinkFlowLab/system1-omni/pull/120
[o121]: https://github.com/ThinkFlowLab/system1-omni/issues/121
[o122]: https://github.com/ThinkFlowLab/system1-omni/pull/122
[o123]: https://github.com/ThinkFlowLab/system1-omni/pull/123
[o124]: https://github.com/ThinkFlowLab/system1-omni/pull/124
[a13]: https://github.com/ThinkFlowLab/system1-agents/issues/13
[a14]: https://github.com/ThinkFlowLab/system1-agents/issues/14
[a18]: https://github.com/ThinkFlowLab/system1-agents/pull/18
[a24]: https://github.com/ThinkFlowLab/system1-agents/issues/24
[a25]: https://github.com/ThinkFlowLab/system1-agents/issues/25
[a30]: https://github.com/ThinkFlowLab/system1-agents/issues/30
[a32]: https://github.com/ThinkFlowLab/system1-agents/pull/32
[a33]: https://github.com/ThinkFlowLab/system1-agents/pull/33
[a35]: https://github.com/ThinkFlowLab/system1-agents/pull/35
[a36]: https://github.com/ThinkFlowLab/system1-agents/pull/36
[a44]: https://github.com/ThinkFlowLab/system1-agents/issues/44
[a46]: https://github.com/ThinkFlowLab/system1-agents/pull/46
[a52]: https://github.com/ThinkFlowLab/system1-agents/pull/52
[a53]: https://github.com/ThinkFlowLab/system1-agents/pull/53
