# GPU serving benchmark

First harness for #39: replay the same labelled requests against CUDA-backed
`laya-serve`, vllm-jev, and System1-Omni. Python 3.11+ and `httpx==0.28.1` are
required (`python -m pip install -r benchmarks/requirements.txt`). Reuse an
existing benchmark environment when available.

The runner is an HTTP client: it neither launches models nor reserves or accesses
a GPU. Start each server separately through the host's verified GPU scheduler,
using the same exact reserved device and pinned Laya checkpoint. Verify CUDA
execution from the worker's configuration/logs and scheduler evidence. An HTTP
response alone cannot prove the backend used CUDA.

## Freeze requests and controls

The JSONL format is one object per request:

```json
{"id":"sample-1","request":{"state":"Refund my duplicate charge.","questions":{"refund":{"type":"noul","instructions":"Does the customer request a refund?"}}},"expected":{"refund":true}}
```

Use a criteria map and a label target for Choice; an ordered criteria list and a
numeric level target for Score. Every question needs a target. The runner adds
only the backend model alias; all other request fields are replayed unchanged.
Keep source dataset revision, selected record IDs, sampling seed, transformation
code revision, token lengths and truncation decisions with the manifest. Freeze
short/near-limit and option-count slices as separate manifests.

`smoke.jsonl` contains four synthetic contract checks, including a mixed request.
Its scores are illustrative labels, not a general accuracy benchmark or a useful
throughput workload. Prepare a larger held-out manifest for measurements, such as
a fixed 500-question MMLU test sample. Dataset acquisition and conversion are not
implemented by this first runner.

```sh
python benchmarks/bench.py validate benchmarks/smoke.jsonl
```

Before measuring, save run metadata as JSON. Required fields are `gpu`, `gpu_ids`,
`driver`, `cuda`, `precision`, `model_revision`, `runtime_revision`, `cache_policy`,
`cuda_evidence`, and `reservation`. These are operator-supplied evidence, not
automatically verified hardware measurements. Use exact revisions and include
working-tree patches, dependency versions, checkpoint configuration, tokenizer,
temperatures, affinity, launch commands and preparation/readiness timings.
Record peak GPU memory and its measurement method externally under the task's
reservation; the client does not sample GPU memory.

Fix precision, context/truncation policy, inputs and cache conditions across
backends. Record necessary differences. Keep cold-cache experiments separate;
this runner always performs readiness inference and warmup and thus measures
warmed serving. It does not reset caches. Never drop shared host caches.

## Run against a ready GPU server

```sh
python benchmarks/bench.py run requests.jsonl \
  --endpoint http://127.0.0.1:8000/v1/systemone --model english \
  --metadata run-metadata.json --phase feasibility \
  --concurrency 1 --output /tmp/laya-feasibility

python benchmarks/bench.py run requests.jsonl \
  --endpoint http://127.0.0.1:8000/v1/systemone --model english \
  --metadata run-metadata.json --phase measured \
  --concurrency 1 --output /tmp/laya-c1-run1
```

Use the checkpoint's served alias for vllm-jev/native implementations. For the
frontend-overhead comparison, reuse the same external CUDA worker and change
only the endpoint. Optional authentication comes from `OMNI_JEV_TEST_TOKEN`;
the token is not saved. Redirects, proxy environment variables and retries are
disabled. `--timeout` is a total deadline for each HTTP exchange, including body
receipt; response validation is included in recorded client latency.

Start with the weight-free preflight, then one feasibility run per configuration.
Exclude feasibility from performance comparisons. Use two measured runs at each
of concurrency 1, 8 and 16, with the same request ordering. Reuse the server;
report variation rather than silently extending an inconclusive run budget.
Use separate output directories; existing results are never overwritten.

Concurrency is a closed-loop count of in-flight HTTP requests. Fixed workers
consume the manifest in order; completion order is naturally variable. Each
worker sends its next request only when the previous one completes. There is no
arrival-rate simulation or client queue-delay measurement.

Outputs:

- `config.json`: manifest/runner SHA256, runtime metadata and client settings.
- `requests.jsonl`: exact manifest snapshot.
- `warmup.json`: first-inference readiness check and warmup responses, excluded
  from load timing. Warmup failure stops the run.
- `responses.jsonl`: one record per input, in input order, with raw response,
  HTTP status, client latency, validated answers and any error.
- `summary.json`: successful requests/s and decisions/s, successful-request
  p50/p95 latency (nearest-rank p95), failure counts by kind, and quality metrics
  with denominators. A decision is a question, not an internal candidate branch.

Any request error makes the run exit nonzero after saving results. Malformed
responses count as failures. Quality and latency summaries cover successful
requests only; failure latencies remain in raw results. Always publish failure
counts alongside those summaries. Different failure rates invalidate a simple
speedup claim.

## Compare output fidelity

Predeclare tolerances; these flags are illustrative, not established merge gates:

```sh
python benchmarks/bench.py compare /tmp/reference /tmp/candidate \
  --max-probability-drift 0.002 --max-score-drift 0.002 --max-flips 0
```

Comparison requires identical manifests and covers all request IDs. It fails on
request errors or exceeded tolerances, reporting probability/Score drift and
Choice/Noul flips separately from ground-truth quality. This is an output-fidelity
check, not automatic proof of equivalent hardware or a performance winner.

## Self-review and A/B evidence

Use the repository's [self-review skill](../.agents/skills/self-review/SKILL.md)
with its contribution guide and PR template. Decide whether measurements are
needed before starting GPU work:

- Performance or accuracy improvement claims require reproducible comparison
  evidence against a relevant reference.
- For scheduler, batching, cache, kernel, precision, model or serving changes,
  identify a concrete performance/numerical risk and choose a focused comparison
  when it can resolve that risk. Output-affecting changes need reference-output
  checks and applicable labelled quality; equal accuracy alone is not parity.
- Documentation, unrelated changes and benchmark tooling without performance
  claims do not automatically need GPU experiments. Run the applicable checks
  and state why an A/B test is unnecessary.

Freeze revisions, the isolated variable, controls, success criteria and stop
condition before execution. Follow the run budget above: one feasibility run,
then two measured repetitions per configuration unless a different budget was
predeclared. Preserve every result and report variation. Reuse prior evidence
only if its revisions, inputs and controls cover the reviewed change.

Review errors and metric denominators alongside latency and throughput. Keep
SDK batch timing separate from HTTP latency and frontend overhead separate from
native acceleration. Report missing measurements as unverified, and remove or
qualify unsupported claims. Stop at the declared error/run limit; do not silently
relax tolerances, discard failures or add runs to obtain a preferred outcome.

### Known validator limitation

The initial GPU run at `9ae70ce` rejected two rounded probability distributions
summing to `0.9998999999999999`: floating-point roundoff put their deviation just
outside `abs_tol=1e-4`. See the [run report on PR #40](https://github.com/ThinkFlowLab/system1-omni/pull/40#issuecomment-5894999739).
That run stopped before completing the A/B comparison. Check for a fix and
regression tests before relying on this runner for full measurements. A validator
change needs a recorded revised protocol; partial results are not a validated
performance baseline.

## Remaining work in #39

Completed GPU comparisons, a checked-in frozen MMLU manifest, automated memory collection, NLL/Brier
metrics, aggregated repeated-run reports and a separate offline SDK comparison
remain to be delivered. Reuse the upstream [Laya batch benchmark] and
[evaluation harness]. SDK batch time must not be mixed with HTTP latency, and
batch time divided by size is not request latency. A frontend/external-worker
result is not a native-engine speedup. Metal and vision benchmarks follow CUDA.

[Laya batch benchmark]: https://github.com/NandhaKishorM/laya/blob/main/benchmarks/bench_predict_batch.py
[evaluation harness]: https://github.com/NandhaKishorM/laya/blob/main/docs/evals.md

## Tests (no model or GPU)

```sh
python -m unittest discover -s tests/benchmarks -p 'test_*.py' -v
```
