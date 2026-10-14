# Weekly local GPU regression suite

Run `benchmarks/weekly.py` on a GPU machine to maintain model accuracy and warm
end-to-end HTTP latency. GitHub Actions validates the frozen inputs and exercises
the runner with a CPU mock service; it does not claim model/GPU coverage.

The runner reserves exact device IDs with `gpu run`, starts the configured model
worker and Rust frontend, waits for health, validates real inference, then runs
one complete feasibility pass and **two measured passes** on the same services.
It stops a model on startup, warmup or request failure and preserves raw results.
It terminates only service process groups it created and the scheduler releases
the reservation. It never clears shared caches or overwrites previous results.

## Prepare once

Reuse a Python 3.11+ environment with `benchmarks/requirements.txt`, the host's
verified `gpu` scheduler, and `nvidia-smi`. Prepare checkpoint exports and build
the worker/frontend from the source being tested using each model's recipe.
Preparation, downloads, builds and image preencoding are outside measured latency.
Native LAYA requires Hopper and cannot run on an Ada GPU; use its actual Python
CUDA serving path where available. CLM must use a real encoder, not the stub.

Copy `suite.example.json` to `.local/e2e/suite.json`. Every coverage row initially
has an explicit `unavailable` reason. To enable a path, remove that field and add
its serving configuration, for example:

```json
{
  "id": "open-jev-27b",
  "manifest": "benchmarks/e2e/text.jsonl",
  "model": "open-jev-27b-v1.1",
  "protocol": "systemone",
  "health": "http://127.0.0.1:18280/health",
  "endpoint": "http://127.0.0.1:18280/v1/systemone",
  "commands": [["target/release/omni-open-jev-native"], ["target/release/omni-jev"]],
  "env": {
    "OPEN_JEV_MODEL": "/absolute/path/to/pinned/merged-export",
    "OPEN_JEV_PORT": "18200",
    "OPEN_JEV_CUDA_LIB": "/absolute/path/to/libqwen3_5_cuda.so",
    "CUA_S1_GRAPH": "0",
    "OMNI_JEV_BIND": "127.0.0.1:18280",
    "OMNI_JEV_BACKEND_URL": "http://127.0.0.1:18200"
  },
  "metadata": {
    "gpu": "actual GPU model",
    "gpu_ids": [0],
    "driver": "actual driver version",
    "cuda": "actual CUDA runtime/toolkit version",
    "precision": "bf16",
    "model_revision": "exact base + adapter + head revisions",
    "cache_policy": "warm; graph off; no shared-cache drops",
    "cuda_evidence": "verified CUDA launch configuration and evidence location"
  }
}
```

Commands are argument arrays executed from the repository root, without a shell.
Use a prepared Python interpreter's absolute path for Python workers. Add an
embeddings server command before the CLM service where needed. Use dedicated
loopback ports; the runner rejects an already responding health endpoint.
For a service with an additional encoder, set `readiness` to its ready URL list
(for example an embeddings server's `/v1/models`). All those services must be
ready before the first inference. Executable availability is checked before GPU
reservation. Input manifests are copied once before launching services.
Do not override `CUDA_VISIBLE_DEVICES`: the scheduler owns device assignment.
Record launch precision, model pins, cache settings and any CPU/NUMA binding in
the configuration. Keep controls fixed across weeks and note other machine load.
Hardware inventory, source SHA/patch/hashes, exact configuration, service logs,
startup time and raw request/response records are saved with each run. Metadata
is operator-supplied evidence; an HTTP response alone does not prove CUDA use.
Reserved device IDs, names and driver versions are checked against `nvidia-smi`;
GPU UUIDs, host and CPU affinity also participate in baseline compatibility.

Use `choice.jsonl` for choice-only Cua-S1 and Valen workers, and
`choice-text.jsonl` for LFM2, whose state must be a text string. Use
`text.jsonl` for choice/score/noul models. JEV-VL text uses `choice.jsonl`, model
`autotrust/JEV-27B-VL`, and `protocol: "jev-vl"`; the adapter sends its
single-question API and validates option order, selected index and probabilities.
Image paths require a separately frozen labelled manifest with inline image
state; the included text corpus does not validate vision. JEV-VL image encoding
is offline, so its serving latency cannot stand for live-image end-to-end latency.
OmniJev has no inference worker at this revision and remains unavailable.

## First baseline, then weekly checks

Before device work, declare the hypothesis, fixed controls, gates and run budget.
The default budget is one feasibility pass and two measured passes; startup or
request failure stops that model, and there are no automatic retries or extra
passes. The example gates allow no accuracy drop or decision flips, 0.02 absolute
probability/score drift and Score MAE increase, and 15% latency growth. These are
**editable policy examples, not empirically established universal tolerances**.
Choose thresholds before running and do not relax them after seeing results.

```sh
python benchmarks/weekly.py validate --suite .local/e2e/suite.json

# Verify scheduler availability, then create an immutable baseline candidate.
python benchmarks/weekly.py run --suite .local/e2e/suite.json \
  --gpu-ids 0 --wait 10m --timeout 2h --output .local/e2e/baseline

# After reviewing accuracy, parity evidence, failures and configuration:
python benchmarks/weekly.py run --suite .local/e2e/suite.json \
  --gpu-ids 0 --wait 10m --timeout 2h --baseline .local/e2e/baseline \
  --output .local/e2e/week-2026-43
```

Use a new output directory each week. `--models open-jev-27b,lfm2` deliberately
selects those paths and records all excluded coverage IDs in `suite.json`.
Omitting it checks the entire coverage matrix; unavailable entries make that
full suite incomplete. Add a newly prepared path through a new baseline run;
missing or incompatible baselines fail rather than silently resetting history.
The validation protocol and model Score precision must also match. Set
`score_decimals: 2` for Decider's documented two-decimal Score; the default is
four. The weighted Score check allows only the analytical rounding bound: half
a decimal step at the declared Score precision
plus each probability's half-step multiplied by its level index. A change to
this numerical protocol requires a new baseline rather than comparing summaries
produced under different validators.
Select the approved baseline explicitly; the runner never promotes one itself.
To test a newer source snapshot with the frozen harness, use `--repo` pointing
to an isolated checkout of fetched upstream `main`. Build its CUDA library and
workers before the run, use absolute frozen manifest paths, and point model
commands/PYTHONPATH/CUDA library paths to that checkout. Keep prepared weights
and external reference environments unchanged. The runtime SHA/patch belongs
to the tested checkout; the harness code is separately hashed and archived, and
its hashes must match the baseline. A weekly run of stale binaries does not
test new source changes.

Exit codes: **0** means every selected path passed against its baseline;
**1** means failure; **2** means baseline recorded or coverage unavailable.
An initial baseline records observed model quality and does not establish
correctness against an independent reference. Before accepting it, use the
model recipe's reference validation. Subsequent comparisons check probability
and decision parity against that frozen baseline as well as labelled quality.

`report.md` lists coverage; `report.json` contains both measured passes and
min/max variation. Per-model `regression.json` shows gate failures, baseline and
candidate metrics, denominators and parity. Feasibility and warmup are excluded
from reported measured latency. HTTP latency includes serialization, frontend,
worker processing, inference, response receipt and response validation. It is
neither kernel time nor startup time. A failed request cannot improve the score
by disappearing from the denominator: any request failure fails the path.
Latency checks require every new pass's p50/p95 to meet the baseline two-pass
mean plus the declared allowance. Inconclusive/noisy results stay visible; the
runner does not pick a winning pass or extend the experiment budget.

Set `reference_model` to another configured profile ID to require independent
implementation parity too, for example Cua-S1 native against its Python reference.
Both profiles must be included in the run. Equal aggregate accuracy alone does
not establish parity: flips and probability drift are checked on both measured
passes and saved in `reference-parity.json`. Any failed reference check fails the
path and the overall run. Complete measurements remain usable as an explicitly
selected observational baseline even when a quality gate failed; that selection
does not turn the prior failure into a pass or approve the model's correctness.

## Frozen corpus provenance

The maintained manifests are derived from the public `easy`, `original`, `hard`
files at JevBench commit `f8ce71361165846101d02ebc83ad44e47ae44fc3`, in that
tier/file order. `text.jsonl` contains all 231 records: 139 Choice, 74 Noul and
18 Score; `choice.jsonl` contains the 139 Choice records unchanged.
`choice-text.jsonl` preserves all 139 Choice records and labels, rendering the
18 structured states as compact JSON text (`ensure_ascii=False`, sorted keys,
separators `(',', ':')`); existing strings stay unchanged. No records are dropped
or truncated to fit a model. This is a different, explicitly frozen workload.
`images.jsonl` contains four deterministic 32×32 RGB PNGs (red, green, blue,
white), with colour labels defined by their pixel values before inference.
It is a small vision contract workload, with n=4 reported explicitly; it does
not establish general visual accuracy or production latency percentiles.
For JEV-VL, the adapter turns the image object into its required media parts
list. Feed that wire-format state to the preencoder too; a plain object is JSON
text in JEV-VL, not an image. Preencoded image serving remains a separate timing
boundary from live vision execution.

CLM's default CPU embedding LRU cache can hide encoder execution even with the
GPU projection cache disabled. The optional test launcher
[`clm_uncached_worker.py`](../../tests/e2e/clm_uncached_worker.py) constructs the
upstream CLM API with `Embedder(cache_size=0)` and `Engine(action_cache=0)`.
Use it with a real encoder and its readiness URL when testing encoder latency.
This suite covers CLM's external HTTP worker, not the native `clm-run` CLI.
Each original question becomes `questions.q`; its original state and question
content are preserved. Expected `yes`/`no` Noul labels become booleans; Choice
labels and Score level indices remain unchanged. Per-record provenance is kept.
The [MIT license](LICENSE.jevbench) accompanies the inputs.

These public authored diagnostic tasks are a reproducible regression workload,
not an unseen production-quality estimate. Labels are independent of model
outputs. A model's low baseline accuracy remains low accuracy; preserving it
does not prove the model is accurate. Add task-specific held-out and image
workloads before making broader quality claims.
