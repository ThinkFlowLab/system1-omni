# LFM2.5-350M constrained scoring recipe

Runs the in-repo LFM2.5-350M hybrid-state candidate engine behind the Rust
frontend, and provides real-GPU verification and benchmarking scripts. The
engine pre-fills a question once, forks both attention KV and convolution
history per candidate batch, and scores complete JSON values by summed
full-vocabulary conditional log probabilities.

This recipe covers text `state` and `choice` questions only. Each public
question is evaluated independently; only candidates within one question share a
prefill. Question IDs are never placed in the prompt.

## Pinned artifacts

| Artifact | Pin |
| --- | --- |
| Base model | `LiquidAI/LFM2.5-350M@9e6c6ccf47cd318696e137d381a7ded8fe4df09f` |
| Reference engine | `notnotsamuel/LFM2.5-350M-RLCD@deb589d803d141cabd158ef55f6617b128529f36` (`rlcd/engine.py`) |

Reference code is loaded with `importlib` from a local checkout; no code is
re-downloaded during a run. Fetch only the code files:

```sh
hf download notnotsamuel/LFM2.5-350M-RLCD \
  --revision deb589d803d141cabd158ef55f6617b128529f36 \
  --include 'rlcd/*' --local-dir .local/lfm/reference
```

## Setup

The recorded Linux CUDA environment uses Python 3.12 with the full dependency
lock in `src/models/lfm2/requirements.lock.txt` (PyTorch 2.14.0, Transformers 5.17.0,
`huggingface-hub` 1.31.0, `jsonschema` 4.26.0). Install the pinned file into an
isolated environment, for example:

```sh
python3.12 -m venv .venv-lfm
.venv-lfm/bin/python -m pip install -r src/models/lfm2/requirements.lock.txt
source .venv-lfm/bin/activate
```

Python, PyTorch, Transformers and CUDA versions are recorded in each manifest. Do not
assume the `candidate_batch_size` is a fixed memory budget: it bounds simultaneous
branches, not bytes.

## Run the worker and frontend

Run the worker from the directory that contains `engine.py` and `worker.py`:

```sh
cd src/models/lfm2
python worker.py --candidate-batch-size 16 --device cuda --host 127.0.0.1 --port 8000
```

Then, from the repository root:

```sh
cargo build --release --locked
OMNI_JEV_BIND=127.0.0.1:8080 OMNI_JEV_BACKEND_URL=http://127.0.0.1:8000 \
  ./target/release/omni-jev
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "lfm2.5-350m",
  "state": "I was charged twice. Please refund the duplicate charge. No hurry.",
  "questions": {"refund": {"type": "choice", "instructions": "Does the customer ask for a refund?",
    "criteria": {"yes": "A refund is requested", "no": "No refund is requested"}}}
}'
```

`confidence` is required in the response but is normalized-entropy concentration
of the candidate distribution, not a calibrated correctness probability and not
the proprietary Jev formula. No calibration evaluation is included.
`usage.input_tokens` sums the prompt tokens of each independently pre-filled
question; there is no cross-question shared prefill or cache.

## Verify on a real GPU

`verify.py` loads one set of read-only weights, loads the pinned reference engine
with `importlib`, and builds an independent uncached oracle by concatenating
`prefix + suffix + value` and running a full `use_cache=False` forward with FP32
full-vocabulary log-softmax. The candidate scorer is never used as its own
oracle. The reference engine file is checked against its pinned SHA-256.

```sh
python recipe/lfm2/verify.py \
  --reference-dir .local/lfm/reference \
  --device cuda --output recipe/lfm2/verify_report.json
```

Required checks, all of which must pass for exit code 0:

- `score_parity`: on the pinned `SUPPORT`, `SENTIMENT`, `ROUTING` and stress
  schemas, raw scores within 0.15 of the oracle and identical selections.
- `cache_isolation`: the forked cache keeps exactly 6 attention and 10
  convolution layers, cached token distributions match the full forward at
  `atol=0.015`, `rtol=0.08`, argmax is equal, and the shared cache is unchanged.
- `api_multi_question`: the same `q1` answer holds when the question is alone,
  combined, reordered, or renamed; `usage.input_tokens` is the sum of the
  questions' prompt tokens; health and `POST /v1/systemone` match direct calls.
- `near_ties`: a fixed set of ambiguous probes reports measured margins (the
  minimum observed margin and any selection flips at every batch size; a near tie is never assumed).
- Candidate batch sizes `1/8/16/32/64/all` are consistent for every case.

The JSON report stores raw scores, maximum errors, selections, token ids,
margins, the execution environment and source hashes. Gold labels from the
pinned diagnostics are hand-authored and are reported individually only; they
are not an aggregate accuracy estimate. A failed run writes a `fail` report and
exits non-zero.

## Benchmark on a real GPU

`bench.py` calls `Engine.score` directly with internal schemas (a fixed `answer`
string enum or a raw multi-field schema). It serializes no request and never
touches the public worker path, so its numbers are internal-scorer numbers. The
workflow matrix crosses candidate count `2/16/64/255` with short/long context and
short/mixed candidate lengths, using fixed synthetic distractor text. Actual token
counts are recorded per cell. A separate `multi_field` table can be added with
`--suites` and is never mixed with the string-enum table.

The uncached oracle intentionally computes logits for the complete input,
including prefix positions. It is a correctness baseline, not a tuned
uncached serving implementation; its timing includes that output-projection
cost. Compare full versus bounded branching directly when attributing the
batching trade-off.

The recipe reports three distinct measurements and never combines them:

- **Internal scorer**: `bench.py`, and `verify.py` `score_parity`, call
  `Engine.score` in-process on internal schemas; no request is serialized.
- **Public worker path**: `bench_worker.py` runs the real `Worker.systemone`,
  including request/response JSON encoding and full response construction, with no HTTP
  transport. This is the public performance number.
- **Multi-process frontend**: `compare_with_frontend.py` only checks that the Rust
  frontend returns byte-identical status, `Content-Type` and body to the worker.
  It is a correctness check, not a latency measurement.

```sh
python recipe/lfm2/bench.py \
  --reference-dir .local/lfm/reference --device cuda --quick \
  --warmup 3 --samples 20 --output recipe/lfm2/bench_results.jsonl
```

Drop `--quick` for the full matrix. `--quick` results are labelled `quick` in the
manifest and summary and are never reported as full. Each measured sample
synchronizes before and after `perf_counter`, resets peak memory statistics, and
records latency plus allocated and reserved peaks; `p50`/`p95` and the raw sample
list are reported per cell and variant. Variant and cell order is rotated across
samples to avoid a fixed-order bias. Out-of-memory runs are recorded as `oom`;
numbers are never fabricated. Inference selections are checked against the
uncached oracle for every sample.

Results append to JSONL and can resume after an interruption:

```sh
python recipe/lfm2/bench.py --reference-dir .local/lfm/reference --device cuda \
  --resume --output recipe/lfm2/bench_results.jsonl
```

`bench_worker.py` loads one FP16 weight set and shares it across candidate batch
sizes. The matrix uses 16-candidate choice questions at question count `1/3/8`,
short/long state and candidate batch `1/8/all`. Every measured output is compared
with the same-config warmup answer (choice and all probabilities within `1e-6`),
`usage.input_tokens` is checked against the sum of independently measured
per-question prompt tokens, and question ids are checked to stay out of the
prompt. Raw samples append as JSONL, with `p50`/`p95` and allocated/reserved peaks
in the summary.

```sh
python recipe/lfm2/bench_worker.py --device cuda \
  --warmup 3 --samples 20 --output recipe/lfm2/bench_worker_results.jsonl
```

A sidecar manifest records GPU, driver, Python, PyTorch, Transformers, model and
reference pins, and source file hashes. The summary is written atomically next to
the JSONL. The measured outcomes and limitations are in [RESULTS.md](RESULTS.md).

## Compare worker and frontend

`compare_with_frontend.py` sends health, valid choice, stability, unicode and
malformed/invalid requests to the worker directly and through a running Rust
frontend, then compares status, `Content-Type` and raw body bytes. It never starts
or stops either service and talks only to loopback URLs.

```sh
python recipe/lfm2/compare_with_frontend.py \
  --worker http://127.0.0.1:8000 --frontend http://127.0.0.1:8080 \
  --output recipe/lfm2/compare_frontend.json
```

## Files

| File | Role |
| --- | --- |
| `verify.py` | Real-GPU correctness verification and JSON report. |
| `bench.py` | Real-GPU latency/memory benchmark of the internal `Engine.score` and JSONL plus summary. |
| `bench_worker.py` | Real-GPU latency/memory benchmark of the public `Worker.systemone` path and JSONL plus summary. |
| `compare_with_frontend.py` | Worker/frontend byte-parity checks against running loopback services; not a benchmark. |
| `_harness.py` | Test-only shared loading, oracle and manifest helpers. |
| `_diagnostics.py` | Pinned diagnostic schemas/cases fixture (not a benchmark). |

## Recorded validation

[RESULTS.md](RESULTS.md) contains the complete L40S matrix, correctness checks,
public worker results and measured latency/memory trade-offs. The [result
bundle](results/l40s-20260928/README.md) preserves all raw samples, reports,
source hashes, required-check logs and the original environment freeze.

From the repository root, run the CPU tests with the same environment:
```sh
python -m pytest -q src/models/lfm2/tests
```
The pinned reference score gate is empirical on the listed inputs; it is not a
uniform FP16 error bound or a calibrated model-quality claim.
