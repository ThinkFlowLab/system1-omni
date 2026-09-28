# LFM2.5-350M worker

Standalone Python worker for System1-Omni choice questions. It runs
`POST /v1/systemone` and `GET /health` behind the Rust frontend and serves the
fixed model alias `lfm2.5-350m` (base `LiquidAI/LFM2.5-350M` at revision
`9e6c6ccf47cd318696e137d381a7ded8fe4df09f`). Requests are handled serially on
one GPU; there is no request framework, only the standard-library HTTP server.

## Model ownership and attribution

The engine is adapted from the RLCD reference implementation
[`notnotsamuel/LFM2.5-350M-RLCD`](https://huggingface.co/notnotsamuel/LFM2.5-350M-RLCD)
(MIT, Copyright (c) 2026 notnotsamuel) and keeps its license notice in
[LICENSE.reference](LICENSE.reference). The downloaded base weights are unchanged from `LiquidAI/LFM2.5-350M`
and stay under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-350M).
Inference only: no training, fine-tuning or parameter updates.

## Run

From the repository root, install the pinned requirements and start the worker:

```sh
python -m pip install -r src/models/lfm2/requirements.lock.txt
python src/models/lfm2/worker.py --candidate-batch-size 8 --device cuda --dtype float16 \
  --host 127.0.0.1 --port 8000
```

`--candidate-batch-size` is required and must be a positive integer; there is no
default. The remaining flags default to the values shown. The first start
downloads the pinned checkpoint and tokenizer.

Health:

```sh
curl http://127.0.0.1:8000/health
# {"status": "ok"}
```

## Request

`state` is a non-empty text string and `questions` is a non-empty object of
choice questions. Each question has a required `type: "choice"`, `instructions`
(string, object or array), and `criteria` (object with 1..255 keys; each value a
string, object, array or null). The criterion keys are the candidate answers and
their original order is preserved.

```sh
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "lfm2.5-350m",
    "state": "I was charged twice. Please refund the duplicate charge.",
    "questions": {
      "refund": {
        "type": "choice",
        "instructions": "Does the customer ask for a refund?",
        "criteria": {"yes": "asks for a refund", "no": "does not"}
      }
    }
  }'
```

Only `choice` is supported. `score` and `noul` questions are rejected with 422;
every part of the request is validated before any GPU work starts.

## Response

Illustrative values, not measured model output:

```json
{
  "model": "LiquidAI/LFM2.5-350M@9e6c6ccf47cd318696e137d381a7ded8fe4df09f",
  "answers": {
    "refund": {
      "type": "choice",
      "choice": "yes",
      "probabilities": {"yes": 0.72, "no": 0.28},
      "confidence": 0.1445
    }
  },
  "usage": {"input_tokens": 214, "output_tokens": 0}
}
```

`probabilities` is `softmax` over each candidate's full-sequence log-likelihood
sum, returned as a map from each criterion key to its probability. `choice` is the highest score, with
ties resolved in favour of the first-listed criterion. `confidence` is
`1 - H / ln(n)` for `n > 1` and `1.0` for a single candidate; it is a heuristic,
not calibrated. `usage.input_tokens` is the **prompt token count only** — the sum
of per-question prefix lengths. It is not the number of forward passes or
candidate tokens; those are reported separately by the engine's `telemetry`
field (`branches`, `branch_tokens_padded`, `forward_calls`).

These scores depend on candidate length, wording and tokenization. Softmax only
compares the candidates supplied in that question; neither it nor the entropy
statistic estimates semantic correctness or reproduces Jev calibration.

Each question is prefilled independently, so questions never share a KV/conv
cache. Within a question, all candidates share one prefill and are scored in
batches of `--candidate-batch-size`; each batch forks the prefill cache and the
original cache is never mutated.

## Errors

| Status | Cause |
| --- | --- |
| 400 | malformed JSON, duplicate object keys, `NaN`/`Infinity` |
| 422 | wrong model, type, shape, or an unsupported `score`/`noul` question |
| 500 | internal failure; the response never contains a traceback |

## Method

The prompt writes the complete JSON schema in the system message and the context
in the user message, then appends the assistant opening `{\n`. For each
criterion, a field suffix and the JSON-quoted value plus trailing newline are
encoded separately and scored as a full sequence with an FP32 full-vocabulary
`log_softmax` sum and no length normalization. The pinned LFM2 model has six
attention layers and ten short-convolution layers; forking deep-copies the cache
and calls `reorder_cache`, which copies both KV tensors and convolution state via
`index_select` (independent storage, no shared mutable views).

## Tests

```sh
python -m pytest src/models/lfm2/tests
```

`tests/test_engine.py` builds a tiny randomly initialized `Lfm2ForCausalLM` on
CPU (no download) and checks hybrid-cache isolation and that batched scores and
argmax match an independent uncached oracle. `tests/test_worker.py` uses a fake
engine and an in-process loopback server to check validation, prompt
independence from batching/ordering/renaming, response fields and the CLI.
